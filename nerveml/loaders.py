"""Dataset intake and the integrity scan that runs before any measurement.

Two categories, deliberately separated. Conditions that would make the scan
meaningless raise immediately - there is no useful subject-held-out number to
report when there is only one subject. Conditions that merely weaken the result
become warnings that travel into the report alongside the metrics.
"""

from dataclasses import dataclass, field

import pandas as pd

from nerveml.validation import SUBJECT_COLUMN, TARGET_COLUMN

TRIAL_COLUMN = "trial_id"
SESSION_COLUMN = "session_id"
RESERVED_COLUMNS = (
    SUBJECT_COLUMN,
    TRIAL_COLUMN,
    TARGET_COLUMN,
    SESSION_COLUMN,
)

# Below this minority-class share the balanced-accuracy folds get very noisy.
IMBALANCE_THRESHOLD = 0.20

# ── Leakage-smell detection ───────────────────────────────────────────────────
# Column names (case-insensitive exact match or substring) that suggest the
# column encodes the label rather than a neural feature.  The DEAP/SEED loaders
# already exclude these, but generic CSVs may accidentally include them.
# Kept narrow (high-precision > high-recall) to avoid noisy false positives on
# legitimate EEG feature names like "complexity_score" or "response_amplitude".
_LABEL_SMELL_KEYWORDS = frozenset({
    # Affective/emotion targets (DEAP, SEED, MAHNOB, AMIGOS, …)
    "valence", "arousal", "dominance", "liking",
    "emotion", "emotion_label",
    "mood", "affect", "sentiment",
    # Physiological-state targets used as secondary sensitive attributes
    "sleep_stage", "sleep_phase",
    "motor_state",
    "fatigue", "stress", "pain", "anxiety", "depression",
    "medication",
    # Generic label-column name patterns
    "target_label", "class_label", "emotion_class",
})

# Point-biserial |r| threshold above which we warn about label-feature leakage.
CORRELATION_LEAKAGE_THRESHOLD = 0.95

# ── DEAP-style loader constants ───────────────────────────────────────────────
# DEAP (Dataset for Emotion Analysis using Physiological signals) uses 1-9
# continuous ratings. Subject columns vary across public releases.
_DEAP_SUBJECT_CANDIDATES = ("participant_id", "participant", "subject_id", "subject")
_DEAP_LABEL_COLUMNS = ("valence", "arousal", "dominance", "liking")
_DEAP_LABEL_LOWER = frozenset(c.lower() for c in _DEAP_LABEL_COLUMNS)
_RESERVED_LOWER = frozenset(c.lower() for c in RESERVED_COLUMNS)
_DEAP_EXCLUDE_LOWER = _DEAP_LABEL_LOWER | _RESERVED_LOWER

# ── SEED-style loader constants ───────────────────────────────────────────────
# SJTU Emotion EEG Dataset. 3-class emotion (negative/neutral/positive).
# SEED-I encodes as -1/0/1; some derived feature tables use 0/1/2.
_SEED_SUBJECT_CANDIDATES = ("sub_id", "subject_id", "subject", "participant")
_SEED_EMOTION_CANDIDATES = ("emotion", "emotion_label", "label")


@dataclass
class DatasetSummary:
    n_subjects: int
    n_trials: int
    n_features: int
    class_balance: dict
    missing_values: int
    duplicate_trial_ids: int
    trials_per_subject: dict
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "n_subjects": self.n_subjects,
            "n_trials": self.n_trials,
            "n_features": self.n_features,
            "class_balance": {str(k): int(v) for k, v in self.class_balance.items()},
            "missing_values": self.missing_values,
            "duplicate_trial_ids": self.duplicate_trial_ids,
            "min_trials_per_subject": min(self.trials_per_subject.values()),
            "max_trials_per_subject": max(self.trials_per_subject.values()),
            "warnings": self.warnings,
        }


def _warning(code, message):
    return {"code": code, "message": message}


def _leaky_label_names(feature_columns):
    """Return feature column names whose lowercased form exactly matches or
    contains a known label-vocabulary keyword.

    Exact-match catches "valence", "emotion", etc.  Substring search catches
    "valence_raw", "norm_arousal", "arousal_score", etc.  Any hit is a strong
    indicator that a label column was accidentally included in the feature set.
    """
    hits = []
    for col in feature_columns:
        lower = col.lower()
        if any(kw == lower or kw in lower for kw in _LABEL_SMELL_KEYWORDS):
            hits.append(col)
    return hits


def _high_correlation_features(df, feature_columns):
    """Return a list of (column, abs_r) pairs where |point-biserial r| with
    the binary target exceeds CORRELATION_LEAKAGE_THRESHOLD.

    We use the algebraic equivalence between point-biserial r and Pearson r
    on a 0/1-encoded binary variable, computing efficiently via `.corr()`.
    Constant features (std=0) are silently skipped — they can't be leaky.
    """
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    hits = []
    for col in feature_columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.std() == 0 or target.std() == 0:
            continue
        r = vals.corr(target)
        if pd.notna(r) and abs(r) >= CORRELATION_LEAKAGE_THRESHOLD:
            hits.append((col, round(abs(r), 4)))
    return hits


def validate_dataset(df, feature_columns, n_splits=5):
    """Inspect a dataset before scanning it. Raises when a scan is pointless."""
    for column in (SUBJECT_COLUMN, TARGET_COLUMN):
        if column not in df.columns:
            raise ValueError(f"required column {column!r} is missing")

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"feature columns not present in data: {missing_features}")

    classes = sorted(df[TARGET_COLUMN].dropna().unique())
    if len(classes) < 2:
        raise ValueError(
            f"{TARGET_COLUMN} must contain two classes, found {len(classes)}"
        )
    if len(classes) > 2:
        raise ValueError(
            f"{TARGET_COLUMN} must be binary in this prototype, found "
            f"{len(classes)} distinct values"
        )

    n_subjects = df[SUBJECT_COLUMN].nunique()
    if n_subjects < 2:
        raise ValueError(
            f"subject-held-out evaluation needs at least 2 subjects, found "
            f"{n_subjects}; without an independent grouping variable this scan "
            "cannot distinguish memorisation from generalisation"
        )

    counts = df[TARGET_COLUMN].value_counts()
    class_balance = {k: int(v) for k, v in counts.sort_index().items()}
    missing_values = int(df[feature_columns].isna().sum().sum())
    duplicate_trial_ids = (
        int(df[TRIAL_COLUMN].duplicated().sum()) if TRIAL_COLUMN in df.columns else 0
    )

    warnings = []
    if missing_values:
        warnings.append(
            _warning(
                "missing_values",
                f"{missing_values} missing feature values; imputation will be "
                "fitted inside training folds only.",
            )
        )
    if duplicate_trial_ids:
        warnings.append(
            _warning(
                "duplicate_trial_ids",
                f"{duplicate_trial_ids} duplicated trial ids; repeated rows can "
                "place the same measurement on both sides of a split.",
            )
        )
    if n_subjects < n_splits:
        warnings.append(
            _warning(
                "too_few_subjects",
                f"{n_subjects} subjects for {n_splits} grouped folds; fold count "
                "will be reduced and estimates will be unstable.",
            )
        )

    minority_share = counts.min() / len(df)
    if minority_share < IMBALANCE_THRESHOLD:
        warnings.append(
            _warning(
                "class_imbalance",
                f"minority class is {minority_share:.1%} of trials; prefer "
                "balanced accuracy and ROC-AUC over raw accuracy.",
            )
        )

    per_subject_classes = df.groupby(SUBJECT_COLUMN)[TARGET_COLUMN].nunique()
    single_class = sorted(per_subject_classes[per_subject_classes < 2].index)
    if single_class:
        warnings.append(
            _warning(
                "single_class_subjects",
                f"{len(single_class)} subject(s) contain only one class "
                f"({', '.join(map(str, single_class[:5]))}); their held-out folds "
                "cannot produce a ROC-AUC.",
            )
        )

    # Leakage-smell: column names that match known label vocabulary.
    label_name_hits = _leaky_label_names(feature_columns)
    if label_name_hits:
        listed = ", ".join(f"'{c}'" for c in label_name_hits[:5])
        tail = f" (and {len(label_name_hits) - 5} more)" if len(label_name_hits) > 5 else ""
        warnings.append(
            _warning(
                "label_name_in_features",
                f"Feature column(s) {listed}{tail} match known label vocabulary. "
                "If these columns encode the classification target rather than a "
                "neural signal, the scan will overestimate decodability "
                "(label leakage — prototype heuristic, verify manually).",
            )
        )

    # Leakage-smell: features that are nearly perfectly correlated with the target.
    corr_hits = _high_correlation_features(df, feature_columns)
    if corr_hits:
        listed = ", ".join(f"'{c}' (|r|={r})" for c, r in corr_hits[:3])
        tail = f" (and {len(corr_hits) - 3} more)" if len(corr_hits) > 3 else ""
        warnings.append(
            _warning(
                "feature_label_high_correlation",
                f"Feature(s) {listed}{tail} have near-perfect correlation with the "
                f"target (|r| ≥ {CORRELATION_LEAKAGE_THRESHOLD}). This is a strong "
                "indicator of label leakage: the feature may encode the label "
                "directly rather than a neural correlate "
                "(prototype heuristic, verify manually).",
            )
        )

    return DatasetSummary(
        n_subjects=n_subjects,
        n_trials=len(df),
        n_features=len(feature_columns),
        class_balance=class_balance,
        missing_values=missing_values,
        duplicate_trial_ids=duplicate_trial_ids,
        trials_per_subject=df[SUBJECT_COLUMN].value_counts().to_dict(),
        warnings=warnings,
    )


def load_feature_csv(path, feature_columns=None):
    """Load a precomputed feature table.

    This is the generic intake path: any dataset reduced to one row per trial
    with subject_id, trial_id and target_label works here, whether it came from
    DEAP, a customer export, or the synthetic generator.
    """
    df = pd.read_csv(path)

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in RESERVED_COLUMNS]
    else:
        unknown = [c for c in feature_columns if c not in df.columns]
        if unknown:
            raise ValueError(f"feature columns not present in {path}: {unknown}")

    return df, list(feature_columns)


# ── shared helpers for format-specific loaders ────────────────────────────────


def _match_column(df, candidates):
    """Return the first candidate name found in df, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        found = lower_map.get(cand.lower())
        if found is not None:
            return found
    return None


def _synthesize_trial_id(df, subject_col):
    """Return a copy of df with a per-subject sequential trial_id added."""
    df = df.copy()
    df[TRIAL_COLUMN] = (
        df.groupby(subject_col).cumcount().astype(str).radd(
            df[subject_col].astype(str) + "_t"
        )
    )
    return df


# ── DEAP-style loader ─────────────────────────────────────────────────────────


def load_deap_feature_csv(path, target="valence", threshold=5.0, feature_columns=None):
    """Load a DEAP-style precomputed feature CSV and normalise to NerveML format.

    DEAP rates emotional video stimuli on valence, arousal, dominance, and
    liking (all on a 1-9 scale). This loader binarises one continuous rating
    at *threshold* (above → 1, at-or-below → 0), strips the other DEAP rating
    columns from the feature set, and renames the subject/target columns to
    NerveML's internal names.

    Parameters
    ----------
    path : str or Path
        CSV with one row per trial. The subject column may be named
        participant_id, participant, subject_id, or subject (case-insensitive).
    target : {"valence", "arousal", "dominance", "liking"}
        Which DEAP rating to treat as the classification target.
    threshold : float
        Binarisation cut-off on the 1-9 scale. Default 5 is the natural
        midpoint; ratings strictly above this become class 1.
    feature_columns : list[str] or None
        Explicit feature columns; inferred from non-label columns if None.

    Returns
    -------
    df : pd.DataFrame
        Normalised table with subject_id, trial_id, target_label, and features.
    feature_columns : list[str]

    Notes
    -----
    The result is a prototype heuristic, not a legal, clinical, or regulatory
    certification. Binarisation at a fixed threshold compresses continuous
    affect ratings; treat the boundary as a modelling convenience, not a
    ground-truth category boundary.
    """
    df = pd.read_csv(path)

    subject_col = _match_column(df, _DEAP_SUBJECT_CANDIDATES)
    if subject_col is None:
        raise ValueError(
            f"no subject column found; expected one of {_DEAP_SUBJECT_CANDIDATES}, "
            f"got {list(df.columns)}"
        )

    target_col = _match_column(df, [target])
    if target_col is None:
        raise ValueError(
            f"target column {target!r} not found; got {list(df.columns)}"
        )

    raw = pd.to_numeric(df[target_col], errors="raise")
    binary = (raw > threshold).astype(int)

    out = df.copy()
    if subject_col != SUBJECT_COLUMN:
        out = out.rename(columns={subject_col: SUBJECT_COLUMN})
    out[TARGET_COLUMN] = binary.values

    if TRIAL_COLUMN not in out.columns:
        out = _synthesize_trial_id(out, SUBJECT_COLUMN)

    if feature_columns is None:
        # Exclude all DEAP label columns and reserved columns by lowercase name.
        exclude = _DEAP_EXCLUDE_LOWER | {target_col.lower()}
        feature_columns = [c for c in out.columns if c.lower() not in exclude]
    else:
        unknown = [c for c in feature_columns if c not in df.columns]
        if unknown:
            raise ValueError(f"feature columns not present in {path}: {unknown}")

    return out, list(feature_columns)


# ── SEED-style loader ─────────────────────────────────────────────────────────


def load_seed_feature_csv(
    path,
    label_column=None,
    positive_class=None,
    drop_neutral=True,
    feature_columns=None,
):
    """Load a SEED-style precomputed feature CSV and normalise to NerveML format.

    SEED (SJTU Emotion EEG Dataset) typically provides differential entropy
    features per frequency band per channel with 3-class emotion labels:
    negative (−1 or 0), neutral (0 or 1), positive (1 or 2).

    By default (drop_neutral=True, positive_class=None) neutral trials are
    discarded and the remaining two classes are mapped to 0/1. To instead keep
    all trials, pass positive_class (the value representing the positive class
    in a one-vs-rest scheme).

    Parameters
    ----------
    path : str or Path
    label_column : str or None
        Column containing emotion labels; inferred from common SEED names if None.
    positive_class : int or str or None
        If given, this value becomes class 1 and all others become class 0
        (one-vs-rest). Overrides drop_neutral.
    drop_neutral : bool
        When True and positive_class is None, neutral trials are dropped so only
        positive-vs-negative remain.
    feature_columns : list[str] or None

    Returns
    -------
    df : pd.DataFrame
    feature_columns : list[str]

    Notes
    -----
    Prototype heuristic. The neutral-class boundary is estimated as the middle
    value of the observed label set and is dataset-convention-dependent.
    """
    df = pd.read_csv(path)

    subject_col = _match_column(df, _SEED_SUBJECT_CANDIDATES)
    if subject_col is None:
        raise ValueError(
            f"no subject column found; expected one of {_SEED_SUBJECT_CANDIDATES}, "
            f"got {list(df.columns)}"
        )

    if label_column is None:
        label_column = _match_column(df, _SEED_EMOTION_CANDIDATES)
    if label_column is None or label_column not in df.columns:
        raise ValueError(
            f"emotion label column not found; tried {_SEED_EMOTION_CANDIDATES}, "
            f"got {list(df.columns)}"
        )

    raw = df[label_column]
    unique_vals = sorted(raw.dropna().unique())
    n_classes = len(unique_vals)

    if positive_class is not None:
        # One-vs-rest: keep all rows.
        if positive_class not in unique_vals:
            raise ValueError(
                f"positive_class={positive_class!r} not in label values {unique_vals}"
            )
        binary = (raw == positive_class).astype(int)
        out = df.copy()
    elif n_classes == 2:
        # Already binary — map lower value → 0, higher → 1.
        binary = (raw == unique_vals[1]).astype(int)
        out = df.copy()
    elif n_classes == 3:
        if not drop_neutral:
            raise ValueError(
                f"3-class SEED labels found: {unique_vals}. Pass positive_class for "
                "one-vs-rest, or leave drop_neutral=True (default) to discard neutral."
            )
        neutral_val = unique_vals[1]  # middle value is neutral regardless of encoding
        out = df[raw != neutral_val].copy()
        remaining_raw = out[label_column]
        remaining_vals = sorted(remaining_raw.unique())
        label_map = {remaining_vals[0]: 0, remaining_vals[1]: 1}
        binary = remaining_raw.map(label_map)
    else:
        raise ValueError(
            f"expected 2 or 3 unique emotion labels, got {n_classes}: {unique_vals}"
        )

    if subject_col != SUBJECT_COLUMN:
        out = out.rename(columns={subject_col: SUBJECT_COLUMN})
    out[TARGET_COLUMN] = binary.values

    if TRIAL_COLUMN not in out.columns:
        out = _synthesize_trial_id(out, SUBJECT_COLUMN)

    # Exclude subject, reserved, and the emotion label column.
    seed_exclude_lower = _RESERVED_LOWER | {label_column.lower(), subject_col.lower()}

    if feature_columns is None:
        feature_columns = [c for c in out.columns if c.lower() not in seed_exclude_lower]
    else:
        unknown = [c for c in feature_columns if c not in df.columns]
        if unknown:
            raise ValueError(f"feature columns not present in {path}: {unknown}")

    return out, list(feature_columns)


# ── format detector ───────────────────────────────────────────────────────────


def detect_dataset_format(path):
    """Sniff a CSV and suggest which NerveML loader to use.

    Reads only the header row to identify the dataset convention. Returns a
    dict with keys:

    - ``format``: ``"nerveml"`` | ``"deap"`` | ``"seed"`` | ``"unknown"``
    - ``confidence``: ``"high"`` | ``"low"``
    - ``columns``: list of column names found in the file
    - ``detected_targets`` (DEAP only): which rating columns were found

    This is a heuristic based on column names only, not data values.
    """
    df = pd.read_csv(path, nrows=0)
    cols = list(df.columns)
    cols_lower = {c.lower() for c in cols}

    # Already in NerveML internal format.
    if {SUBJECT_COLUMN, TARGET_COLUMN}.issubset(cols_lower):
        return {"format": "nerveml", "confidence": "high", "columns": cols}

    # DEAP signals: multiple rating columns + a recognisable subject column.
    deap_ratings = [c for c in _DEAP_LABEL_COLUMNS if c.lower() in cols_lower]
    deap_subject = any(c.lower() in cols_lower for c in _DEAP_SUBJECT_CANDIDATES)
    if deap_subject and len(deap_ratings) >= 2:
        return {
            "format": "deap",
            "confidence": "high" if len(deap_ratings) >= 3 else "low",
            "columns": cols,
            "detected_targets": deap_ratings,
        }

    # SEED signals: an emotion/label column + a subject column.
    seed_subject = any(c.lower() in cols_lower for c in _SEED_SUBJECT_CANDIDATES)
    seed_emotion = any(c.lower() in cols_lower for c in _SEED_EMOTION_CANDIDATES)
    if seed_subject and seed_emotion:
        return {"format": "seed", "confidence": "high", "columns": cols}

    return {"format": "unknown", "confidence": "low", "columns": cols}
