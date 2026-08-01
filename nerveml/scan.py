"""One scan, end to end, producing the report described in spec section 22.

Order matters here: validate before measuring, measure both protocols with the
same features and model family, then generate the null for the grouped protocol
only. The report is assembled last from values that were actually computed -
nothing in it is a placeholder.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from nerveml.artifact_baseline import run_artifact_baseline
from nerveml.confounds import split_fingerprint
from nerveml.identity import identity_attack
from nerveml.interpret import FEATURE_CAVEAT, rank_features
from nerveml.loaders import (
    detect_dataset_format,
    load_deap_feature_csv,
    load_feature_csv,
    load_seed_feature_csv,
    validate_dataset,
)
from nerveml.models import build_model
from nerveml.permutation import permutation_test
from nerveml.risk_rules import assess_risk
from nerveml.scenarios import SCENARIOS, context as scenario_context
from nerveml.scenarios import load_scenario
from nerveml.synth import MODES, feature_columns, make_synthetic
from nerveml.uncertainty import bootstrap_ci
from nerveml.validation import (
    SUBJECT_COLUMN,
    evaluate_grouped,
    evaluate_trial_random,
    generalization_gap,
)

REPORT_SCHEMA_VERSION = "0.2.0"

# When the data says which recording each row came from, the re-identification
# probe holds whole recordings out. That is the harder and more honest attack,
# so it is the default whenever the column exists.
SESSION_COLUMN = "session_id"

# Columns that, when present in the feature table, indicate a natural held-out
# boundary distinct from the primary grouping column. Each is evaluated with
# evaluate_grouped so the report shows how well the label generalises across
# sessions, sites, devices, and stimuli, not just across subjects.
CANDIDATE_GROUPING_AXES = ["session_id", "site_id", "device_id", "stimulus_id"]

# Stage keys in execution order, with the label a UI should show. A caller can
# only report progress it is actually told about, so the scan announces each
# boundary rather than leaving an interface to guess at one.
STAGES = {
    "validate": "Checking dataset integrity",
    "trial_random": "Trial-random cross-validation",
    "grouped": "Held-out-unit cross-validation",
    "permutation": "Permutation null",
    "identity": "Re-identification probe",
    "confounds": "Splitting the fingerprint into amplitude and shape",
    "artifact_baseline": "Artifact-proxy baselines (EOG/EMG)",
    "interpret": "Ranking feature associations",
    "multi_axis": "Multi-axis held-out validation",
    "secondary_probe": "Secondary sensitive-attribute probes",
}

# Known sensitive-attribute column names. When found in the feature table they
# are probed even if they fall outside the generic cardinality-based detection
# (e.g. a low-variance arousal score that happens to have many unique values).
KNOWN_SECONDARY_ATTRS = frozenset({
    # Emotion / affect dimensions
    "arousal", "dominance", "liking",
    # Sleep physiology
    "sleep_stage", "sleep_score",
    # Motor / movement state
    "motor_state", "movement",
    # Fatigue / tiredness
    "fatigue", "fatigue_level", "tiredness",
    # Medication
    "medication", "medication_status",
    # Stress / cognitive state
    "stress", "stress_level",
    "cognitive_load", "workload",
    # Demographic identity — sensitive attributes per GDPR Art. 9 / privacy spec
    "identity", "participant_identity", "subject_identity",
    "gender", "sex",
    "age_group", "age_bin", "age_range",
    "ethnicity", "race",
    "nationality",
    # Disability — GDPR Art. 9 special category
    "disability_status", "disability",
    # Health data — GDPR Art. 9 special category
    "health_condition", "health_status",
    # Sexual orientation — GDPR Art. 9 special category
    "sexual_orientation",
    # Religious / philosophical beliefs — GDPR Art. 9 special category
    "religion", "religious_affiliation",
    # Political opinions — GDPR Art. 9 special category
    "political_opinion", "political_affiliation",
})

# Structural / metadata columns never treated as secondary attribute targets.
_SKIP_SECONDARY_COLS = frozenset(
    CANDIDATE_GROUPING_AXES + ["trial_id", "session_id"]
)

# A secondary-attribute column must have no more than this many unique values.
_MAX_SECONDARY_CARDINALITY = 10

# Minority class must make up at least this share of rows.
_MIN_SECONDARY_MINORITY_SHARE = 0.10


def _pooled_balanced_accuracy(units, predictions):
    """Balanced accuracy over the pooled trials of the given units."""
    truth = np.concatenate([predictions[u][0] for u in units])
    guess = np.concatenate([predictions[u][1] for u in units])
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(truth, guess))


def _axis_gap_ci(naive_bacc, per_group):
    """Bootstrap CI for generalization gap at the per-axis level.

    Resamples axis units (sessions / sites / devices / stimuli) from the
    per-group score list, computes the mean grouped balanced accuracy for each
    bootstrap draw, and returns the gap [lo, hi] as (naive_bacc - hi_bacc,
    naive_bacc - lo_bacc). naive_bacc is treated as a fixed constant because
    the primary naive model is not refit here.
    """
    scores = [
        e["balanced_accuracy"]
        for e in per_group
        if e.get("balanced_accuracy") is not None
    ]
    ci = bootstrap_ci(scores)
    if ci[0] is None or ci[1] is None:
        return (None, None)
    return (float(naive_bacc - ci[1]), float(naive_bacc - ci[0]))


def _gap_interval(naive, grouped, n_resamples=2000, seed=0):
    """Interval for the gap, by resampling subjects and re-pooling.

    The gap is a difference of two pooled scores, and a pooled score is not the
    mean of its per-subject scores - that is the whole point of the within-unit
    decomposition. So per-subject differences are not a paired version of the
    gap, and an interval built from them describes a different quantity.

    Resampling subjects and recomputing both pooled scores over the drawn
    trials is the correct construction, and it still needs no refitting: the
    held-out predictions are already in hand.
    """
    shared = sorted(set(naive.unit_predictions) & set(grouped.unit_predictions))
    if len(shared) < 2:
        return (None, None)

    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_resamples):
        drawn = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
        a = _pooled_balanced_accuracy(drawn, naive.unit_predictions)
        b = _pooled_balanced_accuracy(drawn, grouped.unit_predictions)
        if not (np.isnan(a) or np.isnan(b)):
            gaps.append(a - b)

    if len(gaps) < 2:
        return (None, None)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return (float(low), float(high))


def _confidence_intervals(naive, grouped, seed=0):
    """95% intervals over subjects, needing no refitting."""
    return {
        "grouped_roc_auc": bootstrap_ci([e["roc_auc"] for e in grouped.per_group]),
        "grouped_balanced_accuracy": bootstrap_ci(
            [e["balanced_accuracy"] for e in grouped.per_group]
        ),
        "generalization_gap": _gap_interval(naive, grouped, seed=seed),
    }


def _secondary_probe_interpretation(bacc):
    """Bounded-claim text for a secondary-attribute probe result."""
    if bacc >= 0.70:
        return (
            f"Balanced accuracy {bacc:.3f} — appears decodable across held-out "
            "units under this protocol (prototype heuristic, not a regulatory "
            "or clinical determination)."
        )
    if bacc >= 0.60:
        return (
            f"Balanced accuracy {bacc:.3f} — weakly decodable under this protocol; "
            "association, not a causal or diagnostic claim."
        )
    return (
        f"Balanced accuracy {bacc:.3f} — no convincing cross-unit evidence of "
        "decodability under this protocol."
    )


def _find_secondary_attributes(df, feats, group_col):
    """Columns in df that qualify as secondary sensitive-attribute probe targets.

    Scans ALL DataFrame columns (including those in feats — a generic CSV loader
    may have treated a label column as a feature). A column qualifies when it is
    not the primary label, not the grouping column, not a candidate grouping axis
    or session/trial id, has at most _MAX_SECONDARY_CARDINALITY distinct non-NaN
    values, and its minority class makes up at least _MIN_SECONDARY_MINORITY_SHARE
    of rows.  Continuous feature columns are excluded by the cardinality filter.
    """
    from nerveml.validation import TARGET_COLUMN, SUBJECT_COLUMN

    skip = {TARGET_COLUMN, SUBJECT_COLUMN, group_col} | _SKIP_SECONDARY_COLS

    candidates = []
    for col in df.columns:
        if col in skip:
            continue
        vals = df[col].dropna()
        n_unique = vals.nunique()
        if n_unique < 2 or n_unique > _MAX_SECONDARY_CARDINALITY:
            continue
        counts = vals.value_counts(normalize=True)
        if counts.min() < _MIN_SECONDARY_MINORITY_SHARE:
            continue
        candidates.append(col)

    return candidates


def _run_secondary_probes(df, feats, attrs, model, group_col, n_splits, seed):
    """Run evaluate_grouped for each secondary attribute and collect probe dicts.

    Each probe uses the same features minus the probed column (to avoid trivial
    self-prediction when the attribute was also included as a feature by the
    loader), and replaces the primary label with the secondary attribute column.
    Probes that fail (e.g. too few groups) are silently skipped.
    """
    from nerveml.validation import TARGET_COLUMN, SUBJECT_COLUMN

    results = []
    for attr in attrs:
        try:
            # Exclude the probed attribute from the feature set so that the
            # model cannot trivially memorise the label from itself.
            probe_feats = [f for f in feats if f != attr]
            if not probe_feats:
                continue
            cols = list({group_col, attr}.union(probe_feats))
            df_probe = df[cols].copy()
            df_probe = df_probe.rename(columns={attr: TARGET_COLUMN})
            probe_n_splits = min(n_splits, df_probe[group_col].nunique())
            grouped_probe = evaluate_grouped(
                df_probe,
                probe_feats,
                group_column=group_col,
                model=model,
                n_splits=probe_n_splits,
                seed=seed,
            )
            results.append({
                "attribute": attr,
                "n_units": int(df[group_col].nunique()),
                "grouped_balanced_accuracy": round(grouped_probe.balanced_accuracy, 4),
                "grouped_roc_auc": round(grouped_probe.roc_auc, 4),
                "interpretation": _secondary_probe_interpretation(
                    grouped_probe.balanced_accuracy
                ),
            })
        except Exception:
            pass

    return results


def _looks_like_band_power(feature_columns):
    """Do these names split into a channel and a band?

    The amplitude/shape decomposition needs a channel to sum over. Feature
    tables that do not have one cannot answer the question, and a scan should
    say nothing rather than invent an answer.
    """
    return all(
        "_" in name and name.rpartition("_")[0] and name.rpartition("_")[2]
        for name in feature_columns
    ) and len({name.rpartition("_")[0] for name in feature_columns}) < len(
        feature_columns
    )


def _worst_group(per_group):
    """The held-out unit the model did worst on.

    A mean over units is exactly the statistic that can hide one participant
    the model fails completely, so the report names that participant.
    """
    scored = [e for e in per_group if e["balanced_accuracy"] is not None]
    if not scored:
        return None
    return min(scored, key=lambda entry: entry["balanced_accuracy"])


def _detect_extra_axes(df, primary_group_column, n_splits):
    """Return CANDIDATE_GROUPING_AXES columns that are present in df and have
    at least n_splits unique values.  The primary grouping column is excluded so
    the primary and extra results remain independent.
    """
    extra = []
    for col in CANDIDATE_GROUPING_AXES:
        if col == primary_group_column:
            continue
        if col not in df.columns:
            continue
        if df[col].nunique() >= n_splits:
            extra.append(col)
    return extra


# ── Probe registry ────────────────────────────────────────────────────────────
# The two CV protocols always run. Everything after them — re-identification, the
# amplitude/shape split, the artifact baselines, the secondary-attribute probes —
# is optional and gated on the data. Each is a ProbeSpec with an `applies`
# predicate and a `run`, so run_scan iterates a list instead of carrying four
# hand-wired blocks, and adding a probe is appending a spec.


@dataclass
class ProbeContext:
    """Everything a probe needs, plus the results earlier probes produced.

    Probes run in registry order, so a later probe's `applies`/`run` may read an
    earlier result from `results` — the amplitude/shape split, for instance,
    only runs once the re-identification probe has returned something.
    """

    df: "pd.DataFrame"
    feats: list
    group_column: str
    model: object
    model_kind: str
    n_splits: int
    seed: int
    session_column: object
    grouped_bacc: float
    identity_probe: bool
    results: dict = field(default_factory=dict)


@dataclass
class ProbeSpec:
    """One optional probe: which report key it fills, its stage, when it runs."""

    key: str
    stage: str
    applies: Callable  # (ProbeContext) -> bool
    run: Callable      # (ProbeContext) -> result stored under `key`


def _run_identity_probe(ctx):
    # ValueError means too few identities or records to attack; a missing probe
    # is more honest than a fabricated one, and the integrity scan has warned.
    try:
        return identity_attack(
            ctx.df,
            ctx.feats,
            identity_column=ctx.group_column,
            session_column=ctx.session_column,
            model_kind=ctx.model_kind,
            n_splits=ctx.n_splits,
            seed=ctx.seed,
        )
    except ValueError:
        return None


def _run_confounds_probe(ctx):
    try:
        return split_fingerprint(
            ctx.df,
            ctx.feats,
            identity_column=ctx.group_column,
            session_column=ctx.session_column,
            model_kind=ctx.model_kind,
            n_splits=ctx.n_splits,
            seed=ctx.seed,
        )
    except ValueError:
        return None


def _run_artifact_probe(ctx):
    try:
        return run_artifact_baseline(
            ctx.df,
            ctx.feats,
            group_column=ctx.group_column,
            full_bacc=ctx.grouped_bacc,
            model=ctx.model,
            n_splits=ctx.n_splits,
            seed=ctx.seed,
        )
    except (ValueError, Exception):
        return None


def _secondary_applies(ctx):
    """True when at least one secondary-attribute column is present.

    The detected columns are cached on the context so run does not repeat the
    scan, and so the stage is announced only when there is work to do.
    """
    attrs = _find_secondary_attributes(ctx.df, ctx.feats, ctx.group_column)
    ctx.results["_secondary_attrs"] = attrs
    return bool(attrs)


def _run_secondary_probe(ctx):
    return _run_secondary_probes(
        ctx.df,
        ctx.feats,
        ctx.results.get("_secondary_attrs", []),
        ctx.model,
        ctx.group_column,
        ctx.n_splits,
        ctx.seed,
    )


PROBE_REGISTRY = [
    ProbeSpec(
        key="identity",
        stage="identity",
        applies=lambda ctx: ctx.identity_probe,
        run=_run_identity_probe,
    ),
    ProbeSpec(
        key="composition",
        stage="confounds",
        applies=lambda ctx: (
            ctx.results.get("identity") is not None
            and _looks_like_band_power(ctx.feats)
        ),
        run=_run_confounds_probe,
    ),
    ProbeSpec(
        key="artifact",
        stage="artifact_baseline",
        applies=lambda ctx: _looks_like_band_power(ctx.feats),
        run=_run_artifact_probe,
    ),
    ProbeSpec(
        key="secondary_probes",
        stage="secondary_probe",
        applies=_secondary_applies,
        run=_run_secondary_probe,
    ),
]


def run_probes(ctx, stage):
    """Run every applicable probe in registry order, filling ctx.results.

    stage announces each probe's boundary exactly when the probe runs, so the
    stage stream is unchanged from the hand-wired version.
    """
    for spec in PROBE_REGISTRY:
        if spec.applies(ctx):
            stage(spec.stage)
            ctx.results[spec.key] = spec.run(ctx)
    return ctx.results


def _load(
    dataset,
    n_subjects,
    n_trials,
    n_features,
    seed,
    csv_format="auto",
    deap_target="valence",
    deap_threshold=5.0,
    seed_label_column=None,
    seed_positive_class=None,
    seed_drop_neutral=True,
):
    if dataset in SCENARIOS:
        df, feats = load_scenario(dataset, seed=seed)
        return df, feats, "demonstration"

    if dataset in MODES:
        df = make_synthetic(dataset, n_subjects, n_trials, n_features, seed)
        return df, feature_columns(df), "synthetic"

    path = Path(dataset)
    if path.suffix.lower() == ".csv" and path.exists():
        resolved = csv_format
        if resolved == "auto":
            resolved = detect_dataset_format(path)["format"]

        if resolved == "deap":
            df, feats = load_deap_feature_csv(
                path, target=deap_target, threshold=deap_threshold
            )
        elif resolved == "seed":
            df, feats = load_seed_feature_csv(
                path,
                label_column=seed_label_column,
                positive_class=seed_positive_class,
                drop_neutral=seed_drop_neutral,
            )
        else:
            # "nerveml", "unknown", or any unrecognised value — generic loader.
            df, feats = load_feature_csv(path)
        return df, feats, "feature_table"

    raise ValueError(
        f"unknown dataset {dataset!r}; expected one of "
        f"{tuple(SCENARIOS) + MODES} or a path to an existing feature CSV"
    )


def run_scan(
    dataset="subject_leakage",
    n_subjects=20,
    n_trials=40,
    n_features=16,
    model_kind="random_forest",
    group_column=SUBJECT_COLUMN,
    n_splits=5,
    n_permutations=100,
    top_k=10,
    seed=0,
    target_description="binary self-reported label",
    on_stage=None,
    on_fold=None,
    n_jobs=-1,
    identity_probe=True,
    csv_format="auto",
    deap_target="valence",
    deap_threshold=5.0,
    seed_label_column=None,
    seed_positive_class=None,
    seed_drop_neutral=True,
):
    """Run the full validation and privacy scan, returning the report dict.

    on_fold, when provided, is called after each CV fold in the two primary
    evaluation protocols (trial-random and held-out-unit) as
    on_fold(fold_idx, n_folds, bacc) with 0-based fold_idx. It is NOT called
    during permutation, multi-axis, or secondary-probe runs to avoid spam.
    """
    stage = on_stage if on_stage is not None else lambda _name: None

    stage("validate")
    df, feats, modality = _load(
        dataset,
        n_subjects,
        n_trials,
        n_features,
        seed,
        csv_format=csv_format,
        deap_target=deap_target,
        deap_threshold=deap_threshold,
        seed_label_column=seed_label_column,
        seed_positive_class=seed_positive_class,
        seed_drop_neutral=seed_drop_neutral,
    )
    summary = validate_dataset(df, feats, n_splits=n_splits)

    # Fewer subjects than requested folds would make GroupKFold raise; the
    # integrity scan has already warned about it.
    n_splits = min(n_splits, summary.n_subjects)
    model = build_model(model_kind, seed=seed)

    stage("trial_random")
    naive = evaluate_trial_random(
        df, feats, group_column=group_column, model=model, n_splits=n_splits, seed=seed,
        on_fold=on_fold,
    )

    stage("grouped")
    grouped = evaluate_grouped(
        df, feats, group_column=group_column, model=model, n_splits=n_splits, seed=seed,
        on_fold=on_fold,
    )
    gap = generalization_gap(naive, grouped)
    intervals = _confidence_intervals(naive, grouped, seed=seed)

    extra_axes = _detect_extra_axes(df, group_column, n_splits)
    multi_axis_results = []
    if extra_axes:
        stage("multi_axis")
        for axis in extra_axes:
            try:
                axis_grouped = evaluate_grouped(
                    df, feats,
                    group_column=axis,
                    model=model,
                    n_splits=min(n_splits, df[axis].nunique()),
                    seed=seed,
                )
                axis_ci_bacc = bootstrap_ci(
                    [e["balanced_accuracy"] for e in axis_grouped.per_group]
                )
                axis_ci_auc = bootstrap_ci(
                    [e["roc_auc"] for e in axis_grouped.per_group]
                )
                axis_ci_gap = _axis_gap_ci(
                    naive.balanced_accuracy, axis_grouped.per_group
                )
                multi_axis_results.append({
                    "axis": axis,
                    "n_units": int(df[axis].nunique()),
                    "scheme": axis_grouped.scheme,
                    "grouped_balanced_accuracy": round(axis_grouped.balanced_accuracy, 4),
                    "grouped_roc_auc": round(axis_grouped.roc_auc, 4),
                    "grouped_balanced_accuracy_std": round(
                        axis_grouped.balanced_accuracy_std, 4
                    ),
                    "generalization_gap": round(generalization_gap(naive, axis_grouped), 4),
                    "worst_group": _worst_group(axis_grouped.per_group),
                    "per_group": axis_grouped.per_group,
                    "confidence_intervals": {
                        "grouped_balanced_accuracy": [
                            None if v is None else round(v, 4) for v in axis_ci_bacc
                        ],
                        "grouped_roc_auc": [
                            None if v is None else round(v, 4) for v in axis_ci_auc
                        ],
                        "generalization_gap": [
                            None if v is None else round(v, 4) for v in axis_ci_gap
                        ],
                    },
                })
            except ValueError:
                # Axis exists but evaluation failed (e.g. class imbalance per fold);
                # the primary validation already warns about dataset issues.
                pass

    stage("permutation")
    null = permutation_test(
        df,
        feats,
        group_column=group_column,
        # Hundreds of fits: parallelise over permutations and keep each model
        # single-threaded, rather than letting the two levels contend.
        model=build_model(model_kind, seed=seed, n_jobs=1),
        n_permutations=n_permutations,
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
    )
    # The optional probes — re-identification, amplitude/shape split, artifact
    # baselines, secondary attributes — run through the registry, in order, each
    # gated by its own predicate. See PROBE_REGISTRY.
    probe_ctx = ProbeContext(
        df=df,
        feats=feats,
        group_column=group_column,
        model=model,
        model_kind=model_kind,
        n_splits=n_splits,
        seed=seed,
        session_column=(
            SESSION_COLUMN
            if SESSION_COLUMN in df.columns and group_column != SESSION_COLUMN
            else None
        ),
        grouped_bacc=grouped.balanced_accuracy,
        identity_probe=identity_probe,
    )
    run_probes(probe_ctx, stage)
    identity = probe_ctx.results.get("identity")
    composition = probe_ctx.results.get("composition")
    artifact = probe_ctx.results.get("artifact")
    secondary_probes = probe_ctx.results.get("secondary_probes") or []

    risk = assess_risk(
        grouped_auc=grouped.roc_auc,
        generalization_gap=gap,
        p_value=null.p_value,
        pooled_balanced_accuracy=naive.balanced_accuracy,
        within_unit_balanced_accuracy=naive.within_unit_balanced_accuracy,
        identity_lift=identity.lift if identity else None,
        identity_accuracy=identity.accuracy if identity else None,
        grouped_auc_ci=intervals["grouped_roc_auc"],
        gap_ci=intervals["generalization_gap"],
        n_units=summary.n_subjects,
        artifact_share_eog=artifact.eog_artifact_share if artifact else None,
        artifact_share_emg=artifact.emg_artifact_share if artifact else None,
        secondary_probes=secondary_probes if secondary_probes else None,
        dataset_warnings=summary.warnings or None,
    )

    stage("interpret")
    top_features = rank_features(df, feats, kind=model_kind, top_k=top_k, seed=seed)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": {
            "name": str(dataset),
            "modality": modality,
            "target": target_description,
            "context": (
                scenario_context(dataset) if dataset in SCENARIOS else None
            ),
            **summary.to_dict(),
        },
        "evaluation": {
            "trial_random_balanced_accuracy": round(naive.balanced_accuracy, 4),
            "trial_random_roc_auc": round(naive.roc_auc, 4),
            # The same trial-random predictions scored inside each unit instead
            # of pooled: what the model knows about labels, with what it knows
            # about people taken away.
            "trial_random_within_unit_balanced_accuracy": round(
                naive.within_unit_balanced_accuracy, 4
            ),
            "grouped_within_unit_balanced_accuracy": round(
                grouped.within_unit_balanced_accuracy, 4
            ),
            "grouped_balanced_accuracy": round(grouped.balanced_accuracy, 4),
            "grouped_roc_auc": round(grouped.roc_auc, 4),
            "grouped_balanced_accuracy_std": round(grouped.balanced_accuracy_std, 4),
            "generalization_gap": round(gap, 4),
            "trial_random_within_unit_roc_auc": round(naive.within_unit_roc_auc, 4),
            "grouped_within_unit_roc_auc": round(grouped.within_unit_roc_auc, 4),
            "confidence_intervals": {
                name: [None if v is None else round(v, 4) for v in bounds]
                for name, bounds in intervals.items()
            },
            "grouping_unit": group_column,
            # Consumers read the strict protocol's name from here rather than
            # assuming it is about subjects.
            "strict_scheme": grouped.scheme,
            "per_group": grouped.per_group,
            "worst_group": _worst_group(grouped.per_group),
            "protocols": [naive.to_dict(), grouped.to_dict()],
        },
        "multi_axis_validation": multi_axis_results,
        "secondary_probes": secondary_probes,
        "identity_inference": identity.to_dict() if identity else None,
        "fingerprint_composition": (
            composition.to_dict() if composition else None
        ),
        "artifact_baseline": artifact.to_dict() if artifact else None,
        "permutation_test": null.to_dict(),
        "null_distribution": [round(s, 4) for s in null.null_scores],
        "risk_flags": risk.to_dict(),
        "top_features": top_features,
        "feature_caveat": FEATURE_CAVEAT,
        "permitted_claim": risk.permitted_claim,
        "unsupported_claims": risk.unsupported_claims,
        "recommendations": risk.recommendations,
        "config": {
            "dataset": str(dataset),
            "model_kind": model_kind,
            "group_column": group_column,
            "n_splits": n_splits,
            "n_permutations": n_permutations,
            "seed": seed,
        },
    }


def fold_metrics_frame(report):
    """Per-fold scores for both protocols, one row per fold."""
    rows = []
    for protocol in report["evaluation"]["protocols"]:
        for index, (bacc, auc) in enumerate(
            zip(protocol["fold_balanced_accuracy"], protocol["fold_roc_auc"])
        ):
            rows.append(
                {
                    "scheme": protocol["scheme"],
                    "fold": index,
                    "balanced_accuracy": bacc,
                    "roc_auc": auc,
                }
            )
    return pd.DataFrame(rows)


def write_report(report, output_dir):
    """Write audit_report.json and fold_metrics.csv. Returns the paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "audit_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "fold_metrics.csv"
    fold_metrics_frame(report).to_csv(csv_path, index=False)

    return {"json": json_path, "csv": csv_path}
