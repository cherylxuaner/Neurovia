"""Artifact-confound baselines for the classification probe.

Runs the same subject-held-out cross-validation on feature subsets that proxy
physiological artifacts rather than intended neural signal:

  EOG proxy  — frontal and lateral-frontal channels that pick up eye-movement
               artifacts.  Two sub-classes are covered:
                 • Vertical EOG  (blinks, up/down saccades): Fp*, AF* channels
                   positioned above the eyes in the 10-20 system.
                 • Lateral EOG (horizontal saccades): F7, F8, FT7, FT8 channels
                   positioned near the outer canthi of the eyes and sensitive to
                   left/right gaze shifts, which are a common confound in visual
                   attention and stimulus-locked paradigms.
  EMG proxy  — gamma-band features (30-45 Hz) where muscle artifacts dominate.

If either proxy achieves near-full classification performance, the probe may
be decoding a physiological artifact rather than the intended neural construct.

This is an *upper-bound* attribution: high proxy performance is consistent with
artifact confounding but does not prove it. Low proxy performance does not rule
out other confounds.
"""

from dataclasses import dataclass
from typing import Optional

from nerveml.validation import evaluate_grouped

# Frontal channel prefixes most reliably contaminated by VERTICAL eye-movement
# artifacts (blinks, up/down saccades).
_EOG_PREFIXES = ("fp", "af")

# Additional lateral-frontal channels that proxy HORIZONTAL eye-movement
# artifacts (left/right saccades).  These are exact channel-name matches
# (case-insensitive) rather than prefix matches to avoid including non-EOG
# channels such as F3/F4/Fz which share the "f" prefix.
#   F7, F8  — standard 10-20 inferior-frontal channels near the outer canthi.
#   FT7, FT8 — extended 10-20 fronto-temporal channels, also laterally placed.
# Channels excluded despite sometimes being cited as EOG-sensitive:
#   TP7, TP8 — temporo-parietal channels; posterior to the temporal lobe and
#     too far from the orbits to reliably pick up horizontal saccades.  Their
#     ocular sensitivity is lab- and amplifier-dependent, so including them
#     would introduce false-positive artifact attribution in most setups.
#   T7, T8   — same rationale; reflect auditory/temporal cortex activity and
#     muscle artifacts from jaw/neck muscles rather than eye-movement EOG.
_LATERAL_EOG_CHANNELS = frozenset({"f7", "f8", "ft7", "ft8"})

# Gamma is the EEG band most dominated by muscle artifacts.
_EMG_BAND = "gamma"

# Skip a proxy when fewer than this many features would be used — too few
# features give unstable estimates that are worse than no estimate.
MIN_PROXY_FEATURES = 2


def _parse_channel_band(col):
    """Split a {channel}_{band} column name. Returns (channel, band) or (None, None)."""
    channel, _, band = col.rpartition("_")
    if not channel or not band:
        return None, None
    return channel, band


def eog_proxy_features(feature_columns):
    """Return features from frontal/lateral-frontal channels — proxies for EOG.

    Captures both vertical eye-movement artifacts via Fp*/AF* prefix matching
    and horizontal (lateral) saccade artifacts via exact matching on F7/F8/
    FT7/FT8 channel names.  F3/F4/Fz and other central-frontal channels are
    intentionally excluded — only the laterally positioned electrodes qualify.
    """
    out = []
    for col in feature_columns:
        channel, _ = _parse_channel_band(col)
        if channel:
            ch = channel.lower()
            if ch.startswith(_EOG_PREFIXES) or ch in _LATERAL_EOG_CHANNELS:
                out.append(col)
    return out


def emg_proxy_features(feature_columns):
    """Return gamma-band features — proxies for EMG muscle artifacts."""
    out = []
    for col in feature_columns:
        _, band = _parse_channel_band(col)
        if band and band.lower() == _EMG_BAND:
            out.append(col)
    return out


@dataclass
class ProxyResult:
    name: str
    n_features: int
    balanced_accuracy: float
    roc_auc: float
    note: str

    def to_dict(self):
        return {
            "name": self.name,
            "n_features": self.n_features,
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "note": self.note,
        }


@dataclass
class ArtifactBaseline:
    full_balanced_accuracy: float
    eog: Optional[ProxyResult]
    emg: Optional[ProxyResult]

    def _artifact_share(self, proxy):
        """Fraction of above-chance performance attributable to the proxy."""
        if proxy is None:
            return None
        gap = self.full_balanced_accuracy - 0.5
        if gap <= 1e-6:
            return None
        share = (proxy.balanced_accuracy - 0.5) / gap
        return round(max(0.0, min(1.0, share)), 3)

    @property
    def eog_artifact_share(self):
        return self._artifact_share(self.eog)

    @property
    def emg_artifact_share(self):
        return self._artifact_share(self.emg)

    def _interpretation(self):
        parts = []
        if self.eog is not None and self.eog_artifact_share is not None:
            share_pct = round(self.eog_artifact_share * 100)
            parts.append(
                f"The frontal/lateral-frontal (EOG-proxy) model achieved "
                f"{self.eog.balanced_accuracy:.1%} balanced accuracy under "
                f"subject-held-out CV. Up to {share_pct}% of the performance "
                f"above chance is consistent with eye-movement confounds "
                f"(vertical and/or horizontal saccades). "
                f"This is an upper bound, not a causal decomposition."
            )
        if self.emg is not None and self.emg_artifact_share is not None:
            share_pct = round(self.emg_artifact_share * 100)
            parts.append(
                f"The gamma-band (EMG-proxy) model achieved "
                f"{self.emg.balanced_accuracy:.1%} balanced accuracy. "
                f"Up to {share_pct}% of above-chance performance is consistent "
                f"with muscle-artifact confounds."
            )
        if not parts:
            return "Artifact-proxy features were unavailable for this dataset."
        return " ".join(parts)

    def to_dict(self):
        return {
            "full_balanced_accuracy": round(self.full_balanced_accuracy, 4),
            "eog_proxy": self.eog.to_dict() if self.eog else None,
            "emg_proxy": self.emg.to_dict() if self.emg else None,
            "artifact_share_eog": self.eog_artifact_share,
            "artifact_share_emg": self.emg_artifact_share,
            "interpretation": self._interpretation(),
        }


def run_artifact_baseline(
    df,
    feature_columns,
    group_column,
    full_bacc,
    model=None,
    n_splits=5,
    seed=0,
):
    """Run EOG- and EMG-proxy CV models on the classification probe.

    Uses the already-computed full balanced accuracy (full_bacc) so the full
    model is not re-evaluated. Proxy models use the same group_column and
    n_splits for a direct comparison.

    Returns None when neither proxy has enough features to produce a result.
    """
    eog_cols = eog_proxy_features(feature_columns)
    emg_cols = emg_proxy_features(feature_columns)

    if len(eog_cols) < MIN_PROXY_FEATURES and len(emg_cols) < MIN_PROXY_FEATURES:
        return None

    def _run(cols):
        cv = evaluate_grouped(
            df, cols, group_column=group_column,
            model=model, n_splits=n_splits, seed=seed,
        )
        return cv.balanced_accuracy, cv.roc_auc

    eog_result = None
    if len(eog_cols) >= MIN_PROXY_FEATURES:
        bacc, auc = _run(eog_cols)
        eog_result = ProxyResult(
            name="eog",
            n_features=len(eog_cols),
            balanced_accuracy=bacc,
            roc_auc=auc,
            note="frontal + lateral-frontal channels (Fp*, AF*, F7/F8/FT7/FT8) proxy eye-movement artifacts (vertical and horizontal)",
        )

    emg_result = None
    if len(emg_cols) >= MIN_PROXY_FEATURES:
        bacc, auc = _run(emg_cols)
        emg_result = ProxyResult(
            name="emg",
            n_features=len(emg_cols),
            balanced_accuracy=bacc,
            roc_auc=auc,
            note="gamma-band features (30-45 Hz) proxy muscle artifacts",
        )

    return ArtifactBaseline(
        full_balanced_accuracy=full_bacc,
        eog=eog_result,
        emg=emg_result,
    )
