"""Transparent heuristics that turn metrics into plain-language warnings.

Every threshold is a module constant and every flag carries the rule that
produced it, so a customer can disagree with a verdict by pointing at the rule
rather than at a black box. These are prototype UX heuristics; they are not
clinical, legal or scientific standards.
"""

from dataclasses import dataclass, field

from nerveml.uncertainty import units_needed

HEURISTIC_NOTICE = (
    "These thresholds are prototype heuristics for interpreting a single scan. "
    "They are not clinical, legal, or scientific standards, and they do not "
    "certify privacy or compliance."
)

# Spec section 11.2.
AUC_LOW = 0.55
AUC_ELEVATED = 0.65
GAP_POSSIBLE = 0.10
GAP_HIGH = 0.20
ALPHA = 0.05

# Artifact-baseline thresholds.
# artifact_share = (proxy_bacc − 0.5) / (full_bacc − 0.5) — fraction of above-
# chance performance attributable to the physiological-noise proxy.
# At 0.5 (50%) the probe's signal is split equally between neural and artifact.
ARTIFACT_SHARE_HIGH = 0.50

# Secondary-attribute decodability threshold.
# Aligned with the "weakly decodable" tier in _secondary_probe_interpretation
# (scan.py): grouped_balanced_accuracy >= 0.60 is non-trivial cross-unit
# decodability and warrants a risk flag.
SECONDARY_BACC_ELEVATED = 0.60

# Dataset warning codes that indicate possible label leakage.
# These match the codes emitted by loaders.validate_dataset().
LEAKAGE_TYPE_LABEL_NAME = "label_name_in_features"
LEAKAGE_TYPE_HIGH_CORR = "feature_label_high_correlation"

# A pooled score this far above the within-unit score, with the within-unit
# score itself near chance, means the pooled number was earned by telling units
# apart rather than by telling labels apart.
POOLED_MINUS_WITHIN = 0.15
WITHIN_NEAR_CHANCE = 0.55

# Re-identification this far above chance means the signal carries identity
# whatever was stripped from the metadata.
IDENTITY_LIFT = 2.0

# The closest an estimate is allowed to sit to a threshold before the sample
# needed to decide which side it is on becomes an unhelpful number to print.
MIN_MARGIN = 0.005

UNDECIDED = "inconclusive_at_this_sample_size"

UNSUPPORTED_CLAIMS = [
    "This system reads a person's true emotion or mental state.",
    "This result certifies that the dataset is anonymous or safe to release.",
    "This result proves the dataset cannot be used to identify a participant.",
    "The listed features are causal neural correlates of the target attribute.",
]


@dataclass
class RiskAssessment:
    sensitive_inference: str
    subject_dependence: str
    null_evidence: str
    permitted_claim: str
    units_needed: int = None
    flags: list = field(default_factory=list)
    unsupported_claims: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    heuristic_notice: str = HEURISTIC_NOTICE

    @property
    def flag_codes(self):
        return [flag["code"] for flag in self.flags]

    def to_dict(self):
        return {
            "sensitive_inference_evidence": self.sensitive_inference,
            "possible_subject_dependence": self.subject_dependence != "none",
            "subject_dependence_level": self.subject_dependence,
            "null_evidence": self.null_evidence,
            "units_needed": self.units_needed,
            "flags": self.flags,
            "heuristic_notice": self.heuristic_notice,
        }


def _flag(code, severity, message, rule):
    return {"code": code, "severity": severity, "message": message, "rule": rule}


def _classify_inference(grouped_auc, beats_null, interval=None):
    """Verdict from the estimate, or from the whole interval when there is one.

    With an interval, a verdict is only stated when every value it contains
    agrees. An interval spanning a threshold means a different sample of the
    same size could land on either side, and calling that a verdict is the
    overclaiming this product exists to catch.
    """
    if interval is not None and None not in interval:
        low, high = interval
        if high < AUC_LOW:
            return "low_or_inconclusive"
        if low > AUC_ELEVATED and beats_null:
            return "elevated"
        if AUC_LOW <= low and high <= AUC_ELEVATED:
            return "moderate"
        return UNDECIDED

    if grouped_auc < AUC_LOW:
        return "low_or_inconclusive"
    if grouped_auc > AUC_ELEVATED and beats_null:
        return "elevated"
    return "moderate"


def _classify_dependence(gap, interval=None):
    if interval is not None and None not in interval:
        low, high = interval
        if low >= GAP_HIGH:
            return "high"
        if low >= GAP_POSSIBLE:
            return "possible"
        if high < GAP_POSSIBLE:
            return "none"
        return "undecided"

    if gap >= GAP_HIGH:
        return "high"
    if gap >= GAP_POSSIBLE:
        return "possible"
    return "none"


def _permitted_claim(level, grouped_auc, p_value, interval=None):
    if interval is not None and None not in interval:
        evidence = (
            f"(subject-held-out ROC-AUC {grouped_auc:.2f}, 95% CI "
            f"{interval[0]:.2f} to {interval[1]:.2f}, permutation p {p_value:.3f})"
        )
    else:
        evidence = (
            f"(subject-held-out ROC-AUC {grouped_auc:.2f}, permutation p {p_value:.3f})"
        )
    if level == UNDECIDED:
        return (
            "This sample cannot decide whether the self-reported label is "
            f"decodable across held-out subjects {evidence}. The interval spans "
            "the threshold, so a different sample of the same size could land "
            "on either side."
        )
    if level == "elevated":
        return (
            "Under this protocol, the self-reported label was decodable across "
            f"held-out subjects {evidence}."
        )
    if level == "moderate":
        return (
            "Under this protocol, the self-reported label was at most weakly "
            f"decodable across held-out subjects {evidence}; the evidence does "
            "not support a general cross-person decoder."
        )
    return (
        "Under this protocol, the self-reported label was not convincingly "
        f"decodable across held-out subjects {evidence}."
    )


def _recommendations(level, dependence, beats_null):
    items = []
    if dependence in ("possible", "high"):
        items.append(
            "Repeat the scan holding out session, site, device, and stimulus in "
            "addition to subject, to locate which grouping drives the gap."
        )
        items.append(
            "Report the subject-held-out score, not the trial-random score, in "
            "any external claim about cross-person performance."
        )
    if not beats_null:
        items.append(
            "Increase the permutation count before concluding either way; the "
            "current null test does not separate the result from chance."
        )
    if level == "elevated":
        items.append(
            "Treat the target attribute as inferable from this data and review "
            "consent and intended-use documentation before sharing."
        )
        items.append(
            "Run an artifact-only baseline (EOG/EMG or movement channels alone) "
            "to check whether the signal is physiological rather than neural."
        )
    items.append(
        "Validate on an independent cohort before making claims outside this "
        "dataset."
    )
    return items


def assess_risk(
    *,
    grouped_auc,
    generalization_gap,
    p_value,
    pooled_balanced_accuracy=None,
    within_unit_balanced_accuracy=None,
    identity_lift=None,
    identity_accuracy=None,
    grouped_auc_ci=None,
    gap_ci=None,
    n_units=None,
    artifact_share_eog=None,
    artifact_share_emg=None,
    secondary_probes=None,
    dataset_warnings=None,
):
    """Apply the prototype risk heuristics to one scan's metrics.

    pooled_balanced_accuracy and within_unit_balanced_accuracy are the
    trial-random score computed over all trials at once and computed inside
    each unit separately. Supplying both enables the decomposition flag; a
    caller that has not computed them still gets every other verdict.

    artifact_share_eog / artifact_share_emg are the fraction of above-chance
    classification performance attributable to frontal EOG-proxy or gamma EMG-
    proxy features respectively. Values >= ARTIFACT_SHARE_HIGH trigger a
    warning flag.

    secondary_probes is the list of secondary sensitive-attribute probe dicts
    as written into report["secondary_probes"] by scan.py. Any attribute with
    grouped_balanced_accuracy >= SECONDARY_BACC_ELEVATED triggers a flag.

    dataset_warnings is the list of warning dicts from validate_dataset()
    (each with "code" and "message" keys). Leakage-smell codes
    (LEAKAGE_TYPE_LABEL_NAME, LEAKAGE_TYPE_HIGH_CORR) produce risk flags so
    that leakage smells surface in the risk_flags section, not only in the
    dataset.warnings list which operators may overlook.
    """
    beats_null = p_value < ALPHA
    level = _classify_inference(grouped_auc, beats_null, grouped_auc_ci)
    dependence = _classify_dependence(generalization_gap, gap_ci)

    needed = None
    if level == UNDECIDED and grouped_auc_ci is not None and n_units:
        # What has to shrink is not the interval to some fixed width, but the
        # interval to inside the nearest threshold it currently straddles. An
        # estimate sitting almost on a threshold needs an enormous sample to
        # say which side it falls, and printing a small number would be a lie.
        crossed = [
            edge
            for edge in (AUC_LOW, AUC_ELEVATED)
            if grouped_auc_ci[0] <= edge <= grouped_auc_ci[1]
        ]
        margin = max(
            MIN_MARGIN, min(abs(grouped_auc - edge) for edge in crossed)
        ) if crossed else MIN_MARGIN
        half_width = (grouped_auc_ci[1] - grouped_auc_ci[0]) / 2
        needed = units_needed(n_units, half_width, margin)

    interval_note = (
        f", 95% CI {grouped_auc_ci[0]:.3f} to {grouped_auc_ci[1]:.3f}"
        if grouped_auc_ci is not None and None not in grouped_auc_ci
        else ""
    )
    flags = [
        _flag(
            f"sensitive_inference_{level}",
            "warning" if level == "elevated" else "info",
            {
                "elevated": "The target attribute appears inferable across "
                "held-out subjects under this protocol.",
                "moderate": "Some information about the target may be present, "
                "but it does not yet support cross-person claims.",
                "low_or_inconclusive": "No convincing cross-subject evidence of "
                "sensitive inference under this test.",
                UNDECIDED: "This sample cannot decide. The interval spans a "
                "threshold, so the verdict would depend on which subjects "
                "happened to be recruited.",
            }[level],
            f"grouped ROC-AUC {grouped_auc:.3f}{interval_note} against "
            f"thresholds {AUC_LOW}/{AUC_ELEVATED} with permutation p "
            f"{p_value:.3f}",
        )
    ]

    if level == UNDECIDED:
        detail = f"{n_units} units" if n_units else "the current sample"
        onward = (
            f"; roughly {needed} would tighten it inside the nearest threshold"
            if needed
            else ""
        )
        flags.append(
            _flag(
                "undecided_at_this_sample_size",
                "warning",
                "Report the interval rather than the verdict. At this sample "
                "size the scan cannot separate the possibilities.",
                f"95% CI {grouped_auc_ci[0]:.3f} to {grouped_auc_ci[1]:.3f} "
                f"spans a decision threshold with {detail}{onward}",
            )
        )

    if dependence == "undecided":
        flags.append(
            _flag(
                "generalization_gap_undecided",
                "info",
                "The gap's interval crosses a threshold, so subject dependence "
                "is neither established nor ruled out here.",
                f"gap {generalization_gap:.3f}, 95% CI {gap_ci[0]:.3f} to "
                f"{gap_ci[1]:.3f}, against thresholds {GAP_POSSIBLE}/{GAP_HIGH}",
            )
        )

    if dependence == "high":
        flags.append(
            _flag(
                "high_generalization_warning",
                "warning",
                "The trial-random result is likely misleading for cross-person "
                "claims.",
                f"naive minus grouped balanced accuracy {generalization_gap:.3f} "
                f">= {GAP_HIGH}",
            )
        )
    if dependence in ("possible", "high"):
        flags.append(
            _flag(
                "possible_subject_dependence",
                "warning",
                "Performance may rely substantially on person-specific "
                "structure rather than the intended signal.",
                f"naive minus grouped balanced accuracy {generalization_gap:.3f} "
                f">= {GAP_POSSIBLE}",
            )
        )
    decomposed = (
        pooled_balanced_accuracy is not None
        and within_unit_balanced_accuracy is not None
    )
    if (
        decomposed
        and within_unit_balanced_accuracy < WITHIN_NEAR_CHANCE
        and pooled_balanced_accuracy - within_unit_balanced_accuracy
        >= POOLED_MINUS_WITHIN
    ):
        flags.append(
            _flag(
                "between_unit_discrimination_only",
                "warning",
                "The headline score comes from telling units apart, not from "
                "telling labels apart: inside any single unit the model is at "
                "chance. A score like this cannot track one individual over "
                "time, whatever the pooled number says.",
                f"pooled trial-random balanced accuracy "
                f"{pooled_balanced_accuracy:.3f} minus within-unit "
                f"{within_unit_balanced_accuracy:.3f} >= {POOLED_MINUS_WITHIN}, "
                f"with within-unit below {WITHIN_NEAR_CHANCE}",
            )
        )

    if identity_lift is not None and identity_lift >= IDENTITY_LIFT:
        flags.append(
            _flag(
                "re_identifiable",
                "warning",
                "Held-out records could be attached to the individuals who "
                "produced them, from the signal alone. Removing names and ids "
                "does not remove this.",
                f"re-identification accuracy {identity_accuracy:.3f} is "
                f"{identity_lift:.1f}x chance, at or above {IDENTITY_LIFT}x",
            )
        )

    if not beats_null:
        flags.append(
            _flag(
                "insufficient_null_evidence",
                "info",
                "The subject-held-out result does not clearly exceed its "
                "permutation null.",
                f"permutation p {p_value:.3f} >= {ALPHA}",
            )
        )

    # Artifact-baseline flags.  High proxy share means a material fraction of
    # the above-chance performance may come from physiological noise rather than
    # a neural correlate of the target.
    if artifact_share_eog is not None and artifact_share_eog >= ARTIFACT_SHARE_HIGH:
        flags.append(
            _flag(
                "high_eog_artifact_share",
                "warning",
                "A large share of the above-chance classification performance "
                "is reproduced by frontal/lateral-frontal (EOG-proxy) features "
                "alone. The signal may partly reflect eye-movement artifacts "
                "rather than the intended neural construct.",
                f"EOG-proxy artifact share {artifact_share_eog:.2f} >= "
                f"{ARTIFACT_SHARE_HIGH} (prototype heuristic)",
            )
        )

    if artifact_share_emg is not None and artifact_share_emg >= ARTIFACT_SHARE_HIGH:
        flags.append(
            _flag(
                "high_emg_artifact_share",
                "warning",
                "A large share of the above-chance classification performance "
                "is reproduced by gamma-band (EMG-proxy) features alone. "
                "The signal may partly reflect muscle artifacts rather than "
                "the intended neural construct.",
                f"EMG-proxy artifact share {artifact_share_emg:.2f} >= "
                f"{ARTIFACT_SHARE_HIGH} (prototype heuristic)",
            )
        )

    # Secondary sensitive-attribute flags.  Any attribute decodable above the
    # "weakly decodable" tier is an independent privacy concern — the features
    # carry information about a sensitive variable beyond the primary target.
    for probe in (secondary_probes or []):
        if not isinstance(probe, dict):
            continue
        attr = probe.get("attribute", "unknown")
        bacc = probe.get("grouped_balanced_accuracy")
        if bacc is not None and bacc >= SECONDARY_BACC_ELEVATED:
            flags.append(
                _flag(
                    "secondary_attribute_decodable",
                    "warning",
                    f"The sensitive attribute '{attr}' appears decodable from "
                    "the same EEG features used for the primary probe "
                    "(association under this protocol, not a causal claim).",
                    f"secondary probe grouped balanced accuracy {bacc:.3f} >= "
                    f"{SECONDARY_BACC_ELEVATED} (prototype heuristic)",
                )
            )

    # Leakage-smell flags from validate_dataset.  These warnings are also stored
    # in report["dataset"]["warnings"], but operators scanning the risk_flags
    # section may never reach the dataset section; surfacing them here ensures
    # they are not missed.  The message is passed through verbatim so the
    # operator sees the same bounded-claim wording in both places.
    for dw in (dataset_warnings or []):
        if not isinstance(dw, dict):
            continue
        code = dw.get("code", "")
        message = dw.get("message", "")
        if code == LEAKAGE_TYPE_LABEL_NAME:
            flags.append(
                _flag(
                    "label_leakage_smell",
                    "warning",
                    message,
                    "dataset column name matches known label vocabulary "
                    "(prototype heuristic, verify manually)",
                )
            )
        elif code == LEAKAGE_TYPE_HIGH_CORR:
            flags.append(
                _flag(
                    "high_correlation_leakage_smell",
                    "warning",
                    message,
                    f"feature near-perfectly correlated with target "
                    f"(prototype heuristic, verify manually)",
                )
            )

    return RiskAssessment(
        sensitive_inference=level,
        subject_dependence=dependence,
        null_evidence="exceeds_null" if beats_null else "insufficient",
        permitted_claim=_permitted_claim(level, grouped_auc, p_value, grouped_auc_ci),
        units_needed=needed,
        flags=flags,
        unsupported_claims=list(UNSUPPORTED_CLAIMS),
        recommendations=_recommendations(level, dependence, beats_null),
    )
