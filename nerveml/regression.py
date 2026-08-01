"""Regression scan: compare two audit reports and flag metric changes.

A regression is a degradation in a tracked metric beyond a tolerance threshold
between a baseline scan and a candidate scan. Designed for CI: after any code
or preprocessing change, run a scan and compare it against the last known-good
baseline; exit non-zero when regressions are found.

Metric directions
-----------------
grouped_balanced_accuracy, grouped_roc_auc
    Higher is better. A drop means the scanner is less able to verify genuine
    signal, which may mean the preprocessing or model change broke something.

generalization_gap
    Lower is better. A growing gap means more within-unit structure is inflating
    the naive score — a sign that a data-pipeline change introduced new leakage.

permutation_test.p_value
    Lower is better. A significance flip (≤0.05 → >0.05) strips the result of
    its statistical license, which is a regression regardless of delta size.

identity_inference.accuracy
    Lower re-identification accuracy means less fingerprinting risk. A rise on
    the same dataset under the same configuration is a regression.

evaluation.worst_group.balanced_accuracy
    Higher is better. The balanced accuracy of the single held-out unit the
    model performed worst on. Aggregate metrics can hide a unit that degrades
    entirely, so this tracks the floor separately. Tolerance is wider (0.05)
    than the aggregate tolerance (0.02) because per-unit performance is noisier.
    Absent when no per-group scores were collected (e.g., synthetic data with no
    structured held-out axis); surfaces as "unavailable" in that case.

artifact_baseline.artifact_share_eog / artifact_share_emg
    Lower is better. A rising artifact share means more above-chance performance
    is attributable to physiological-noise proxy features (frontal EOG channels
    or gamma-band EMG), which is a data-quality regression. Only compared when
    at least one of the two reports contains an artifact_baseline section.

secondary_probes[{attribute}].grouped_balanced_accuracy / grouped_roc_auc
    Lower is better for each secondary attribute (arousal, sleep_stage,
    motor_state, fatigue, medication, …). A rising score means the model change
    made that sensitive attribute *more* decodable from the EEG features, which
    is a privacy regression. Only compared when at least one of the two reports
    contains a secondary_probes list. Attributes present in only one report
    surface as "unavailable" rather than being silently skipped.

Unavailable metrics (a section is None in one or both reports) are noted but
never counted as regressions, since the comparison is undefined.
"""

from dataclasses import dataclass, field
from typing import Optional

# (dotted path into the report dict, higher_is_better, tolerance)
# tolerance=None applies a significance-flip rule rather than a delta check.
TRACKED_METRICS = [
    ("evaluation.grouped_balanced_accuracy", True, 0.02),
    ("evaluation.grouped_roc_auc", True, 0.02),
    ("evaluation.trial_random_balanced_accuracy", True, 0.03),
    ("evaluation.generalization_gap", False, 0.03),
    ("permutation_test.p_value", False, None),
    ("identity_inference.accuracy", False, 0.05),
    # Worst held-out unit. Wider tolerance (0.05) than aggregate (0.02) because
    # per-unit performance is noisier. Absent → "unavailable", never regression.
    ("evaluation.worst_group.balanced_accuracy", True, 0.05),
]

# Per-axis metrics tracked inside multi_axis_validation entries.
# Metric keys match what scan.py stores per entry; paths surfaced externally are
# multi_axis_validation[{axis}].{key} so callers can override tolerances by axis.
MULTI_AXIS_METRIC_SPECS = [
    ("grouped_balanced_accuracy", True, 0.02),
    ("grouped_roc_auc", True, 0.02),
    ("generalization_gap", False, 0.03),
]

# CI lower-bound specs for multi-axis validation.
# Paths: multi_axis_validation[{axis}].ci_lo.{key}
# Only emitted when at least one axis entry carries valid (non-None) CI bounds —
# old reports without confidence_intervals produce no ci_lo rows at all.
# Tolerance is wider (0.05) than the point-estimate tolerance (0.02) because
# bootstrapped bounds have higher run-to-run variance than the point estimates.
# Mixed directionality: higher_is_better=True for accuracy metrics (a falling
# lower bound is bad), higher_is_better=False for generalization_gap (a rising
# lower bound means even the best-case gap has widened → regression).
MULTI_AXIS_CI_METRICS = [
    ("grouped_balanced_accuracy", True, 0.05),
    ("grouped_roc_auc", True, 0.05),
    ("generalization_gap", False, 0.05),
]

# CI upper-bound specs for lower-is-better multi-axis metrics.
# Paths: multi_axis_validation[{axis}].ci_hi.{key}
# A rising upper bound on the gap means even the "best-case" gap is widening —
# a privacy-risk regression even when the point estimate is within tolerance.
MULTI_AXIS_CI_HI_METRICS = [
    ("generalization_gap", False, 0.05),
]

# Artifact-baseline metrics tracked when the report includes an artifact_baseline
# section (band-power datasets with frontal/gamma proxy features).
# artifact_share_eog/emg is (proxy_bacc − 0.5) / (full_bacc − 0.5) — lower is
# better (less above-chance performance attributable to physiological noise).
# Tolerance of 0.10 allows normal run-to-run variance without false alarms.
ARTIFACT_METRIC_SPECS = [
    ("artifact_share_eog", False, 0.10),
    ("artifact_share_emg", False, 0.10),
]

# Per-attribute secondary-probe metrics.
# grouped_balanced_accuracy and grouped_roc_auc are both lower-is-better here:
# a RISING value means the sensitive attribute became MORE decodable after the
# change, which is a privacy regression. Tolerance 0.05 (noisier than aggregate).
# Metric path format: secondary_probes[{attribute}].{key}
SECONDARY_PROBE_METRIC_SPECS = [
    ("grouped_balanced_accuracy", False, 0.05),
    ("grouped_roc_auc", False, 0.05),
]

SIGNIFICANCE_THRESHOLD = 0.05


def _get(report, dotted_path):
    """Walk a dotted path into a nested dict. Returns None if any step is absent."""
    value = report
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


@dataclass
class MetricDiff:
    """One metric's movement between baseline and candidate."""

    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    threshold: Optional[float]
    status: str   # "regression" | "improvement" | "stable" | "unavailable"
    note: str = ""

    def to_dict(self):
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 4) if self.baseline is not None else None,
            "candidate": (
                round(self.candidate, 4) if self.candidate is not None else None
            ),
            "delta": round(self.delta, 4) if self.delta is not None else None,
            "threshold": self.threshold,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class RegressionResult:
    """Full comparison between a baseline and a candidate audit report."""

    metrics: list = field(default_factory=list)   # list[MetricDiff]
    passed: bool = True
    permitted_claim_changed: bool = False
    baseline_claim: str = ""
    candidate_claim: str = ""

    @property
    def regressions(self):
        return [m for m in self.metrics if m.status == "regression"]

    @property
    def improvements(self):
        return [m for m in self.metrics if m.status == "improvement"]

    @property
    def stable(self):
        return [m for m in self.metrics if m.status == "stable"]

    @property
    def unavailable(self):
        return [m for m in self.metrics if m.status == "unavailable"]

    def to_dict(self):
        return {
            "passed": self.passed,
            "n_regressions": len(self.regressions),
            "n_improvements": len(self.improvements),
            "n_stable": len(self.stable),
            "n_unavailable": len(self.unavailable),
            "permitted_claim_changed": self.permitted_claim_changed,
            "baseline_claim": self.baseline_claim,
            "candidate_claim": self.candidate_claim,
            "regressions": [m.to_dict() for m in self.regressions],
            "improvements": [m.to_dict() for m in self.improvements],
            "stable": [m.to_dict() for m in self.stable],
            "unavailable": [m.to_dict() for m in self.unavailable],
        }


def _classify(baseline_val, candidate_val, higher_is_better, tolerance):
    """Classify one metric's delta as regression, improvement, stable, or unavailable."""
    if baseline_val is None or candidate_val is None:
        return "unavailable", None, ""

    delta = float(candidate_val) - float(baseline_val)

    if tolerance is None:
        # Significance-flip rule: used for p_value.
        was_sig = baseline_val <= SIGNIFICANCE_THRESHOLD
        now_sig = candidate_val <= SIGNIFICANCE_THRESHOLD
        if was_sig and not now_sig:
            return "regression", delta, (
                f"significance lost: p={candidate_val:.4f} > {SIGNIFICANCE_THRESHOLD}"
            )
        if not was_sig and now_sig:
            return "improvement", delta, (
                f"significance gained: p={candidate_val:.4f} <= {SIGNIFICANCE_THRESHOLD}"
            )
        return "stable", delta, ""

    if higher_is_better:
        if delta < -tolerance:
            return "regression", delta, (
                f"dropped by {abs(delta):.4f} (tolerance {tolerance})"
            )
        if delta > tolerance:
            return "improvement", delta, f"gained {delta:.4f}"
        return "stable", delta, ""
    else:
        if delta > tolerance:
            return "regression", delta, (
                f"rose by {delta:.4f} (tolerance {tolerance})"
            )
        if delta < -tolerance:
            return "improvement", delta, f"fell by {abs(delta):.4f}"
        return "stable", delta, ""


def _multi_axis_diffs(baseline, candidate, overrides):
    """Build MetricDiff objects for every axis in multi_axis_validation.

    Axes present in only one report produce 'unavailable' diffs rather than
    being silently skipped, so the caller knows when coverage changed.

    CI lower-bound diffs (metric path ``ci_lo.<key>``) are emitted per axis
    when at least one entry carries valid (non-None) CI data for that metric.
    CI upper-bound diffs (metric path ``ci_hi.<key>``) are emitted for
    lower-is-better metrics (generalization_gap) when CI data is available.
    Entries from old-format reports without ``confidence_intervals`` produce no
    ci_lo/ci_hi rows at all, preserving backward compatibility.
    """
    b_list = baseline.get("multi_axis_validation") or []
    c_list = candidate.get("multi_axis_validation") or []

    b_by_axis = {e["axis"]: e for e in b_list if isinstance(e, dict) and "axis" in e}
    c_by_axis = {e["axis"]: e for e in c_list if isinstance(e, dict) and "axis" in e}

    diffs = []
    for axis in sorted(set(b_by_axis) | set(c_by_axis)):
        b_entry = b_by_axis.get(axis)
        c_entry = c_by_axis.get(axis)

        # Point-estimate metrics (always emitted per axis)
        for metric_key, higher_is_better, default_tol in MULTI_AXIS_METRIC_SPECS:
            path = f"multi_axis_validation[{axis}].{metric_key}"
            tol = overrides.get(path, default_tol)
            b_val = b_entry.get(metric_key) if b_entry else None
            c_val = c_entry.get(metric_key) if c_entry else None
            status, delta, note = _classify(b_val, c_val, higher_is_better, tol)
            diffs.append(MetricDiff(
                metric=path,
                baseline=b_val,
                candidate=c_val,
                delta=delta,
                threshold=tol,
                status=status,
                note=note,
            ))

        # CI lower-bound metrics (only when at least one entry has valid CI data)
        for metric_key, higher_is_better, default_tol in MULTI_AXIS_CI_METRICS:
            path = f"multi_axis_validation[{axis}].ci_lo.{metric_key}"
            tol = overrides.get(path, default_tol)

            b_ci = (b_entry or {}).get("confidence_intervals", {}).get(metric_key)
            c_ci = (c_entry or {}).get("confidence_intervals", {}).get(metric_key)

            b_lo = b_ci[0] if (isinstance(b_ci, list) and len(b_ci) >= 1
                                and b_ci[0] is not None) else None
            c_lo = c_ci[0] if (isinstance(c_ci, list) and len(c_ci) >= 1
                                and c_ci[0] is not None) else None

            # Skip entirely when neither entry has valid CI data
            if b_lo is None and c_lo is None:
                continue

            status, delta, note = _classify(b_lo, c_lo, higher_is_better, tol)
            diffs.append(MetricDiff(
                metric=path,
                baseline=b_lo,
                candidate=c_lo,
                delta=delta,
                threshold=tol,
                status=status,
                note=note,
            ))

        # CI upper-bound metrics for lower-is-better quantities (e.g. gap).
        # A rising upper bound means the "best-case" gap is widening — a
        # regression even when the point estimate is nominally within tolerance.
        for metric_key, higher_is_better, default_tol in MULTI_AXIS_CI_HI_METRICS:
            path = f"multi_axis_validation[{axis}].ci_hi.{metric_key}"
            tol = overrides.get(path, default_tol)

            b_ci = (b_entry or {}).get("confidence_intervals", {}).get(metric_key)
            c_ci = (c_entry or {}).get("confidence_intervals", {}).get(metric_key)

            b_hi = b_ci[1] if (isinstance(b_ci, list) and len(b_ci) >= 2
                                and b_ci[1] is not None) else None
            c_hi = c_ci[1] if (isinstance(c_ci, list) and len(c_ci) >= 2
                                and c_ci[1] is not None) else None

            # Skip entirely when neither entry has valid CI data
            if b_hi is None and c_hi is None:
                continue

            status, delta, note = _classify(b_hi, c_hi, higher_is_better, tol)
            diffs.append(MetricDiff(
                metric=path,
                baseline=b_hi,
                candidate=c_hi,
                delta=delta,
                threshold=tol,
                status=status,
                note=note,
            ))

    return diffs


def _artifact_diffs(baseline, candidate, overrides):
    """Build MetricDiff objects for artifact-baseline metrics.

    Returns an empty list when neither report has an artifact_baseline section,
    so non-band-power datasets never see spurious n/a rows in the compare table.
    When one report has the section and the other does not, the missing metrics
    surface as 'unavailable' rather than being silently skipped.
    """
    b_ab = baseline.get("artifact_baseline") or {}
    c_ab = candidate.get("artifact_baseline") or {}

    if not b_ab and not c_ab:
        return []

    diffs = []
    for key, higher_is_better, default_tol in ARTIFACT_METRIC_SPECS:
        path = f"artifact_baseline.{key}"
        tol = overrides.get(path, default_tol)
        b_val = b_ab.get(key) if b_ab else None
        c_val = c_ab.get(key) if c_ab else None
        status, delta, note = _classify(b_val, c_val, higher_is_better, tol)
        diffs.append(MetricDiff(
            metric=path,
            baseline=b_val,
            candidate=c_val,
            delta=delta,
            threshold=tol,
            status=status,
            note=note,
        ))
    return diffs


def _secondary_probe_diffs(baseline, candidate, overrides):
    """Build MetricDiff objects for secondary sensitive-attribute probes.

    Returns an empty list when neither report has a non-empty secondary_probes
    list, so datasets without secondary labels never see spurious n/a rows.
    When an attribute appears in only one report it surfaces as 'unavailable'
    rather than being silently skipped.

    Higher decodability of a sensitive attribute is WORSE (privacy regression),
    so both metrics use higher_is_better=False.
    """
    b_list = baseline.get("secondary_probes") or []
    c_list = candidate.get("secondary_probes") or []

    if not b_list and not c_list:
        return []

    b_by_attr = {e["attribute"]: e for e in b_list if isinstance(e, dict) and "attribute" in e}
    c_by_attr = {e["attribute"]: e for e in c_list if isinstance(e, dict) and "attribute" in e}

    diffs = []
    for attr in sorted(set(b_by_attr) | set(c_by_attr)):
        b_entry = b_by_attr.get(attr)
        c_entry = c_by_attr.get(attr)

        for metric_key, higher_is_better, default_tol in SECONDARY_PROBE_METRIC_SPECS:
            path = f"secondary_probes[{attr}].{metric_key}"
            tol = overrides.get(path, default_tol)
            b_val = b_entry.get(metric_key) if b_entry else None
            c_val = c_entry.get(metric_key) if c_entry else None
            status, delta, note = _classify(b_val, c_val, higher_is_better, tol)
            diffs.append(MetricDiff(
                metric=path,
                baseline=b_val,
                candidate=c_val,
                delta=delta,
                threshold=tol,
                status=status,
                note=note,
            ))
    return diffs


def compare_reports(baseline, candidate, thresholds=None, on_stage=None):
    """Compare a baseline audit report against a candidate and detect regressions.

    Parameters
    ----------
    baseline, candidate:
        Dicts as returned by nerveml.scan.run_scan(), or loaded from
        audit_report.json files.
    thresholds:
        Optional dict overriding per-metric tolerances. Keys are dotted metric
        paths matching TRACKED_METRICS; values are float tolerances.
    on_stage:
        Optional callable(stage_key: str) invoked before each comparison
        category. Keys: "aggregate", "multi_axis", "artifact",
        "secondary_probes", "claim". Used by the CLI --verbose flag; safe to
        ignore (default None means silent).

    Returns
    -------
    RegressionResult
        result.passed is False when at least one regression is found.
        result.to_dict() is fully JSON-serialisable.
    """
    def _stage(key):
        if on_stage is not None:
            on_stage(key)

    overrides = thresholds or {}
    diffs = []
    has_regression = False

    _stage("aggregate")
    for path, higher_is_better, default_tol in TRACKED_METRICS:
        tol = overrides.get(path, default_tol)
        b_val = _get(baseline, path)
        c_val = _get(candidate, path)

        status, delta, note = _classify(b_val, c_val, higher_is_better, tol)
        if status == "regression":
            has_regression = True

        # For worst-unit regressions, annotate which unit changed so CI
        # operators can distinguish "same unit got worse" from "new unit is now
        # the floor".  Applied whenever a regression fires — stable/improvement
        # cases are unambiguous without the extra context.
        if path == "evaluation.worst_group.balanced_accuracy" and status == "regression":
            b_grp = _get(baseline, "evaluation.worst_group.group")
            c_grp = _get(candidate, "evaluation.worst_group.group")
            if b_grp is not None or c_grp is not None:
                if b_grp == c_grp:
                    note = f"{note}; worst unit unchanged: {b_grp}"
                else:
                    note = f"{note}; worst unit: baseline={b_grp}, candidate={c_grp}"

        diffs.append(
            MetricDiff(
                metric=path,
                baseline=b_val,
                candidate=c_val,
                delta=delta,
                threshold=tol,
                status=status,
                note=note,
            )
        )

    _stage("multi_axis")
    for diff in _multi_axis_diffs(baseline, candidate, overrides):
        if diff.status == "regression":
            has_regression = True
        diffs.append(diff)

    _stage("artifact")
    for diff in _artifact_diffs(baseline, candidate, overrides):
        if diff.status == "regression":
            has_regression = True
        diffs.append(diff)

    _stage("secondary_probes")
    for diff in _secondary_probe_diffs(baseline, candidate, overrides):
        if diff.status == "regression":
            has_regression = True
        diffs.append(diff)

    _stage("claim")
    b_claim = baseline.get("permitted_claim", "")
    c_claim = candidate.get("permitted_claim", "")

    return RegressionResult(
        metrics=diffs,
        passed=not has_regression,
        permitted_claim_changed=b_claim != c_claim,
        baseline_claim=b_claim,
        candidate_claim=c_claim,
    )
