"""The probe registry: order, gating, and result accumulation.

Behaviour equivalence with the old hand-wired blocks is covered by the existing
scan/report tests (they assert the same report). These tests pin the seam
itself, so a future probe added to PROBE_REGISTRY has a contract to meet.
"""

from nerveml.scan import (
    PROBE_REGISTRY,
    ProbeContext,
    ProbeSpec,
    run_probes,
)


def _ctx(**overrides):
    base = dict(
        df=None, feats=[], group_column="subject_id", model=None,
        model_kind="logistic_regression", n_splits=5, seed=0,
        session_column=None, grouped_bacc=0.5, identity_probe=True,
    )
    base.update(overrides)
    return ProbeContext(**base)


def test_registry_is_the_four_probes_in_fixed_order():
    keys = [spec.key for spec in PROBE_REGISTRY]
    assert keys == ["identity", "composition", "artifact", "secondary_probes"]
    stages = [spec.stage for spec in PROBE_REGISTRY]
    assert stages == ["identity", "confounds", "artifact_baseline", "secondary_probe"]


def test_run_probes_only_announces_applicable_stages():
    announced = []
    spec_a = ProbeSpec("a", "stage_a", applies=lambda ctx: True, run=lambda ctx: "ra")
    spec_b = ProbeSpec("b", "stage_b", applies=lambda ctx: False, run=lambda ctx: "rb")

    import nerveml.scan as scan
    original = scan.PROBE_REGISTRY
    scan.PROBE_REGISTRY = [spec_a, spec_b]
    try:
        ctx = _ctx()
        run_probes(ctx, lambda name: announced.append(name))
    finally:
        scan.PROBE_REGISTRY = original

    assert announced == ["stage_a"]          # b was gated out
    assert ctx.results["a"] == "ra"
    assert "b" not in ctx.results


def test_later_probe_sees_earlier_result():
    """A spec's applies/run can read a result an earlier spec stored."""
    seen = {}

    def _second_run(c):
        seen["saw_identity"] = c.results["identity"]
        return "COMP"

    first = ProbeSpec("identity", "identity", applies=lambda c: True, run=lambda c: "ID")
    second = ProbeSpec(
        "composition", "confounds",
        applies=lambda c: c.results.get("identity") == "ID",
        run=_second_run,
    )

    import nerveml.scan as scan
    original = scan.PROBE_REGISTRY
    scan.PROBE_REGISTRY = [first, second]
    try:
        ctx = _ctx()
        run_probes(ctx, lambda name: None)
    finally:
        scan.PROBE_REGISTRY = original

    assert ctx.results["composition"] == "COMP"
    assert seen["saw_identity"] == "ID"


def test_identity_probe_flag_gates_the_identity_spec():
    spec = next(s for s in PROBE_REGISTRY if s.key == "identity")
    assert spec.applies(_ctx(identity_probe=True)) is True
    assert spec.applies(_ctx(identity_probe=False)) is False


def test_confounds_needs_identity_and_band_power():
    spec = next(s for s in PROBE_REGISTRY if s.key == "composition")
    # No identity result yet → does not apply, whatever the features look like.
    assert spec.applies(_ctx(feats=["C3_alpha", "C4_beta"])) is False
