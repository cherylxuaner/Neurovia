"""Spec section 11.2, as tests.

These thresholds are prototype heuristics for the user experience, not
scientific or legal standards, and the report has to say so. Every boundary is
pinned here because an off-by-one comparison silently changes what the product
tells a customer about their data.
"""

import pytest

from nerveml.risk_rules import (
    ALPHA,
    ARTIFACT_SHARE_HIGH,
    AUC_ELEVATED,
    AUC_LOW,
    GAP_HIGH,
    GAP_POSSIBLE,
    LEAKAGE_TYPE_HIGH_CORR,
    LEAKAGE_TYPE_LABEL_NAME,
    SECONDARY_BACC_ELEVATED,
    assess_risk,
)

SIGNIFICANT = 0.01


def assess(auc=0.5, gap=0.0, p_value=1.0):
    return assess_risk(grouped_auc=auc, generalization_gap=gap, p_value=p_value)


def test_weak_grouped_auc_is_inconclusive():
    assert assess(auc=0.52).sensitive_inference == "low_or_inconclusive"


def test_middling_grouped_auc_is_moderate():
    assert assess(auc=0.60, p_value=SIGNIFICANT).sensitive_inference == "moderate"


def test_strong_grouped_auc_with_null_evidence_is_elevated():
    assert assess(auc=0.70, p_value=SIGNIFICANT).sensitive_inference == "elevated"


def test_strong_grouped_auc_without_null_evidence_is_not_elevated():
    # A high score that cannot beat its own null is not evidence of inference.
    result = assess(auc=0.70, p_value=0.30)

    assert result.sensitive_inference == "moderate"
    assert "insufficient_null_evidence" in result.flag_codes


@pytest.mark.parametrize(
    "auc,expected",
    [
        (AUC_LOW - 0.001, "low_or_inconclusive"),
        (AUC_LOW, "moderate"),
        (AUC_ELEVATED, "moderate"),
        (AUC_ELEVATED + 0.001, "elevated"),
    ],
)
def test_inference_thresholds_fall_on_the_documented_side(auc, expected):
    assert assess(auc=auc, p_value=SIGNIFICANT).sensitive_inference == expected


@pytest.mark.parametrize(
    "gap,expected",
    [
        (GAP_POSSIBLE - 0.001, "none"),
        (GAP_POSSIBLE, "possible"),
        (GAP_HIGH - 0.001, "possible"),
        (GAP_HIGH, "high"),
    ],
)
def test_gap_thresholds_fall_on_the_documented_side(gap, expected):
    assert assess(gap=gap).subject_dependence == expected


def test_p_value_exactly_at_alpha_counts_as_insufficient():
    assert "insufficient_null_evidence" in assess(auc=0.70, p_value=ALPHA).flag_codes


def test_a_large_gap_raises_a_high_generalization_warning():
    result = assess(gap=0.25)

    assert result.subject_dependence == "high"
    assert "high_generalization_warning" in result.flag_codes


def test_a_clean_result_raises_no_dependence_flags():
    result = assess(auc=0.72, gap=0.02, p_value=SIGNIFICANT)

    assert result.subject_dependence == "none"
    assert "possible_subject_dependence" not in result.flag_codes
    assert "high_generalization_warning" not in result.flag_codes


def test_every_flag_states_the_rule_that_produced_it():
    result = assess(auc=0.70, gap=0.25, p_value=0.30)

    assert result.flags
    for flag in result.flags:
        assert flag["rule"]
        assert flag["message"]
        assert flag["severity"] in {"info", "warning"}


def test_at_least_one_flag_is_always_produced():
    # Spec P0: the scan must emit a warning and a bounded interpretation.
    assert assess(auc=0.50, gap=0.0, p_value=1.0).flags


def test_permitted_claim_uses_bounded_language():
    claim = assess(auc=0.70, p_value=SIGNIFICANT).permitted_claim

    assert "self-reported" in claim
    assert "held-out subjects" in claim
    assert "true emotion" not in claim.lower()


def test_inconclusive_results_do_not_claim_decodability():
    claim = assess(auc=0.51, p_value=0.4).permitted_claim

    assert "not" in claim.lower()


def test_unsupported_claims_always_reject_mind_reading():
    unsupported = assess(auc=0.99, p_value=SIGNIFICANT).unsupported_claims

    assert any("true emotion" in c.lower() for c in unsupported)
    assert any("identif" in c.lower() for c in unsupported)


def test_recommendations_are_always_offered():
    assert assess(auc=0.52).recommendations
    assert assess(auc=0.80, gap=0.30, p_value=SIGNIFICANT).recommendations


def test_high_dependence_recommends_holding_out_more_than_subjects():
    text = " ".join(assess(gap=0.30).recommendations).lower()

    assert "session" in text or "site" in text


def test_a_score_built_only_on_telling_people_apart_is_flagged():
    result = assess_risk(
        grouped_auc=0.45,
        generalization_gap=0.40,
        p_value=0.5,
        pooled_balanced_accuracy=0.844,
        within_unit_balanced_accuracy=0.496,
    )

    assert "between_unit_discrimination_only" in result.flag_codes
    flag = next(f for f in result.flags if f["code"] == "between_unit_discrimination_only")
    assert flag["severity"] == "warning"
    assert "0.844" in flag["rule"] and "0.496" in flag["rule"]


def test_a_genuine_signal_is_not_flagged_as_between_unit_only():
    result = assess_risk(
        grouped_auc=0.92,
        generalization_gap=0.01,
        p_value=0.01,
        pooled_balanced_accuracy=0.845,
        within_unit_balanced_accuracy=0.845,
    )

    assert "between_unit_discrimination_only" not in result.flag_codes


def test_the_decomposition_is_optional():
    # Callers that have not computed it still get an assessment.
    assert assess(auc=0.60, p_value=SIGNIFICANT).sensitive_inference == "moderate"


def test_a_verdict_needs_the_whole_interval_on_one_side():
    # Point estimate 0.70 would read "elevated"; an interval reaching down to
    # 0.52 means twenty more subjects could say the opposite.
    result = assess_risk(
        grouped_auc=0.70,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        grouped_auc_ci=(0.52, 0.88),
    )

    assert result.sensitive_inference == "inconclusive_at_this_sample_size"
    assert "undecided_at_this_sample_size" in result.flag_codes


def test_an_interval_clear_of_the_threshold_still_gets_its_verdict():
    result = assess_risk(
        grouped_auc=0.80,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        grouped_auc_ci=(0.72, 0.88),
    )

    assert result.sensitive_inference == "elevated"
    assert "undecided_at_this_sample_size" not in result.flag_codes


def test_an_interval_wholly_below_the_low_threshold_is_conclusive():
    result = assess_risk(
        grouped_auc=0.50, generalization_gap=0.0, p_value=1.0,
        grouped_auc_ci=(0.44, 0.53),
    )

    assert result.sensitive_inference == "low_or_inconclusive"


def test_an_estimate_sitting_on_a_threshold_needs_far_more_units():
    # 0.555 against a threshold at 0.55: deciding which side it falls on is a
    # much harder question than merely tightening the interval a little.
    on_the_line = assess_risk(
        grouped_auc=0.555, generalization_gap=0.0, p_value=SIGNIFICANT,
        grouped_auc_ci=(0.521, 0.606), n_units=20,
    )
    well_clear = assess_risk(
        grouped_auc=0.60, generalization_gap=0.0, p_value=SIGNIFICANT,
        grouped_auc_ci=(0.52, 0.68), n_units=20,
    )

    assert on_the_line.units_needed > well_clear.units_needed
    # And it must never claim the sample already in hand would do.
    assert on_the_line.units_needed > 20


def test_an_undecided_verdict_says_how_many_units_would_settle_it():
    result = assess_risk(
        grouped_auc=0.70,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        grouped_auc_ci=(0.52, 0.88),
        n_units=20,
    )

    flag = next(f for f in result.flags if f["code"] == "undecided_at_this_sample_size")
    assert "20" in flag["rule"]
    assert result.units_needed > 20


def test_a_gap_whose_interval_crosses_zero_is_not_called_dependence():
    result = assess_risk(
        grouped_auc=0.60,
        generalization_gap=0.12,
        p_value=SIGNIFICANT,
        gap_ci=(-0.04, 0.28),
    )

    assert result.subject_dependence == "undecided"
    assert "possible_subject_dependence" not in result.flag_codes


def test_a_gap_whose_interval_clears_the_threshold_is_called():
    result = assess_risk(
        grouped_auc=0.60,
        generalization_gap=0.25,
        p_value=SIGNIFICANT,
        gap_ci=(0.21, 0.29),
    )

    assert result.subject_dependence == "high"


def test_without_intervals_the_point_estimate_verdicts_still_work():
    # Callers that cannot compute intervals are not blocked from a verdict.
    assert assess(auc=0.70, p_value=SIGNIFICANT).sensitive_inference == "elevated"
    assert assess(gap=0.25).subject_dependence == "high"


def test_the_permitted_claim_carries_the_interval():
    result = assess_risk(
        grouped_auc=0.80,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        grouped_auc_ci=(0.72, 0.88),
    )

    assert "0.72" in result.permitted_claim and "0.88" in result.permitted_claim


def test_assessment_declares_itself_a_prototype_heuristic():
    assert "prototype" in assess().heuristic_notice.lower()


# ── Artifact-baseline risk flags ─────────────────────────────────────────────

def _assess_artifact(eog=None, emg=None):
    return assess_risk(
        grouped_auc=0.70,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        artifact_share_eog=eog,
        artifact_share_emg=emg,
    )


def test_high_eog_share_raises_flag():
    result = _assess_artifact(eog=ARTIFACT_SHARE_HIGH)
    assert "high_eog_artifact_share" in result.flag_codes


def test_eog_share_below_threshold_raises_no_flag():
    result = _assess_artifact(eog=ARTIFACT_SHARE_HIGH - 0.01)
    assert "high_eog_artifact_share" not in result.flag_codes


def test_high_emg_share_raises_flag():
    result = _assess_artifact(emg=ARTIFACT_SHARE_HIGH)
    assert "high_emg_artifact_share" in result.flag_codes


def test_emg_share_below_threshold_raises_no_flag():
    result = _assess_artifact(emg=ARTIFACT_SHARE_HIGH - 0.01)
    assert "high_emg_artifact_share" not in result.flag_codes


def test_none_artifact_shares_raise_no_flags():
    result = _assess_artifact(eog=None, emg=None)
    assert "high_eog_artifact_share" not in result.flag_codes
    assert "high_emg_artifact_share" not in result.flag_codes


def test_artifact_flag_is_warning_severity():
    result = _assess_artifact(eog=0.80)
    flag = next(f for f in result.flags if f["code"] == "high_eog_artifact_share")
    assert flag["severity"] == "warning"


def test_artifact_flag_cites_share_and_threshold():
    result = _assess_artifact(eog=0.72)
    flag = next(f for f in result.flags if f["code"] == "high_eog_artifact_share")
    assert "0.72" in flag["rule"]
    assert str(ARTIFACT_SHARE_HIGH) in flag["rule"]


def test_both_artifact_flags_fire_independently():
    result = _assess_artifact(eog=0.80, emg=0.75)
    assert "high_eog_artifact_share" in result.flag_codes
    assert "high_emg_artifact_share" in result.flag_codes


# ── Secondary-attribute risk flags ───────────────────────────────────────────

def _make_probe(attr, bacc):
    return {"attribute": attr, "grouped_balanced_accuracy": bacc}


def _assess_secondary(probes):
    return assess_risk(
        grouped_auc=0.70,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        secondary_probes=probes,
    )


def test_elevated_secondary_probe_raises_flag():
    result = _assess_secondary([_make_probe("arousal", SECONDARY_BACC_ELEVATED)])
    assert "secondary_attribute_decodable" in result.flag_codes


def test_secondary_probe_below_threshold_raises_no_flag():
    result = _assess_secondary([_make_probe("arousal", SECONDARY_BACC_ELEVATED - 0.01)])
    assert "secondary_attribute_decodable" not in result.flag_codes


def test_none_secondary_probes_raises_no_flag():
    result = _assess_secondary(None)
    assert "secondary_attribute_decodable" not in result.flag_codes


def test_empty_secondary_probes_raises_no_flag():
    result = _assess_secondary([])
    assert "secondary_attribute_decodable" not in result.flag_codes


def test_secondary_flag_names_the_attribute():
    result = _assess_secondary([_make_probe("sleep_stage", 0.72)])
    flag = next(f for f in result.flags if f["code"] == "secondary_attribute_decodable")
    assert "sleep_stage" in flag["message"]


def test_multiple_elevated_secondary_probes_each_get_a_flag():
    probes = [
        _make_probe("arousal", 0.68),
        _make_probe("fatigue", 0.71),
        _make_probe("stress", 0.50),   # below threshold
    ]
    result = _assess_secondary(probes)
    codes = result.flag_codes
    # Should fire twice (arousal and fatigue), not for stress
    arousal_flags = [f for f in result.flags
                     if f["code"] == "secondary_attribute_decodable"
                     and "arousal" in f["message"]]
    fatigue_flags = [f for f in result.flags
                     if f["code"] == "secondary_attribute_decodable"
                     and "fatigue" in f["message"]]
    assert len(arousal_flags) == 1
    assert len(fatigue_flags) == 1
    # Stress should NOT have triggered a flag
    stress_flags = [f for f in result.flags
                    if f["code"] == "secondary_attribute_decodable"
                    and "stress" in f["message"]]
    assert len(stress_flags) == 0


def test_secondary_flag_is_warning_severity():
    result = _assess_secondary([_make_probe("medication", 0.70)])
    flag = next(f for f in result.flags if f["code"] == "secondary_attribute_decodable")
    assert flag["severity"] == "warning"


def test_secondary_flag_uses_bounded_language():
    result = _assess_secondary([_make_probe("arousal", 0.65)])
    flag = next(f for f in result.flags if f["code"] == "secondary_attribute_decodable")
    assert "association" in flag["message"].lower() or "protocol" in flag["message"].lower()


# ── Dataset leakage-smell risk flags ─────────────────────────────────────────

def _make_dw(code, message="test warning"):
    return {"code": code, "message": message}


def _assess_leakage(warnings):
    return assess_risk(
        grouped_auc=0.70,
        generalization_gap=0.0,
        p_value=SIGNIFICANT,
        dataset_warnings=warnings,
    )


def test_label_name_warning_raises_label_leakage_smell_flag():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_LABEL_NAME)])
    assert "label_leakage_smell" in result.flag_codes


def test_high_corr_warning_raises_high_correlation_leakage_smell_flag():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_HIGH_CORR)])
    assert "high_correlation_leakage_smell" in result.flag_codes


def test_none_dataset_warnings_raises_no_leakage_flags():
    result = _assess_leakage(None)
    assert "label_leakage_smell" not in result.flag_codes
    assert "high_correlation_leakage_smell" not in result.flag_codes


def test_empty_dataset_warnings_raises_no_leakage_flags():
    result = _assess_leakage([])
    assert "label_leakage_smell" not in result.flag_codes
    assert "high_correlation_leakage_smell" not in result.flag_codes


def test_unrelated_warning_code_raises_no_leakage_flag():
    result = _assess_leakage([_make_dw("class_imbalance")])
    assert "label_leakage_smell" not in result.flag_codes
    assert "high_correlation_leakage_smell" not in result.flag_codes


def test_leakage_flag_is_warning_severity():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_LABEL_NAME)])
    flag = next(f for f in result.flags if f["code"] == "label_leakage_smell")
    assert flag["severity"] == "warning"


def test_high_corr_flag_is_warning_severity():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_HIGH_CORR)])
    flag = next(f for f in result.flags if f["code"] == "high_correlation_leakage_smell")
    assert flag["severity"] == "warning"


def test_leakage_flag_message_passes_through_from_validate_dataset():
    msg = "Feature 'valence' (|r|=0.99) has near-perfect correlation with the target."
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_HIGH_CORR, message=msg)])
    flag = next(f for f in result.flags if f["code"] == "high_correlation_leakage_smell")
    assert flag["message"] == msg


def test_leakage_flag_rule_mentions_prototype_heuristic():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_LABEL_NAME)])
    flag = next(f for f in result.flags if f["code"] == "label_leakage_smell")
    assert "prototype heuristic" in flag["rule"]


def test_both_leakage_types_fire_independently():
    warnings = [
        _make_dw(LEAKAGE_TYPE_LABEL_NAME, "name smell"),
        _make_dw(LEAKAGE_TYPE_HIGH_CORR, "correlation smell"),
    ]
    result = _assess_leakage(warnings)
    assert "label_leakage_smell" in result.flag_codes
    assert "high_correlation_leakage_smell" in result.flag_codes


def test_multiple_label_name_warnings_each_get_a_flag():
    warnings = [
        _make_dw(LEAKAGE_TYPE_LABEL_NAME, "first smell"),
        _make_dw(LEAKAGE_TYPE_LABEL_NAME, "second smell"),
    ]
    result = _assess_leakage(warnings)
    label_flags = [f for f in result.flags if f["code"] == "label_leakage_smell"]
    assert len(label_flags) == 2


def test_leakage_flag_label_name_rule_mentions_column_name():
    result = _assess_leakage([_make_dw(LEAKAGE_TYPE_LABEL_NAME)])
    flag = next(f for f in result.flags if f["code"] == "label_leakage_smell")
    assert "column" in flag["rule"] or "label" in flag["rule"]
