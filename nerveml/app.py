"""Streamlit interface (spec section 13).

Presentation only. Every number rendered here comes from a report produced by
nerveml.scan, and nothing is displayed before a scan has actually run - the
demo must never show a placeholder that could be mistaken for a result.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nerveml.loaders import load_feature_csv, validate_dataset
from nerveml.risk_rules import LEAKAGE_TYPE_HIGH_CORR, LEAKAGE_TYPE_LABEL_NAME

_LEAKAGE_SMELL_CODES = frozenset({LEAKAGE_TYPE_LABEL_NAME, LEAKAGE_TYPE_HIGH_CORR})
from nerveml.models import MODEL_KINDS
from nerveml.plots import (
    plot_fold_scores,
    plot_identity,
    plot_null_distribution,
    plot_per_group,
    plot_scorecard,
    plot_top_features,
)
from nerveml.scan import STAGES, fold_metrics_frame, run_scan
from nerveml.scenarios import SCENARIOS, describe
from nerveml.synth import MODES

UPLOAD_OPTION = "upload a feature CSV"

RISK_HELP = {
    "elevated": "The target appears inferable across held-out subjects.",
    "moderate": "Some information may be present; not enough for cross-person claims.",
    "low_or_inconclusive": "No convincing cross-subject evidence under this test.",
    "inconclusive_at_this_sample_size": (
        "The confidence interval spans a decision threshold, so this sample "
        "cannot settle the question either way."
    ),
}
# A verdict this interface has no wording for must not take the page down.
UNKNOWN_RISK_HELP = "See the risk report below for what this verdict means."

st.set_page_config(page_title="NerveML", layout="wide")


def sidebar():
    """Screens 1 and 2: dataset selection and evaluation configuration."""
    with st.sidebar:
        st.subheader("1 · Dataset")
        # Scenarios first: a demo should open on the situation the tool is
        # sold into, not on the fixture that isolates the mechanism.
        dataset = st.selectbox(
            "Source",
            list(SCENARIOS) + list(MODES) + [UPLOAD_OPTION],
            key="dataset",
        )
        if dataset in SCENARIOS:
            st.caption(describe(dataset)["pitch"])

        uploaded = None
        if dataset == UPLOAD_OPTION:
            uploaded = st.file_uploader("Feature table (CSV)", type="csv", key="upload")
            st.caption(
                "One row per trial, with subject_id, trial_id and target_label. "
                "Remaining columns are treated as features."
            )

        synthetic_only = uploaded is not None or dataset in SCENARIOS
        st.number_input("Subjects", 4, 200, 20, key="subjects", disabled=synthetic_only)
        st.number_input(
            "Trials per subject", 10, 400, 40, key="trials", disabled=synthetic_only
        )

        st.subheader("2 · Evaluation")
        st.text_input(
            "Independent grouping column",
            value="subject_id",
            key="group_column",
            help="The unit held out whole: subject_id, session_id, site_id, "
            "device_id, stimulus_id.",
        )
        st.selectbox("Model", MODEL_KINDS, key="model")
        st.number_input("Cross-validation folds", 2, 20, 5, key="folds")
        st.number_input("Permutations", 10, 1000, 100, key="permutations")
        st.caption(
            "Both protocols use these features and this model family. Only the "
            "splitting rule differs."
        )

        run = st.button("Run scan", type="primary", key="run", width="stretch")
    return dataset, uploaded, run


def show_dataset_summary(dataset, uploaded, folds):
    if uploaded is None:
        return
    df, feats = load_feature_csv(uploaded)
    summary = validate_dataset(df, feats, n_splits=folds)
    st.subheader("Dataset inventory")
    st.write(
        f"{summary.n_subjects} subjects · {summary.n_trials} trials · "
        f"{summary.n_features} features · class balance {summary.class_balance}"
    )
    for warning in summary.warnings:
        if warning.get("code") in _LEAKAGE_SMELL_CODES:
            st.error(f"**Label leakage smell ({warning['code']}):** {warning['message']}")
        else:
            st.warning(warning["message"])


def dataset_leakage_warnings_section(report):
    """Show leakage-smell warnings from the stored report before result cards.

    These warnings mean the scan numbers may be inflated by label leakage, so
    they need to appear before the metrics, not buried in the dataset JSON.
    """
    warnings = report.get("dataset", {}).get("warnings", [])
    leakage = [w for w in warnings if w.get("code") in _LEAKAGE_SMELL_CODES]
    if not leakage:
        return
    for w in leakage:
        st.error(
            f"**Label leakage smell detected ({w['code']})** — {w['message']}"
        )


def client_context(report):
    """What the team said about the data, before anything was measured."""
    context = report["dataset"].get("context")
    if not context:
        return

    st.markdown(f"**Submitted for review — {context['customer']}**")
    st.markdown(context["pitch"])
    st.markdown(f"*Reported:* {context['reported_metric']}")
    st.caption(
        "Constructed demonstration data. The client is fictional and the "
        "recordings are synthetic; the failure inside them is real, and the "
        "scan was not told about it."
    )


def result_cards(report):
    """Screen 4: the four numbers that carry the insight."""
    evaluation = report["evaluation"]
    risk = report["risk_flags"]
    naive = evaluation["trial_random_balanced_accuracy"]
    grouped = evaluation["grouped_balanced_accuracy"]
    unit = evaluation["grouping_unit"].removesuffix("_id")
    gap = evaluation["generalization_gap"]

    columns = st.columns(4)
    columns[0].metric(
        "trial-random balanced accuracy",
        f"{naive:.3f}",
        help="Weaker protocol: the same subject can appear in train and test.",
    )
    columns[1].metric(
        f"{unit}-held-out balanced accuracy",
        f"{grouped:.3f}",
        delta=f"{-gap:+.3f} vs trial-random",
        delta_color="inverse" if gap > 0 else "normal",
        help=f"Strict protocol: every {unit} is wholly in train or wholly in test.",
    )
    interval = evaluation["confidence_intervals"]["generalization_gap"]
    columns[2].metric(
        "generalization gap",
        f"{gap:+.3f}",
        delta=(
            None
            if None in interval
            else f"95% CI {interval[0]:+.3f} to {interval[1]:+.3f}"
        ),
        delta_color="off",
        help="Trial-random minus held-out. Descriptive, not an accusation.",
    )
    columns[3].metric(
        "sensitive inference risk",
        risk["sensitive_inference_evidence"].replace("_", " "),
        help=RISK_HELP.get(
            risk["sensitive_inference_evidence"], UNKNOWN_RISK_HELP
        ),
    )


def re_identification(report):
    """The privacy half: can a held-out record be attached to its producer?"""
    identity = report.get("identity_inference")
    if identity is None:
        return

    st.subheader("Re-identification")
    columns = st.columns(4)
    columns[0].metric(
        "re-identification accuracy",
        f"{identity['accuracy']:.3f}",
        help="Share of held-out records attached to the right individual.",
    )
    columns[1].metric("chance", f"{identity['chance']:.3f}")
    columns[2].metric("lift over chance", f"{identity['lift_over_chance']:.1f}x")
    columns[3].metric("identities", f"{identity['n_identities']}")

    top = identity["most_identifiable"]
    weakest = min(identity["per_identity"], key=lambda e: e["recall"])
    st.markdown(
        f"Most identifiable: **{top['group']}** at {top['recall']:.2f} recall. "
        f"Least: **{weakest['group']}** at {weakest['recall']:.2f}."
    )
    composition = report.get("fingerprint_composition")
    if composition is not None:
        st.markdown("**What is the fingerprint made of?**")
        left, right = st.columns(2)
        left.metric(
            "amplitude only",
            f"{composition['amplitude_only']['accuracy']:.3f}",
            help="Per-channel level alone: skull thickness, impedance, cap fit.",
        )
        right.metric(
            "spectral shape only",
            f"{composition['spectral_shape_only']['accuracy']:.3f}",
            help="Each channel's overall level divided out.",
        )
        st.caption(composition["interpretation"])

    st.pyplot(plot_identity(report), width="stretch")

    # The number means nothing without the threat model it was measured under.
    st.info(f"**Attack model.** {identity['attack_model']}")


def multi_axis_validation_section(report):
    """Extra held-out axes when the dataset carried session/site/device/stimulus columns."""
    entries = report.get("multi_axis_validation", [])
    if not entries:
        return

    st.subheader("Multi-axis held-out validation")
    rows = []
    for entry in entries:
        worst = entry.get("worst_group") or {}
        ci = entry.get("confidence_intervals", {})
        bacc_ci = ci.get("grouped_balanced_accuracy", [None, None])
        if bacc_ci and bacc_ci[0] is not None:
            ci_str = f"[{bacc_ci[0]:.3f}, {bacc_ci[1]:.3f}]"
        else:
            ci_str = "—"
        gap_ci = ci.get("generalization_gap", [None, None])
        if gap_ci and gap_ci[0] is not None:
            gap_ci_str = f"[{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}]"
        else:
            gap_ci_str = "—"
        rows.append({
            "axis": entry["axis"],
            "n_units": entry["n_units"],
            "grouped_balanced_accuracy": entry["grouped_balanced_accuracy"],
            "bal-acc 95% CI": ci_str,
            "grouped_roc_auc": entry["grouped_roc_auc"],
            "generalization_gap": entry["generalization_gap"],
            "gap 95% CI": gap_ci_str,
            "worst_group": worst.get("group", "—"),
            "worst_bacc": (
                f"{worst['balanced_accuracy']:.3f}" if worst.get("balanced_accuracy") is not None
                else "—"
            ),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(
        "Each row is an independent strict held-out evaluation under the same "
        "protocol as the primary split. These are associations under the stated "
        "attack model, not independent validity claims."
    )


def evidence(report):
    """Screen 5: the evidence behind the cards."""
    st.subheader("Evidence")
    left, right = st.columns(2)
    with left:
        st.pyplot(plot_fold_scores(report), width="stretch")
    with right:
        st.pyplot(plot_null_distribution(report), width="stretch")

    worst = report["evaluation"]["worst_group"]
    if worst is not None:
        st.markdown(
            f"The mean above hides **{worst['group']}**, whose held-out "
            f"balanced accuracy was {worst['balanced_accuracy']:.3f} across "
            f"{worst['n_trials']} trials."
        )
    per_group = report["evaluation"].get("per_group", [])
    if per_group:
        unit = report["evaluation"]["grouping_unit"].removesuffix("_id")
        df = pd.DataFrame(
            sorted(per_group, key=lambda e: e["balanced_accuracy"]),
        )[["group", "balanced_accuracy", "roc_auc", "n_trials"]].rename(columns={
            "group": unit,
            "balanced_accuracy": "balanced accuracy",
            "roc_auc": "ROC-AUC",
            "n_trials": "trials",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Sorted worst-first. Balanced accuracy of 0.50 is chance-level. "
            "Association under this attack model — not a claim about the "
            "individual."
        )
    st.pyplot(plot_per_group(report), width="stretch")

    st.pyplot(plot_top_features(report), width="stretch")
    st.caption(report["feature_caveat"])

    artifact_baseline_section(report)
    secondary_probes_section(report)


def secondary_probes_section(report):
    """Secondary sensitive-attribute probes found alongside the primary label."""
    probes = report.get("secondary_probes", [])
    if not probes:
        return

    st.markdown("#### Secondary sensitive-attribute probes")
    rows = []
    for p in probes:
        rows.append({
            "attribute": p["attribute"],
            "n_units": p["n_units"],
            "grouped_balanced_accuracy": p["grouped_balanced_accuracy"],
            "grouped_roc_auc": p["grouped_roc_auc"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "These extra columns were found alongside the primary label and probed "
        "with the same features under the same held-out protocol. Balanced "
        "accuracy above 0.50 indicates the features carry information about that "
        "attribute — an association, not a causal or diagnostic claim."
    )


def artifact_baseline_section(report):
    """Render the EOG/EMG artifact-proxy baseline, if available."""
    ab = report.get("artifact_baseline")
    if not ab:
        return

    st.markdown("#### Artifact-confound baseline")
    rows = []
    for key, label in (("eog_proxy", "EOG proxy (frontal channels)"),
                       ("emg_proxy", "EMG proxy (gamma band)")):
        p = ab.get(key)
        if p:
            share = ab.get(f"artifact_share_{p['name']}")
            rows.append({
                "proxy": label,
                "features": p["n_features"],
                "balanced accuracy": p["balanced_accuracy"],
                "ROC-AUC": p["roc_auc"],
                "artifact share": f"{share:.0%}" if share is not None else "n/a",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(ab["interpretation"])


@st.cache_data(show_spinner=False)
def _pdf_bytes(report):
    """Render the PDF once per report, with its figures embedded."""
    import tempfile
    from pathlib import Path

    from nerveml.pdf import write_pdf_report
    from nerveml.plots import write_figures

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        figures = write_figures(report, folder)
        path = write_pdf_report(report, folder / "risk_report.pdf", figures=figures)
        return path.read_bytes()


def report_section(report):
    """Screen 6: flags, bounded claims, remediation, downloads."""
    st.subheader("Risk report")

    for flag in report["risk_flags"]["flags"]:
        render = st.warning if flag["severity"] == "warning" else st.info
        render(f"**{flag['code']}** — {flag['message']}\n\nRule: {flag['rule']}")

    st.markdown("#### Supported claim")
    st.markdown(f"> {report['permitted_claim']}")

    st.markdown("#### Not supported by this scan")
    st.markdown("\n".join(f"- {claim}" for claim in report["unsupported_claims"]))

    st.markdown("#### Recommended next steps")
    st.markdown("\n".join(f"- {item}" for item in report["recommendations"]))

    st.caption(report["risk_flags"]["heuristic_notice"])

    left, middle, right = st.columns(3)
    middle.download_button(
        "Download risk_report.pdf",
        _pdf_bytes(report),
        file_name="risk_report.pdf",
        mime="application/pdf",
        width="stretch",
    )
    left.download_button(
        "Download audit_report.json",
        json.dumps(report, indent=2, ensure_ascii=False),
        file_name="audit_report.json",
        mime="application/json",
        width="stretch",
    )
    right.download_button(
        "Download fold_metrics.csv",
        fold_metrics_frame(report).to_csv(index=False),
        file_name="fold_metrics.csv",
        mime="text/csv",
        width="stretch",
    )

    with st.expander("Dataset summary and per-fold metrics"):
        st.json(report["dataset"], expanded=False)
        st.dataframe(fold_metrics_frame(report), width="stretch")


def main():
    st.title("NerveML — neural data privacy and validation scanner")
    st.caption(
        "Prototype. Not a clinical, legal, or regulatory certification system."
    )

    dataset, uploaded, run = sidebar()
    show_dataset_summary(dataset, uploaded, st.session_state.get("folds", 5))

    if run:
        source = uploaded if uploaded is not None else dataset
        if dataset == UPLOAD_OPTION and uploaded is None:
            st.error("Select a CSV file, or pick one of the demo datasets.")
        else:
            # Screen 3. The labels come from the scan's own stage boundaries,
            # so the indicator cannot drift out of step with the work.
            with st.status("Starting scan…", expanded=True) as status:
                try:
                    st.session_state["report"] = run_scan(
                        dataset=source,
                        n_subjects=st.session_state["subjects"],
                        n_trials=st.session_state["trials"],
                        model_kind=st.session_state["model"],
                        group_column=st.session_state["group_column"],
                        n_splits=st.session_state["folds"],
                        n_permutations=st.session_state["permutations"],
                        seed=0,
                        on_stage=lambda name: status.update(label=STAGES[name]),
                    )
                    status.update(label="Scan complete", state="complete")
                except ValueError as error:
                    st.session_state.pop("report", None)
                    status.update(label="Scan stopped", state="error")
                    st.error(str(error))

    report = st.session_state.get("report")
    if report is None:
        st.info(
            "Choose a dataset and run a scan. Nothing is shown until a scan has "
            "produced real numbers."
        )
        return

    client_context(report)
    dataset_leakage_warnings_section(report)
    st.pyplot(plot_scorecard(report), width="stretch")
    result_cards(report)
    re_identification(report)
    evidence(report)
    multi_axis_validation_section(report)
    report_section(report)


main()
