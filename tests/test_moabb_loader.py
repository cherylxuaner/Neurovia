"""MOABB dataset loader — contract checks without the network, plus a slow
end-to-end download test.

moabb is an optional dependency; every test here skips cleanly when it is not
installed, and the one test that downloads is marked slow.
"""

import pandas as pd
import pytest

from nerveml.moabb_loader import KNOWN_DATASETS, load_moabb_dataset


def test_known_datasets_are_positive_subject_counts():
    assert KNOWN_DATASETS
    assert all(isinstance(n, int) and n > 0 for n in KNOWN_DATASETS.values())
    assert "BNCI2014_001" in KNOWN_DATASETS


def test_cache_is_served_without_moabb(tmp_path):
    """A cached CSV is returned directly, so no moabb import or download."""
    csv = tmp_path / "cached.csv"
    frame = pd.DataFrame({
        "subject_id": ["S001", "S001"],
        "session_id": [0, 0],
        "trial_id": ["t0", "t1"],
        "target_label": [0, 1],
        "C3_alpha": [0.1, 0.2],
    })
    frame.to_csv(csv, index=False)
    out = load_moabb_dataset("BNCI2014_001", cache_path=csv)
    assert list(out["target_label"]) == [0, 1]


def test_unknown_dataset_rejected():
    pytest.importorskip("moabb")
    with pytest.raises(ValueError, match="unknown MOABB dataset"):
        load_moabb_dataset("NotARealDataset")


@pytest.mark.slow
def test_bnci2014_001_end_to_end(tmp_path):
    """Download two subjects of BCI Competition IV 2a and featurise them."""
    pytest.importorskip("moabb")
    from nerveml.loaders import validate_dataset

    df = load_moabb_dataset("BNCI2014_001", n_subjects=2,
                            cache_path=tmp_path / "bnci.csv")
    assert {"subject_id", "session_id", "trial_id", "target_label"} <= set(df.columns)
    assert set(df["target_label"].unique()) <= {0, 1}
    feats = [c for c in df.columns
             if c not in ("subject_id", "session_id", "trial_id", "target_label")]
    # Feature table must satisfy the same integrity contract as any other input.
    summary = validate_dataset(df, feats, n_splits=2)
    assert summary.n_subjects == 2
