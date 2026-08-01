"""Common Spatial Patterns decoder for raw multichannel epochs.

The feature-table audit pipeline reduces each epoch to log band power per
channel (``features.py``), which throws away the spatial covariance structure
that Common Spatial Patterns needs. This module keeps the raw epochs
``(n_trials, n_channels, n_samples)`` and learns supervised spatial filters that
maximise the variance ratio between the two classes.

Why this exists: on real motor-imagery EEG the band-power + logistic baseline
sits at chance across held-out subjects (that is honest — those features are
deliberately crude, and their weakness is what makes the identity result so
striking by contrast). CSP is the standard decoder that actually lifts
left-versus-right motor imagery above chance, so this module lets us report the
*optimised* decoder's number beside the baseline's, and show that even a good
task decoder does not close the gap to the near-perfect re-identification score.

The leakage red-line still holds. CSP is *supervised* — it uses the labels to
find its filters — so fitting it on the whole dataset before splitting would
leak exactly the way this product exists to catch. Every CSP fit here happens
inside a single CV training fold, via an sklearn ``Pipeline`` that is cloned per
fold. The transform is pure NumPy/SciPy; MNE is only needed to read raw EDF.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

# Guards the log-variance against a filtered component with zero variance.
_FLOOR = 1e-12

# Motor imagery lives in the mu (8-13 Hz) and beta (13-30 Hz) sensorimotor
# rhythms. CSP on the broadband signal fails because low-frequency drift
# dominates the variance it maximises, so a band-pass to this range is not a
# tuning knob but the standard, necessary front of a CSP decoder.
_DEFAULT_LOW_HZ = 8.0
_DEFAULT_HIGH_HZ = 30.0


class BandpassFilter(BaseEstimator, TransformerMixin):
    """Zero-phase band-pass over the time axis of raw epochs.

    Unsupervised — it never sees the labels — so applying it before the split
    leaks nothing; it lives in the pipeline only so the whole decoder travels as
    one object. A no-op when sampling_rate is None, which keeps the synthetic
    tests (whose sample axis has no real time base) filter-free.
    """

    def __init__(self, sampling_rate=None, low_hz=_DEFAULT_LOW_HZ,
                 high_hz=_DEFAULT_HIGH_HZ, order=4):
        self.sampling_rate = sampling_rate
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.sampling_rate is None:
            return X
        nyq = 0.5 * self.sampling_rate
        b, a = butter(self.order, [self.low_hz / nyq, self.high_hz / nyq], btype="band")
        # filtfilt along the last (time) axis, zero-phase so no group delay.
        return filtfilt(b, a, X, axis=-1)


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns spatial filter for binary classification.

    Fits ``n_components`` spatial filters by simultaneous diagonalisation of the
    two class covariance matrices, then returns the log-variance of the filtered
    signals as features — the standard CSP feature for a downstream linear
    classifier.

    Filters are chosen from both ends of the generalised-eigenvalue spectrum:
    eigenvalues near 1 mark filters whose output varies most for class 0, near 0
    for class 1. The most discriminative filters live at the extremes, so we
    take the components whose eigenvalue is furthest from 0.5.

    Parameters
    ----------
    n_components : int
        Number of spatial filters to keep. Capped at the channel count.
    reg : float
        Ridge added to each class covariance's diagonal before the eigenproblem,
        so a rank-deficient fold (fewer samples than channels) stays solvable.
    """

    def __init__(self, n_components=6, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    @staticmethod
    def _class_covariance(trials):
        """Mean trace-normalised covariance over the trials of one class.

        Trace-normalising each trial before averaging stops a few high-amplitude
        trials from dominating the class covariance — the amplitude-vs-signal
        failure mode ``features.py`` warns about, handled here at the covariance
        level instead of with a log.
        """
        covs = np.empty((trials.shape[0], trials.shape[1], trials.shape[1]))
        for i, trial in enumerate(trials):
            cov = trial @ trial.T
            trace = np.trace(cov)
            covs[i] = cov / trace if trace > 0 else cov
        return covs.mean(axis=0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 3:
            raise ValueError(
                f"CSP expects epochs shaped (n_trials, n_channels, n_samples), "
                f"got {X.shape}"
            )
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"CSP is a binary spatial filter; got {len(classes)} classes"
            )
        self.classes_ = classes
        n_channels = X.shape[1]
        n_components = min(self.n_components, n_channels)

        c0 = self._class_covariance(X[y == classes[0]])
        c1 = self._class_covariance(X[y == classes[1]])
        eye = np.eye(n_channels)
        c0 = c0 + self.reg * eye
        c1 = c1 + self.reg * eye

        # Generalised eigenproblem C0 w = lambda (C0 + C1) w. eigh returns
        # eigenvalues ascending in [0, 1]; the composite is symmetric so the
        # eigenvectors are real and the solve is stable.
        eigvals, eigvecs = eigh(c0, c0 + c1)

        # Most discriminative filters sit at the spectrum's extremes.
        order = np.argsort(np.abs(eigvals - 0.5))[::-1][:n_components]
        # (n_channels, n_components): columns are spatial filters.
        self.filters_ = eigvecs[:, order]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(
                f"CSP expects epochs shaped (n_trials, n_channels, n_samples), "
                f"got {X.shape}"
            )
        out = np.empty((X.shape[0], self.filters_.shape[1]))
        for i, trial in enumerate(X):
            projected = self.filters_.T @ trial  # (n_components, n_samples)
            var = projected.var(axis=1)
            var = var / (var.sum() + _FLOOR)  # normalise so features are comparable
            out[i] = np.log(var + _FLOOR)
        return out


def build_csp_decoder(n_components=6, reg=1e-6, sampling_rate=None,
                      low_hz=_DEFAULT_LOW_HZ, high_hz=_DEFAULT_HIGH_HZ):
    """Band-pass + CSP + LDA — the classic motor-imagery decoder.

    Returned unfitted, as a single sklearn Pipeline, so a caller cloning it per
    CV fold fits the CSP filters on training epochs alone. Pass sampling_rate to
    enable the mu+beta band-pass (needed on real EEG); leave it None for
    synthetic epochs with no real time base.
    """
    return Pipeline([
        ("bandpass", BandpassFilter(sampling_rate=sampling_rate,
                                    low_hz=low_hz, high_hz=high_hz)),
        ("csp", CSP(n_components=n_components, reg=reg)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


# Sub-bands for filterbank CSP, tiling the mu+beta range. Each is CSP-filtered
# and classified from its own spatial patterns, then concatenated — motor
# imagery's discriminative rhythm sits in a narrower band for some people than a
# single 8-30 Hz pass captures, and the filterbank lets the classifier weight
# whichever sub-band carries it.
_DEFAULT_FILTERBANK = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0),
                       (20.0, 26.0), (26.0, 30.0))


class FilterbankCSP(BaseEstimator, TransformerMixin):
    """CSP applied in several frequency sub-bands, features concatenated.

    A CSP is fitted per sub-band on the training epochs (so still in-fold), and
    the per-band log-variance features are stacked. With sampling_rate=None the
    band-passes are no-ops, so this collapses to repeated single-band CSP; that
    mode is for callers without a real time base and is not the intended use.
    """

    def __init__(self, sampling_rate=None, bands=_DEFAULT_FILTERBANK,
                 n_components=4, reg=1e-6):
        self.sampling_rate = sampling_rate
        self.bands = bands
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.banks_ = []
        for low, high in self.bands:
            bandpass = BandpassFilter(
                sampling_rate=self.sampling_rate, low_hz=low, high_hz=high
            )
            filtered = bandpass.transform(X)
            csp = CSP(n_components=self.n_components, reg=self.reg).fit(filtered, y)
            self.banks_.append((bandpass, csp))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = [csp.transform(bp.transform(X)) for bp, csp in self.banks_]
        return np.concatenate(parts, axis=1)


def build_fbcsp_decoder(sampling_rate=None, bands=_DEFAULT_FILTERBANK,
                        n_components=4, reg=1e-6):
    """Filterbank CSP followed by LDA. Unfitted, one pipeline, in-fold safe."""
    return Pipeline([
        ("fbcsp", FilterbankCSP(sampling_rate=sampling_rate, bands=bands,
                                n_components=n_components, reg=reg)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


@dataclass
class DecodeResult:
    """Trial-random vs subject-held-out decoding for one decoder on raw epochs."""

    decoder: str
    n_trials: int
    n_channels: int
    n_subjects: int
    trial_random_roc_auc: float
    trial_random_balanced_accuracy: float
    grouped_roc_auc: float
    grouped_balanced_accuracy: float
    grouped_roc_auc_std: float
    fold_grouped_roc_auc: list = field(default_factory=list)

    @property
    def generalization_gap(self):
        """Trial-random minus subject-held-out balanced accuracy."""
        return self.trial_random_balanced_accuracy - self.grouped_balanced_accuracy

    def to_dict(self):
        return {
            "decoder": self.decoder,
            "n_trials": self.n_trials,
            "n_channels": self.n_channels,
            "n_subjects": self.n_subjects,
            "trial_random_roc_auc": round(self.trial_random_roc_auc, 4),
            "trial_random_balanced_accuracy": round(
                self.trial_random_balanced_accuracy, 4
            ),
            "grouped_roc_auc": round(self.grouped_roc_auc, 4),
            "grouped_balanced_accuracy": round(self.grouped_balanced_accuracy, 4),
            "grouped_roc_auc_std": round(self.grouped_roc_auc_std, 4),
            "generalization_gap": round(self.generalization_gap, 4),
        }


def _score_splits(decoder, X, y, splits):
    """Fit the decoder on each train fold, score the held-out fold. In-fold only."""
    from sklearn.base import clone

    baccs, aucs = [], []
    for train_idx, test_idx in splits:
        est = clone(decoder)
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        baccs.append(balanced_accuracy_score(y[test_idx], pred))
        if len(np.unique(y[test_idx])) < 2:
            aucs.append(float("nan"))
        else:
            score = est.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], score))
    return np.array(baccs), np.array(aucs)


def decode_benchmark(
    epochs, labels, subjects, decoder=None, n_splits=5, seed=0, n_components=6,
    sampling_rate=None, method="csp",
):
    """Trial-random vs subject-held-out decoding on raw epochs.

    epochs is (n_trials, n_channels, n_samples). Mirrors the two protocols the
    audit runs on the feature table, but for a raw-epoch decoder (CSP+LDA by
    default), so the optimised decoder's held-out number is directly comparable
    to the band-power baseline's. Every fit is inside a CV training fold.
    """
    X = np.asarray(epochs, dtype=float)
    y = np.asarray(labels)
    groups = np.asarray(subjects)
    if X.ndim != 3:
        raise ValueError(
            f"expected epochs (n_trials, n_channels, n_samples), got {X.shape}"
        )
    if decoder is None:
        if method == "fbcsp":
            decoder = build_fbcsp_decoder(
                sampling_rate=sampling_rate, n_components=n_components
            )
        elif method == "csp":
            decoder = build_csp_decoder(
                n_components=n_components, sampling_rate=sampling_rate
            )
        else:
            raise ValueError(f"unknown method {method!r}; expected 'csp' or 'fbcsp'")

    n_subjects = len(np.unique(groups))
    grouped_splits_n = min(n_splits, n_subjects)
    if grouped_splits_n < 2:
        raise ValueError(
            "subject-held-out decoding needs at least 2 subjects, "
            f"found {n_subjects}"
        )

    trial_random = StratifiedKFold(
        n_splits=min(n_splits, np.bincount(y.astype(int)).min()),
        shuffle=True,
        random_state=seed,
    )
    tr_bacc, tr_auc = _score_splits(
        decoder, X, y, list(trial_random.split(X, y))
    )

    grouped = GroupKFold(n_splits=grouped_splits_n)
    gr_bacc, gr_auc = _score_splits(
        decoder, X, y, list(grouped.split(X, y, groups))
    )

    return DecodeResult(
        decoder=_decoder_name(decoder),
        n_trials=int(X.shape[0]),
        n_channels=int(X.shape[1]),
        n_subjects=int(n_subjects),
        trial_random_roc_auc=float(np.nanmean(tr_auc)),
        trial_random_balanced_accuracy=float(np.mean(tr_bacc)),
        grouped_roc_auc=float(np.nanmean(gr_auc)),
        grouped_balanced_accuracy=float(np.mean(gr_bacc)),
        grouped_roc_auc_std=float(np.nanstd(gr_auc)),
        fold_grouped_roc_auc=[float(v) for v in gr_auc],
    )


def _decoder_name(decoder):
    """A short label for a pipeline, for the result record."""
    if isinstance(decoder, Pipeline):
        return "+".join(step for step, _ in decoder.steps)
    return type(decoder).__name__
