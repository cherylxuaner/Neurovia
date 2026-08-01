"""Demonstration datasets shaped like a customer's, not like a test fixture.

Scanning a public dataset shows what the tool does. It does not show what it is
for. These put the tool in the situation it is sold into: a team has built
something, has a number they are about to put in front of a customer, and the
question is whether that number survives contact with a new person.

Each scenario is constructed and says so. What makes that honest is that the
failure inside it is a real and common one, the numbers are not hand-set, and
the scan is never told which scenario it is looking at - it has to find the
failure the same way it would in the field.

Feature names follow the channel_band convention of real recordings, so the
same code paths run as on the PhysioNet loader.
"""

import numpy as np
import pandas as pd

CHANNELS = ("AF7", "AF8", "TP9", "TP10", "Fz", "Cz", "Pz", "Oz")
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

SCENARIOS = {
    "focus_tracker": {
        "customer": "Halcyon Labs (fictional) - consumer EEG headband",
        "pitch": (
            "A wearable company's attention classifier. Two years of data from "
            "a headband, and an internal number the team is about to show a "
            "retail partner."
        ),
        "reported_metric": (
            "0.92 balanced accuracy, from a shuffled split of all recorded "
            "trials - the split their training script does by default."
        ),
        "planted_failure": (
            "Participants differ in how they self-report attention, and the "
            "headband records each person's own signature. The classifier can "
            "recognise the wearer and answer from that alone, which works "
            "perfectly on the people it was trained on and not at all on a new "
            "customer."
        ),
        "n_subjects": 40,
        "n_trials": 50,
        "label_rate": 0.92,
        "subject_offset": 4.0,
        "shared_signal": 0.0,
    },
    "calibrated_detector": {
        "customer": "Halcyon Labs (fictional) - consumer EEG headband, v2",
        "pitch": (
            "The same company after a rebuild: fewer channels, a target that "
            "does not track who the wearer is."
        ),
        "reported_metric": "0.90 balanced accuracy from the same shuffled split.",
        "planted_failure": (
            "None. This one generalises, and the scan has to say so - a "
            "detector that warns about everything is worth nothing."
        ),
        "n_subjects": 40,
        "n_trials": 50,
        "label_rate": 0.5,
        "subject_offset": 0.3,
        "shared_signal": 1.6,
    },
}


def _balanced_rates(offsets, rate):
    """Give each subject a label rate that their signature cannot predict.

    Assigning rates by index leaves them free to correlate with the random
    offsets, and with as many features as subjects a model can find a direction
    through the training people that happens to separate the two rates. It then
    scores above chance on a held-out person for no reason at all, which
    muddies the very comparison the scenario exists to show.

    Pairing each subject with their nearest neighbour in signature space and
    splitting the pair between the two rates removes that: the people closest
    together in the data are the ones who disagree.
    """
    order = np.argsort(offsets @ offsets.mean(axis=0))
    rates = np.empty(len(offsets))
    for position, subject in enumerate(order):
        rates[subject] = rate if position % 2 == 0 else 1 - rate
    return rates


def describe(name):
    """What a customer would say about this dataset before the scan runs."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {list(SCENARIOS)}")
    return SCENARIOS[name]


CONTEXT_KEYS = ("customer", "pitch", "reported_metric", "planted_failure")


def context(name):
    """What the team would say about this data before the scan runs.

    Carried into the report so a reader meets the claim before the finding,
    which is the order they would meet it in a real engagement.
    """
    described = describe(name)
    return {key: described[key] for key in CONTEXT_KEYS}


def load_scenario(name, seed=0):
    """Build one scenario. Returns (dataframe, feature column names)."""
    spec = describe(name)
    rng = np.random.default_rng(seed)

    columns = [f"{channel}_{band}" for channel in CHANNELS for band in BANDS]
    n_features = len(columns)

    offsets = rng.normal(0, spec["subject_offset"], size=(spec["n_subjects"], n_features))
    direction = rng.normal(0, 1, size=n_features)
    direction /= np.linalg.norm(direction)
    rates = _balanced_rates(offsets, spec["label_rate"])

    frames = []
    for subject in range(spec["n_subjects"]):
        noise = rng.normal(0, 1, size=(spec["n_trials"], n_features))

        if spec["shared_signal"] > 0:
            # A direction every wearer shares: the label is a property of the
            # signal, so it transfers to someone the model has never seen.
            score = noise @ direction
            labels = (score > np.median(score)).astype(int)
            features = offsets[subject] + noise + spec["shared_signal"] * np.outer(
                labels - 0.5, direction
            )
        else:
            # The label is a property of the person. Alternating rates keep the
            # dataset globally balanced, so a majority guess gains nothing.
            n_high = round(rates[subject] * spec["n_trials"])
            labels = np.array([1] * n_high + [0] * (spec["n_trials"] - n_high))
            rng.shuffle(labels)
            features = offsets[subject] + noise

        participant = f"P{subject:02d}"
        frame = pd.DataFrame(features, columns=columns)
        frame.insert(0, "subject_id", participant)
        frame.insert(
            1,
            "trial_id",
            [f"{participant}_t{t:03d}" for t in range(spec["n_trials"])],
        )
        frame.insert(2, "target_label", labels)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), columns
