import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
import warnings
#from mne.decoding import CSP
from csp.csp import MyCsp
from mne.preprocessing import ICA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

SUBJECT = "S029"

EXPERIMENTS = {
    "EXPERIMENT 0 - REAL_LEFT_RIGHT": ["R03", "R07", "R11"],
    "EXPERIMENT 1 - IMAGERY_LEFT_RIGHT": ["R04", "R08", "R12"],
    "EXPERIMENT 2 - REAL_HANDS_FEET": ["R05", "R09", "R13"],
    "EXPERIMENT 3 - IMAGERY_HANDS_FEET": ["R06", "R10", "R14"],
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CROP_WINDOWS = [
    (1.0, 2.0),
    (0.5, 2.5),
    (1.0, 3.0),
    (0.5, 3.5),
]

def load_raw(subject, runs):
    raw_files = []

    for run in runs:
        path = f"physionet.org/{subject}/{subject}{run}.edf"
        raw_run = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw_files.append(raw_run)

    raw = mne.concatenate_raws(raw_files)

    print(f"raw : {raw}")
    print(f"Sampling Freq : {raw.info['sfreq']} Hz")
    print(f"Duration      : {raw.times[-1]:.2f} s")
    print(f"n_channels    : {raw.info['nchan']}")
    print(f"Data shape    : {raw.get_data().shape} (n_channels, n_samples)")
    print("--------------")

    return raw


def preprocess(raw):
    raw = raw.copy()

    raw.rename_channels({ch: ch.strip(".") for ch in raw.ch_names})

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="ignore"
    )

    raw_ica = raw.copy().filter(1.0, None)
    ica = ICA(n_components=20, random_state=42, max_iter=800)
    ica.fit(raw_ica)

    raw.filter(7, 30)

    eog_idx, _ = ica.find_bads_eog(raw_ica, ch_name="Fp1", threshold=3.0)
    ica.exclude = eog_idx
    ica.apply(raw)

    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    return raw


def plot_raw_channels(raw, n_channels=10, duration_sec=5):
    data = raw.get_data(picks="eeg") * 1e6
    times = raw.times
    channel_names = raw.copy().pick("eeg").ch_names

    n_channels = min(n_channels, len(channel_names))

    num_samples = int(duration_sec * raw.info["sfreq"])
    data = data[:n_channels, :num_samples]
    times = times[:num_samples]

    data = data - data.mean(axis=1, keepdims=True)

    spacing = np.percentile(np.abs(data), 95) * 3
    offsets = np.arange(n_channels)[::-1] * spacing

    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(n_channels):
        ax.plot(times, data[i] + offsets[i], linewidth=0.8)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[:n_channels])

    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"First {duration_sec} Seconds of Filtered EEG Signal "
        f"(first {n_channels} channels)"
    )

    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_psd_bands(raw):
    bands = {
        "alpha": (8, 13, "#4CAF50"),
        "beta": (13, 30, "#7F77DD"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (name, (fmin, fmax, color)) in zip(axes, bands.items()):
        psd = raw.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            reject_by_annotation=False,
            verbose=False,
        )

        psd.plot(picks="eeg", axes=ax, show=False)

        for line in ax.get_lines():
            line.set_color(color)
            line.set_alpha(0.45)

        ax.set_title(name, color=color, fontweight="bold")
        ax.grid(True, alpha=0.25)

    fig.suptitle("PSD Bands: Alpha and Beta", fontsize=13)
    plt.show()


def get_epochs(raw):
    events, event_id = mne.events_from_annotations(raw)
    event_id = {str(k): int(v) for k, v in event_id.items()}

    if "T1" not in event_id or "T2" not in event_id:
        print("T1/T2 not found in annotations, skipping.")
        return None, None, None

    t1_label = event_id["T1"]
    t2_label = event_id["T2"]

    events_filtered = []

    for e in events:
        if e[2] == t1_label:
            events_filtered.append([e[0], 0, t1_label])
        elif e[2] == t2_label:
            events_filtered.append([e[0], 0, t2_label])

    events_filtered = np.array(events_filtered)
    events_filtered = events_filtered[np.argsort(events_filtered[:, 0])]

    epochs = mne.Epochs(
        raw,
        events_filtered,
        event_id={"T1": t1_label, "T2": t2_label},
        tmin=-1.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    motor_picks = [ch for ch in raw.ch_names if ch.startswith("C")]
    print(f"Motor picks : {motor_picks}")

    y = epochs.events[:, -1]

    return epochs, motor_picks, y


def make_pipeline():
    return Pipeline(
        [
            ("CSP", MyCsp(n_components=6)),
            ("LDA", LinearDiscriminantAnalysis()),
        ]
    )


def search_best_crop(epochs, motor_picks, y):
    best_crop_mean = 0
    best_crop = CROP_WINDOWS[0]

    print("=== Crop Window Search ===")

    for tmin_c, tmax_c in CROP_WINDOWS:
        epochs_tmp = epochs.copy().crop(tmin=tmin_c, tmax=tmax_c)
        X_tmp = epochs_tmp.get_data(picks=motor_picks)

        pipeline_tmp = make_pipeline()

        scores_tmp = cross_val_score(
            pipeline_tmp,
            X_tmp,
            y,
            cv=CV,
            scoring="accuracy",
        )

        print(
            f"  crop=({tmin_c}, {tmax_c}): "
            f"mean={scores_tmp.mean():.4f}  std={scores_tmp.std():.4f}"
        )

        if scores_tmp.mean() > best_crop_mean:
            best_crop_mean = scores_tmp.mean()
            best_crop = (tmin_c, tmax_c)

    print(f"{GREEN}Best crop: {best_crop} -> {best_crop_mean:.4f}{RESET}")
    print("==========================")

    return best_crop


def train_evaluate(epochs, motor_picks, y, best_crop, visualize=False):
    epochs_train = epochs.copy().crop(tmin=best_crop[0], tmax=best_crop[1])
    X = epochs_train.get_data(picks=motor_picks)

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    pipeline = make_pipeline()

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=CV,
        scoring="accuracy",
    )

    print(f"CV scores      : {scores}")
    print(f"Mean accuracy  : {scores.mean():.4f}")
    print(f"Std            : {scores.std():.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    final_pipeline = make_pipeline()
    final_pipeline.fit(X_tr, y_tr)

    y_pred = final_pipeline.predict(X_te)

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_te,
            y_pred,
            target_names=["T1", "T2"],
        )
    )
    print(f"Accuracy: {accuracy_score(y_te, y_pred):.4f}")

    return final_pipeline, accuracy_score(y_te, y_pred)


def score_over_time(epochs, motor_picks):
    sfreq = epochs.info["sfreq"]

    w_length = int(sfreq * 0.5)
    w_step = int(sfreq * 0.1)

    epochs_data = epochs.get_data(picks=motor_picks)
    labels = epochs.events[:, -1]

    motor_imagery_onset = 1.5
    onset_sample = int(motor_imagery_onset * sfreq)

    w_start = np.arange(
        max(0, onset_sample - w_length),
        epochs_data.shape[2] - w_length,
        w_step,
    )

    scores_windows = []

    for train_idx, test_idx in CV.split(epochs_data, labels):
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        csp = MyCsp(n_components=6)
        lda = LinearDiscriminantAnalysis()

        X_train = csp.fit_transform(epochs_data[train_idx], y_train)
        lda.fit(X_train, y_train)

        window_scores = []

        for n in w_start:
            X_win = csp.transform(epochs_data[test_idx][:, :, n:n + w_length])
            window_scores.append(lda.score(X_win, y_test))

        scores_windows.append(window_scores)

    scores_windows = np.array(scores_windows)

    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    mean_scores = scores_windows.mean(axis=0)
    std_scores = scores_windows.std(axis=0)

    plt.figure(figsize=(10, 4))

    plt.plot(
        w_times,
        mean_scores,
        label="Score",
        color="#7F77DD",
        linewidth=2,
    )

    plt.fill_between(
        w_times,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        color="#7F77DD",
    )

    plt.axvline(
        motor_imagery_onset,
        linestyle="--",
        color="black",
        label="Motor Imagery Onset (~1.5s)",
    )

    plt.axhline(
        0.5,
        linestyle="-",
        color="black",
        label="Chance (0.5)",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Score Over Time")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def visualize_epochs(epochs):
    epochs.plot_sensors(show_names=True, show=True)

    fig = plt.figure(figsize=(6, 5))
    ax_3d = fig.add_subplot(111, projection="3d")
    epochs.plot_sensors(show_names=True, kind="3d", axes=ax_3d, show=True)

    average_T1 = epochs["T1"].average()
    average_T2 = epochs["T2"].average()

    times_to_plot = [1.6, 2.0, 2.5, 3.0]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    average_T1.plot_topomap(
        times=times_to_plot,
        axes=axes[0],
        show=False,
        colorbar=False,
    )

    average_T2.plot_topomap(
        times=times_to_plot,
        axes=axes[1],
        show=False,
        colorbar=False,
    )

    axes[0][0].set_ylabel("T1", fontsize=12)
    axes[1][0].set_ylabel("T2", fontsize=12)

    plt.suptitle("T1 vs T2 - Motor Cortex Activation", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    results = {}
    print(f"SUBJECT: {SUBJECT}")

    for exp_name, runs in EXPERIMENTS.items():
        print("\n" + "=" * 60)
        print(exp_name)
        print("=" * 60)

        try:
            raw = load_raw(SUBJECT, runs)
            raw = preprocess(raw)

            if args.visualize:
                plot_raw_channels(raw)
                plot_psd_bands(raw)

            epochs, motor_picks, y = get_epochs(raw)

            if epochs is None:
                continue

            best_crop = search_best_crop(epochs, motor_picks, y)

            final_pipeline, acc = train_evaluate(
                epochs,
                motor_picks,
                y,
                best_crop,
                visualize=args.visualize,
            )

            if args.visualize:
                score_over_time(epochs, motor_picks)
                visualize_epochs(epochs)

            results[exp_name] = {
                "accuracy": acc,
                "crop": best_crop,
                "filter": (7, 30),
            }

        except FileNotFoundError:
            print(f"{RED}Missing file for {SUBJECT} | {exp_name}{RESET}")
            continue

        except Exception as e:
            print(f"{RED}Error for {SUBJECT} | {exp_name}: {e}{RESET}")
            continue

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results.items():
        acc = result["accuracy"]
        crop = result["crop"]
        band = result["filter"]

        print(
            f"{name}: "
            f"acc={acc:.4f}, "
            f"filter={band[0]}-{band[1]} Hz, "
            f"crop={crop}"
        )

    if len(results) > 0:
        mean_acc = np.mean([r["accuracy"] for r in results.values()])
        print(f"{GREEN}Mean accuracy: {mean_acc:.4f}{RESET}")
    else:
        print("\nNo valid results.")
