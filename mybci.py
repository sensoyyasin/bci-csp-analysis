import argparse
import warnings
import numpy as np
import mne
import matplotlib.pyplot as plt

from csp.csp import MyCsp
from mne.preprocessing import ICA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

CROP_WINDOWS = [
    (1.0, 2.0),
    (0.5, 2.5),
    (1.0, 3.0),
    (0.5, 3.5),
]

EXPERIMENTS = {
    0: ["R03", "R07", "R11"],
    1: ["R04", "R08", "R12"],
    2: ["R05", "R09", "R13"],
    3: ["R06", "R10", "R14"],
}


def make_subject_id(subject_nb):
    return f"S{int(subject_nb):03d}"


def make_pipeline():
    return Pipeline([
        ("CSP", MyCsp(n_components=6)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def load_raw(subject, runs):
    raws = []

    for run in runs:
        path = f"physionet.org/{subject}/{subject}{run}.edf"
        raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))

    return mne.concatenate_raws(raws)


def preprocess(raw):
    raw = raw.copy()
    raw.rename_channels({ch: ch.strip(".") for ch in raw.ch_names})

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="ignore",
    )

    raw_ica = raw.copy().filter(1.0, None, verbose=False)

    ica = ICA(n_components=20, random_state=42, max_iter=800)
    ica.fit(raw_ica)

    raw.filter(7, 30, verbose=False)

    try:
        eog_idx, _ = ica.find_bads_eog(
            raw_ica,
            ch_name="Fp1",
            threshold=3.0,
        )
        ica.exclude = eog_idx
    except Exception:
        ica.exclude = []

    ica.apply(raw)

    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()

    return raw


def get_epochs(raw):
    events, event_id = mne.events_from_annotations(raw)
    event_id = {str(k): int(v) for k, v in event_id.items()}

    if "T1" not in event_id or "T2" not in event_id:
        return None, None, None

    t1_label = event_id["T1"]
    t2_label = event_id["T2"]

    events_filtered = []

    for event in events:
        if event[2] == t1_label:
            events_filtered.append([event[0], 0, t1_label])
        elif event[2] == t2_label:
            events_filtered.append([event[0], 0, t2_label])

    if len(events_filtered) == 0:
        return None, None, None

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
    y = epochs.events[:, -1]

    return epochs, motor_picks, y


def load_dataset(subject_nb, experiment_nb):
    subject = make_subject_id(subject_nb)
    runs = EXPERIMENTS[int(experiment_nb)]

    raw = load_raw(subject, runs)
    raw = preprocess(raw)

    return get_epochs(raw)


def search_best_crop(epochs, motor_picks, y):
    best_crop = CROP_WINDOWS[0]
    best_score = 0.0

    for crop in CROP_WINDOWS:
        epochs_tmp = epochs.copy().crop(tmin=crop[0], tmax=crop[1])
        X = epochs_tmp.get_data(picks=motor_picks)

        scores = cross_val_score(
            make_pipeline(),
            X,
            y,
            cv=CV,
            scoring="accuracy",
        )

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_crop = crop

    return best_crop


def train_mode(subject_nb, experiment_nb):
    epochs, motor_picks, y = load_dataset(subject_nb, experiment_nb)

    if epochs is None:
        print("T1/T2 not found.")
        return

    best_crop = search_best_crop(epochs, motor_picks, y)

    epochs_train = epochs.copy().crop(
        tmin=best_crop[0],
        tmax=best_crop[1],
    )

    X = epochs_train.get_data(picks=motor_picks)

    scores = cross_val_score(
        make_pipeline(),
        X,
        y,
        cv=CV,
        scoring="accuracy",
    )

    print(f"Best crop window: {best_crop}")
    print(np.round(scores, 4))
    print(f"cross_val_score: {scores.mean():.4f}")


def plot_pr_and_roc_curves(y_true, y_score, labels, title_suffix, filename_suffix):
    positive_label = labels[1]
    y_true_binary = (y_true == positive_label).astype(int)

    precision, recall, _ = precision_recall_curve(
        y_true_binary,
        y_score,
    )

    pr_auc = average_precision_score(
        y_true_binary,
        y_score,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {title_suffix}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    pr_output_name = f"precision_recall_curve_{filename_suffix}.png"
    plt.savefig(pr_output_name, dpi=150)
    plt.show()

    print(f"{GREEN}Precision-Recall AUC / Average Precision: {pr_auc:.4f}{RESET}")
    print(f"Saved: {pr_output_name}")

    fpr, tpr, _ = roc_curve(
        y_true_binary,
        y_score,
    )

    roc_auc = roc_auc_score(
        y_true_binary,
        y_score,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title_suffix}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    roc_output_name = f"roc_curve_{filename_suffix}.png"
    plt.savefig(roc_output_name, dpi=150)
    plt.show()

    print(f"{GREEN}ROC AUC: {roc_auc:.4f}{RESET}")
    print(f"Saved: {roc_output_name}")


def predict_mode(subject_nb, experiment_nb):
    epochs, motor_picks, y = load_dataset(subject_nb, experiment_nb)

    if epochs is None:
        print("T1/T2 not found.")
        return

    best_crop = search_best_crop(epochs, motor_picks, y)

    epochs_train = epochs.copy().crop(
        tmin=best_crop[0],
        tmax=best_crop[1],
    )

    X = epochs_train.get_data(picks=motor_picks)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    labels = np.unique(y_test)
    label_names = [f"T{i + 1}" for i in range(len(labels))]

    print(f"Best crop window: {best_crop}")
    print("epoch nb: [prediction] [truth] equal?")

    for i, (pred, truth) in enumerate(zip(y_pred, y_test)):
        print(f"epoch {i:02d}: [{pred}] [{truth}] {pred == truth}")

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Subject {subject_nb:03d}, Experiment {experiment_nb}")
    plt.tight_layout()

    cm_output_name = f"confusion_matrix_S{subject_nb:03d}_exp{experiment_nb}.png"
    plt.savefig(cm_output_name, dpi=150)
    plt.show()

    print(f"Saved: {cm_output_name}")

    plot_pr_and_roc_curves(
        y_true=y_test,
        y_score=y_score,
        labels=labels,
        title_suffix=f"Subject {subject_nb:03d}, Experiment {experiment_nb}",
        filename_suffix=f"S{subject_nb:03d}_exp{experiment_nb}",
    )


def run_single_experiment(subject_nb, experiment_nb):
    epochs, motor_picks, y = load_dataset(subject_nb, experiment_nb)

    if epochs is None:
        return None

    best_crop = search_best_crop(epochs, motor_picks, y)

    epochs_train = epochs.copy().crop(
        tmin=best_crop[0],
        tmax=best_crop[1],
    )

    X = epochs_train.get_data(picks=motor_picks)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    return acc, y_test, y_pred, y_score


def summary_mode():
    subjects = range(1, 110)

    experiment_scores = {exp: [] for exp in EXPERIMENTS.keys()}

    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for exp_nb in EXPERIMENTS.keys():
        for subject_nb in subjects:
            try:
                result = run_single_experiment(subject_nb, exp_nb)

                if result is None:
                    continue

                acc, y_test, y_pred, y_score = result

                experiment_scores[exp_nb].append(acc)
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                all_y_score.extend(y_score)

                print(
                    f"experiment {exp_nb}: "
                    f"subject {subject_nb:03d}: "
                    f"accuracy = {acc:.4f}"
                )

            except FileNotFoundError:
                continue

            except Exception:
                continue

    print("\nMean accuracy of the experiments for all subjects:")

    means = []

    for exp_nb, scores in experiment_scores.items():
        if len(scores) == 0:
            print(f"experiment {exp_nb}: accuracy = no valid data")
            continue

        mean_score = np.mean(scores)
        means.append(mean_score)

        print(f"experiment {exp_nb}: accuracy = {mean_score:.4f}")

    if len(means) == 0:
        print("No valid results.")
        return

    print(f"Mean accuracy of {len(means)} experiments: {np.mean(means):.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    labels = np.unique(all_y_true)
    label_names = [f"T{i + 1}" for i in range(len(labels))]

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(
        all_y_true,
        all_y_pred,
        average="macro",
        zero_division=0,
    )
    overall_precision = precision_score(
        all_y_true,
        all_y_pred,
        average="macro",
        zero_division=0,
    )
    overall_recall = recall_score(
        all_y_true,
        all_y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"\n{GREEN}" + "=" * 60)
    print("Overall Performance For all subjects and experiments")
    print("=" * 60 + f"{RESET}")

    print(f"Overall Accuracy        : {overall_acc:.4f}")
    print(f"Overall F1 Score macro  : {overall_f1:.4f}")
    print(f"Overall Precision macro : {overall_precision:.4f}")
    print(f"Overall Recall macro    : {overall_recall:.4f}")

    print(f"\n{GREEN}Classification Report:{RESET}")
    print(
        classification_report(
            all_y_true,
            all_y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)

    print("\nOverall Confusion Matrix")
    print("Rows = true labels, columns = predicted labels")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Overall Confusion Matrix - All Subjects and Experiments")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    output_name = "overall_confusion_matrix_all_subjects_all_experiments.png"
    plt.savefig(output_name, dpi=150)
    plt.show()

    print(f"{GREEN}Saved: {output_name}{RESET}")

    plot_pr_and_roc_curves(
        y_true=all_y_true,
        y_score=all_y_score,
        labels=labels,
        title_suffix="All Subjects and Experiments",
        filename_suffix="all_subjects_all_experiments",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("subject", nargs="?", type=int)
    parser.add_argument("experiment", nargs="?", type=int)
    parser.add_argument("mode", nargs="?", choices=["train", "predict"])

    args = parser.parse_args()

    if args.subject is not None and args.experiment is not None:
        if args.mode == "train":
            train_mode(args.subject, args.experiment)

        elif args.mode == "predict":
            predict_mode(args.subject, args.experiment)

        else:
            print(f"{RED}Error: mode must be 'train' or 'predict'{RESET}")
            print("Usage: python mybci.py <subject> <experiment> train|predict")

    elif args.subject is None and args.experiment is None and args.mode is None:
        summary_mode()

    else:
        print("Invalid arguments")
