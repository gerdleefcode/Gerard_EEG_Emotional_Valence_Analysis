"""
EEG Emotional Response — with MNE support and automatic figure saving.

Main changes:
- No IPython display() usage (replaced by show_df()).
- Optional MNE analysis (psd_welch, topomap of band power, evoked averages) when MNE is installed.
- All matplotlib/seaborn/MNE generated figures are saved to FIG_DIR automatically.

Dependencies:
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
- Optional: mne (install with `pip install mne`)
"""

import os
import math
import warnings
import datetime
import mne
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# -------------- USER CONFIG --------------
DATA_PATH = "/Users/stageacomeback/Desktop/Gerard Lee/PolyU SPEED RA/EEG Brain Signal Analysis/Self Testing (Emotions)/emotions.csv"    # path to your CSV file
LABEL_COL = None              # if None -> try to auto-detect
GROUP_COL = None              # name of grouping column for epochs/trials (long format)
TIME_COL = None               # name of time column (if present)
SAMPLE_RATE = 256             # default sample rate in Hz (fallback)
DATA_FORMAT = "auto"          # "auto", "long", or "wide"

# Figure saving options
SAVE_FIGURES = True
FIG_DIR = Path("figures")
FIG_FORMAT = "png"
FIG_DPI = 150
FIG_PREFIX = "fig"

# MNE options
USE_MNE = True               # set to False to skip MNE attempts even if installed
# -----------------------------------------

# If saving figures, select a non-interactive backend before importing pyplot
if SAVE_FIGURES:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="notebook")

from scipy.stats import skew, kurtosis
from scipy.signal import welch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

# Try to import mne if requested
has_mne = False
if USE_MNE:
    try:
        import mne
        from mne.time_frequency import psd_welch
        has_mne = True
    except Exception as e:
        print("MNE not available or failed to import. MNE-based analyses will be skipped.")
        print("To enable MNE analyses, install MNE: pip install mne")
        has_mne = False

# Figure saving helper
FIG_DIR.mkdir(parents=True, exist_ok=True)
_fig_counter = 0


def save_fig(name: str, fig=None, tight=True, close=True):
    """
    Save the current matplotlib figure (or provided fig) into FIG_DIR with a timestamped filename.
    Returns the saved Path.
    """
    global _fig_counter
    _fig_counter += 1
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_").replace("/", "_")
    fname = FIG_DIR / f"{FIG_PREFIX}_{_fig_counter:03d}_{ts}_{safe_name}.{FIG_FORMAT}"
    if fig is None:
        fig = plt.gcf()
    try:
        if tight:
            fig.tight_layout()
        fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
        if close:
            plt.close(fig)
        print(f"[FIGURE SAVED] {fname}")
    except Exception as e:
        print(f"[FIG SAVE ERROR] Could not save figure {fname}: {e}")
    return fname


# Replacement for IPython.display.display in a script
def show_df(df: pd.DataFrame, n: int = 5, message: str = None) -> None:
    if message:
        print(message)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(n).to_string(index=False))
    print()


# ---------- Utility detection helpers ----------
def detect_label_column(df, prefer=None):
    candidates = [c for c in df.columns if c.lower() in ("emotion", "emotions", "label", "labels",
                                                          "target", "class", "feeling", "y", "emotion_label")]
    if prefer and prefer in df.columns:
        return prefer
    if candidates:
        return candidates[0]
    # fallback: choose a non-numeric column with low cardinality
    for c in df.columns:
        if df[c].dtype == "object" or df[c].dtype.name == "category":
            if df[c].nunique() <= 50:
                return c
    # fallback: numeric small unique
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if df[c].nunique() <= 20:
            return c
    return None


def detect_group_and_time(df):
    group_cols = [c for c in df.columns if c.lower() in ("trial", "epoch", "segment", "window", "id", "session", "recording")]
    time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp", "t", "sample", "index")]
    return (group_cols[0] if group_cols else None, time_cols[0] if time_cols else None)


# ---------- Feature extraction utilities ----------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def spectral_entropy(psd, base=2):
    psd = np.asarray(psd)
    psd_sum = psd.sum()
    if psd_sum <= 0:
        return 0.0
    psd_norm = psd / psd_sum
    psd_norm = psd_norm + 1e-12
    ent = -np.sum(psd_norm * np.log(psd_norm)) / np.log(base)
    return float(ent)


def hjorth_params(x):
    x = np.asarray(x)
    if x.size < 3:
        return 0.0, 0.0, 0.0
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    activity = float(var_x)
    mobility = float(math.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
    complexity = float(math.sqrt(var_ddx / var_dx) / mobility) if (var_dx > 0 and mobility > 0) else 0.0
    return activity, mobility, complexity


def band_power_from_psd(f, Pxx, band):
    low, high = band
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    bp = np.trapz(Pxx[mask], f[mask])
    return float(bp)


def compute_features_for_epoch(epoch_df, sample_rate=SAMPLE_RATE, channel_cols=None, bands=BANDS):
    """
    epoch_df: pandas DataFrame for a single epoch (rows=time samples).
    channel_cols: channel column names (list).
    returns: dict of features for that epoch.
    """
    exclude = {c for c in (LABEL_COL, GROUP_COL, TIME_COL) if c}
    if channel_cols is None:
        channel_cols = [c for c in epoch_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(epoch_df[c])]
    n_samples = epoch_df.shape[0]
    features = {}
    for ch in channel_cols:
        x = epoch_df[ch].astype(float).fillna(0.0).values
        if x.size == 0:
            continue
        features[f"{ch}_mean"] = float(np.mean(x))
        features[f"{ch}_std"] = float(np.std(x))
        features[f"{ch}_var"] = float(np.var(x))
        features[f"{ch}_min"] = float(np.min(x))
        features[f"{ch}_max"] = float(np.max(x))
        features[f"{ch}_median"] = float(np.median(x))
        features[f"{ch}_ptp"] = float(np.ptp(x))
        features[f"{ch}_skew"] = float(skew(x)) if x.size >= 3 else 0.0
        features[f"{ch}_kurtosis"] = float(kurtosis(x)) if x.size >= 4 else 0.0
        features[f"{ch}_rms"] = float(np.sqrt(np.mean(x**2)))
        zc = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
        features[f"{ch}_zcr"] = float(zc)
        a, m, cpx = hjorth_params(x)
        features[f"{ch}_hjorth_activity"] = float(a)
        features[f"{ch}_hjorth_mobility"] = float(m)
        features[f"{ch}_hjorth_complexity"] = float(cpx)
        # PSD via scipy welch (fallback)
        nperseg = min(512, max(64, n_samples))
        try:
            f, Pxx = welch(x, fs=sample_rate, nperseg=nperseg)
        except Exception:
            f, Pxx = welch(x, fs=sample_rate)
        total_band_power = 0.0
        for band_name, band_range in bands.items():
            bp = band_power_from_psd(f, Pxx, band_range)
            features[f"{ch}_bp_{band_name}"] = bp
            total_band_power += bp
        features[f"{ch}_bp_total"] = float(total_band_power)
        features[f"{ch}_spec_entropy"] = float(spectral_entropy(Pxx))
        features[f"{ch}_peak_freq"] = float(f[np.argmax(Pxx)]) if Pxx.size else 0.0

    if channel_cols and len(channel_cols) >= 2:
        sigs = np.vstack([epoch_df[ch].astype(float).fillna(0.0).values for ch in channel_cols])
        if sigs.shape[1] >= 2:
            corr = np.corrcoef(sigs)
            abs_corr = np.abs(corr)
            n = abs_corr.shape[0]
            if n > 1:
                avg_abs_corr = (np.sum(abs_corr) - n) / (n * (n - 1))
            else:
                avg_abs_corr = 0.0
        else:
            avg_abs_corr = 0.0
        features["avg_abs_channel_corr"] = float(avg_abs_corr)
    else:
        features["avg_abs_channel_corr"] = 0.0

    for band_name, _ in bands.items():
        vals = [features[f"{ch}_bp_{band_name}"] for ch in channel_cols if f"{ch}_bp_{band_name}" in features]
        if len(vals) == 0:
            features[f"mean_bp_{band_name}"] = 0.0
            features[f"std_bp_{band_name}"] = 0.0
        else:
            features[f"mean_bp_{band_name}"] = float(np.mean(vals))
            features[f"std_bp_{band_name}"] = float(np.std(vals))

    def ratio(a, b):
        try:
            return float(a / b) if b and b != 0 else 0.0
        except Exception:
            return 0.0

    features["ratio_alpha_theta"] = ratio(features.get("mean_bp_alpha", 0.0), features.get("mean_bp_theta", 0.0))
    features["ratio_alpha_beta"] = ratio(features.get("mean_bp_alpha", 0.0), features.get("mean_bp_beta", 0.0))

    return features


# ---------- Load CSV ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}. Place your file or change DATA_PATH.")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH} -> shape = {df.shape}")
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nFirst rows:")
show_df(df, n=6)

# Auto-detect label/group/time
if LABEL_COL is None:
    LABEL_COL = detect_label_column(df)
    print(f"Auto-detected label column: {LABEL_COL}")
if GROUP_COL is None or TIME_COL is None:
    auto_group, auto_time = detect_group_and_time(df)
    if GROUP_COL is None and auto_group:
        GROUP_COL = auto_group
        print(f"Auto-detected group column: {GROUP_COL}")
    if TIME_COL is None and auto_time:
        TIME_COL = auto_time
        print(f"Auto-detected time column: {TIME_COL}")

# Determine data format
if DATA_FORMAT == "auto":
    if GROUP_COL is not None:
        DATA_FORMAT = "long"
    else:
        numeric_cols = df.select_dtypes(include=np.number).shape[1]
        if df.shape[0] > 5000 and numeric_cols <= 6:
            DATA_FORMAT = "long"
        else:
            DATA_FORMAT = "wide"
print(f"Decided DATA_FORMAT = {DATA_FORMAT}")

# ---------- Basic EDA (and save figures) ----------
if LABEL_COL and LABEL_COL in df.columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=LABEL_COL, order=df[LABEL_COL].value_counts().index)
    plt.title("Label distribution")
    plt.xticks(rotation=45)
    save_fig("label_distribution")
else:
    print("Label column not found; set LABEL_COL manually if available.")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:30]}")
print(df[numeric_cols].describe().T)

# ---------- Feature extraction (long format -> epoch-level features) ----------
channel_cols = []
group_ids = []
if DATA_FORMAT == "long":
    if GROUP_COL is None:
        raise ValueError("DATA_FORMAT is 'long' but GROUP_COL is not set/detected. Set GROUP_COL to identify each epoch/trial.")
    exclude = {c for c in (LABEL_COL, GROUP_COL, TIME_COL) if c}
    channel_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Detected {len(channel_cols)} numeric channel columns: {channel_cols[:20]}")
    feats = []
    group_ids = []
    groups = df.groupby(GROUP_COL)
    for gid, g in groups:
        feat = compute_features_for_epoch(g, sample_rate=SAMPLE_RATE, channel_cols=channel_cols)
        if LABEL_COL and LABEL_COL in g.columns:
            lbl = g[LABEL_COL].mode().iloc[0]
        else:
            lbl = None
        feat[LABEL_COL if LABEL_COL else "label"] = lbl
        feat[GROUP_COL] = gid
        feats.append(feat)
        group_ids.append(gid)
    feat_df = pd.DataFrame(feats)
    print(f"Feature extraction complete. Features shape = {feat_df.shape}")
    show_df(feat_df, n=5, message="Extracted features (first rows):")
    modeling_df = feat_df.copy()
else:
    modeling_df = df.copy()
    print(f"Using wide-format dataset with shape {modeling_df.shape}")

# ---------- Prepare X, y ----------
if LABEL_COL is None or LABEL_COL not in modeling_df.columns:
    raise ValueError("Label column not found in modeling_df. Set LABEL_COL appropriately.")

modeling_df = modeling_df.dropna(subset=[LABEL_COL])
exclude_feats = {c for c in (GROUP_COL, TIME_COL, LABEL_COL) if c}
feature_cols = [c for c in modeling_df.select_dtypes(include=np.number).columns if c not in exclude_feats]
print(f"Using {len(feature_cols)} numeric features (first 40 shown): {feature_cols[:40]}")

X = modeling_df[feature_cols].fillna(0.0).values
y_raw = modeling_df[LABEL_COL].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print(f"Label classes: {list(label_encoder.classes_)}")

# Correlation heatmap (top variance features)
if X.shape[1] > 1:
    plt.figure(figsize=(10, 8))
    var_idx = np.argsort(np.var(X, axis=0))[::-1][:40]
    corr_df = pd.DataFrame(X[:, var_idx], columns=[feature_cols[i] for i in var_idx]).corr()
    sns.heatmap(corr_df, cmap="vlag", center=0)
    plt.title("Correlation (top-variance features)")
    save_fig("correlation_heatmap")

# PCA 2D
if X.shape[1] >= 2 and X.shape[0] >= 5:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 6))
    palette = sns.color_palette("tab10", np.unique(y).size)
    sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=[label_encoder.classes_[i] for i in y], palette=palette, alpha=0.8)
    plt.title("PCA (2D) of features colored by emotion")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    save_fig("pca_2d_by_label")

# ---------- Modeling: cross-validated baseline classifiers ----------
print("\nModeling: cross-validated scores for baseline classifiers")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]),
    "RandomForest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
}

cv_results = {}
for name, model in models.items():
    print(f"CV for {name} ...")
    try:
        res = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1)
    except Exception as e:
        print(f"  cross_validate failed for {name}: {e}")
        continue
    cv_results[name] = res
    print(f"  accuracy: {res['test_accuracy'].mean():.4f} ± {res['test_accuracy'].std():.4f}")
    print(f"  precision_macro: {res['test_precision_macro'].mean():.4f} ± {res['test_precision_macro'].std():.4f}")
    print(f"  recall_macro: {res['test_recall_macro'].mean():.4f} ± {res['test_recall_macro'].std():.4f}")
    print(f"  f1_macro: {res['test_f1_macro'].mean():.4f} ± {res['test_f1_macro'].std():.4f}")
    print()

# ---------- Final train/test evaluation ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_model_name = "RandomForest"
best_model = models[best_model_name]
print(f"Training final model ({best_model_name}) ...")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("Classification report (test set):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix figure
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title(f"Confusion matrix ({best_model_name})")
save_fig("confusion_matrix")

# Feature importance for RandomForest
if best_model_name == "RandomForest":
    try:
        rf = best_model.named_steps["clf"]
        importances = rf.feature_importances_
        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        topk = 25 if len(fi) >= 25 else len(fi)
        plt.figure(figsize=(8, min(0.3 * topk + 1, 12)))
        sns.barplot(x=fi.values[:topk], y=fi.index[:topk], palette="viridis")
        plt.title("Top feature importances (RandomForest)")
        plt.xlabel("Importance")
        plt.ylabel("")
        save_fig("feature_importances_topk")
    except Exception as e:
        print("Could not compute feature importances:", e)

# ROC AUC (multi-class)
try:
    y_prob = best_model.predict_proba(X_test)
    from sklearn.preprocessing import label_binarize
    classes_idx = np.arange(len(label_encoder.classes_))
    y_test_bin = label_binarize(y_test, classes=classes_idx)
    auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    print(f"Macro-averaged ROC AUC (test set): {auc:.4f}")
except Exception as e:
    print("Could not compute ROC AUC (model may not support predict_proba):", e)

# ---------- MNE-based analyses (if long format and MNE available) ----------
if has_mne and DATA_FORMAT == "long" and channel_cols:
    print("\nRunning MNE analyses (PSD, band-power topomaps, evoked averages) ...")
    # Build epoch arrays in the same order as group_ids used for feature extraction
    epoch_arrs = []
    labels_for_epochs = []
    lengths = []
    for gid in group_ids:
        g = df[df[GROUP_COL] == gid]
        if TIME_COL and TIME_COL in g.columns:
            try:
                g_sorted = g.sort_values(TIME_COL)
            except Exception:
                g_sorted = g
        else:
            g_sorted = g
        # stack channels -> shape (n_channels, n_times)
        arr = np.vstack([g_sorted[ch].astype(float).fillna(0.0).values for ch in channel_cols])
        epoch_arrs.append(arr)
        lengths.append(arr.shape[1])
        lbl = g_sorted[LABEL_COL].mode().iloc[0] if LABEL_COL in g_sorted.columns else None
        labels_for_epochs.append(lbl)
    if len(epoch_arrs) == 0:
        print("No epoch arrays constructed; skipping MNE analysis.")
    else:
        max_len = max(lengths)
        # pad/truncate to max_len for consistency
        padded = []
        for arr in epoch_arrs:
            if arr.shape[1] < max_len:
                arr2 = np.pad(arr, ((0, 0), (0, max_len - arr.shape[1])), mode="constant", constant_values=0.0)
            else:
                arr2 = arr[:, :max_len]
            padded.append(arr2)
        epoch_data = np.stack(padded, axis=0).astype(np.float64)  # (n_epochs, n_channels, n_times)

        # infer sfreq from TIME_COL if possible
        sfreq = SAMPLE_RATE
        if TIME_COL and TIME_COL in df.columns:
            dts = []
            for gid in group_ids:
                g = df[df[GROUP_COL] == gid]
                if TIME_COL in g.columns:
                    t = pd.to_numeric(g[TIME_COL], errors="coerce").dropna().values
                    if len(t) >= 2:
                        dt = np.median(np.diff(t))
                        if dt > 0:
                            dts.append(dt)
            if len(dts) > 0:
                median_dt = float(np.median(dts))
                if median_dt > 0:
                    sfreq = 1.0 / median_dt
                    print(f"Inferred sample rate from TIME_COL ≈ {sfreq:.2f} Hz (median dt={median_dt:.4f}s)")
                else:
                    print("Could not infer sample rate from TIME_COL; using SAMPLE_RATE fallback.")
            else:
                print("TIME_COL present but could not compute dt; using SAMPLE_RATE fallback.")
        else:
            print("No TIME_COL; using SAMPLE_RATE fallback.")

        # create mne Info
        info = mne.create_info(ch_names=channel_cols, sfreq=sfreq, ch_types=["eeg"] * len(channel_cols))
        # try to set a standard montage if channel names match
        montage_set = False
        try:
            std_montage = mne.channels.make_standard_montage("standard_1020")
            intersection = set(channel_cols) & set(std_montage.ch_names)
            if len(intersection) >= max(3, int(0.5 * len(channel_cols))):
                info.set_montage(std_montage)
                montage_set = True
                print("Applied standard_1020 montage (partial/complete match).")
            else:
                print("Channel names do not match standard_1020 sufficiently; skipping montage assignment.")
        except Exception as e:
            print("Could not set montage:", e)
            montage_set = False

        # create EpochsArray
        try:
            epochs_mne = mne.EpochsArray(epoch_data, info, tmin=0.0, verbose=False)
            print(f"Created EpochsArray: {epochs_mne}")
        except Exception as e:
            print("Failed to create MNE EpochsArray:", e)
            epochs_mne = None

        if epochs_mne is not None:
            # compute PSD with MNE (per epoch, per channel)
            try:
                # limit fmax to Nyquist and to desired range
                fmax = min(45.0, epochs_mne.info["sfreq"] / 2.0)
                psds, freqs = psd_welch(epochs_mne, fmin=0.5, fmax=fmax, n_jobs=1, verbose=False)
                # psds shape (n_epochs, n_channels, n_freqs)
                print(f"Computed PSDs with shape {psds.shape}, freqs len = {len(freqs)}")
            except Exception as e:
                print("mne.time_frequency.psd_welch failed:", e)
                psds = None
                freqs = None

            # Plot mean PSD per label (averaged over channels)
            if psds is not None:
                unique_labels = list(pd.unique(labels_for_epochs))
                plt.figure(figsize=(9, 6))
                for lbl in unique_labels:
                    idxs = [i for i, x in enumerate(labels_for_epochs) if str(x) == str(lbl)]
                    if not idxs:
                        continue
                    # average across epochs and channels -> 1D freq vector
                    mean_psd = psds[idxs].mean(axis=(0, 1))
                    plt.semilogy(freqs, mean_psd, label=str(lbl))
                plt.xlim(0.5, fmax)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("PSD (power)")
                plt.title("Mean PSD per label (averaged over channels)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                save_fig("mne_psd_by_label_channel_avg")

                # For each band, compute average band power per channel and plot topomap (if montage exists)
                for band_name, band_range in BANDS.items():
                    low, high = band_range
                    mask = (freqs >= low) & (freqs <= high)
                    if not np.any(mask):
                        continue
                    for lbl in unique_labels:
                        idxs = [i for i, x in enumerate(labels_for_epochs) if str(x) == str(lbl)]
                        if not idxs:
                            continue
                        subset = psds[idxs]  # (n_epochs_label, n_channels, n_freqs)
                        # integrate across frequency axis to get band power: shape (n_epochs_label, n_channels)
                        band_power_epochs = np.trapz(subset[..., mask], freqs[mask], axis=-1)
                        band_power_mean = band_power_epochs.mean(axis=0)  # (n_channels,)
                        # create an Evoked-like object to use topomap plotting
                        try:
                            evoked_band = mne.EvokedArray(band_power_mean[:, np.newaxis], epochs_mne.info, tmin=0.0)
                            # plot topomap at the single time point
                            fig = evoked_band.plot_topomap(times=[0.0], ch_type="eeg", show=False)
                            # save the figure
                            save_fig(f"mne_topomap_{band_name}_{lbl}")
                        except Exception as e:
                            # if topomap fails (likely due to missing montages or sensor positions), fallback to barplot
                            plt.figure(figsize=(10, 4))
                            sns.barplot(x=channel_cols, y=band_power_mean, palette="viridis")
                            plt.title(f"Band power ({band_name}) per channel — label={lbl}")
                            plt.xticks(rotation=90)
                            save_fig(f"mne_barband_{band_name}_{lbl}")

                # Evoked (time-domain average) per label
                for lbl in unique_labels:
                    idxs = [i for i, x in enumerate(labels_for_epochs) if str(x) == str(lbl)]
                    if not idxs:
                        continue
                    try:
                        epochs_label = epochs_mne[idxs]
                        ev = epochs_label.average()
                        # plot evoked time-series (channels colored)
                        fig = ev.plot(spatial_colors=True, show=False)
                        save_fig(f"mne_evoked_{lbl}")
                        # optionally plot topomap at a couple of times if montage present
                        if montage_set:
                            times_to_plot = np.linspace(ev.times[0], ev.times[-1], min(5, len(ev.times)))
                            fig2 = ev.plot_topomap(times=times_to_plot, ch_type="eeg", show=False)
                            save_fig(f"mne_evoked_topomap_{lbl}")
                    except Exception as e:
                        print(f"Failed to compute/plot evoked for label {lbl}: {e}")

            else:
                print("psds is None; skipping MNE PSD/topomap/evoked plots.")
else:
    if not has_mne:
        print("\nMNE not installed; MNE analyses skipped.")
    elif DATA_FORMAT != "long":
        print("\nMNE analyses are only implemented for DATA_FORMAT == 'long' (time samples grouped by GROUP_COL).")

# ---------- Save feature table ----------
SAVE_FEATURES_CSV = True
if SAVE_FEATURES_CSV:
    out_csv = "emotions_features_extracted.csv"
    save_df = pd.DataFrame(X, columns=feature_cols)
    save_df[LABEL_COL] = label_encoder.inverse_transform(y)
    if GROUP_COL in modeling_df.columns:
        if GROUP_COL in modeling_df.columns:
            save_df[GROUP_COL] = modeling_df[GROUP_COL].values
    save_df.to_csv(out_csv, index=False)
    print(f"Saved features CSV -> {out_csv}")

print("Done. Adjust SAMPLE_RATE, LABEL_COL, GROUP_COL, TIME_COL, DATA_FORMAT, and USE_MNE as needed.")