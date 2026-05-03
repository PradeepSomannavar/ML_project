import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_kddcup99

RANDOM_STATE = 42

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

NUMERICAL_COLS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_data(data_dir="data", sample_size=None):
    print("[1/7] Loading KDDCup'99 dataset...")
    raw_data_path = os.path.join(data_dir, "kddcup.data_10_percent.gz")

    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, header=None, names=COLUMNS)
    else:
        print("  Downloading dataset via sklearn fetch_kddcup99...")
        kdd = fetch_kddcup99(as_frame=True)
        df = kdd.frame
        df.columns = COLUMNS

    for col in CATEGORICAL_COLS + ["label"]:
        if df[col].dtype == object:
            mask = df[col].apply(lambda x: isinstance(x, bytes))
            df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x.decode("utf-8"))

    df["label"] = df["label"].str.rstrip(".")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE, replace=False)
        df = df.sort_index().reset_index(drop=True)
        print(f"  Sampled {sample_size} rows from {len(df)} total")

    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")
    return df


def handle_missing_values(df):
    print("[2/7] Handling missing values (mean substitution)...")
    for col in NUMERICAL_COLS:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    print(f"  Missing values remaining: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    print("[3/7] Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"  Removed {before - after} duplicates ({before} -> {after})")
    return df


def encode_labels(df):
    print("[4/7] Encoding categorical features and labels...")
    le_proto = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()

    df["protocol_type"] = le_proto.fit_transform(df["protocol_type"])
    df["service"] = le_service.fit_transform(df["service"])
    df["flag"] = le_flag.fit_transform(df["flag"])

    label_encoders = {
        "protocol_type": le_proto,
        "service": le_service,
        "flag": le_flag
    }

    le_binary = LabelEncoder()
    df["label_binary"] = le_binary.fit_transform(df["label"].apply(lambda x: "normal" if x == "normal" else "attack"))

    le_multiclass = LabelEncoder()
    df["label_multiclass"] = le_multiclass.fit_transform(df["label"])

    label_encoders["binary"] = le_binary
    label_encoders["multiclass"] = le_multiclass

    print(f"  Binary classes: {le_binary.classes_}")
    print(f"  Multi-class classes: {le_multiclass.classes_}")
    return df, label_encoders


def apply_normalization(X_train, X_test):
    print("[5/7] Applying Z-score normalization (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Normalization complete")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    print("[6/7] Applying SMOTE oversampling on training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  Training samples: {len(y_train)} -> {len(y_train_res)}")
    print(f"  Class distribution after SMOTE: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")
    return X_train_res, y_train_res, smote


def apply_pca(X_train, X_test, variance_threshold=0.95):
    print(f"[7/7] Applying PCA ({variance_threshold*100}% variance retention)...")
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"  Reduced features: {X_train.shape[1]} -> {X_train_pca.shape[1]}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return X_train_pca, X_test_pca, pca


def preprocess_pipeline(data_dir="data", sample_size=None):
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data(data_dir, sample_size=sample_size)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df, label_encoders = encode_labels(df)

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values
    y_binary = df["label_binary"].values
    y_multiclass = df["label_multiclass"].values

    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )
    _, _, y_train_multi, y_test_multi = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )

    X_train_norm, X_test_norm, scaler = apply_normalization(X_train, X_test)
    X_train_res, y_train_res, smote = apply_smote(X_train_norm, y_train_bin)
    X_train_pca, X_test_pca, pca = apply_pca(X_train_res, X_test_norm)

    X_train_res_pca = pca.transform(X_train_res)

    artifacts = {
        "scaler": scaler,
        "smote": smote,
        "pca": pca,
        "label_encoders": label_encoders,
        "feature_cols": feature_cols
    }

    data_splits = {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "X_train_res_pca": X_train_res_pca,
        "y_train_bin": y_train_bin,
        "y_test_bin": y_test_bin,
        "y_train_multi": y_train_multi,
        "y_test_multi": y_test_multi
    }

    print("\nPreprocessing complete!")
    print(f"  Train samples (after SMOTE+PCA): {X_train_pca.shape}")
    print(f"  Test samples: {X_test_pca.shape}")
    print("=" * 60)

    return data_splits, artifacts, df
