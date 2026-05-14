"""
src/preprocessing.py
----------------------
Handles all data preprocessing steps for the HR Attrition dataset.

Steps performed:
  1. Load the raw CSV dataset
  2. Drop columns that add no predictive value
  3. Encode the target variable (Attrition: Yes/No → 1/0)
  4. Label-encode remaining categorical columns
  5. Scale numerical features with StandardScaler
  6. Split data into train and test sets

This module is imported by train_model.py and app.py.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# Columns that carry no useful predictive signal
COLUMNS_TO_DROP = [
    'EmployeeCount',   # Always 1 — no variance
    'EmployeeNumber',  # Just an ID
    'Over18',          # All employees are over 18
    'StandardHours',   # Always 80 — no variance
]

# Path where the fitted scaler will be saved
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'label_encoders.pkl')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.pkl')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')


def load_data(filepath=None):
    """
    Load the raw HR Attrition CSV file into a PD DF.

    Args:
        filepath (str): Path to the CSV. If None -->uses default path.

    Returns:
        pd.DataFrame: Raw dataset
    """
    if filepath is None:
        filepath = DATA_PATH

    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def drop_useless_columns(df):
    """
    Remove columns that contribute no useful information to the model.

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    existing_cols = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"[INFO] Dropped columns: {existing_cols}")
    return df


def handle_missing_values(df):

    missing_before = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values: Before={missing_before}, After={missing_after}")
    return df


def encode_features(df, fit=True, encoders=None):
    """
    Encode categorical columns into numbers using LabelEncoder.
    The target column 'Attrition' is encoded separately: Yes=1, No=0.

    Args:
        df (pd.DataFrame): Dataset
        fit (bool): If True, fit new encoders. If False, use saved encoders.
        encoders (dict): Pre-fitted encoders (used during prediction)

    Returns:
        pd.DataFrame: Encoded dataset
        dict: Fitted label encoders
    """
    
    # Encode target variable explicitly
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # all remaining categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if fit:
        # Training phase: fit a new encoder for each column
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        print(f"[INFO] Label-encoded columns: {categorical_cols}")
    else:
        # use previously fitted encoders
        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    return df, encoders


def scale_features(X_train, X_test=None, fit=True, scaler=None):
    """
    Standardise numerical features to have mean=0 and std=1.
    """
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = None

    return X_train_scaled, X_test_scaled, scaler


def preprocess_for_training(filepath=None):
    """
    Full preprocessing pipeline for model training.
    Loads, cleans, encodes, scales, and splits the dataset.

    Args:
        filepath (str): Path to raw CSV

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, encoders
    """
    # Load raw data
    df = load_data(filepath)

    # Drop useless columns
    df = drop_useless_columns(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical features
    df, encoders = encode_features(df, fit=True)

    # Separate features x from target y
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    feature_names = X.columns.tolist()

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # features scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, fit=True)

    # save
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_COLS_PATH)
    print(f"[INFO] Saved scaler → {SCALER_PATH}")
    print(f"[INFO] Saved encoders → {ENCODER_PATH}")
    print(f"[INFO] Saved feature columns → {FEATURE_COLS_PATH}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, encoders


def preprocess_single_input(input_dict):
   
    # Load saved artefacts
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)

    df_input = pd.DataFrame([input_dict])

    # Encode categorical columns using saved encoders
    df_input, _ = encode_features(df_input, fit=False, encoders=encoders)

    # Ensure columns are in the exact same order as during training
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0 

    df_input = df_input[feature_columns]

    # Scale
    X_scaled, _, _ = scale_features(df_input, fit=False, scaler=scaler)

    return X_scaled
