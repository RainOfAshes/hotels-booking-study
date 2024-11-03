from typing import Tuple, Dict

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_booking_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """
    Preprocesses the booking data by mapping, renaming, and selecting relevant features.

    Parameters:
        df (pd.DataFrame): Raw booking dataset.

    Returns:
        tuple: A tuple containing:
            - preprocessed_df (pd.DataFrame): The processed DataFrame ready for training or analysis.
            - features_dict (dict): Dictionary categorizing features for binary, numerical, categorical, and label columns.
    """
    # Map special requests to a binary indicator
    df["has_special_requests"] = df["no_of_special_requests"].apply(bool).apply(int)

    # Define and apply mappings for market segment
    market_map = {'Offline': 'Offline', 'Online': 'Online', 'Corporate': 'Corporate',
                  'Complementary': 'Other', 'Aviation': 'Other'}
    df["normalized_market_segment_type"] = df["market_segment_type"].map(market_map)
    market_segment_value = {'Online': 0, 'Offline': 1, 'Corporate': 2, 'Other': 3}
    df["normalized_market_segment_value"] = df["normalized_market_segment_type"].map(market_segment_value).astype(int)

    # Convert has_children to integer
    df["has_children"] = df["has_children"].apply(int)

    # Map booking status to binary labels
    booking_status_map = {"Canceled": 1, "Not_Canceled": 0}
    df["booking_status_label"] = df["booking_status"].map(booking_status_map).astype(int)

    df['arrival_datetime'] = pd.to_datetime(df['arrival_datetime'], errors='coerce')
    df['year_quarter'] = df['arrival_datetime'].dt.quarter

    weekdays = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    df["arrival_day_name"] = df["arrival_day_name"].map(weekdays)

    # Rename columns for consistency
    df.rename(columns={"required_car_parking_space": "has_car",
                       "repeated_guest": "is_repeated_guest"}, inplace=True)

    # Select and reorder necessary features
    needed_features_order = [
        'is_repeated_guest', 'has_car', "has_special_requests",
        'lead_time', 'has_children', 'normalized_market_segment_value',
        'year_quarter', 'arrival_month',
        # 'arrival_day_name',
        'total_nights', 'total_guests',
        'booking_status_label'
    ]
    preprocessed_df = df[needed_features_order].copy().reset_index(drop=True)

    # Define feature categories for model input
    binary_features = ["is_repeated_guest", 'has_car', "has_special_requests", 'has_children']
    # 'normalized_market_segment_value', 'year_quarter' are ordinal categorical  features,
    # since their values have significance.
    numerical_features = ['lead_time', 'total_nights', 'total_guests',
                          'normalized_market_segment_value', 'year_quarter']
    categorical_features = ['arrival_month',
                            # 'arrival_day_name'
                            ]
    label_columns = ['booking_status_label']

    features_dict = {
        "binary_features": binary_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "label_columns": label_columns
    }

    return preprocessed_df, features_dict


def split_data_train_val_test(df, label_column='booking_status_label',
               test_size=0.2, val_size=0.1, random_state=42) -> Tuple:
    # Extract features and labels
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    # First split into training+validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    val_relative_size = val_size / (1 - test_size)  # Adjust val size relative to train+val set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, stratify=y_train_val, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "accuracy": accuracy_score(y, y_pred)
    }
    return metrics


def evaluate_model_with_threshold(model, X, y, threshold=0.5):
    # Get probability predictions for the positive class
    y_prob = model.predict_proba(X)[:, 1]  # Probability of positive class

    # Apply the custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate evaluation metrics
    metrics = {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "accuracy": accuracy_score(y, y_pred)
    }
    return metrics


def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    return model, train_metrics, val_metrics
