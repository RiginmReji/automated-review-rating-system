# models/train_test_split.py

"""
Dataset balancing, train-test split, and TF-IDF vectorization.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def balance_dataset(df, target_col='Rating', n_samples=5000, random_state=42):
    """
    Balance dataset by sampling equal number of rows per class.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Column name of target.
        n_samples (int): Number of samples per class.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced dataframe.
    """
    balanced = df.groupby(target_col).sample(n=n_samples, random_state=random_state)
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_train_test(df, text_col='Review_text', target_col='Rating', test_size=0.2, max_features=5000):
    """
    Perform stratified train-test split and TF-IDF vectorization.

    Args:
        df (pd.DataFrame): Input dataframe (already balanced).
        text_col (str): Text feature column.
        target_col (str): Target label column.
        test_size (float): Fraction of test set.
        max_features (int): Max TF-IDF features.

    Returns:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """
    X = df[text_col]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

