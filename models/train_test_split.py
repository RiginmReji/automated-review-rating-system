import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def balance_dataset(df, target_col="Rating", n_samples=5000, random_state=42):
    balanced = df.groupby(target_col).sample(n=n_samples, random_state=random_state)
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

def prepare_train_test(df, text_col="Review_text", target_col="Rating", test_size=0.2, max_features=5000):
    X = df[text_col]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned_reviews.csv")
    balanced_df = balance_dataset(df, target_col="Rating", n_samples=5000)
    X_train, X_test, y_train, y_test, vectorizer = prepare_train_test(balanced_df)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Class balance in Train:\n", y_train.value_counts())
    print("Class balance in Test:\n", y_test.value_counts())

