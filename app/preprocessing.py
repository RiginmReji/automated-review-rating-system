# app/preprocessing.py

"""
Text preprocessing functions for the Automated Review Rating System.
Includes text cleaning, stopwords removal, lemmatization, and length filtering.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean the input text by:
    - Lowercasing
    - Removing punctuation and special characters
    - Removing extra whitespace
    - Removing stopwords
    - Lemmatizing words

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


def filter_reviews_by_length(df, text_col='Review_text', min_words=3, max_words=200):
    """
    Remove reviews that are too short or too long.

    Args:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Column name containing review text.
        min_words (int): Minimum words to keep.
        max_words (int): Maximum words to keep.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    df_filtered = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
    df_filtered = df_filtered.drop(columns=['word_count'])
    return df_filtered
