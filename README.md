# Automated Review Rating System

## Project Overview and Objective
- Predicts product review ratings (1–5 stars) from textual customer reviews using machine learning.
- Goal: Automatically classify customer reviews based on sentiment and assign a rating.
- Use case: Helps e-commerce platforms analyze user feedback and identify trends quickly.

## Dataset Description
- Dataset: Amazon product reviews (collected from Kaggle).
- Columns used:
  - `Review_text`: Text of the review
  - `Rating`: Rating of the product (1–5 stars)
- Dataset size (after cleaning): depends on your cleaned dataset.
- Preprocessing removed unnecessary columns like `ProductId`, `UserId`, `ProfileName`, etc.

## Preprocessing Steps
1. **Removed duplicates and conflicting reviews**  
   - Removed reviews with the same text but different ratings.
2. **Text normalization**  
   - Lowercased all text  
   - Removed punctuation and special characters
3. **Stopwords removal**  
   - Removed common English stopwords using NLTK
4. **Lemmatization**  
   - Converted words to their base form (preferred over stemming for better readability)
5. **Filtered review length**  
   - Removed very short reviews (<3 words)  
   - Removed excessively long reviews (>200 words)

## Visualizations Used
- Rating distribution (bar chart)  
- Review length vs Rating (min/max words per rating)  
- Balanced dataset distribution after sampling

## Balancing Strategy
- Dataset was imbalanced across ratings.  
- Applied undersampling/oversampling to create a balanced dataset:
  - Each rating class has **5000 samples**

## Train-Test Split Methodology
- Performed **stratified train-test split** (80% train, 20% test)  
  - Ensures all classes are balanced in both sets
- Applied **TF-IDF vectorization** to text features
  - Fit vectorizer only on the training set, then transformed the test set

## Notes on Decisions
- **Lemmatization vs Stemming:** Lemmatization preserves word meaning and improves model performance.
- **TF-IDF vectorizer:** Chosen over Bag-of-Words to give importance to less frequent, meaningful words.
- **Balanced dataset:** Ensures model doesn’t favor majority ratings.
