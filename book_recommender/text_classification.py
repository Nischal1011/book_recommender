import pandas as pd
import numpy as np
import re
from transformers import pipeline
from tqdm import tqdm

# Load dataset
books = pd.read_csv("book_recommender/data/books_cleaned.csv")

# Initialize zero-shot classification model
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Fiction categories for zero-shot classification
fiction_categories = ['Fiction', 'NonFiction']

# Function to categorize book descriptions using zero-shot classification
def classify_text(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions['scores'])
    return predictions['labels'][max_index]

# Function to classify categories into simple labels (Fiction, NonFiction, Other)
def categorize_category(row):
    if pd.isna(row):
        return np.nan
    elif re.search(r'(?<!\w)(Fiction|Fantasy|Science fiction|Historical fiction|Detective)(?!\w)', row, re.IGNORECASE):
        return 'Fiction'
    elif re.search(r'(?<!\w)(Biography|History|Essay|Nonfiction)(?!\w)', row, re.IGNORECASE):
        return 'NonFiction'
    else:
        return 'Other'

# Apply categorization to 'categories' column
books['simple_categories'] = books['categories'].apply(categorize_category)

# Filter out 'Other' categories for reclassification
books['simple_categories'] = books['simple_categories'].replace('Other', np.nan)

# Function to generate predictions for books missing categories
def generate_predictions_for_missing_categories(missing_data, categories):
    isbns = []
    predicted_cats = []
    for i in tqdm(range(len(missing_data))):
        sequence = missing_data['description'][i]
        predicted_cats.append(classify_text(sequence, categories))
        isbns.append(missing_data['isbn13'][i])
    return pd.DataFrame({'isbn13': isbns, 'missing_prediction': predicted_cats})

# Function to compute the accuracy of predictions
def compute_accuracy(actual_cats, predicted_cats):
    predictions_df = pd.DataFrame({'actual_categories': actual_cats, 'predicted_categories': predicted_cats})
    predictions_df['correct_predictions'] = np.where(predictions_df['actual_categories'] == predictions_df['predicted_categories'], 1, 0)
    accuracy = predictions_df['correct_predictions'].sum() / len(predictions_df)
    return accuracy

# Split books into Fiction and NonFiction descriptions
fiction_books = books[books['simple_categories'] == 'Fiction']
nonfiction_books = books[books['simple_categories'] == 'NonFiction']

# Generate predictions for Fiction and NonFiction books
actual_cats, predicted_cats = [], []

# Predict Fiction categories
for description in tqdm(fiction_books['description']):
    actual_cats.append('Fiction')
    predicted_cats.append(classify_text(description, fiction_categories))

# Predict NonFiction categories
for description in tqdm(nonfiction_books['description']):
    actual_cats.append('NonFiction')
    predicted_cats.append(classify_text(description, fiction_categories))

# Compute prediction accuracy
accuracy = compute_accuracy(actual_cats, predicted_cats)
print(f"Prediction accuracy: {accuracy * 100:.2f}%")

# Handle missing categories by generating predictions for them
missing_cats = books[books['simple_categories'].isna()][['isbn13', 'description']].reset_index(drop=True)
missing_prediction_df = generate_predictions_for_missing_categories(missing_cats, fiction_categories)

# Merge the predictions with the original data
books = books.merge(missing_prediction_df, on='isbn13', how='left')
books['simple_categories'] = np.where(books['simple_categories'].isna(), books['missing_prediction'], books['simple_categories'])

# Save final results to CSV
books.to_csv("books_with_classification.csv", index=False)
