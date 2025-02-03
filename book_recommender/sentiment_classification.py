import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

# Load the dataset
books = pd.read_csv("book_recommender/data/books_with_classification.csv")

# Initialize the emotion classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7)

# Emotion labels
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Function to calculate max emotion score for a description
def calculate_max_emotion_scores(predictions):
    max_scores = {}
    for prediction in predictions:
        for p in prediction:
            label, score = p["label"], p["score"]
            if label not in max_scores or score > max_scores[label]:
                max_scores[label] = score
    return {label: max_scores.get(label, 0.0) for label in emotion_labels}

# Function to process description and extract sentence-level emotions
def get_sentence_emotions(description):
    sentences = description.split(".")
    return classifier(sentences)

# Prepare to store results
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

# Loop through the books and process each one
for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    description = books["description"][i]
    
    # Get emotion predictions for each sentence
    predictions = get_sentence_emotions(description)
    
    # Calculate the max emotion scores for this book's description
    max_scores = calculate_max_emotion_scores(predictions)
    
    # Store the max emotion scores for each label
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

# Create a DataFrame with emotion scores
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn

# Merge the emotion scores back with the original books DataFrame
books = pd.merge(books, emotions_df, on="isbn13")

# Save the final result to CSV
books.to_csv("books_with_emotions.csv", index=False)
