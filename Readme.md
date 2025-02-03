# Book Recommender

A modern book recommendation system that helps you discover your next favorite read based on your search query, preferred category, and mood. The project leverages state-of-the-art natural language processing, vector search, and interactive UI techniques.


![image](https://github.com/user-attachments/assets/3a228f38-7d7c-494d-9f7e-7c2eb063a2d7)


## Overview

This project uses multiple advanced tools and libraries to process, analyze, and recommend books. Key components include:

- **Data Ingestion and Cleaning:**  
  Loads and preprocesses book data using Pandas and NumPy. Datasets are cleaned and enriched (for example, combining book titles and subtitles) before further processing.

- **Vector Search & Embeddings:**  
  Utilizes LangChain's integration with OpenAI Embeddings and Chroma to convert book descriptions into vector representations. These representations are then stored in a vector database and queried for semantic similarity, enabling effective recommendations.

- **UI Dashboard:**  
  Built with Gradio to provide an interactive web interface. Custom CSS styling and layout improvements ensure a modern, user-friendly experience.

- **Zero-Shot Text Classification:**  
  Uses HuggingFace’s `transformers` library for zero-shot classification to simplify diverse book categories into broader classifications (e.g., Fiction vs. NonFiction).

- **Sentiment & Emotion Analysis:**  
  Analyzes the sentiment of book descriptions using emotion classification pipelines, allowing for mood-based filtering of recommendations.

- **Data Exploration & Visualization:**  
  Employs Seaborn and Matplotlib for EDA (Exploratory Data Analysis) to better understand data patterns, missing values, and distributions across various book attributes.

## Tools and Libraries

- **Python:** Core programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **dotenv:** For managing environment variables.
- **Gradio:** To create an interactive dashboard for book recommendations.
- **LangChain:**  
  - `langchain-community.document_loaders` to load raw text files.  
  - `langchain-text-splitters` for dividing text into chunks.  
  - `langchain-openai` for generating document embeddings.  
  - `langchain-chroma` to store and query the embeddings in a vector database.
- **Transformers (HuggingFace):** Used for zero-shot text classification and emotion detection.
- **KaggleHub:** To download and manage datasets from Kaggle.
- **Seaborn & Matplotlib:** For data visualization.
- **Torch:** Underlying framework supporting deep learning models.

## Project Structure

- **book_recommender/gradio_dashboard.py:**  
  Contains the code for the Gradio dashboard, including custom CSS and UI elements to accept queries, filter by category or mood, and display recommended books.

- **book_recommender/data_exploration.py:**  
  Provides scripts to explore and visualize the dataset, highlighting data quality, missing values, and variable distributions.

- **book_recommender/vector_search.py:**  
  Handles the conversion of book descriptions into embeddings and executes similarity searches using the Chroma vector database.

- **book_recommender/text_classification.py:**  
  Implements zero-shot classification to simplify book categories (e.g., Fiction vs. NonFiction).

- **book_recommender/sentiment_classification.py:**  
  Analyzes the sentiment of book descriptions on a sentence level to aggregate emotion scores for each book.

- **book_recommender/data:**  
  should contain processed data files such as `books_cleaned.csv`, `books_with_classification.csv`, and `books_with_emotions.csv`.

- **requirements.txt:**  
  Lists all required packages and their versions for reproducibility.

- **.gitignore:**  
  Specifies files and directories that should be excluded from version control (e.g., `.env`, virtual environments).

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project root directory as needed.

4. **Launch the Dashboard:**
   ```bash
   python book_recommender/gradio_dashboard.py
   ```
   This command will launch the Gradio UI, allowing you to interact with the Book Recommender in your browser.

