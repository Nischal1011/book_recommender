import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os

load_dotenv()

# Load books and set thumbnails
books = pd.read_csv("book_recommender/data/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), 
    "book_recommender/data/OnePageBookCoverImage.jpg", 
    books['large_thumbnail']
)

CHROMA_DB_PATH = "book_recommender/chroma_db"

def get_or_create_db():
    if os.path.exists(CHROMA_DB_PATH):
        try:
            print("Loading existing Chroma database...")
            return Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=OpenAIEmbeddings()
            )
        except Exception as e:
            print(f"Error loading database: {e}")
    
    print("Creating new Chroma database...")
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    
    return Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DB_PATH
    )

db_books = get_or_create_db()

def retrieve_book_recommendation(query: str, category: str = None, tone: str = None,
                                 initial_top_k=50, final_top_k=16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [int(rec.page_content.strip('"\'').split()[0]) for rec in recs]
    matching_books = books[books['isbn13'].isin(book_list)]
    
    book_recs = matching_books.copy()
    
    if category and category != "All":
        book_recs = book_recs[book_recs['simple_categories'] == category]
    
    if tone and tone != "All":
        tone_mapping = {
            "Happy": "joy", "Surprising": "surprise",
            "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"
        }
        if tone in tone_mapping:
            book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)
    
    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_book_recommendation(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row.get('description', 'No description available')
        truncated_desc = ' '.join(str(description).split()[:30]) + '...'
        
        authors = str(row.get('authors', 'Unknown Author')).split(';')
        if len(authors) >= 2:
            authors_str = f"{', '.join(authors[:-1])} and {authors[-1]}"
        else:
            authors_str = authors[0]
        
        caption = f"📖 {row['title']}\n✍️ {authors_str}\n📝 {truncated_desc}"
        results.append((row['large_thumbnail'], caption))
    
    return results

# UI Configuration
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

css = """
.gallery {
    gap: 20px !important;
}
.gallery-item {
    height: 400px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
}
.gallery-item img {
    height: 300px !important;
    object-fit: cover !important;
}
.gallery-item .caption {
    padding: 12px !important;
    color: white !important;
    font-size: 0.9em !important;
    line-height: 1.4 !important;
}
"""

with gr.Blocks(css=css) as dashboard:
    gr.Markdown("# 📚 Discover Your Next Favorite Book")
    
    with gr.Row():
        user_query = gr.Textbox(
            label="Describe your reading mood",
            placeholder="e.g. 'A thrilling mystery with complex characters'",
            max_lines=3
        )
        
    with gr.Row():
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Filter by Category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Filter by Mood",
            value="All"
        )
        submit_btn = gr.Button("Find Books 🔍", variant="primary")
    
    gallery = gr.Gallery(
        label="Recommended Books",
        columns=5,
        rows=2,
        object_fit="cover",
        height="auto"
    )

    submit_btn.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

if __name__ == "__main__":
    dashboard.launch(share=True)