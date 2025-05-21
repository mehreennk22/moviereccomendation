# 🎬 Movie Recommendation System

This project builds a hybrid **Movie Recommendation System** using both **Collaborative Filtering** (Alternating Least Squares - ALS) and **Content-Based Filtering** (TF-IDF + Cosine Similarity) techniques on the MovieLens dataset. It also includes Exploratory Data Analysis (EDA) to uncover insights about movie ratings and user preferences.

## 📌 Features

- 📈 Exploratory Data Analysis (EDA) on user ratings and movies
- 🤝 Collaborative Filtering using ALS (PySpark MLlib)
- 🧠 Content-Based Filtering using TF-IDF on genres
- 🔍 Hybrid Recommendation functionality using Streamlit UI
- 📊 RMSE-based model evaluation
- 💡 Clean and interactive user interface

## 📁 Project Structure


## ⚙️ How It Works

### 1. Collaborative Filtering (ALS)
- Trains an ALS model to learn latent user and movie factors.
- Predicts user ratings as the dot product of latent vectors.
- Evaluates model using **Root Mean Squared Error (RMSE)**.

### 2. Content-Based Filtering
- Creates a TF-IDF matrix of movie genres.
- Recommends similar movies based on **cosine similarity** with the input movie.

### 3. Hybrid Streamlit App
- User selects a movie from dropdown.
- App recommends:
  - Similar movies based on genres (content-based)
  - Top-rated unseen movies for the user (collaborative)

## 📊 Exploratory Data Analysis Highlights

- Distribution of user ratings
- Most-rated and highest-rated movies
- Average ratings over time
- Insights into user and movie behavior

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PySpark
- Pandas, NumPy, scikit-learn
- Streamlit

### Installation

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py
