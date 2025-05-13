# 🎬 CTR & Genre-Based Movie Recommendation System

This project is a Streamlit-powered web application that predicts Click-Through Rates (CTR) for movies and offers personalized recommendations using genre preferences and movie overviews. It combines data from the **MovieLens 1M dataset** and the latest **TMDB 2024 dataset**, with NLP-powered enhancements using **TF-IDF** for better content-based filtering.

## 🚀 Features

- ✅ **CTR Prediction** using trained XGBoost classifier.
- 🎯 **Genre-Based Recommendations** tailored to user preferences.
- 🧠 **TF-IDF Integration** to extract semantic similarity from movie overviews.
- 📊 **Explainable AI** with SHAP values for CTR predictions.
- 🎥 **Streamlit Web App Interface** – clean and interactive frontend.
- 🧩 Robust preprocessing and feature engineering.

---

## 📁 Datasets Used

### 1. [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- Contains 1 million ratings from 6,000 users on 4,000 movies.
- User demographics, ratings, and timestamp metadata.

### 2. [TMDB (2024 Scraped)](https://www.themoviedb.org/)
- Used to enrich movie metadata (overview, genre, popularity, release date).
- Scraped and cleaned to match MovieLens data.

---

## 🔍 Model Overview

- **Target Variable:** Binary CTR (1 = Clicked/Watched, 0 = Not Clicked).
- **Input Features:**
  - One-hot encoded genres
  - TF-IDF vectorized overviews
  - Engineered features like `is_family_friendly`, `is_action_heavy`
- **Model Used:** XGBoost Classifier
- **Explainability:** SHAP integration for post-prediction insights

---

## 🛠 Tech Stack

| Component         | Tech Used                         |
|------------------|------------------------------------|
| Interface         | Streamlit                         |
| Backend Model     | XGBoost                           |
| NLP Engine        | TF-IDF (Scikit-learn)             |
| Data Manipulation | Pandas, NumPy                     |
| Explainability    | SHAP                              |
| Deployment        | Streamlit Sharing / Local Hosting |

---

## 🧠 How It Works

1. **User selects genres** or a specific movie.
2. **TF-IDF scores** calculate content similarity between overviews.
3. **CTR Model** predicts likelihood of user engagement.
4. **Recommendations** are shown based on CTR scores + similarity.

---

