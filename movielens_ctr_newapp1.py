
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="🎥 Movie CTR Predictor", layout="wide")
st.title("🎯 Movie Click-Through Rate (CTR) Predictor & Recommender")
st.markdown("🔍 *Estimate how likely a user is to click on a recommended movie.*")

# Load model and data
model = joblib.load("movielens_ctr.joblib")
train_df = pd.read_csv("merged_movie_data.csv")  # 🔁 Ensure this file exists
merged_df=pd.read_csv("merged_movie_file.csv")

# Expected features for prediction
expected_features = ['Adventure','Drama', 'Film-Noir', 'IMAX', 'Musical', 'Sci-Fi', 'War', 'Western',
                     'has_no_genres', 'user_avg_rating', 'user_rating_count', 'is_documentary',
                     'History', 'is_light_hearted','Action', 'Crime', 'Fantasy','movie_rating_count',
                     'is_family_friendly', 'is_romantic', 'is_thriller', 'is_scifi_fantasy',
                     'is_action_heavy', 'is_dark_intense']



explainer = shap.TreeExplainer(model)

# Genre Recommendation Function
def recommend_by_genres(input_genres, top_n=10):
    input_genres = [g.strip().lower() for g in input_genres.split(',')]
    genre_cols = [col for col in merged_df.columns if col.lower() in input_genres]
    if not genre_cols:
        return pd.DataFrame(columns=["title", "popularity_norm"])

    filtered_df = merged_df[merged_df[genre_cols].sum(axis=1) > 0].drop_duplicates(subset='title')

    if len(filtered_df) == 0:
        return merged_df[merged_df[genre_cols].sum(axis=1) > 0].drop_duplicates(subset='title').head(1)

    top_movies = filtered_df.sort_values(by='popularity_norm', ascending=False).head(top_n)
    return top_movies[['title', 'popularity_norm'] + genre_cols]

# Sidebar Inputs
st.sidebar.header("📥 Enter Movie & User Info")

user_id = st.sidebar.number_input("👤 User ID", min_value=1)
title = st.sidebar.text_input("🎬 Movie Title")

genres = ["Action", "Adventure", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "IMAX", "Musical", "Sci-Fi", "War", "Western"]
selected_genres = st.sidebar.multiselect("🎭 Select Genres", genres)

# Thematic checkboxes
is_family_friendly = st.sidebar.checkbox("👪 Family Friendly")
is_romantic = st.sidebar.checkbox("💕 Romantic")
is_thriller = st.sidebar.checkbox("🔪 Thriller")
is_scifi_fantasy = st.sidebar.checkbox("👽 Sci-Fi / Fantasy")
is_action_heavy = st.sidebar.checkbox("💥 Action Heavy")
is_dark_intense = st.sidebar.checkbox("🌑 Dark & Intense")
is_documentary = st.sidebar.checkbox("🎓 Documentary")

# Profile features
st.sidebar.markdown("🧠 **User & Movie Stats**")
user_avg_rating = st.sidebar.slider("📊 User Avg Rating", 0.0, 5.0, 3.5)
user_rating_count = st.sidebar.number_input("🗳️ User Rating Count", 1, 1000, 450)
movie_avg_rating = st.sidebar.slider("🌟 Movie Avg Rating", 0.0, 5.0, 3.5)
movie_rating_count = st.sidebar.number_input("🎫 Movie Rating Count", 1, 10000, 300)

# Input preparation
def prepare_input():
    row = {
        'Action': int('Action' in selected_genres),
        'Adventure': int('Adventure' in selected_genres),
        'Crime': int('Crime' in selected_genres),
        'Drama': int('Drama' in selected_genres),
        'Fantasy': int('Fantasy' in selected_genres),
        'Film-Noir': int('Film-Noir' in selected_genres),
        'IMAX': int('IMAX' in selected_genres),
        'Musical': int('Musical' in selected_genres),
        'Sci-Fi': int('Sci-Fi' in selected_genres),
        'War': int('War' in selected_genres),
        'Western': int('Western' in selected_genres),
        'has_no_genres': int(len(selected_genres) == 0),
        'user_avg_rating': user_avg_rating,
        'user_rating_count': user_rating_count,
        'movie_rating_count': movie_rating_count,
        'is_family_friendly': int(is_family_friendly),
        'is_romantic': int(is_romantic),
        'is_thriller': int(is_thriller),
        'is_scifi_fantasy': int(is_scifi_fantasy),
        'is_action_heavy': int(is_action_heavy),
        'is_dark_intense': int(is_dark_intense),
        'is_documentary': int(is_documentary),        
        # Defaults for engineered features
        'History': 0,
        'is_light_hearted': 0,
                
    }
    return pd.DataFrame([row])

# Prediction Logic
if st.sidebar.button("🚀 Predict CTR"):
    input_df = prepare_input()
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Prediction Result")
        st.markdown(f"🎬 **Movie:** `{title}`")
        st.markdown(f"👤 **User ID:** `{user_id}`")
        st.markdown(f"📈 **Predicted Click:** {'✅ YES' if prediction == 1 else '❌ NO'}")
        st.markdown(f"🔢 **Click Probability:** `{pred_prob:.2%}`")

        if pred_prob > 0.70:
            st.success("🔥 High CTR! Recommend this prominently.")
        elif pred_prob > 0.45:
            st.info("🤔 Moderate interest. Worth suggesting.")
        else:
            st.warning("❄️ Low CTR. Consider other options.")

    with col2:
        st.subheader("🧠 SHAP Force Plot (Explanation)")
        shap_values = explainer.shap_values(input_df)
        plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
        st.image("shap_force_plot.png", caption="🧐 Feature Impact on CTR Prediction")

    if pred_prob < 0.4:
        st.subheader("💡 Recommendations")
        st.markdown('''
- 🎯 **Recommend higher-rated movies** based on user and movie rating trends  
- 🎭 **Explore lighter or popular genres** like `Drama`, `Comedy`, or `Adventure`  
- 📚 **Use past behavior** to personalize suggestions more effectively  
- 🌟 Consider highlighting movies with strong **family-friendly** or **romantic** themes  
        ''')

# Optional Genre Recommendation Section
st.subheader("🎬 Genre-Based Recommendations")
input_genres_text = st.text_input("Enter genres (comma-separated):", value="Drama, Action")

if st.button("🎯 Recommend Movies"):
    recs = recommend_by_genres(input_genres_text)
    if not recs.empty:
        st.write("🎥 Top Recommendations Based on Genre:")
        st.dataframe(recs)
    else:
        st.warning("No movies found for the entered genres.")
