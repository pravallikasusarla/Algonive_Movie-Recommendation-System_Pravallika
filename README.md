# Algonive_Movie-Recommendation-System_Pravallika

**Project Overview**

This project develops a Movie Recommendation System that suggests movies based on user preferences and movie similarities.
It uses Content-Based Filtering and User-Based Collaborative Filtering to provide personalized recommendations.
The system is built using machine learning techniques and deployed as an interactive web application using Streamlit.


**Objectives**

Analyze and preprocess movie and user rating datasets

Apply Content-Based Filtering using TF-IDF and Cosine Similarity

Implement User-Based Collaborative Filtering to find similar users

Build a unified recommendation engine for both approaches

Create a Streamlit interface for interactive movie suggestions


**Tools and Technologies Used**

Python
Pandas
NumPy
Scikit-learn
Streamlit


**Key Insights**

The system effectively identifies movies with similar genres and content.

Users with comparable rating patterns receive relevant recommendations.

Combining content-based and collaborative methods improves accuracy.

The interface provides a simple and interactive way for users to explore movies.


**Future Improvements**

Add sentiment analysis of user reviews to refine recommendation accuracy

Include regional or multilingual movie datasets (Telugu, Hindi, etc.)

Integrate real-time APIs such as TMDb for updated movie information

Implement user login and preference tracking for better personalization


**Dataset**

Dataset used: https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset
This dataset includes movie titles, genres, and user rating data.


**How to Run Locally**

1. Clone this repository:

git clone https://github.com/pravallikasusarla/movie-recommendation-system.git
cd movie-recommendation-system


2. Install dependencies:

pip install -r requirements.txt


3. Run the Streamlit app:

streamlit run app.py

