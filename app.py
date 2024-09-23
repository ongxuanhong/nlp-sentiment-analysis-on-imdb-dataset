import streamlit as st
import joblib

# Load pre-trained TF-IDF vectorizer and RandomForest classifier
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
rf_model = joblib.load("models/randomforest_model.pkl")

# Title of the app
st.title("Sentiment Analysis App")
st.write("Enter a comment to predict its sentiment (positive or negative).")

# Input box for new review
user_input = st.text_area("Enter your review here", "")

# Add a button to trigger prediction
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess user input and predict
        user_input_tfidf = tfidf.transform([user_input])
        prediction = rf_model.predict(user_input_tfidf)
        predicted_sentiment = "Positive" if prediction[0] == 1 else "Negative"

        # Show prediction result
        st.write(f"Predicted Sentiment: **{predicted_sentiment}**")
    else:
        st.write("Please enter a review to predict sentiment.")
