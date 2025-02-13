import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("refined_ecommerce_product_data.csv")  # Replace with your dataset file name

# Preprocessing function
def preprocess_data(df):
    # Combine text features
    df['text_features'] = df['Category'] + " " + df['Sub_Category'] + " " + df['Review_Sentiment']
    
    # TF-IDF Vectorization for text features
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(df[['Price', 'Customer_Age', 'Purchase_History', 'Review_Rating']])
    
    # Encode categorical features (Customer_Gender)
    encoder = OneHotEncoder()
    gender_encoded = encoder.fit_transform(df[['Customer_Gender']]).toarray()
    
    # Combine all features
    combined_features = np.hstack([tfidf_matrix.toarray(), numerical_features, gender_encoded])
    return combined_features, tfidf, scaler, encoder

# Recommendation function
def get_recommendations(input_features, features_matrix, df, top_n=5):
    # Ensure input_features is 2D
    if input_features.ndim > 2:
        input_features = np.squeeze(input_features, axis=0)
    
    # Compute similarities
    similarities = cosine_similarity(input_features.reshape(1, -1), features_matrix)
    similar_indices = similarities.argsort()[0][-top_n-1:-1][::-1]
    return df.iloc[similar_indices]

# Streamlit App
def main():
    st.title("E-commerce Product Recommender")
    
    # Load data
    df = load_data()
    features_matrix, tfidf, scaler, encoder = preprocess_data(df)
    
    # Get unique values for dropdowns
    categories = df['Category'].unique().tolist()
    sub_categories = df['Sub_Category'].unique().tolist()
    review_sentiments = df['Review_Sentiment'].unique().tolist()
    genders = df['Customer_Gender'].unique().tolist()
    
    # User input
    with st.form("product_input"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", categories)
            sub_category = st.selectbox("Sub Category", sub_categories)
            review_sentiment = st.selectbox("Review Sentiment", review_sentiments)
        with col2:
            price = st.number_input("Price", min_value=0.0)
            customer_age = st.number_input("Customer Age", min_value=0)
            purchase_history = st.number_input("Purchase History", min_value=0)
            review_rating = st.number_input("Review Rating", min_value=1, max_value=5)
            customer_gender = st.selectbox("Customer Gender", genders)
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        # Process input
        input_text = f"{category} {sub_category} {review_sentiment}"
        input_tfidf = tfidf.transform([input_text])
        input_numerical = scaler.transform([[price, customer_age, purchase_history, review_rating]])
        input_gender = encoder.transform([[customer_gender]]).toarray()
        
        # Combine all input features
        input_features = np.hstack([input_tfidf.toarray(), input_numerical, input_gender])
        
        # Ensure input_features is 2D
        if input_features.ndim > 2:
            input_features = np.squeeze(input_features, axis=0)
        
        # Get recommendations
        recommendations = get_recommendations(input_features, features_matrix, df)
        
        # Display results
        st.subheader("Recommended Products")
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['Product_Name']}**")
            st.write(f"Category: {row['Category']}")
            st.write(f"Sub Category: {row['Sub_Category']}")
            st.write(f"Price: ${row['Price']}")
            st.write(f"Customer Age: {row['Customer_Age']}")
            st.write(f"Customer Gender: {row['Customer_Gender']}")
            st.write(f"Purchase History: {row['Purchase_History']}")
            st.write(f"Review Rating: {row['Review_Rating']}")
            st.write(f"Review Sentiment: {row['Review_Sentiment']}")
            st.divider()

if __name__ == "__main__":
    main()