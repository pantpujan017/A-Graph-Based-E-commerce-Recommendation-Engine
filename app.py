import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from scipy.sparse.linalg import svds

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("refined_ecommerce_product_data.csv")

def get_collaborative_recommendations(user_item_matrix):
    """Generate collaborative filtering recommendations using SVD"""
    matrix = user_item_matrix.values
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_norm = matrix - user_ratings_mean.reshape(-1, 1)
    
    min_dim = min(matrix_norm.shape)
    n_factors = min(min_dim - 1, 10)
    
    if n_factors <= 0:
        raise ValueError("Not enough data for collaborative filtering")
    
    U, sigma, Vt = svds(matrix_norm, k=n_factors)
    sigma = np.diag(sigma)
    predictions = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    
    return pd.DataFrame(predictions, index=user_item_matrix.index, columns=user_item_matrix.columns)

def get_category_based_recommendations(df, category, sub_category, price_range, rating_filter, n_recommendations=10):
    """Get recommendations based on selected category and sub-category"""
    # First filter by category and sub-category
    filtered_df = df[
        (df['Category'] == category) &
        (df['Sub_Category'] == sub_category) &
        (df['Price'].between(price_range[0], price_range[1])) &
        (df['Review_Rating'] >= rating_filter)
    ]
    
    if len(filtered_df) >= n_recommendations:
        # If we have enough products in the exact category/sub-category
        recommendations = filtered_df.nlargest(n_recommendations, 'Review_Rating')
    else:
        # Get products from exact sub-category first
        recommendations = filtered_df
        
        # Then get products from same category but different sub-category
        remaining_needed = n_recommendations - len(recommendations)
        same_category_diff_sub = df[
            (df['Category'] == category) &
            (df['Sub_Category'] != sub_category) &
            (df['Price'].between(price_range[0], price_range[1])) &
            (df['Review_Rating'] >= rating_filter)
        ].nlargest(remaining_needed, 'Review_Rating')
        
        recommendations = pd.concat([recommendations, same_category_diff_sub])
    
    return recommendations.head(n_recommendations)

def get_related_recommendations(df, category, sub_category, price_range, rating_filter, n_recommendations=10):
    """Get related products recommendations"""
    # Define related categories mapping
    related_categories = {
        'Electronics': ['Computers', 'Mobile Phones', 'Accessories', 'Audio'],
        'Clothing': ['Shoes', 'Accessories', 'Fashion', 'Sports Wear'],
        'Books': ['Stationery', 'Educational', 'Arts'],
        'Furniture': ['Home Decor', 'Lighting', 'Storage'],
        'Sports': ['Fitness Equipment', 'Outdoor Gear', 'Sportswear'],
    }
    
    # Get related categories for the current category
    related_cats = related_categories.get(category, [])
    
    # Filter products from related categories
    related_products = df[
        (df['Category'].isin(related_cats)) &
        (df['Price'].between(price_range[0], price_range[1])) &
        (df['Review_Rating'] >= rating_filter)
    ].nlargest(n_recommendations, 'Review_Rating')
    
    if len(related_products) < n_recommendations:
        # If we don't have enough related products, add top-rated products from same category
        remaining = n_recommendations - len(related_products)
        top_rated_same_category = df[
            (df['Category'] == category) &
            (~df.index.isin(related_products.index)) &
            (df['Price'].between(price_range[0], price_range[1])) &
            (df['Review_Rating'] >= rating_filter)
        ].nlargest(remaining, 'Review_Rating')
        
        related_products = pd.concat([related_products, top_rated_same_category])
    
    return related_products.head(n_recommendations)

def create_user_item_matrix(df):
    return df.pivot_table(
        index='Customer_ID',
        columns='Product_Name',
        values='Review_Rating',
        fill_value=0
    )

def display_recommendations(recommendations, section_title):
    """Display recommendations in a consistent format"""
    st.subheader(section_title)
    if len(recommendations) == 0:
        st.warning("No products found matching your criteria.")
        return
    
    for idx, row in recommendations.iterrows():
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{row['Product_Name']}**")
                st.write(f"Category: {row['Category']} | Sub-Category: {row['Sub_Category']}")
                st.write(f"Price: ${row['Price']:.2f} | Rating: {row['Review_Rating']:.1f}⭐")
            with col2:
                st.write(f"Sentiment: {row['Review_Sentiment']}")
            st.divider()

def preprocess_data(df):
    # Text features
    df['text_features'] = df['Category'] + " " + df['Sub_Category'] + " " + df['Review_Sentiment']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    
    # Numerical features
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(df[['Price', 'Customer_Age', 'Purchase_History', 'Review_Rating']])
    
    # Categorical features
    encoder = OneHotEncoder()
    gender_encoded = encoder.fit_transform(df[['Customer_Gender']]).toarray()
    
    # Combined features
    combined_features = np.hstack([tfidf_matrix.toarray(), numerical_features, gender_encoded])
    
    try:
        user_item_matrix = create_user_item_matrix(df)
        cf_predictions = get_collaborative_recommendations(user_item_matrix)
        st.success("Collaborative filtering initialized successfully!")
    except Exception as e:
        st.warning(f"Collaborative filtering initialization failed: {str(e)}")
        cf_predictions = pd.DataFrame()
    
    return combined_features, tfidf, scaler, encoder, cf_predictions

def main():
    st.title("E-commerce Product Recommender")
    
    df = load_data()
    features_matrix, tfidf, scaler, encoder, cf_predictions = preprocess_data(df)
    
    categories = df['Category'].unique().tolist()
    sub_categories = df['Sub_Category'].unique().tolist()
    
    with st.form("product_input"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", categories)
            sub_category = st.selectbox("Sub Category", sub_categories)
            customer_id = st.number_input("Customer ID", min_value=1)
        with col2:
            price_range = st.slider("Price Range", 
                                  min_value=float(df['Price'].min()),
                                  max_value=float(df['Price'].max()),
                                  value=(float(df['Price'].min()), float(df['Price'].max())))
            rating_filter = st.slider("Minimum Rating", min_value=1, max_value=5, value=3)
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        # 1. Category-based recommendations
        category_recs = get_category_based_recommendations(
            df, category, sub_category, price_range, rating_filter
        )
        display_recommendations(
            category_recs,
            f"Top Recommendations in {category} - {sub_category}"
        )
        
        # 2. Collaborative filtering recommendations
        if not cf_predictions.empty and customer_id in cf_predictions.index:
            # Get user's predictions
            user_predictions = cf_predictions.loc[customer_id].sort_values(ascending=False)
            
            # Filter predicted products by category and other criteria
            predicted_products = df[
                df['Product_Name'].isin(user_predictions.index) &
                (df['Category'] == category) &
                (df['Price'].between(price_range[0], price_range[1])) &
                (df['Review_Rating'] >= rating_filter)
            ]
            
            if len(predicted_products) < 10:
                # Add top-rated products from same category if needed
                remaining = 10 - len(predicted_products)
                additional_products = df[
                    (df['Category'] == category) &
                    (~df['Product_Name'].isin(predicted_products['Product_Name'])) &
                    (df['Price'].between(price_range[0], price_range[1])) &
                    (df['Review_Rating'] >= rating_filter)
                ].nlargest(remaining, 'Review_Rating')
                
                predicted_products = pd.concat([predicted_products, additional_products])
            
            display_recommendations(
                predicted_products.head(10),
                "Recommended Based on Similar Users (Top 10)"
            )
        
        # 3. Related products recommendations
        related_recs = get_related_recommendations(
            df, category, sub_category, price_range, rating_filter
        )
        display_recommendations(
            related_recs,
            "Products You Might Be Interested In"
        )

if __name__ == "__main__":
    main()