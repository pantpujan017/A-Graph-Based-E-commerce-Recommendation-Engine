import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            # Create constraint for product_id
            session.run("CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE")
            # Create constraint for customer_id
            session.run("CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE")

    def import_data(self, df):
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create product nodes
            for _, row in df.iterrows():
                cypher_query = """
                CREATE (p:Product {
                    product_id: $product_id,
                    product_name: $product_name,
                    category: $category,
                    sub_category: $sub_category,
                    price: toFloat($price),
                    review_rating: toFloat($review_rating),
                    review_sentiment: $review_sentiment
                })
                """
                # Convert row to dict for Neo4j parameters
                product_data = {
                    'product_id': int(row.name),  # Using index as product_id
                    'product_name': row['Product_Name'],
                    'category': row['Category'],
                    'sub_category': row['Sub_Category'],
                    'price': float(row['Price']),
                    'review_rating': float(row['Review_Rating']),
                    'review_sentiment': row['Review_Sentiment']
                }
                session.run(cypher_query, product_data)
            
            # Create customer nodes and purchase relationships
            unique_customers = df['Customer_ID'].unique()
            for customer_id in unique_customers:
                # Create customer node
                customer_rows = df[df['Customer_ID'] == customer_id]
                first_row = customer_rows.iloc[0]
                
                cypher_query = """
                CREATE (c:Customer {
                    customer_id: $customer_id,
                    age: toInteger($age),
                    gender: $gender,
                    purchase_history: toInteger($purchase_history)
                })
                """
                customer_data = {
                    'customer_id': int(customer_id),
                    'age': int(first_row['Customer_Age']),
                    'gender': first_row['Customer_Gender'],
                    'purchase_history': int(first_row['Purchase_History'])
                }
                session.run(cypher_query, customer_data)
                
                # Create PURCHASED relationships with ratings
                for _, row in customer_rows.iterrows():
                    cypher_query = """
                    MATCH (c:Customer {customer_id: $customer_id}), 
                          (p:Product {product_id: $product_id})
                    CREATE (c)-[:PURCHASED {rating: toFloat($rating)}]->(p)
                    """
                    purchase_data = {
                        'customer_id': int(customer_id),
                        'product_id': int(row.name),
                        'rating': float(row['Review_Rating'])
                    }
                    session.run(cypher_query, purchase_data)
            
            # Create SIMILAR relationships between products based on attributes
            session.run("""
                MATCH (p1:Product), (p2:Product)
                WHERE p1.product_id < p2.product_id
                WITH p1, p2,
                CASE WHEN p1.category = p2.category THEN 2 ELSE 0 END +
                CASE WHEN p1.sub_category = p2.sub_category THEN 3 ELSE 0 END +
                CASE WHEN abs(p1.price - p2.price) < 50 THEN 1 ELSE 0 END +
                CASE WHEN abs(p1.review_rating - p2.review_rating) < 1 THEN 1 ELSE 0 END +
                CASE WHEN p1.review_sentiment = p2.review_sentiment THEN 1 ELSE 0 END as similarity
                WHERE similarity > 2
                CREATE (p1)-[:SIMILAR {score: similarity}]->(p2)
            """)

    def get_category_based_recommendations(self, category, sub_category, price_min, price_max, rating_filter, n_recommendations=10):
        with self.driver.session() as session:
            cypher_query = """
            MATCH (p:Product)
            WHERE p.category = $category 
            AND p.sub_category = $sub_category
            AND p.price >= $price_min AND p.price <= $price_max
            AND p.review_rating >= $rating_filter
            RETURN p.product_id as product_id, p.product_name as product_name, 
                   p.category as category, p.sub_category as sub_category,
                   p.price as price, p.review_rating as review_rating,
                   p.review_sentiment as review_sentiment
            ORDER BY p.review_rating DESC
            LIMIT $limit
            """
            
            result = session.run(cypher_query, {
                'category': category,
                'sub_category': sub_category,
                'price_min': price_min,
                'price_max': price_max,
                'rating_filter': rating_filter,
                'limit': n_recommendations
            })
            
            records = [dict(record) for record in result]
            
            # If not enough records, get products from same category but different sub-category
            if len(records) < n_recommendations:
                remaining = n_recommendations - len(records)
                cypher_query_remaining = """
                MATCH (p:Product)
                WHERE p.category = $category 
                AND p.sub_category <> $sub_category
                AND p.price >= $price_min AND p.price <= $price_max
                AND p.review_rating >= $rating_filter
                RETURN p.product_id as product_id, p.product_name as product_name, 
                       p.category as category, p.sub_category as sub_category,
                       p.price as price, p.review_rating as review_rating,
                       p.review_sentiment as review_sentiment
                ORDER BY p.review_rating DESC
                LIMIT $limit
                """
                
                result_remaining = session.run(cypher_query_remaining, {
                    'category': category,
                    'sub_category': sub_category,
                    'price_min': price_min,
                    'price_max': price_max,
                    'rating_filter': rating_filter,
                    'limit': remaining
                })
                
                records.extend([dict(record) for record in result_remaining])
                
            return records

    def get_collaborative_recommendations(self, customer_id, category, price_min, price_max, rating_filter, n_recommendations=10):
        with self.driver.session() as session:
            # First, check if the customer exists
            customer_check = session.run(
                "MATCH (c:Customer {customer_id: $customer_id}) RETURN count(c) as count",
                customer_id=customer_id
            ).single()
            
            if customer_check['count'] == 0:
                # Customer doesn't exist, return empty list
                return []
            
            # Get collaborative filtering recommendations
            cypher_query = """
            MATCH (c:Customer {customer_id: $customer_id})-[p:PURCHASED]->(product)<-[p2:PURCHASED]-(other_customer)-[p3:PURCHASED]->(recommended)
            WHERE NOT (c)-[:PURCHASED]->(recommended)
            AND recommended.category = $category
            AND recommended.price >= $price_min AND recommended.price <= $price_max
            AND recommended.review_rating >= $rating_filter
            WITH recommended, avg(p3.rating) as score
            RETURN recommended.product_id as product_id, recommended.product_name as product_name,
                recommended.category as category, recommended.sub_category as sub_category,
                recommended.price as price, recommended.review_rating as review_rating,
                recommended.review_sentiment as review_sentiment,
                score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            result = session.run(cypher_query, {
                'customer_id': customer_id,
                'category': category,
                'price_min': price_min,
                'price_max': price_max,
                'rating_filter': rating_filter,
                'limit': n_recommendations
            })
            
            records = [dict(record) for record in result]
            
            # If not enough collaborative recommendations, add top-rated products from same category
            if len(records) < n_recommendations:
                remaining = n_recommendations - len(records)
                already_recommended_ids = [r['product_id'] for r in records]
                
                cypher_query_top_rated = """
                MATCH (p:Product)
                WHERE p.category = $category
                AND p.price >= $price_min AND p.price <= $price_max
                AND p.review_rating >= $rating_filter
                AND NOT p.product_id IN $already_recommended
                RETURN p.product_id as product_id, p.product_name as product_name, 
                    p.category as category, p.sub_category as sub_category,
                    p.price as price, p.review_rating as review_rating,
                    p.review_sentiment as review_sentiment
                ORDER BY p.review_rating DESC
                LIMIT $limit
                """
                
                result_top_rated = session.run(cypher_query_top_rated, {
                    'category': category,
                    'price_min': price_min,
                    'price_max': price_max,
                    'rating_filter': rating_filter,
                    'already_recommended': already_recommended_ids,
                    'limit': remaining
                })
                
                records.extend([dict(record) for record in result_top_rated])
                
            return records

    def get_related_recommendations(self, category, sub_category, price_min, price_max, rating_filter, n_recommendations=10):
        # Define related categories mapping
        related_categories = {
            'Electronics': ['Computers', 'Mobile Phones', 'Accessories', 'Audio'],
            'Clothing': ['Shoes', 'Accessories', 'Fashion', 'Sports Wear'],
            'Books': ['Stationery', 'Educational', 'Arts'],
            'Furniture': ['Home Decor', 'Lighting', 'Storage'],
            'Sports': ['Fitness Equipment', 'Outdoor Gear', 'Sportswear'],
        }
        
        related_cats = related_categories.get(category, [])
        
        with self.driver.session() as session:
            cypher_query = """
            MATCH (p:Product)
            WHERE p.category IN $related_categories
            AND p.price >= $price_min AND p.price <= $price_max
            AND p.review_rating >= $rating_filter
            RETURN p.product_id as product_id, p.product_name as product_name, 
                   p.category as category, p.sub_category as sub_category,
                   p.price as price, p.review_rating as review_rating,
                   p.review_sentiment as review_sentiment
            ORDER BY p.review_rating DESC
            LIMIT $limit
            """
            
            result = session.run(cypher_query, {
                'related_categories': related_cats,
                'price_min': price_min,
                'price_max': price_max,
                'rating_filter': rating_filter,
                'limit': n_recommendations
            })
            
            records = [dict(record) for record in result]
            
            # If not enough related records, add top-rated products from same category
            if len(records) < n_recommendations:
                remaining = n_recommendations - len(records)
                already_recommended_ids = [r['product_id'] for r in records]
                
                cypher_query_same_category = """
                MATCH (p:Product)
                WHERE p.category = $category
                AND NOT p.product_id IN $already_recommended
                AND p.price >= $price_min AND p.price <= $price_max
                AND p.review_rating >= $rating_filter
                RETURN p.product_id as product_id, p.product_name as product_name, 
                       p.category as category, p.sub_category as sub_category,
                       p.price as price, p.review_rating as review_rating,
                       p.review_sentiment as review_sentiment
                ORDER BY p.review_rating DESC
                LIMIT $limit
                """
                
                result_same_category = session.run(cypher_query_same_category, {
                    'category': category,
                    'already_recommended': already_recommended_ids,
                    'price_min': price_min,
                    'price_max': price_max,
                    'rating_filter': rating_filter,
                    'limit': remaining
                })
                
                records.extend([dict(record) for record in result_same_category])
                
            return records

# Function to convert Neo4j result to DataFrame
def result_to_dataframe(result_list):
    if not result_list:
        return pd.DataFrame()
    
    return pd.DataFrame(result_list)

def display_recommendations(recommendations_df, section_title):
    """Display recommendations in a consistent format"""
    st.subheader(section_title)
    if len(recommendations_df) == 0:
        st.warning("No products found matching your criteria.")
        return
    
    for idx, row in recommendations_df.iterrows():
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{row['product_name']}**")
                st.write(f"Category: {row['category']} | Sub-Category: {row['sub_category']}")
                st.write(f"Price: ${row['price']:.2f} | Rating: {row['review_rating']:.1f}⭐")
            with col2:
                st.write(f"Sentiment: {row['review_sentiment']}")
            st.divider()

def main():
    st.title("E-commerce Product Recommender with Neo4j")
    
    # Neo4j connection settings
    with st.sidebar:
        st.subheader("Neo4j Connection Settings")
        uri = st.text_input("Neo4j URI", "neo4j://localhost:7687")
        user = st.text_input("Username", "neo4j")
        password = st.text_input("Password", type="password")
        
        # Initialize Neo4j connection
        if st.button("Connect to Neo4j"):
            try:
                neo4j_conn = Neo4jConnection(uri, user, password)
                st.session_state['neo4j_conn'] = neo4j_conn
                st.success("Connected to Neo4j successfully!")
            except Exception as e:
                st.error(f"Error connecting to Neo4j: {str(e)}")
    
    # File upload section
    st.subheader("Import Product Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.write("Preview of the data:")
            st.dataframe(df.head())
            
            # Import button
            if st.button("Import to Neo4j"):
                if 'neo4j_conn' not in st.session_state:
                    st.error("Please connect to Neo4j first!")
                else:
                    try:
                        with st.spinner("Setting up database..."):
                            st.session_state['neo4j_conn'].create_constraints()
                        with st.spinner("Importing data..."):
                            st.session_state['neo4j_conn'].import_data(df)
                        st.success("Data imported successfully!")
                        st.session_state['data_imported'] = True
                    except Exception as e:
                        st.error(f"Error importing data to Neo4j: {str(e)}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Recommendation section (only if data is imported)
    if st.session_state.get('data_imported', False):
        st.subheader("Get Recommendations")
        
        with st.form("product_input"):
            # Get unique categories and subcategories from the database
            categories = ['Electronics', 'Clothing', 'Books', 'Furniture', 'Sports']  # Placeholder - ideally fetch from Neo4j
            sub_categories = ['Mobile Phones', 'Shoes', 'Fiction', 'Tables', 'Fitness Equipment']  # Placeholder
            
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category", categories)
                sub_category = st.selectbox("Sub Category", sub_categories)
                customer_id = st.number_input("Customer ID", min_value=1, value=1)
            with col2:
                price_range = st.slider("Price Range", 
                                        min_value=float(0),
                                        max_value=float(2000),
                                        value=(float(0), float(2000)))
                rating_filter = st.slider("Minimum Rating", min_value=1, max_value=5, value=3)
            
            submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            if 'neo4j_conn' not in st.session_state:
                st.error("Please connect to Neo4j first!")
            else:
                # 1. Category-based recommendations
                category_recs = st.session_state['neo4j_conn'].get_category_based_recommendations(
                    category, sub_category, price_range[0], price_range[1], rating_filter
                )
                category_recs_df = result_to_dataframe(category_recs)
                display_recommendations(
                    category_recs_df,
                    f"Top Recommendations in {category} - {sub_category}"
                )
                
                # 2. Collaborative filtering recommendations
                cf_recs = st.session_state['neo4j_conn'].get_collaborative_recommendations(
                    customer_id, category, price_range[0], price_range[1], rating_filter
                )
                cf_recs_df = result_to_dataframe(cf_recs)
                if not cf_recs_df.empty:
                    display_recommendations(
                        cf_recs_df,
                        "Recommended Based on Similar Users (Top 10)"
                    )
                else:
                    st.warning("No collaborative recommendations found for this customer.")
                
                # 3. Related products recommendations
                related_recs = st.session_state['neo4j_conn'].get_related_recommendations(
                    category, sub_category, price_range[0], price_range[1], rating_filter
                )
                related_recs_df = result_to_dataframe(related_recs)
                display_recommendations(
                    related_recs_df,
                    "Products You Might Be Interested In"
                )

if __name__ == "__main__":
    # Initialize session state
    if 'data_imported' not in st.session_state:
        st.session_state['data_imported'] = False
    
    main()