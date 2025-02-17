import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -------------------------------
# Step 1: Create Personal Test Drive Data
# -------------------------------
def create_test_drive_data():
    """Create personal test drive data manually."""
    data = {
        'Car': ['2025 Volvo EX30', '2025 Polestar 2', '2025 BMW i4', '2025 Hyundai Ioniq 6', '2025 Tesla Model 3'],
        'Price': [47895, 66200, 71875, 55650, 52000],  # in USD
        'Range': [275, 254, 307, 342, 346],  # in miles
        'Battery Capacity': [64, 79, 68.7, 63, 75],  # in kWh
        'Charging Time': [0.5, 0.5, 0.6, 0.9, 0.5],  # in hours
        'Performance Score': [9.8, 9.6, 9.5, 6.0, 8.5],  # out of 10
        'Warranty': [4, 4, 4, 4, 4],  # in years
        'Federal Discount': [0, 7500, 7500, 7500, 8500],  # in USD
        'Comfort Level': [8.6, 8, 8.6, 5, 6.6],  # out of 10
    }
    return pd.DataFrame(data)
# Load data
data = create_test_drive_data()

# Add Final Price column
data['Final Price'] = data['Price'] - data['Federal Discount']

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
# Normalize numerical columns
scaler = MinMaxScaler()
numerical_cols = ['Final Price', 'Range', 'Battery Capacity', 'Charging Time', 
                  'Performance Score', 'Warranty', 'Comfort Level']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# -------------------------------
# Step 3: Detailed Scoring
# -------------------------------
# Define weights for each column
column_weights = {
    'Final Price': 0.3,
    'Range': 0.15,
    'Battery Capacity': 0.15,
    'Charging Time': 0.05,
    'Performance Score': 0.25,
    'Warranty': 0.05,
    'Comfort Level': 0.05,
}

# Calculate weighted scores for each column
data['Score'] = (
    data['Final Price'] * column_weights['Final Price'] +
    data['Range'] * column_weights['Range'] +
    data['Battery Capacity'] * column_weights['Battery Capacity'] +
    data['Charging Time'] * column_weights['Charging Time'] +
    data['Performance Score'] * column_weights['Performance Score'] +
    data['Warranty'] * column_weights['Warranty'] +
    data['Comfort Level'] * column_weights['Comfort Level']
)

# Rank cars based on the total score
data = data.sort_values(by='Score', ascending=False)
top_ranked_car = data.iloc[0]  # Best car

# -------------------------------
# Step 4: Clustering
# -------------------------------
# Select features for clustering
features = data[numerical_cols]
kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
data['Cluster'] = kmeans.labels_

# -------------------------------
# Step 5: Visualize Data
# -------------------------------
def plot_visualizations():
    """Generate visualizations for data."""
    # Scatter plot of Final Price vs. Range
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Final Price', y='Range', hue='Cluster', data=data, palette='viridis')
    plt.title('Final Price vs. Range with Clusters')
    plt.xlabel('Final Price (normalized)')
    plt.ylabel('Range (normalized)')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data[numerical_cols + ['Score']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Show visualizations
plot_visualizations()

# -------------------------------
# Step 6: Streamlit Dashboard
# -------------------------------
def streamlit_dashboard():
    """Build an interactive Streamlit dashboard."""
    st.title("EV Recommendation System")

    # Display dataset
    st.write("### EV Dataset")
    st.dataframe(data)

    # Input weights for criteria
    st.sidebar.write("### Adjust Weights for Criteria")
    column_weights['Final Price'] = st.sidebar.slider('Weight for Final Price', 0.0, 1.0, 0.3)
    column_weights['Range'] = st.sidebar.slider('Weight for Range', 0.0, 1.0, 0.25)
    column_weights['Battery Capacity'] = st.sidebar.slider('Weight for Battery Capacity', 0.0, 1.0, 0.2)
    column_weights['Charging Time'] = st.sidebar.slider('Weight for Charging Time', 0.0, 1.0, 0.05)
    column_weights['Performance Score'] = st.sidebar.slider('Weight for Performance Score', 0.0, 1.0, 0.1)
    column_weights['Warranty'] = st.sidebar.slider('Weight for Warranty', 0.0, 1.0, 0.05)
    column_weights['Comfort Level'] = st.sidebar.slider('Weight for Comfort Level', 0.0, 1.0, 0.05)

    # Recalculate scores
    data['Score'] = (
        data['Final Price'] * column_weights['Final Price'] +
        data['Range'] * column_weights['Range'] +
        data['Battery Capacity'] * column_weights['Battery Capacity'] +
        data['Charging Time'] * column_weights['Charging Time'] +
        data['Performance Score'] * column_weights['Performance Score'] +
        data['Warranty'] * column_weights['Warranty'] +
        data['Comfort Level'] * column_weights['Comfort Level']
    )
    data_sorted = data.sort_values(by='Score', ascending=False)

    # Display top car recommendation
    st.write("### Top Recommendation")
    st.write(f"**{data_sorted.iloc[0]['Car']}** is the best car based on your preferences.")
    st.dataframe(data_sorted.head(1))

    # Display summary and trade-offs
    st.write("### Summary and Trade-offs")
    st.write(f"**{data_sorted.iloc[0]['Car']}** offers a balance of range, performance, and price. "
             "However, consider that it may not excel in warranty or comfort level compared to other options.")

# Uncomment the following to run the Streamlit dashboard
streamlit_dashboard()