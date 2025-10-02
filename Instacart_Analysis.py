

# upload data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeClassifier, plot_tree


aisles = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\aisles.csv")
all_order_products = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\all_order_products.csv")
departments = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\departments.csv")
orders = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\orders.csv")
products = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\products.csv")
user_features = pd.read_csv(r"C:\Users\urios\OneDrive\Documents\DataStory6_files\user_features.csv")


## Intro

# Using 2020 customer shopping data from online grocery store, Instacart, the goal of the study is to analyze customer behavior. Through this, I will provide marking, and rentention strategies. The techniques used are as follows:

# PCA & Clustering: Used to reduce dimension and segment customers by their habits.
# RFM Analysis: Classify customers based on Recency, Frequency and Monetary value.
# Decision Tree:

## Data Exploration

# Using the info function, we can see there aren't any nulls. I also observed that there are many variables of interest among the different dataset. As a result, we'll have to combine the datasets through matching keys (for ex. aisle_id, order_id, etc). 
# A concern I have is the large number of column and rows. This will have to be addressed through the PCA technique.

### Summary of Datasets
aisles.info()
all_order_products.info()
departments.info()
orders.info()
products.info()
user_features.info()

### Order Distribution by Day of Week and Hour of Day

# To create the heatmap we need to create a table with just the days of the week (order_dow) and hour of day (order_hour_of_day). 
# In addition, we have to replace the values in both columns for interpretability. The days of the week are currently numbered 0-6 while
# the order hour of day are labled 1-24.

# Create pivot table (rows = day of week, cols = hour of day)
order_heatmap = (
    orders.groupby(["order_dow", "order_hour_of_day"])
    .size()
    .reset_index(name="order_count")
    .pivot(index="order_dow", columns="order_hour_of_day", values="order_count")
)

# Map order_dow numbers to day names
day_map = {
    0: "Sunday", 1: "Monday", 2: "Tuesday",
    3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"
}
order_heatmap.index = order_heatmap.index.map(day_map)

# looking at the heatmap, we can see Sunday and Monday are the most popular days to place orders. Most orders are placed between 9AM and 5PM, 
# with the most activity occurring around 10AM and 2PM. Very few shop late night or early morning before 9AM.

# Plot heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(order_heatmap, cmap="YlGnBu", linewidths=.5, annot=False)

plt.title("Order Distribution by Day of Week and Hour of Day", fontsize=14, pad=15)
hours = list(range(24))
hour_labels = [
    "12AM", "1AM", "2AM", "3AM", "4AM", "5AM",
    "6AM", "7AM", "8AM", "9AM", "10AM", "11AM",
    "12PM", "1PM", "2PM", "3PM", "4PM", "5PM",
    "6PM", "7PM", "8PM", "9PM", "10PM", "11PM"
]

# Show only every 2 hours
plt.xticks(ticks=range(0, 24, 2), labels=[hour_labels[i] for i in range(0, 24, 2)], rotation=0)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)
plt.tight_layout()
plt.show()


### Histogram of Unique Orders per Customer

# The majority of customers place a have placed a few orders (1-10). This tell us there is 
# very few customers that make repeat purchases. Retention may be a possible challenge with this customer dataset.

# Create df of number of unique orders per customer for histogram
orders_per_customer = orders.groupby("user_id")["order_id"].nunique()

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(orders_per_customer, bins=30, edgecolor="black", color="teal")
plt.title("Distribution of Orders per Customer", fontsize=14, pad=15)
plt.xlabel("Number of Orders", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(np.arange(0, orders_per_customer.max()+1, 10))
plt.tight_layout()
plt.show()


### Histogram of Basket Sizes

# The most common order size is between 5-10 items.A few customers place 20+ items, with some rare outlier customers that order 50+ items.
# A potential strategy could be to encourage small-basket customers to encourage larger orders.

# items per order (basket size)
basket_size = all_order_products.groupby("order_id")["product_id"].count()

# quick stats (helps choose axes/bins)
print(basket_size.describe(percentiles=[.1,.25,.5,.75,.9,.95]))

# histogram
max_cap = 60  # cap tail so bars aren’t tiny; adjust as needed
trimmed = basket_size[basket_size <= max_cap]

plt.figure(figsize=(10,6))
plt.hist(trimmed, bins=np.arange(1, max_cap+2), edgecolor="black")
plt.title("Distribution of Items per Order (Basket Size)")
plt.xlabel("Items per Order")
plt.ylabel("Number of Orders")
plt.xticks(np.arange(0, max_cap+1, 5))
plt.grid(axis="y", linestyle="--", alpha=.6)
plt.tight_layout()
plt.show()


### Top 10 and Bottom 10 Products Purchased

# Bananas is the most purchased item. Other fruits and vegetables are included among the top 10 purchased items.
# In the bottom 10, there are a lot of non-grocery items and specialty foods.

# Merge All Order Products with Products to get product_name
df = all_order_products.merge(products, on="product_id", how="left")

# Count product purchases
product_counts = df.groupby("product_name")["order_id"].count().sort_values(ascending=False)

# Top 10 products
top_10 = product_counts.head(10)

# Bottom 10 products
bottom_10 = product_counts.tail(10)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10
top_10.plot(kind='barh', ax=axes[0], color='seagreen')
axes[0].set_title("Top 10 Most Purchased Products")
axes[0].set_xlabel("Number of Orders")
axes[0].invert_yaxis()

# Bottom 10
bottom_10.plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title("Bottom 10 Least Purchased Products")
axes[1].set_xlabel("Number of Orders")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()


## Preprocessing

# Important data preprocessing took place prior to clustering. Some steps included:

# Merging different datasets such as the orders and products datasets
# Reduce dimensionality using PCA
# Scale all features prior to PCA and clustering
# Remove outlier users based on PCA distance
# Filter for top products and active users during RFM scoring


### Standardization

# As mentioned prior, since we have a large number of columns, we'll need to reduce the size via the PCA technique. 
# This will reduce the dimension while keeping the most important features of the data.
# We will first seperate the customer purchase history for the user feature dataset, standardize it and perform PCA.

# Capture and seperate variables of interest 
aisle_cols = user_features.columns[1:-7]   # Skip user_id, exclude day columns
weekday_cols = user_features.columns[-7:]  # Last 7 columns are weekday order counts
feature_cols = list(aisle_cols) + list(weekday_cols)

# Isolate the customer purchase history from the user feature dataset
X = user_features[feature_cols]

# Standardize numeric data prior to clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features[feature_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

### PCA

# After performing PCA and visualizing the variance plot. We see that it takes atleast 120 principal components to capture 
# atleast 95% of the total variance of the dataset. Despite performing PCA, there are still a large number of variables.
# We can use the decision tree model after clustering to improve the interpretability of the results.


# Perform PCA 
pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(X_scaled)

print("Number of PCA components retained:", X_pca.shape[1])


# Visualize how much variance is explained by PCA comp
explained_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(explained_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.axhline(0.95, color='red', linestyle='--', label='95% threshold')
plt.legend()
plt.grid()
plt.show()


## Modeling

### Clustering

# Using the elbow method, I determined that the ideal number of clusters is 6.
# For the clustering technique, I will create 6 clusters. 

# elbow method
inertia = []
K_range = range(2, 15)  

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# Plot elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid()
plt.show()

# Double check clusters were created
kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_pca)

print("Cluster labels assigned to users:", np.unique(cluster_labels))

# Join cluster info to user dataset
user_features['cluster'] = cluster_labels

# Looking at the 6 distinct clusters in 2-dimensional space, something that stands out is the 
# prescence of outliers. THese outliers can skew our results. To address this, I will keep the
# users who are within the 95th percentile to remove any outlier data points, and then perform clustering again.

# Plot clusters
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_pca)

plt.figure(figsize=(8,6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap='Set2', alpha=0.6)
plt.title("Customer Clusters (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()

# Calculate distance to find outliers
distances = pairwise_distances(X_pca, kmeans.cluster_centers_)[np.arange(len(X_pca)), cluster_labels]

# Add distances to DataFrame
user_features['distance_from_centroid'] = distances

# Remove users above 95th percentile distance from their centroid
threshold = np.percentile(distances, 95)
outlier_mask = distances > threshold

print("Number of outliers:", outlier_mask.sum())


### Clustering Again

# In the last section we labeled any users that exceeded the 95th percentile as outliers 
# and stored them into outlier_mask. In this section, we're going to split the overall data
# into inlier and outlier data and then perform clustering again on the inlier set.
# The outlier data will be perserved and stored in a seperate cluster labeled "anomaly cluster".
# This will help enhance the qualtiy of the clustering solution.


# Keep inliers and reset index
X_pca_inliers = X_pca[~outlier_mask]
user_features_inliers = user_features[~outlier_mask].reset_index(drop=True)

# Also isolate outliers
X_pca_outliers = X_pca[outlier_mask]
user_features_outliers = user_features[outlier_mask].reset_index(drop=True)


# Rerun K-means on inliers
# Choose same number of clusters (6, based on elbow method)
kmeans_clean = KMeans(n_clusters=6, random_state=42, n_init='auto')
inlier_labels = kmeans_clean.fit_predict(X_pca_inliers)

# Add cluster labels to inlier DataFrame
user_features_inliers['cluster'] = inlier_labels

# Assign Anomal Cluster
user_features_outliers['cluster'] = 6  # Anomaly cluster


### PCA - Inliers

# Visualizing the inlier clusters, we can see clearer boundaries and less distortion.

# PCA transformation
pca_inliers = PCA(n_components=2)
X_vis_inliers = pca_inliers.fit_transform(X_pca_inliers)

# Create a smaller figure for slide display 
fig, ax = plt.subplots(figsize=(8, 6))  

# Plot each cluster
for cluster in np.unique(inlier_labels):
    mask = inlier_labels == cluster
    ax.scatter(
        X_vis_inliers[mask, 0],
        X_vis_inliers[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.6,
        s=30  # marker size
    )

# Style
ax.set_title("Customer Clusters 0-5", fontsize=12)
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend(title="Cluster", fontsize=8, title_fontsize=9, loc='best')

plt.tight_layout()
plt.show()

### PCA - Outliers

# Here we visualize the anomaly cluster, where most atypical customers are.

# Use same PCA as inlier plot
X_vis_outliers = pca_inliers.transform(X_pca_outliers)

# Resize plot for slides or multi-panel layout
fig, ax = plt.subplots(figsize=(8, 6))  

# Plot outliers
ax.scatter(
    X_vis_outliers[:, 0],
    X_vis_outliers[:, 1],
    color='magenta',
    alpha=0.5,
    label='Anomaly Cluster',
    s=30  # marker size
)

# Add styling
ax.set_title("Created Cluster 6 - which contains outlier points", fontsize=11)
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend(fontsize=8, loc='best')


plt.tight_layout()
plt.show()

### Combining datasets

# Lastly, we can now combine the inlier and outlier datasets. In the next section, we'll be summarizing the
# different cluster profiles.

# Combine outliers + inliers for profiling
user_features_final = pd.concat([user_features_inliers, user_features_outliers], ignore_index=True)


# Create cluster profiles
# Drop non-numeric columns 
feature_cols = user_features_final.select_dtypes(include='number').columns.drop('cluster')


# View cluster sizes
user_features_final['cluster'].value_counts().sort_index()


## Explaining Customer Profiles with Decision Trees

# Using one-vs-all decision trees, we're going to identify key differentiating features including:

# -Showing how specific products, days of the week or purchase behavior seperate clusters.
# -Seperate outlier data and analyze it seperately from inlier data points.

# First - we're going to start by analyzing the inlier clusters (0-5). We're also going to exlude 
# any features that we are not interested in. Then standardized values using z-score scaling.
# Lastly, we're going to use the decision tree models to look at each cluster.

### Decision Tree Inliers 0-1

# Cluster 0 stand outs for their heavy spending on hygiene and body care products such as body lotion, soap and oral hygiene.
# Cluster 1 shows a preference towards fresh fruits, vegetables, and packaged cheese.They are near the top in all RFM categories. 

# Filter out anomalies (cluster 6)
inlier_mask = user_features_final['cluster'] < 6

# Get feature matrix + target
X = user_features_final.loc[inlier_mask].drop(columns=['cluster', 'user_id', 'distance_from_centroid'], errors='ignore')
y = user_features_final.loc[inlier_mask, 'cluster']

#  Scale features 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


cluster_summaries = {
    0: "Heavy spenders on hygiene & body care",
    1: "Value-focused produce buyers"
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, cluster_id in enumerate([0, 1]):
    y_binary = (y == cluster_id).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_scaled, y_binary)

    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=['Other', f'Cluster {cluster_id}'],
        filled=True,
        ax=axes[i],
        fontsize=8
    )

    # Title
    axes[i].set_title(f"Cluster {cluster_id} vs All", fontsize=10)

    # Subtext just below the title (in axes coordinates)
    axes[i].text(
        0.5, -0.12,  
        cluster_summaries[cluster_id],
        ha='center', fontsize=10, transform=axes[i].transAxes
    )

plt.tight_layout()
plt.show()


### Decision Tree Inliers 2-3

# Cluster 2 is characterized by their purchasing patterns around cheese and fresh produce. They also tend to purchase on Thursday.
# Cluster 3 are snack (cookies, chips, frozen items) lovers. Their RFM profile shows a high recency score (77) which shows they’ve purchased an item fairly recently. 

# Cluster summaries
cluster_summaries = {
    2: "Cheese & produce-heavy weekday shoppers",
    3: "Snack & dessert lovers (cookies, chips, frozen items)"
}

# Define figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot decision trees for Cluster 2 and Cluster 3
for i, cluster_id in enumerate([2, 3]):
    y_binary = (y == cluster_id).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_scaled, y_binary)

    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=['Other', f'Cluster {cluster_id}'],
        filled=True,
        ax=axes[i],
        fontsize=8
    )

    # Add title and subtext
    axes[i].set_title(f"Cluster {cluster_id} vs All", fontsize=10)
    axes[i].text(
        0.5, -0.12,  # Adjust vertical position if needed
        cluster_summaries[cluster_id],
        ha='center', fontsize=9, transform=axes[i].transAxes
    )

plt.tight_layout()
plt.show()

### Decision Tree Inliers 4-5

# Cluster 4 customer are veggie-heavy buyers that lean towards fresh and canned produce and packaged vegetables.
# The customers in cluster 5 contain are frequent shoppers of mixed produce and package cheese. They make purchases on Monday. 

# Cluster summaries
cluster_summaries = {
    4: "Veggie-heavy buyers; prefer fresh and canned produce",
    5: "Mixed produce and cheese shoppers; some weekday pattern"
}

# Define figure and axes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

# Plot decision trees for Cluster 4 and Cluster 5
for i, cluster_id in enumerate([4, 5]):
    y_binary = (y == cluster_id).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_scaled, y_binary)

    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=['Other', f'Cluster {cluster_id}'],
        filled=True,
        ax=axes[i],
        fontsize=8
    )

    # Title
    axes[i].set_title(f"Cluster {cluster_id} vs All", fontsize=10)

    # Subtext under title
    axes[i].text(
        0.5, -0.12,  # adjust y for spacing
        cluster_summaries[cluster_id],
        ha='center', fontsize=9, transform=axes[i].transAxes
    )

plt.tight_layout()
plt.show()


### Decision Tree Outlier (Cluster 6)

# Cluster 6 has late week activity, particularly around Monday/Tuesday, which 
# separate them from the rest of cluster groups. These are super-shoppers that buy frequently and in very large quantities.

# Binary target: 1 for cluster 6, 0 for others
X = user_features_final.drop(columns=['cluster', 'user_id', 'distance_from_centroid'], errors='ignore')
y = (user_features_final['cluster'] == 6).astype(int)  # Outlier = 1, all others = 0

# Standardize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Fit decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_scaled, y)

# Plot the tree
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=['Inlier', 'Outlier'],
    filled=True,
    ax=ax,
    fontsize=8
)

# Add title
ax.set_title("Decision Tree: Outliers (Cluster 6) vs All", fontsize=12)

# Add subtext
ax.text(
    0.5, -0.1,
    "Late-week activity, Monday & Tuesday behavior key to outlier detection",
    ha='center', fontsize=9, transform=ax.transAxes
)

plt.tight_layout()
plt.show()

## RFM Analysis 

### Preprocessing

# In this section, we'll prepare our data to create the Recency(R), Frequency (F) and Monetary (M) values. Starting with recency, 
# we'll have to perform steps like organizing the order information in chronical order, filling any nulls with zero,
# and calculate the sum of days since each customer's first oder. Recency is then calculated by substracting each customer's most 
# recent purchase date from the reference day.

#### Recency Calcuation

# Clean and prep the orders table
orders_sorted = orders.sort_values(['user_id', 'order_number'])
orders_sorted['days_since_prior_order'] = orders_sorted['days_since_prior_order'].fillna(0)

# Create cumulative days since first order
orders_sorted['days_since_first_order'] = orders_sorted.groupby('user_id')['days_since_prior_order'].cumsum()

# Calculate reference day (most recent day in dataset = proxy for "today")
reference_day = orders_sorted['days_since_first_order'].max()

# Get recency from latest order per user
recency_df = (
    orders_sorted.groupby('user_id')['days_since_first_order']
    .max()
    .reset_index()
    .rename(columns={'days_since_first_order': 'Recency'})
)
recency_df['Recency'] = reference_day - recency_df['Recency']

#### Frequency and Monetary value

# Here we will carry over all the users that we had calculated a recency value. In addition we'll be merging the orders
# and the all order products dataset to get information neccesarry to get a count of total orders and total products.
# Frequency will be calculated as the number of unique orders.
# Monetary will be calculated as the number of total items.

# Get list of valid users from Recency table
valid_users = recency_df['user_id']

# Frequency & Monetary from merged dataset
user_orders = orders.merge(all_order_products, on='order_id', how='inner')

# Filter user_orders to match the Recency user base
user_orders = user_orders[user_orders['user_id'].isin(valid_users)]


# Calculate Frequency and Monetary
rfm_counts = (
    user_orders.groupby('user_id')
    .agg(
        Frequency=('order_id', pd.Series.nunique),
        Monetary=('product_id', 'count')
    )
    .reset_index()
)


rfm = recency_df.merge(rfm_counts, on='user_id', how='left')

#### Calculate RFM Scores

# Here we create an RFM scoring scheme using a quintile-based bins. Frequency and Monetary are scaled from 1 (low) to 5 (high). Unlike the other metrics, 
# Recency is scaled in inverse style 5 (high) to 1 (low) because the smaller the days between an order the more recent an order was placed.
# In the case of recency, smaller is better.

# Score each R, F, M from 1 (low) to 5 (high)
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combine scores
rfm['RFM_Segment'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
rfm['RFM_Score'] = rfm[['R_score', 'F_score', 'M_score']].astype(int).sum(axis=1)


## RFM Analysis

# In this section I'll identify customer segments and their value by using RFM metrics. The following will take place:

# Calculate Recency (how recently a user purchased), Frequency(how often they purchased), and Monetary (items purchased).
# Combine users RFM scores and look at distribution.
# Group users into segments like Champions, At Risk, Hibernating, etc.
# Look further into user behavior.

## RFM & Cluster Information 

# Merge RFM table with clusters
rfm_cluster = rfm.merge(user_features_final[['user_id', 'cluster']], on='user_id', how='left')

### What food was most popular in each cluster

# This chart looks at purchasing preferences (top 5) across all the clusters. Across all clusters we see 
# the presence of fresh fruits and fresh vegetables. 
# However there are some differences. CLuster 1, 4, and 5 have produce but also contain
# purchases of yogurt and cheese (dairy products). Cluster 3
# contains snack items like chips and pretzels along produce and yogurt.
# Users in cluster 0 show a mix of produce, household goods and beverages. 
# cluster 2 shows high volume of produce purchases. Finally, 
# cluster 6 also shows a volume of produce.


# Data backbone (needed for any aisle plots)
aisle_by_cluster = (
    user_features_final.groupby('cluster')[aisle_cols].sum().T
)

# Single concise panel for all clusters (auto-layout)
cluster_ids = sorted(aisle_by_cluster.columns.tolist())
n = len(cluster_ids)
cols = 3
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows))
axes = axes.flatten()

for i, cluster_id in enumerate(cluster_ids):
    ax = axes[i]
    top_aisles = aisle_by_cluster[cluster_id].nlargest(5)
    top_aisles.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title(f"Top Aisles • Cluster {cluster_id}", fontsize=10)
    ax.set_xlabel("Total Items Purchased", fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.invert_yaxis()

# Remove any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

### RFM Results Breakdown by Cluster


#### Create table for RFM summary table

# 1. Top_Aisle: most purchased aisle per user
top_aisle = (
    all_order_products
    .merge(orders[['order_id', 'user_id']], on='order_id')
    .merge(products[['product_id', 'aisle_id']], on='product_id')
    .groupby('user_id')['aisle_id']
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index(name='Top_Aisle')
)

# 2. Preferred_Day: day of week most often ordered
preferred_day = (
    orders.groupby('user_id')['order_dow']
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index(name='Preferred_Day')
)

# 3. Total_Orders
total_orders = (
    orders.groupby('user_id')['order_id']
    .nunique()
    .reset_index(name='Total_Orders')
)

# 4. Cluster (from user_features_final)
clusters = user_features_final[['user_id', 'cluster']].rename(columns={'cluster': 'Cluster'})

# Merge all into user_features_final
user_features_final = (
    user_features_final
    .merge(top_aisle, on='user_id', how='left')
    .merge(preferred_day, on='user_id', how='left')
    .merge(total_orders, on='user_id', how='left')
    .merge(clusters, on='user_id', how='left')
)

# Final customer summary
user_summary = (
    user_features
    .merge(rfm, on="user_id", how="left")
    .merge(user_features_final[["user_id", "Top_Aisle", "Preferred_Day", "Total_Orders", "Cluster"]], on="user_id", how="left")
)

#### Results

# This table summarizes the normalized RFM scores by cluster. Clusters 1
# and 6 stand out as the most valuable, with near-maximum scores across Recency,
# Frequency, and Monetary, indicating highly active and profitable users. In contrast,
# cluster 2 shows zeros across all metrics, reflecting total inactivity, while 
# cluster 5 also performs poorly with low scores in all dimensions. Clusters 3 
# cluster 4 represent moderate value with average Frequency and Monetary values
# with recency. Lastly, cluster 0 shows modest activity with low frequency and monetary 
# value but average recency. Cluster 1 and 6 represent loyal, high value shoppers while
# clusters 2 and 5 represent disengaged segments.


# means by cluster
rfm_means = (
    user_summary
    .groupby("Cluster")[["Recency","Frequency","Monetary"]]
    .mean()
    .reset_index()
)

# invert Recency (lower is better)
rfm_means["Recency"] = rfm_means["Recency"].max() - rfm_means["Recency"]

# scale to 0–100
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rfm_means[["Recency","Frequency","Monetary"]] = (
    scaler.fit_transform(rfm_means[["Recency","Frequency","Monetary"]]) * 100
)

# melt to tidy table
rfm_export = rfm_means.melt(id_vars="Cluster", var_name="Metric", value_name="Value")

# Show tidy table: Cluster | Metric | Value
rfm_export_table = (
    rfm_export
    .copy()
    .sort_values(["Cluster", "Metric"])
    .round(2)   # round values for readability
)

print(rfm_export_table.to_string(index=False))


## Recommendations

# Personalize marketing by cluster. For ex. highlight produce for health-conscious users and frozen meals for busy households.
# Re-engage at-risk users with targeted email campaigns with reactivation incentives.
# Reward loyal & high value shoppers by having loyalty perks.
# Optimize product placements. Stock top aisles and items and ensure availability. 


## Full Report & Dashboard

# For full report go here: https://github.com/curiostegui/Instacart-customer-segmentation-analysis/blob/main/analysis-report.md
# For Dashboard go here:

###------------------------------------------------------

### Data used for Tableau Dashboard

# Due to size of the dataset and size limit of Tableau, the data had to be filtered significantly. Below is the code
# used to create for Tableau dashboard dataset.

# Merge all_order_products with orders to get user_id
order_products_with_users = all_order_products.merge(
    orders[['order_id', 'user_id']],
    on='order_id',
    how='left'
)

# Now merge with cluster info
order_products_with_clusters = order_products_with_users.merge(
    user_summary[['user_id', 'Cluster']],
    on='user_id',
    how='left'
)

# Filter only clusters 0–6 
order_products_with_clusters = order_products_with_clusters[
    order_products_with_clusters['Cluster'].isin(range(0, 7))
]

# Sample up to 140,000 rows per cluster
max_rows_per_cluster = 140_000
filtered_all_order_products = (
    order_products_with_clusters
    .groupby('Cluster', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), max_rows_per_cluster), random_state=42))
    .reset_index(drop=True)  
)

# Extract filtered user IDs and order IDs
filtered_user_ids = filtered_all_order_products['user_id'].unique()
filtered_order_ids = filtered_all_order_products['order_id'].unique()

# Also save the filtered user dataframe
filtered_user_summary = user_summary[user_summary['user_id'].isin(filtered_user_ids)]

# STEP 2: Filter orders based on those user_ids / order_ids
filtered_orders = orders[orders['order_id'].isin(filtered_order_ids)]

# STEP 3: Filter products, departments, aisles based on used products
filtered_product_ids = filtered_all_order_products['product_id'].unique()
filtered_products = products[products['product_id'].isin(filtered_product_ids)]

filtered_departments = departments[departments['department_id'].isin(filtered_products['department_id'])]
filtered_aisles = aisles[aisles['aisle_id'].isin(filtered_products['aisle_id'])]

# STEP 4: Export to CSV for Tableau

#filtered_user_summary.to_csv("filtered_user_features.csv", index=False)
#filtered_orders.to_csv("filtered_orders.csv", index=False)
#filtered_all_order_products.to_csv("filtered_all_order_products.csv", index=False)
#print(filtered_all_order_products['Cluster'].value_counts())
#filtered_products.to_csv("filtered_products.csv", index=False)
#filtered_departments.to_csv("filtered_departments.csv", index=False)
#filtered_aisles.to_csv("filtered_aisles.csv", index=False)
#rfm_export.to_csv("rfm_radar_export.csv", index=False)

