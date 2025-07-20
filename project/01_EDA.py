import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plot
from scipy.stats import kruskal
from sklearn import manifold
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
# plt.switch_backend('agg')

data = pd.read_csv("data/01_marketing_campaign.csv", sep='\t')
data.head()
# let's do a basic statistics of the entire dataset to understand it better
data_statistics = data.describe()

# remove NA values
print(f"before removing missing values, we have {len(data)} data points")
df = data.dropna().copy()
print(f"after removing missing values, we have {len(df)} data points")

# let's explore the categorical features
df["Education"].value_counts()
# Education
# Graduation    1116 # undergraduate
# PhD            481
# Master         365
# 2n Cycle       200 # means post graduate
# Basic           54 # undergraduate
undergrad = ['Basic', 'Graduation']
# binary encoding: Undergraduate: 0, Graduate: 1
df['Education'] = df['Education'].apply(lambda x: 0 if x in undergrad else 1)
df["Education"].value_counts()

df["Marital_Status"].value_counts()
# Marital_Status
# Married     857 #partnered
# Together    573 #partnered
# Single      471
# Divorced    232
# Widow        76
# Alone         3
# Absurd        2
# YOLO          2
partnered = ['Married', 'Together']
# binary encoding: 'Partnered': 1, single: 0
df["Marital_Status"] = df["Marital_Status"].apply(lambda x: 1 if x in partnered else 0)
df["Marital_Status"].value_counts()

# create new features
# 1. demographics
current_year = 2025
df['Age'] = current_year - df['Year_Birth']
df = df.drop(columns=['Year_Birth'])
# kids
df['Num_Children'] = df['Kidhome'] + df['Teenhome']
df = df.drop(columns=['Kidhome', 'Teenhome'])
# df['Has_Children'] = df['Num_Children'].apply((lambda x: 1 if x > 0 else 0))
# df['Has_Children'].value_counts()
# Convert Dt_Customer to datetime
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
latest_date = df['Dt_Customer'].max()
df['Customer_Since'] = (latest_date - df["Dt_Customer"]).dt.days
df = df.drop(columns=['Dt_Customer'])

# 2. spending behavior
df['Total_Spent'] = df[list(data.filter(regex='Mnt'))].sum(axis=1)
# create ratio columns
df[['RatioWines', 'RatioFruits', 'RatioMeatProducts',
    'RatioFishProducts', 'RatioSweetProducts', 'RatioGoldProds']] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                                                                        'MntFishProducts', 'MntSweetProducts',
                                                                        'MntGoldProds']].div(df.Total_Spent, axis=0)
# remove the Mnt* columns
df = df.drop(columns=df[list(df.filter(regex='Mnt'))].columns.to_list())
# aggregate total accepted campaigns
df['Total_Accepted_Campaign'] = df[list(df.filter(regex='AcceptedCmp'))].sum(axis=1)
# remove the campaign types since we want to cluster on response volume
df = df.drop(columns=df[list(df.filter(regex='AcceptedCmp'))].columns.to_list())

# 3. purchase behavior
df['Total_Purchase'] = df[list(df.filter(regex='Num'))].sum(axis=1)
df['Total_Web_Engagement'] = df[list(df.filter(regex='Web'))].sum(axis=1)
# remove num purchases since we have total_purchases and web engagements
df = df.drop(columns=df[list(df.filter(regex='Num'))].columns.to_list())

# Univarate data exploration

# 1. Plot distributions of continuous variables
continuous_cols = [
    'Income', 'Recency', 'Age', 'Customer_Since', 'Total_Spent',
    'RatioWines', 'RatioFruits', 'RatioMeatProducts', 'RatioFishProducts',
    'RatioSweetProducts', 'RatioGoldProds', 'Total_Accepted_Campaign',
    'Total_Purchase', 'Total_Web_Engagement'
]
categorical_cols = ['Education', 'Marital_Status', 'Complain', 'Response']

# 2. Plot PDFs/histograms
plt.figure(figsize=(18, 12))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# 3. Boxplots to check for outliers
plt.figure(figsize=(18, 12))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(df[col])
    plt.title(f'Boxplot: {col}')
plt.tight_layout()
plt.show()

# 4. Bar plots for categorical features
plt.figure(figsize=(18, 6))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Categorical Variable: {col}')
    plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Multivariate data exploration

# Correlation heatmap of continuous features
plt.figure(figsize=(16, 12))
corr_matrix = df[continuous_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Correlation Heatmap of Continuous Features')
plt.show()

# Mahalanobis distance outlier removal
features_for_mahalanobis = [
    'Income', 'Total_Spent', 'Age',
    'Total_Purchase', 'Total_Web_Engagement'
]
X = df[features_for_mahalanobis]
X_mean = X.mean().values
X_cov = np.cov(X.values, rowvar=False)
X_cov_inv = np.linalg.inv(X_cov)
# calc distances
mahalanobis_distances = X.apply(lambda row: mahalanobis(row, X_mean, X_cov_inv), axis=1)
df['Mahalanobis_Dist'] = mahalanobis_distances
# Chi-square threshold for 99.5% CI
threshold = chi2.ppf(0.995, len(features_for_mahalanobis))
# Remove outliers
df_maha_clean = df[df['Mahalanobis_Dist'] < np.sqrt(threshold)].copy()
# removed 29 outliers

# Boxplots to check for outliers after removal
plt.figure(figsize=(18, 12))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(df_maha_clean[col])
    plt.title(f'After Removing Outliers Boxplot: {col}')
plt.tight_layout()
plt.show()

# df_maha_clean.to_csv("data/removed_outliers.csv")


# Remove redundant/low variance features
df_clean = df_maha_clean.drop(
    columns=['ID', 'Z_CostContact', 'Z_Revenue',
             'Mahalanobis_Dist'])
# identify the numeric features for low variance feature selection
numeric_feats = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Selecting high variance features
selector = VarianceThreshold(threshold=0.01)
selector.fit(df_clean[numeric_feats])
high_variance_indices = selector.get_support(indices=True)
selected_data = df_clean.iloc[:, high_variance_indices]

# plotting low variance features to see if they should be removed:
low_variance_features = list(set(df_clean.columns.to_list()) - set(selected_data.columns.to_list()))

# Plot histograms
plt.figure(figsize=(18, 12))
for i, col in enumerate(low_variance_features, 1):
    plt.subplot(3, 2, i)
    sns.histplot(df_maha_clean[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Decided only to remove complain, everything else is meaningful.
# the values are valid, the skew is meaningful
df_clean = df_clean.drop(columns=['Complain'])
df_clean.to_csv("data/02_removed_outliers_redundant.csv")

# plot correlation after removal of redundant features
plt.figure(figsize=(16, 12))
corr_matrix = df_clean.corr()
corr_matrix.to_csv("data/numeric_correlation_matrix.tsv")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Correlation Heatmap of All Features')
plt.show()

#JM-Apply scaling to variables conducive to scaling ONLY. For example, avoid scaling indicator variables
# Standardize
scaler = StandardScaler()
columns_to_avoid_scaling = ["Education","Marital_Status","Response"]
columns_to_scale = [col for col in df_clean.columns if col not in columns_to_avoid_scaling]
df_clean2 = df_clean.copy()
df_clean2[columns_to_scale] = scaler.fit_transform(df_clean2[columns_to_scale])

X_scaled = df_clean2.copy()

# Optional: wrap back in a DataFrame for downstream use
#X_scaled_df = pd.DataFrame(X_scaled, columns=df_clean.columns, index=df_clean.index)
X_scaled_df = pd.DataFrame(scaler.fit_transform(df_clean), columns = df_clean.columns, index=df_clean.index)
X_scaled_df.to_csv("data/03_scaled_proprocessed_marketing_campaign.csv")

#visualize tsne data
np.random.seed(6740)
tsne = TSNE(n_components=2, random_state=6740)
tsne_dat = tsne.fit_transform(X_scaled_df)
plt.figure()
plt.title("2-Level TSNE on Scaled Data")
plt.scatter(tsne_dat[:,0], tsne_dat[:,1])
plt.show()
# plt.close()

#2-dimensional PCA visualization
pca = PCA(n_components=3)
pca_dat = pd.DataFrame(pca.fit_transform(X_scaled_df), columns=['PC1', 'PC2', 'PC3'])
plt.figure()
plt.title("2-Level PCA on Scaled Data")
plt.scatter(pca_dat['PC1'], pca_dat['PC2'])
plt.show()

#2-dimensional KMeans on first-two principles components
kmeans = KMeans().fit(X_scaled_df)
pca_kmeans = pca_dat.copy()
pca_kmeans['cluster'] = pd.Categorical(kmeans.labels_)
sns.scatterplot(x="PC1", y="PC2", hue='cluster', data=pca_kmeans)



#view the 3-d projected PCA data
plot.figure()
axes=plot.axes(projection='3d')
axes.scatter3D(pca_dat['PC1'], pca_dat['PC2'], pca_dat['PC3'])
axes.title.set_text('3-Level PCA on Scaled Data')

#3-dimensional KMeans clusters on first-three principles components
plt.figure()
fig = px.scatter_3d(pca_kmeans, x='PC1', y='PC2', z='PC3', color=pca_kmeans['cluster'].astype(str), title="3D KMeans Clusters on PCA Data")
fig.show()

#isomap plot
iso=manifold.Isomap(n_neighbors=5, n_components=2)
iso.fit(X_scaled_df)
manifolded = iso.transform(X_scaled_df)
manifolded_df = pd.DataFrame(manifolded, columns=['Component_1',"Component_2"])
plt.figure()
plt.title("2-level Isomap on Scaled Data")
plt.scatter(manifolded_df['Component_1'], manifolded_df['Component_2'])
plt.show()


#elbow plot on PCA components
sse_kemeans = []
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=6740)
    kmeans.fit(pca_dat)
    sse_kemeans.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1,11), sse_kemeans)
plt.xticks(range(1,11))
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title("Cluster Elbow Plot on Principle Components: KMeans")
plt.show()

#elbow plot on ISOMAP components - turned out to not be very helpful, need #n_clusters not #n_components
# residuals = []
# for d in range(1,11):
#     isomap=manifold.Isomap(n_neighbors=10, n_components=d)
#     iso_dat = isomap.fit_transform(X_scaled_df)
#     dist = pairwise_distances(iso_dat)
#     r = np.corrcoef(pairwise_distances(X_scaled_df).ravel(),pairwise_distances(iso_dat).ravel())[0,1]
#     resid = 1 - r**2
#     residuals.append(resid)
# plt.figure()
# plt.plot(range(1,11), residuals)
# plt.xticks(range(1,11))
# plt.xlabel('Number of Components')
# plt.ylabel('Residual Variance')
# plt.title("Components Elbow Plot on Components: ISOMAP")
# plt.show()

#agglomerative clustering approach, use the value of 5 from KMeans elbow-plot
agc = AgglomerativeClustering(n_clusters=5)
predicted_agc = agc.fit_predict(pca_dat)
pca_agc = pca_dat.copy()
pca_agc['cluster'] = predicted_agc

pca_agc['cluster'].value_counts()
pca_agc['cluster'].value_counts() / pca_agc['cluster'].value_counts().sum()


#2d agglomerative
plt.figure()
sns.scatterplot(x="PC1", y="PC2", hue='cluster', data=pca_agc)

#3d agglomerative clustering on 3-d PCA
plt.figure()
axes=plot.axes(projection='3d')
axes.scatter3D(pca_agc['PC1'], pca_agc['PC2'], pca_agc['PC3'], c=pca_agc['cluster'])
axes.title.set_text('3-Level PCA on Scaled Data')

#plot how many respondents fall into each agglomerative cluster

plt.figure()
agc_counts = pd.Series(predicted_agc).value_counts().sort_index()
# agc_counts.plot(kind='bar', c=pca_agc['cluster'])
plt.bar(agc_counts.index, agc_counts.values)
plt.title("Number of Customers Per Cluster - Agglomerative Clustering")
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')

#now plot 2-d, agglomerative clustering with UNSCALED data
plt.figure()
sns.scatterplot(data=df_clean, x='Total_Spent', y='Income', hue=pca_agc['cluster'],palette='tab10')
plt.title("Agglomerative Clusters: Total Spending vs Income")

# plt.figure()
# sns.scatterplot(data=df_clean, x='Total_Spent', y='Customer_Since', hue=pca_agc['cluster'],palette='tab10')
# plt.title("Agglomerative Clusters: Total Spending vs Income")
#
# plt.figure()
# sns.scatterplot(data=df_clean, x='Total_Spent', y='Total_Web_Engagement', hue=pca_agc['cluster'],palette='tab10')
# plt.title("Agglomerative Clusters: Total Spending vs Income")
#
# plt.figure()
# sns.scatterplot(data=df_clean, x='Customer_Since', y='Recency', hue=pca_agc['cluster'],palette='tab10')
# plt.title("Agglomerative Clusters: Total Spending vs Income")
#
# plt.figure()
# sns.scatterplot(data=df_clean, x='Customer_Since', y='Recency', hue=pca_agc['cluster'],palette='tab10')
# plt.title("Agglomerative Clusters: Total Spending vs Income")

#Apply box-plot of spending by cluster to get a feel for cluster-behavior
plt.figure()
plt.title("Agglomerative Clusters: Income")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Income'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Total Spent")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Total_Spent'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Total Purchases")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Total_Purchase'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Total Web Engagement")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Total_Web_Engagement'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Time as Customer")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Customer_Since'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Ratio of Total Spending on Wines")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['RatioWines'] * 100, palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Ratio of Total Spending on Fruits")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['RatioFruits'] * 100, palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Ratio of Total Spending on Meats")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['RatioMeatProducts'] * 100, palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Ratio of Total Spending on Fish")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['RatioFishProducts'] * 100, palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Ratio of Total Spending on Sweets")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['RatioSweetProducts'] * 100, palette='tab10')

#count plots are better suited for binary indicators, try to distinguish any major differences
plt.figure()
plt.title("Agglomerative Clusters: Number of Total Accepted Campaigns")
sns.countplot(x=df_clean['Total_Accepted_Campaign'], hue=pca_agc['cluster'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Education Status")
sns.countplot(x=df_clean['Education'], hue=pca_agc['cluster'], palette='tab10')

plt.figure()
plt.title("Agglomerative Clusters: Married Status")
sns.countplot(x=df_clean['Marital_Status'], hue=pca_agc['cluster'], palette='tab10')

df_clean['cluster'] = pca_agc['cluster'].values
#counting clusters may not be enough due to their differing size, although perhaps we should give more attention to bigger clusters
def plot_percent_per_cluster_indicator(data, cluster, column, palette='tab10'):
    df = data.copy()
    df[cluster] = df[cluster].astype(str)  # Ensure clusters are categorical for plotting
    count = df.groupby([cluster, column]).size().reset_index(name='count')
    total = count.groupby(cluster)['count'].transform('sum')
    count['percent'] = 100 * count['count'] / total
    plt.figure(figsize=(8, 5))
    sns.barplot(data=count, x=cluster, y="percent", hue=column, palette=palette)
    plt.ylabel("Percent of Respondents")
    plt.title(f"{column} Distribution by Cluster")
    plt.tight_layout()
    # print(df['cluster'].value_counts())
    plt.show()
plot_percent_per_cluster_indicator(df_clean, cluster='cluster', column='Education')
plot_percent_per_cluster_indicator(df_clean, cluster='cluster', column='Marital_Status')
plot_percent_per_cluster_indicator(df_clean, cluster='cluster', column='Response')
plot_percent_per_cluster_indicator(df_clean, cluster='cluster', column='Total_Accepted_Campaign')
# print(df_clean.shape)
# print(pca_agc.shape)

plt.figure()
plt.title("Agglomerative Clusters: Age")
sns.boxplot(x=pca_agc['cluster'], y=df_clean['Age'], palette='tab10')




