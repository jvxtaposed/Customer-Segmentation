import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

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
X_scaled_df = pd.DataFrame(X_scaled, columns=df_clean.columns, index=df_clean.index)
X_scaled_df.to_csv("data/03_scaled_proprocessed_marketing_campaign.csv")
