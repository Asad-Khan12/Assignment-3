import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# making data frame from csv file
Population_Growth_Data = pd.read_csv('Population_growth.csv', skiprows=4)
Population_Growth_Data = Population_Growth_Data.fillna(Population_Growth_Data.mean(numeric_only=True))
Population_Growth_Data

# check columns about dataset
Population_Growth_Data.columns

# Validate for null values systematically.
Population_Growth_Data.isnull().sum()

# drop missing columns
Population_Growth_Data.drop(['1960'], axis=1, inplace= True)

Population_Growth_Data.shape

Population_Growth_Data.value_counts('Indicator Name')

# Identify relevant variables for analysis.
variables = ['Population growth (annual %)']

# Subset the data
data1 = Population_Growth_Data[(Population_Growth_Data['Indicator Name'].isin(variables))]

data1.head(2)

Population_Growth_Data.columns

# without normalize
#Choose pertinent data fields.
cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code','1970', '1990', '2000', '2010', '2018', '2020']

# Create dataframe, select specific columns
df = Population_Growth_Data[cols].copy()
np.random.seed(0)
X = np.random.randn(100, 2)

# set country name as index
df.set_index('Country Name', inplace=True)


cols_to_nor = ['1970', '1990', '2000', '2010', '2018', '2020']


cols = ['Country Code', 'Indicator Name', 'Indicator Code','1970', '1990', '2000', '2010', '2018', '2020']

# apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[cols_to_nor])

kmean = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmean.labels_

# Show cluster's initial values
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
#     print(df.columns)
    print(df[df['Cluster'] == i][cols])

from sklearn.metrics import silhouette_score

score = silhouette_score(df[cols_to_nor], df['Cluster'])
print(f"Silhouette score: {score}")

# Normalize data
# Choose columns for data normalization process
norm_cols = ['1970', '1990', '2000', '2010', '2018', '2020']

# Standardize data using normalization
scaler = StandardScaler()
df[norm_cols] = scaler.fit_transform(df[norm_cols])

cols = ['Country Code', 'Indicator Name', 'Indicator Code',
        '1970', '1990', '2000', '2010', '2018', '2020']

# apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[norm_cols])

kmean = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmean.labels_

# Display original cluster values using concise language.
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
#     print(df.columns)
    print(df[df['Cluster'] == i][cols])

from sklearn.metrics import silhouette_score

score = silhouette_score(df[norm_cols], df['Cluster'])
print(f"Silhouette score: {score}")

# Visualize clusters and their centers graphically.
import matplotlib.pyplot as plt

# create a figure and axis object with a specified size
fig, ax = plt.subplots(figsize=(10, 8))

# create a scatter plot using two normalized columns of a dataframe
# and color-code the points based on a 'Cluster' column
scatter = ax.scatter(df[norm_cols[0]], df[norm_cols[-1]], c=df['Cluster'])

# obtain the cluster centers from a KMeans clustering model and scale them back to their original values
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# plot the cluster centers as red circles with a specified size and linewidth
plt.scatter(centers[:, 0], centers[:, -1], marker='o', s=200, linewidths=3, color='r')

# add labels to the x-axis and y-axis
plt.title('KMeans Clustering of Climate Data')
plt.xlabel("1970")
plt.ylabel("2021")
# display the plot
plt.show()
plt.savefig('climate_data.png')

# Curve Fitting Analysis

def exponential_growth(x, a, b):
    return a * np.exp(b * x)

x = np.array(range(1980, 2020)) 
us_data = data1[data1["Country Name"] == "United States"]
us_data.head(1)

y = (np.array(us_data[us_data['Indicator Name']== "Population growth (annual %)"]))[0][4:44]

from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
popt, pcov = curve_fit(exponential_growth, x, y)

from scipy import stats
# Specify the prediction's temporal scope concisely.
Time_Forecasting = np.array(range(2022, 2042))

# Utilize model for predictions in concise language.
Projected_values = exponential_growth(Time_Forecasting, *popt)

# Refactor err_ranges for confidence intervals using concise, professional language.
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n = len(ydata)
    dof = max(0, n - len(popt))
    tval = np.abs(stats.t.ppf(alpha / 2, dof))
    ranges = tval * perr
    return ranges

lower_bounds, upper_bounds = err_ranges(exponential_growth, x, y, popt, pcov)

# Generate plot with optimal fit and confidence interval boundaries.
plt.plot(x, y, 'o', label='data')
plt.plot(Time_Forecasting, Projected_values, 'r-', label='fit')
plt.fill_between(Time_Forecasting, Projected_values - upper_bounds, Projected_values + lower_bounds, alpha=0.3);
plt.title('Best Fitting Function Vs Confidence Range')
plt.legend()
plt.show

# "Categorizing National Clusters"

# select the columns for analysis
cols = ['1970', '1971', '1972', '1973',
       '1974', '1975', '1976', '1977',
       '1978', '1990', '1991', '1992', '1993',
       '1994', '1995', '1996', '1997',
       '1998', '1999', '2000', '2001',
       '2002', '2003', '2004', '2005',
       '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013',
       '2014', '2015', '2017',
       '2018', '2019', '2020', '2021']
data_years = Population_Growth_Data[cols]
data_years

# "Normalize data to standardized form."
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_years)

scaled_data

# Conduct KMeans clustering, 5 clusters required
kmeans = KMeans(n_clusters=5, random_state=42).fit(scaled_data)

# add cluster labels to the original data
Population_Growth_Data['Cluster'] = kmeans.labels_

# Output cluster-wise country count using print statement.
print(Population_Growth_Data.groupby('Cluster')['Country Name'].count())

# Choose one nation per cluster succinctly
sample_countries = Population_Growth_Data.groupby('Cluster').apply(lambda x: x.sample(1))

# Compare countries within a cluster succinctly.
cluster_data_0 = Population_Growth_Data[Population_Growth_Data['Cluster'] == 0]
print(cluster_data_0[cols].mean())

# compare countries from different clusters
cluster_data_1 = Population_Growth_Data[Population_Growth_Data['Cluster'] == 1]
print(cluster_data_1[cols].mean())

# Analyze prevailing patterns and developments.
trend_cluster_0 = cluster_data_0[cols].mean()
trend_cluster_1 = cluster_data_1[cols].mean()
print('Trend similarity between cluster 0 and cluster 1:', np.corrcoef(trend_cluster_0, trend_cluster_1)[0,1])

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# create a scatter plot of the first two principal components
pca = PCA(n_components=2)

Data_for_PCA = pca.fit_transform(scaled_data)
colors = ['blue', 'green', 'tab:orange', 'red', 'black']
for i in range(5):
    plt.scatter(Data_for_PCA[kmeans.labels_==i,0], Data_for_PCA[kmeans.labels_==i,1], color=colors[i])
plt.show()
