import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

df1=pd.read_csv('./question2/Dataset1.csv')
df2=pd.read_csv('./question2/Dataset2.csv')

# elbow method for dataset1
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df1)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('The Elbow Method showing the optimal k for dataset1')
plt.show(block=False)

# elbow method for dataset2
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df2)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('cost')
plt.title('The Elbow Method showing the optimal k for dataset2')
plt.show(block=False)

# Kmeans for dataset1
k=2
kmeans = KMeans(n_clusters=k, random_state=0).fit(df1)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
plt.scatter(df1['X'],df1['Y'], c=kmeans.labels_, cmap='rainbow')
plt.title('kmeans clustering algorithm for dataset1')
plt.show(block=False)

# Kmeans for dataset2
k=4
kmeans = KMeans(n_clusters=k, random_state=0).fit(df2)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
plt.scatter(df2['X'],df2['Y'], c=kmeans.labels_, cmap='rainbow')
plt.title('kmeans clustering algorithm for dataset2')
plt.show(block=False)

# DBSCAN for dataset1
db = DBSCAN(eps = 0.15, min_samples =8).fit(df1)
labels = db.labels_
colors1 = {}
colors1[0] = 'g'
colors1[1] = 'b'
colors1[2] = 'r'
colors1[3] = 'm'
colors1[4] = 'y'
colors1[5] = 'c'
colors1[-1] = 'k'

cvec = [colors1[label] for label in labels]
colors = ['g', 'b', 'r', 'm', 'y', 'c', 'k']

g = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[0])
b = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[1])
r = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[2])
m = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[3])
y = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[4])
c = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[5])
k = plt.scatter(
    df1['X'], df1['Y'], marker='o', color=colors[6])

plt.figure(figsize=(9, 9))
plt.scatter(df1['X'], df1['Y'], c=cvec)
plt.legend((g, b, r, m, y, c, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3 ','Label 4',
           'Label 5', 'Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol=4,
           fontsize=8)
plt.title('DBSCAN clustering algorithm for dataset1')
plt.show(block=False)

# DBSCAN for dataset2
db = DBSCAN(eps = 1.5, min_samples =8).fit(df2)
labels = db.labels_
print(labels)
colors1 = {}
colors1[0] = 'g'
colors1[1] = 'b'
colors1[2] = 'r'
colors1[3] = 'm'
colors1[4] = 'y'
colors1[5] = 'c'
colors1[-1] = 'k'

cvec = [colors1[label] for label in labels]
colors = ['g', 'b', 'r', 'm', 'y', 'c', 'k']

g = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[0])
b = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[1])
r = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[2])
m = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[3])
y = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[4])
c = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[5])
k = plt.scatter(
    df2['X'], df2['Y'], marker='o', color=colors[6])

plt.figure(figsize=(9, 9))
plt.scatter(df2['X'], df2['Y'], c=cvec)
plt.legend((g, b, r, m, y, c, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3 ','Label 4',
           'Label 5', 'Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
plt.title('DBSCAN clustering algorithm for dataset2')
plt.show(block=False)


