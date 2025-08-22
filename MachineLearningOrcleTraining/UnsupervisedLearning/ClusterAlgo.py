import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from apyori import apriori
from sklearn.cluster import KMeans



def main():
    clusterAlgo_Malldata()


def clusterAlgo_Malldata():
    df = pd.read_csv("../../TestData/Mall_Customers.csv")
    # print(df)

    X = df.iloc[:,3:4].values
    # print(X)

    wcss = []
    for i in range(1,6):
        kmeans = KMeans(n_clusters = i, init='k-means++',random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,6),wcss)
    # plt.savefig('wcss_plot.png')

    kmeans = KMeans(n_clusters = 5, init='k-means++',random_state=0)
    y_kmeans = kmeans.fit_transform(X)

    print(y_kmeans)


    # plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    # plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    # plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    # plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__== "__main__":
    main()