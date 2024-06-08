#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

class clustering_methods:
    def k_means_clustering(centroids, dataset):
    #   Description: Perform k means clustering for 2 iterations given as input the dataset and centroids.
    #   Input:
    #       1. centroids - A list of lists containing the initial centroids for each cluster. 
    #       2. dataset - A list of lists denoting points in the space.
    #   Output:
    #       1. results - A dictionary where the key is iteration number and store the cluster assignments in the 
    #           appropriate clusters. Also, update the centroids list after each iteration.

        result = {
            '1': { 'cluster1': [], 'cluster2': [], 'cluster3': [], 'centroids': []},
            '2': { 'cluster1': [], 'cluster2': [], 'cluster3': [], 'centroids': []}
        }
        
        centroid1, centroid2, centroid3 = centroids[0], centroids[1], centroids[2]
        
        for iteration in range(2):
            # your code here
            
            # assign points to clusters - specify just 3 per code above
            cluster1, cluster2, cluster3 = [], [], []
            for point in dataset:
                # calculate euclidian distance from each centroid
                dist1 = np.linalg.norm(np.array(point) - np.array(centroid1))
                dist2 = np.linalg.norm(np.array(point) - np.array(centroid2))
                dist3 = np.linalg.norm(np.array(point) - np.array(centroid3))
                # find nearest centroid
                if dist1 < dist2 and dist1 < dist3:
                    cluster1.append(point)
                elif dist2 < dist1 and dist2 < dist3:
                    cluster2.append(point)
                else:
                    cluster3.append(point)
                    
            # update centroids, list of lists is expected
            centroid1 = list(np.mean(cluster1, axis=0))
            centroid2 = list(np.mean(cluster2, axis=0))
            centroid3 = list(np.mean(cluster3, axis=0))
            
            # store results
            result[str(iteration+1)]['cluster1'] = cluster1
            result[str(iteration+1)]['cluster2'] = cluster2
            result[str(iteration+1)]['cluster3'] = cluster3
            result[str(iteration+1)]['centroids'] = [centroid1, centroid2, centroid3]
            
        return result
    
    def em_clustering(centroids, dataset):
    #   Input: 
    #       1. centroids - A list of lists with each value representing the mean and standard deviation values picked from a gausian distribution.
    #       2. dataset - A list of points randomly picked.
    #   Output:
    #       1. results - Return the updated centroids(updated mean and std values after the EM step) after the first iteration.

        new_centroids = list()

        # your code here

        # convert the dataset to a numpy array for easier calculations
        data = np.array(dataset)

        # expectation step
        expectations = np.zeros((len(data), len(centroids)))
        # define the gaussian probability function in lambda to apply to each data point
        gauss_prob = lambda x, mean, std: (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        # calculate the probability of each data point belonging to each cluster
        for i, (mean, std_dev) in enumerate(centroids):
            expectations[:, i] =  gauss_prob(data, mean, std_dev)
        expectations /= expectations.sum(axis=1, keepdims=True)

        # maximization step
        for i in range(expectations.shape[1]):
            expectation = expectations[:, i]
            new_mean = np.sum(expectation * data) / np.sum(expectation)
            std_dev_final = np.sqrt(np.sum(expectation * (data - new_mean) ** 2) / np.sum(expectation))
            new_centroids.append((new_mean, std_dev_final))

        return new_centroids

if __name__ == "__main__":
    import pprint
    
    cm = clustering_methods()
    
    pp = pprint.PrettyPrinter(depth=4)

    with open('./data/sample_centroids_kmeans.pickle', "rb") as fh:
        centroids = pickle.load(fh)
            
    with open('./data/sample_dataset_kmeans.pickle', "rb") as fh:
        dataset = pickle.load(fh)     

    result = cm.k_means_clustering(centroids,dataset)

    with open('./data/sample_result_kmeans.pickle', "rb") as fh:
        result_expected = pickle.load(fh)

    print(f'The expected result value for the given dataset is:')
    pp.pprint(result_expected)
    print(f'\nYour result value is:')
    pp.pprint(result)
    
    pp = pprint.PrettyPrinter(depth=4)

    with open('./data/sample_centroids_em.pickle', "rb") as fh:
        centroids = pickle.load(fh)
            
    with open('./data/sample_dataset_em.pickle', "rb") as fh:
        dataset = pickle.load(fh)

    new_centroids = cm.em_clustering(centroids,dataset)

    with open('./data/sample_result_em.pickle', "rb") as fh:
        new_centroids_expected = pickle.load(fh)

    print(f'The expected result value for the given dataset is:')
    pp.pprint(new_centroids_expected)
    print(f'\nYour result value is:')
    pp.pprint(new_centroids)

    print(f'\ndataset:')
    pp.pprint(dataset)
    print(f'\noriginal centroids:')
    pp.pprint(centroids)
    