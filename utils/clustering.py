
# Input: NXD, N points with D features for each point
# Try clustering algorithms from: https://scikit-learn.org/stable/modules/clustering.html
# Represent the original clusters

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def get_cluster_algorithms(X, algorithm_type, params):
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    if algorithm_type == 'MiniBatchKMeans':
        return cluster.MiniBatchKMeans(n_clusters=params['n_clusters'],verbose=1)
    elif algorithm_type == 'AffinityPropagation':
        return cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
    elif algorithm_type == 'MeanShift':
        return cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    elif algorithm_type == 'SpectralClustering':
        return cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
    elif algorithm_type == 'Ward':
        return cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    elif algorithm_type == 'AgglomerativeClustering':
        return cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
    elif algorithm_type == 'DBSCAN':
        return cluster.DBSCAN(eps=params['eps'])
    elif algorithm_type == 'OPTICS':
        return cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])
    elif algorithm_type == 'BIRCH':
        return cluster.Birch(n_clusters=params['n_clusters'])
    elif algorithm_type == 'GaussianMixture':
        return mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
    raise ValueError('Algorithm type not in options')


def clustering(pc_features, algorithm_type='DBSCAN', algorythm_params={}):
    """

    :param pc_features: ndarray [N,d]
    :param algorithm_type:
    'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering', 'Ward', 'AgglomerativeClustering', 'DBSCAN', 'OPTICS', 'BIRCH', 'GaussianMixture'
    :param k: number of clusters
    :return: assignment integer of [N,1]
    """
    params = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}
    params.update(algorythm_params)
    algorithm = get_cluster_algorithms(pc_features, algorithm_type, params)
    algorithm.fit(pc_features)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(pc_features)
    return y_pred


def class_activation_maps(pc_features):
    """

    :param pc_features: [N,c] array of features
    :return: y_pred: [N,1] argmax prediction
    """
    y_pred = np.argmax(pc_features, axis=1)
    return y_pred


if __name__ == "__main__":
    pc_numpy = np.loadtxt('airplane_0038.txt', delimiter=",")
    points = pc_numpy[..., :3]
    W = np.random.randn(3,10)
    pred = clustering(points@W, algorithm_type='Ward')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(pred) + 1))))
    colors = np.append(colors, ["#000000"])    # add black color for outliers (if any)
    colors = colors[pred]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors)
    plt.show()
