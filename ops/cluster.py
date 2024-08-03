"""
Clustering Algorithms and Utilities
This module provides a collection of clustering algorithms and related utilities
(relating to step 5 -- clustering). It includes functions for:

1. Affinity-based Clustering: Implementation of the Leiden algorithm for community detection.
2. Hierarchical Clustering: Functions for hierarchical clustering and dendrogram manipulation.
3. Density-based Clustering: DBSCAN and HDBSCAN implementations.
4. Spectral Clustering: Implementation of spectral clustering algorithm.
5. Consensus Clustering: Methods for combining multiple clustering results.
6. Cluster Refinement: Utilities for merging small clusters and improving clustering results.

"""

import leidenalg
import graphtools
import hdbscan
import sklearn
from sklearn import cluster
from igraph import Graph
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy


class AffinityLeiden(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Affinity Leiden clustering algorithm.

    This class implements a clustering algorithm using an affinity matrix and the Leiden algorithm for community detection.
    The algorithm constructs an affinity matrix from input data, then applies the Leiden algorithm to detect clusters.

    Attributes:
        knn (int): Number of nearest neighbors for constructing the graph.
        knn_max (int, optional): Maximum number of nearest neighbors to consider.
        knn_dist (str): Distance metric for nearest neighbor search (default is "euclidean").
        n_pca (int): Number of principal components to use for dimensionality reduction.
        decay (float): Decay parameter for the graph construction.
        n_landmark (int): Number of landmarks for the graph construction.
        resolution_parameter (float): Resolution parameter for the Leiden algorithm.
        n_jobs (int): Number of parallel jobs to run (default is 1).
        verbose (bool): If True, print progress information.
        random_state (int, optional): Seed for the random number generator.
    """

    def __init__(
        self,
        knn=5,
        knn_max=None,
        knn_dist="euclidean",
        n_pca=100,
        decay=40,
        n_landmark=2000,
        resolution_parameter=1,
        n_jobs=1,
        verbose=True,
        random_state=None,
    ):
        """
        Initialize the AffinityLeiden clustering algorithm.

        Parameters:
            knn (int): Number of nearest neighbors for graph construction.
            knn_max (int, optional): Maximum number of nearest neighbors.
            knn_dist (str): Distance metric for nearest neighbor search.
            n_pca (int): Number of principal components.
            decay (float): Decay parameter for the graph.
            n_landmark (int): Number of landmarks.
            resolution_parameter (float): Resolution parameter for Leiden algorithm.
            n_jobs (int): Number of parallel jobs.
            verbose (bool): If True, print progress information.
            random_state (int, optional): Random seed for reproducibility.
        """
        self.knn = knn
        self.knn_max = knn_max
        self.knn_dist = knn_dist
        self.decay = decay
        self.n_pca = n_pca
        self.n_jobs = n_jobs
        self.n_landmark = n_landmark
        self.resolution_parameter = resolution_parameter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the AffinityLeiden model to the data.

        Parameters:
            X (array-like or list): Input data.
            y (array-like, optional): Target values (not used).

        Returns:
            self: The fitted instance of the AffinityLeiden class.
        """
        if isinstance(X, list):
            X = np.array(X)

        if X.ndim < 2:
            raise ValueError("Cannot fit 1D array.")

        if X.shape[0] == 1:
            raise ValueError("Input contains only 1 sample.")

        self.n_features_in_ = X.shape[1]

        # Create the graph from input data
        graph = graphtools.Graph(
            X,
            n_pca=self.n_pca,
            n_landmark=self.n_landmark,
            distance=self.knn_dist,
            knn=self.knn,
            knn_max=self.knn_max,
            decay=self.decay,
            thresh=1e-4,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        # Compute the affinity matrix
        self.affinity_matrix_ = graph.diff_op.toarray()

        # Create a weighted adjacency graph for Leiden algorithm
        affinity_igraph = Graph().Weighted_Adjacency(
            matrix=self.affinity_matrix_.tolist(), mode="undirected"
        )

        # Apply the Leiden algorithm for community detection
        partition = leidenalg.find_partition(
            affinity_igraph,
            partition_type=leidenalg.RBConfigurationVertexPartition,
            weights=affinity_igraph.es["weight"],
            n_iterations=-1,
            seed=self.random_state,
            resolution_parameter=self.resolution_parameter,
        )

        self.labels_ = np.array(partition.membership)
        self.q_ = partition.q
        return self

    @property
    def singularities(self):
        """
        Number of singleton clusters (clusters with only one member).

        Returns:
            int: Number of singleton clusters.
        """
        self._singularities = (
            np.unique(self.labels_, return_counts=True)[1] == 1
        ).sum()
        return self._singularities

    @property
    def n_clusters(self):
        """
        Number of unique clusters identified by the algorithm.

        Returns:
            int: Number of clusters.
        """
        self._n_labels = len(np.unique(self.labels_))
        return self._n_labels

    @property
    def mean_cluster_size(self):
        """
        Mean size of clusters.

        Returns:
            float: Mean size of the clusters.
        """
        self._mean_cluster_size = np.unique(self.labels_, return_counts=True)[1].mean()
        return self._mean_cluster_size

    def set_index(self, index):
        """
        Set custom index for the data.

        Parameters:
            index (array-like): Custom index.

        Returns:
            self: The instance with the custom index set.
        """
        assert len(index) == len(self.labels_)
        self.index = index
        return self

    def adjusted_mutual_info_score(self, s):
        """
        Compute the adjusted mutual information score between true labels and provided labels.

        Parameters:
            s (DataFrame): DataFrame with true labels.

        Returns:
            float: Adjusted mutual information score.
        """
        return sklearn.metrics.adjusted_mutual_info_score(
            self.labels_, s[self.index].values
        )

    def adjusted_rand_score(self, s):
        """
        Compute the adjusted Rand index between true labels and provided labels.

        Parameters:
            s (DataFrame): DataFrame with true labels.

        Returns:
            float: Adjusted Rand index.
        """
        return sklearn.metrics.adjusted_rand_score(self.labels_, s[self.index].values)

def fcluster_to_df(Z, n_clusters, index):
    """
    Convert hierarchical clustering linkage matrix to a DataFrame with cluster labels.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        n_clusters (int): The number of clusters to form.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    clusters = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def leiden_to_df(adjacency, resolution, index, seed=42):
    """
    Perform Leiden clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        resolution (float): Resolution parameter for Leiden algorithm.
        index (Index): Index for the DataFrame.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    igraph = Graph().Weighted_Adjacency(matrix=adjacency, mode="undirected")

    partition = leidenalg.find_partition(
        igraph,
        partition_type=leidenalg.RBConfigurationVertexPartition,
        weights=igraph.es["weight"],
        n_iterations=-1,
        seed=seed,
        resolution_parameter=resolution,
    )

    return pd.DataFrame(partition.membership, index=index, columns=['cluster'])

def dbscan_to_df(adjacency, eps, min_samples, index):
    """
    Perform DBSCAN clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    distance = 1 - (adjacency / adjacency.max())
    clusters = cluster.dbscan(distance, metric='precomputed', eps=eps, min_samples=min_samples)[1]
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def hdbscan_to_df(adjacency, min_cluster_size, min_samples, index):
    """
    Perform HDBSCAN clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        min_cluster_size (int): The minimum size of a cluster.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        index (Index): Index for the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    distance = 1 - (adjacency / adjacency.max())
    clusters = hdbscan.hdbscan(distance, metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)[0]
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def spectral_to_df(adjacency, n_clusters, index, seed=42):
    """
    Perform spectral clustering on an affinity matrix and convert to a DataFrame.

    Parameters:
        adjacency (ndarray): Affinity matrix for clustering.
        n_clusters (int): The number of clusters to form.
        index (Index): Index for the DataFrame.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with cluster labels.
    """
    if not np.array_equal(adjacency, adjacency.T):
        adjacency = (adjacency + adjacency.T) / 2

    clusters = cluster.spectral_clustering(adjacency, n_clusters=n_clusters, random_state=seed)
    return pd.DataFrame(clusters, index=index, columns=['cluster'])

def consensus_matrix(dfs, column="cluster", weights=None, combine_nt=False, nt_threshold=1):
    """
    Compute the consensus matrix from multiple clustering DataFrames.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames with cluster labels.
        column (str): Column name for cluster labels in each DataFrame.
        weights (list of float, optional): Weights for each DataFrame. Must sum to 1.
        combine_nt (bool, optional): Whether to combine non-target clusters.
        nt_threshold (int, optional): Threshold for combining non-target clusters.

    Returns:
        pd.DataFrame: Consensus matrix.
    """
    # Check for identical indices across DataFrames
    [pd.testing.assert_index_equal(dfs[0].index, df.index) for df in dfs[1:]]

    if weights is not None:
        if sum(weights) != 1:
            raise ValueError("`weights` must sum to 1")
        if len(weights) != len(dfs):
            raise ValueError("`weights` must have the same length as `dfs`")
    else:
        weights = [1 / len(dfs)] * len(dfs)

    C = np.zeros((dfs[0].pipe(len),) * 2)

    for df, w in zip(dfs, weights):
        if combine_nt:
            df = combine_nt_clusters(df, col=column, nt_threshold=nt_threshold)
        C += np.equal(*np.meshgrid(*(df[column].values,) * 2)) * w

    df_consensus = pd.DataFrame(C, columns=dfs[0].index, index=dfs[0].index)
    return df_consensus

def subsampled_consensus_matrix(dfs, column="cluster", combine_nt=False, nt_threshold=1, tqdm=False):
    """
    Compute the subsampled consensus matrix from multiple clustering DataFrames.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames with cluster labels.
        column (str): Column name for cluster labels in each DataFrame.
        combine_nt (bool, optional): Whether to combine non-target clusters.
        nt_threshold (int, optional): Threshold for combining non-target clusters.
        tqdm (bool, optional): Whether to display a progress bar.

    Returns:
        pd.DataFrame: Subsampled consensus matrix.
    """
    full_index = set()
    for df in dfs:
        full_index |= set(df.index)

    full_index = pd.MultiIndex.from_tuples(sorted(full_index), names=["gene_symbol", "gene_id"])

    C = np.zeros((len(full_index),) * 2)
    C_count = np.zeros_like(C)

    if tqdm:
        from tqdm.auto import tqdm
        dfs = tqdm(dfs)

    for df in dfs:
        df = df.sort_index()
        if combine_nt:
            df = combine_nt_clusters(df, col=column, nt_threshold=nt_threshold)
        selected = np.argwhere(full_index.isin(df.index))
        C_count[selected, selected.T] += 1
        C[selected, selected.T] += np.equal(*np.meshgrid(*(df[column].values,) * 2)).astype(int)

    df_consensus = pd.DataFrame(C / C_count, columns=full_index, index=full_index)
    return df_consensus

def linkage_clustermap(df_similarity, method="single"):
    """
    Create a clustermap based on a similarity matrix using hierarchical clustering.

    Parameters:
        df_similarity (pd.DataFrame): Similarity matrix.
        method (str): Method for hierarchical clustering (e.g., "single", "complete").

    Returns:
        sns.matrix.ClusterGrid: Clustermap object.
    """
    linkage = getattr(hierarchy, method)(squareform(1 - df_similarity.values))
    cm = sns.clustermap(df_similarity, row_linkage=linkage, col_linkage=linkage)
    return cm

def combine_nt_clusters(df, col="cluster", nt_threshold=1):
    """
    Combine non-target clusters in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with cluster labels.
        col (str): Column name for cluster labels.
        nt_threshold (int): Threshold for combining non-target clusters.

    Returns:
        pd.DataFrame: DataFrame with non-target clusters combined.
    """
    df_ = df.copy()
    if nt_threshold is None:
        df_ = df_.astype({col: "category"})
        s = (
            df_.query('gene_id=="-1"')[col].value_counts(normalize=True).sort_index()
            > df_[col].value_counts(normalize=True).sort_index()
        )
        nt_clusters = list(s.index.categories[s])
    else:
        nt_clusters = list(
            df_.query('gene_id=="-1"')[col]
            .value_counts()
            .rename("nt_count")
            .to_frame()
            .query("nt_count>=@nt_threshold")
            .index
        )
    df_ = df_.astype({col: int})
    df_.loc[df_[col].isin(nt_clusters), col] = -1
    return df_

def fcluster_combine_leaves(Z, t, criterion="distance", depth=2, R=None, monocrit=None):
    """
    Combine clusters in a hierarchical clustering linkage matrix to ensure no leaf clusters remain.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        t (float): The threshold to apply.
        criterion (str): Criterion to use for cluster formation (default is "distance").
        depth (int): The depth of the tree to use for cluster formation.
        R (float, optional): Criterion parameter (not used in this implementation).
        monocrit (float, optional): Monotonicity criterion (not used in this implementation).

    Returns:
        ndarray: Array with cluster labels after combining leaves.
    """
    _ = hierarchy.is_valid_linkage(Z, throw=True)
    N = Z.shape[0] + 1
    T = hierarchy.fcluster(Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit)
    L, M = hierarchy.leaders(Z, T)
    leaf_leaders = list(L[L < N])

    if len(leaf_leaders) == 0:
        return T

    max_cluster = T.max()

    for n, link in enumerate(
        Z[np.logical_or(*(np.in1d(Z[:, l], leaf_leaders) for l in range(2))), :2].astype("i")
    ):
        if n % 10 == 0:
            print(
                f"After {n} iterations, {len(leaf_leaders)} leaf leaders left with {len(np.unique(T))} total clusters"
            )

        if all([l in leaf_leaders for l in link]):
            max_cluster += 1
            T[link] = max_cluster
            _ = [leaf_leaders.remove(l) for l in link]
        elif any([l in leaf_leaders for l in link]):
            node_index = link[0] in leaf_leaders
            node, leaf = link[int(node_index)], link[int(~node_index)]

            if node in L:
                downstream_leaders = [node]
            else:
                tree = hierarchy.to_tree(Z, rd=True)[1][node]

                def check_node(node, nodes_to_check, downstream_leaders, L):
                    if node.id in L:
                        downstream_leaders.append(node.id)
                    else:
                        nodes_to_check.extend([node.left, node.right])
                    return nodes_to_check, downstream_leaders

                downstream_leaders = []
                nodes_to_check = [tree.left, tree.right]

                while len(nodes_to_check) > 0:
                    n_ = nodes_to_check.pop(0)
                    if all([s is None for s in [n_.left, n_.right]]):
                        raise ValueError(
                            "While traversing the tree, a leaf node was reached"
                            f", node {n_.id}. In theory this should not occur."
                        )
                    nodes_to_check, downstream_leaders = check_node(
                        n_, nodes_to_check, downstream_leaders, L
                    )

            max_cluster += 1
            merge_clusters = M[np.in1d(L, downstream_leaders)]
            T[np.in1d(T, merge_clusters)] = max_cluster
            T[leaf] = max_cluster
            _ = leaf_leaders.remove(leaf)
        else:
            continue

        L, M = hierarchy.leaders(Z, T)

        if len(leaf_leaders) == 0:
            break

    leaf_leaders = list(L[L < N])

    if len(leaf_leaders) == 0:
        print(
            f"All leaf leaders combined, resulting in {len(np.unique(T))} total clusters"
        )

        unique, inverse = np.unique(T, return_inverse=True)
        return np.arange(0, unique.shape[0])[inverse]
    else:
        raise ValueError(f"Failed to merge leaf leaders {leaf_leaders}")

def fcluster_multi_threshold(
    Z, ts, criterion="distance", depth=2, R=None, monocrit=None, min_cluster_size=2
):
    """
    Perform hierarchical clustering with multiple thresholds and combine small clusters.

    Parameters:
        Z (ndarray): The linkage matrix from hierarchical clustering.
        ts (list of float): List of thresholds to apply.
        criterion (str): Criterion to use for cluster formation (default is "distance").
        depth (int): The depth of the tree to use for cluster formation.
        R (float, optional): Criterion parameter (not used in this implementation).
        monocrit (float, optional): Monotonicity criterion (not used in this implementation).
        min_cluster_size (int): Minimum size of clusters to keep.

    Returns:
        ndarray: Array with cluster labels after applying multiple thresholds and merging small clusters.
    """
    _ = hierarchy.is_valid_linkage(Z, throw=True)
    T_ = -1 * np.ones(Z.shape[0] + 1)
    N = len(ts)

    for n, t in enumerate(ts):
        T = hierarchy.fcluster(
            Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit
        )
        uniques_, counts_ = np.unique(T, return_counts=True)
        if n == (N - 1):
            keep = T_ == -1
        else:
            keep = np.logical_and(
                np.in1d(T, uniques_[counts_ >= min_cluster_size]), T_ == -1
            )
        T_[keep] = T[keep] + T_.max()

    uniques_, inverse_ = np.unique(T_, return_inverse=True)
    return np.arange(0, uniques_.shape[0])[inverse_]

def merge_small_clusters(y, T, method="complete", min_cluster_size=2):
    """
    Internal function to merge small clusters based on distance matrix `y`.

    Parameters:
        y (ndarray): Distance matrix.
        T (ndarray): Cluster labels.
        method (str): Method for merging clusters ("complete" or "single").
        min_cluster_size (int): Minimum size of clusters to keep.

    Returns:
        ndarray: Array with updated cluster labels after merging small clusters.
    """
    T = T.copy()
    uniques, counts = np.unique(T, return_counts=True)
    size_sort = np.argsort(counts, kind="stable")[::-1]
    uniques, counts = uniques[size_sort], counts[size_sort]

    small_clusters = list(uniques[counts < min_cluster_size])

    if len(small_clusters) == 0:
        raise UserWarning(f"No clusters with fewer than {min_cluster_size} members found.")
        return T

    if method == "complete":
        distance_func = np.max
    elif method == "single":
        distance_func = np.min
    else:
        raise NotImplementedError(f"`method`={method} not implemented")

    while len(small_clusters) > 0:
        s_c = small_clusters.pop(0)
        merge = np.argmin(
            [
                distance_func(y[np.ix_(T == s_c, T == c)])
                for c in uniques[uniques != s_c]
            ]
        )
        T[T == s_c] = uniques[uniques != s_c][merge]

        uniques, counts = np.unique(T, return_counts=True)
        size_sort = np.argsort(counts, kind="stable")[::-1]
        uniques, counts = uniques[size_sort], counts[size_sort]

        small_clusters = list(uniques[counts < min_cluster_size])

    print(f"All small clusters combined, resulting in {len(uniques)} total clusters")
    uniques, inverse = np.unique(T, return_inverse=True)
    return np.arange(0, uniques.shape[0])[inverse]
