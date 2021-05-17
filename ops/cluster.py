import leidenalg
import graphtools
import sklearn
from igraph import Graph
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy


class AffinityLeiden(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
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

        if isinstance(X, list):
            X = np.array(X)

        if X.ndim < 2:
            raise ValueError("Cannot fit 1D array.")

        if X.shape[0] == 1:
            raise ValueError("Input contains only 1 sample.")

        self.n_features_in_ = X.shape[1]

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

        self.affinity_matrix_ = graph.diff_op.todense()

        affinity_igraph = Graph().Weighted_Adjacency(
            matrix=self.affinity_matrix_.tolist(), mode="undirected"
        )

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
        self._singularities = (
            np.unique(self.labels_, return_counts=True)[1] == 1
        ).sum()
        return self._singularities

    @property
    def n_clusters(self):
        self._n_labels = len(np.unique(self.labels_))
        return self._n_labels

    @property
    def mean_cluster_size(self):
        self._mean_cluster_size = np.unique(self.labels_, return_counts=True)[1].mean()
        return self._mean_cluster_size

    def set_index(self, index):
        assert len(index) == len(self.labels_)
        self.index = index
        return self

    def adjusted_mutual_info_score(self, s):
        return sklearn.metrics.adjusted_mutual_info_score(
            self.labels_, s[self.index].values
        )

    def adjusted_rand_score(self, s):
        return sklearn.metrics.adjusted_rand_score(self.labels_, s[self.index].values)


def consensus_matrix(
    dfs, column="cluster", weights=None, combine_nt=False, nt_threshold=1
):

    # check for identical indices
    [pd.testing.assert_index_equal(dfs[0].index, df.index) for df in dfs[1:]]

    if weights is not None:
        if sum(weights) != 1:
            raise ValueError("`weights` must sum to 1")
        if len(weights) != len(dfs):
            raise ValueError("`weights` must have the same length as `dfs`")
    else:
        weights = [
            1 / len(dfs),
        ] * len(dfs)

    C = np.zeros((dfs[0].pipe(len),) * 2)

    for df, w in zip(dfs, weights):
        if combine_nt:
            df = combine_nt_clusters(df, nt_threshold)

        C += np.equal(*np.meshgrid(*(df[column].values,) * 2)) * w

    df_consensus = pd.DataFrame(C, columns=dfs[0].index, index=dfs[0].index)

    return df_consensus


def subsampled_consensus_matrix(
    dfs, column="cluster", combine_nt=False, nt_threshold=1, tqdm=False
):

    full_index = set()
    for df in dfs:
        full_index |= set(df.index)

    full_index = pd.MultiIndex.from_tuples(
        sorted(full_index), names=["gene_symbol", "gene_id"]
    )

    C = np.zeros((len(full_index),) * 2)

    C_count = np.zeros_like(C)

    if tqdm:
        from tqdm.auto import tqdm

        dfs = tqdm(dfs)

    for df in dfs:
        df = df.sort_index()
        if combine_nt:
            df = combine_nt_clusters(df, nt_threshold)
        selected = np.argwhere(full_index.isin(df.index))
        C_count[selected, selected.T] += 1
        C[selected, selected.T] += np.equal(
            *np.meshgrid(*(df[column].values,) * 2)
        ).astype(int)

    #     return C,C_count
    df_consensus = pd.DataFrame(C / C_count, columns=full_index, index=full_index)

    return df_consensus


def linkage_clustermap(df_similarity, method="single"):
    linkage = getattr(hierarchy, method)(squareform(1 - df_similarity.values))
    cm = sns.clustermap(df_similarity, row_linkage=linkage, col_linkage=linkage)
    return cm


def combine_nt_clusters(df, col="cluster", nt_threshold=1):
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
    # AKA no leaf left behind

    # check if Z is a valid linkage matrix
    _ = hierarchy.is_valid_linkage(Z, throw=True)

    N = Z.shape[0] + 1

    # alternative: iteratively increase t, check for remaining leaves

    # move up the tree, merging leaf clusters until all leaves are merged into clusters
    T = hierarchy.fcluster(
        Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit
    )
    L, M = hierarchy.leaders(Z, T)
    leaf_leaders = list(L[L < N])

    # no leaf clusters
    if len(leaf_leaders) == 0:
        return T

    max_cluster = T.max()

    # iterate through all links
    for n, link in enumerate(
        Z[
            np.logical_or(*(np.in1d(Z[:, l], leaf_leaders) for l in range(2))), :2
        ].astype("i")
    ):

        if n % 10 == 0:
            print(
                f"After {n} iterations, {len(leaf_leaders)} leaf leaders left with {len(np.unique(T))} total clusters"
            )

        # find linkages if link is between two leaf_leaders
        if all([l in leaf_leaders for l in link]):
            # make new cluster of leaf leaders
            max_cluster += 1
            T[link] = max_cluster

            # remove from list of leaf_leaders
            _ = [leaf_leaders.remove(l) for l in link]

        # find linkages of leaf leaders with any non-leaf node
        elif any([l in leaf_leaders for l in link]):

            # which one is the leaf leader?
            node_index = link[0] in leaf_leaders
            node, leaf = link[int(node_index)], link[int(~node_index)]

            # other node is a leader
            if node in L:
                downstream_leaders = [node]

            # node is not a leader, have to traverse down the tree until leaders are found
            else:
                # get hierarchy.ClusterNode representation of the node
                tree = hierarchy.to_tree(Z, rd=True)[1][node]

                def check_node(node, nodes_to_check, downstream_leaders, L):
                    """check if a node is a leader, else append successors to nodes_to_check"""
                    if node.id in L:
                        downstream_leaders.append(node.id)
                    else:
                        nodes_to_check.extend([node.left, node.right])
                    return nodes_to_check, downstream_leaders

                # initialize traversal
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

            # update T
            max_cluster += 1
            merge_clusters = M[np.in1d(L, downstream_leaders)]
            T[np.in1d(T, merge_clusters)] = max_cluster
            T[leaf] = max_cluster

            # remove from leaf_leaders
            _ = leaf_leaders.remove(leaf)

        else:
            continue

        # update L,M
        L, M = hierarchy.leaders(Z, T)

        if len(leaf_leaders) == 0:
            break

    leaf_leaders = list(L[L < N])

    # no leaf clusters
    if len(leaf_leaders) == 0:
        print(
            f"All leaf leaders combined, resulting in {len(np.unique(T))} total clusters"
        )

        # relabel
        unique, inverse = np.unique(T, return_inverse=True)

        return np.arange(0, unique.shape[0])[inverse]
    else:
        raise ValueError(f"Failed to merge leaf leaders {leaf_leaders}")


def fcluster_multi_threshold(
    Z, ts, criterion="distance", depth=2, R=None, monocrit=None, min_cluster_size=2
):
    # check if Z is a valid linkage matrix
    _ = hierarchy.is_valid_linkage(Z, throw=True)

    T_ = -1 * np.ones(Z.shape[0] + 1)
    N = len(ts)

    for n, t in enumerate(ts):
        T = hierarchy.fcluster(
            Z, t, criterion=criterion, depth=depth, R=R, monocrit=monocrit
        )
        uniques_, counts_ = np.unique(T, return_counts=True)
        if n == (N - 1):
            # last threshold, keep all resulting cluster labels
            keep = T_ == -1
        else:
            keep = np.logical_and(
                np.in1d(T, uniques_[counts_ >= min_cluster_size]), T_ == -1
            )
        T_[keep] = T[keep] + T_.max()

    # relabel
    uniques_, inverse_ = np.unique(T_, return_inverse=True)

    return np.arange(0, uniques_.shape[0])[inverse_]


def merge_small_clusters(y, T, method="complete", min_cluster_size=2):
    return merge_small_clusters_(y, T, method=method, min_cluster_size=min_cluster_size)
    # uniques,counts = np.unique(T,return_counts=True)
    # small_clusters = uniques[counts<min_cluster_size]
    # valid_clusters = uniques[counts>=min_cluster_size]

    # # no small clusters
    # if len(small_clusters)==0:
    #     raise UserWarning(f'No clusters with fewer than {min_cluster_size} members found.')
    #     return T

    # # for each small cluster:
    # # find closest existing cluster using `y` and `method`

    # if method=='complete':
    # 	distance_func = np.max
    # elif method=='single':
    # 	distance_func = np.min
    # else:
    # 	raise NotImplementedError(f'`method`={method} not implemented')

    # T_ = T.copy()

    # for s_c in small_clusters:
    # 	merge = np.argmin([distance_func(y[np.ix_(T==s_c,T==v_c)]) for v_c in valid_clusters])
    # 	T_[T==s_c] = valid_clusters[merge]

    # uniques_,inverse_,counts_ = np.unique(T_,return_inverse=True,return_counts=True)

    # # no small clusters
    # if len(uniques_[counts_<min_cluster_size])==0:
    # 	print(f'All small clusters combined, resulting in {len(uniques_)} total clusters')

    # 	# relabel
    # 	return np.arange(0,uniques_.shape[0])[inverse_]
    # else:
    #     raise ValueError(f'Failed to merge all clusters with fewer than {min_cluster_size} members.')


def merge_small_clusters_(y, T, method="complete", min_cluster_size=2):
    # prioritizes larger clusters
    T = T.copy()
    uniques, counts = np.unique(T, return_counts=True)
    size_sort = np.argsort(counts, kind="stable")[::-1]
    uniques, counts = uniques[size_sort], counts[size_sort]

    small_clusters = list(uniques[counts < min_cluster_size])

    # no small clusters
    if len(small_clusters) == 0:
        raise UserWarning(
            f"No clusters with fewer than {min_cluster_size} members found."
        )
        return T

    # for each small cluster:
    # find closest existing cluster using `y` and `method`

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

        # re-compute size-sorted clusters below threshold
        uniques, counts = np.unique(T, return_counts=True)
        size_sort = np.argsort(counts, kind="stable")[::-1]
        uniques, counts = uniques[size_sort], counts[size_sort]

        small_clusters = list(uniques[counts < min_cluster_size])

    # no more small clusters
    print(f"All small clusters combined, resulting in {len(uniques)} total clusters")

    uniques, inverse = np.unique(T, return_inverse=True)

    # relabel
    return np.arange(0, uniques.shape[0])[inverse]
