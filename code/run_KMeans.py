from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from generate_data import generate_data, get_theta_max, complete_to_subspace_uniformly
from evaluate_algo import c_clusters, c_subspaces
import numpy as np
import matplotlib.pyplot as plt


def run_KMeans(data_points, K, d, return_subspaces=True):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data_points)
    z = kmeans.labels_
    p = len(data_points[0])
    subspaces = []
    if return_subspaces:
        for k in range(K):
            k_len = len(data_points[z == k])
            if k_len == d:
                B_k = data_points[z == k].transpose()
            elif k_len < d:
                # then the data points with label k span a subspace with dim = k_len - 1
                # (sometimes can be less but very much unlikely in a simulation).
                # we'll complete the matrix representing the subspace to have d columns by adding zero vectors,
                # so it will not change the spanned subspace.
                B_k = np.concatenate([data_points[z == k], np.zeros((d - k_len, p))]).transpose()
            else:
                pca = PCA(n_components=d)
                pca.fit(data_points[z == k])
                B_k = pca.components_.transpose()
            subspaces.append(B_k)
    # sub = np.array([true_subspaces[i].T[0] for i in range(K)])
    # sub2 = np.array([subspaces[i].T[0] for i in range(K)])
    # centers = kmeans.cluster_centers_
    # temp = np.concatenate([data_points, sub, sub2])
    # pca = PCA(n_components=2)
    # pca.fit(temp.T)
    # temp2 = pca.components_[:, :-2 * K]
    # sub_pca = pca.components_[:, -2 * K:-K]
    # sub_pca2 = pca.components_[:, -K:]
    # plt.title('Kmeans')
    # # plt.scatter(temp2[0], temp2[1], c='black', s=2)
    # for i in range(K):
    #     # plt.scatter(temp2[0], temp2[1], c='black', s=2)
    #     plt.scatter(temp2[0][z == i], temp2[1][z == i], s=2)
    #     # plt.scatter([centers_pca[0][i]], [centers_pca[1][i]], s=4, c='black')
    #     plt.plot([-sub_pca[0][i], sub_pca[0][i]], [-sub_pca[1][i], sub_pca[1][i]], alpha=0.5)
    #     plt.plot([-sub_pca2[0][i], sub_pca2[0][i]], [-sub_pca2[1][i], sub_pca2[1][i]], c='black')
    # plt.show()

    return np.array(subspaces), z


if __name__ == '__main__':
    n = 100
    p = 10
    d = 5
    K = 4
    theta_max = get_theta_max(p, d, K)
    data_points, true_subspaces, true_clusters = generate_data(100, 10, 5, 4, 0.1 * theta_max)
    subspaces, clusters = run_KMeans(data_points, K, d)
    print(c_clusters(clusters, true_clusters, K))
    print(c_subspaces(subspaces, true_subspaces, K))

