import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from generate_data import get_theta_max, generate_data
from run_KMeans import run_KMeans
from run_EnSC import run_EnSC
from evaluate_algo import c_subspaces, c_clusters
from multiprocessing import Pool
from sklearn.decomposition import PCA


def get_heatmaps(p, d, K):
    print()
    print('params p, d, K =', p, d, K)
    theta_max = get_theta_max(p, d, K)

    get_n = lambda x: 2 ** x
    n_values = np.array([get_n(x) for x in np.arange(10, 2, -1)])
    # n_values = np.array([1024, 128, 8])
    theta_values = np.array([0.01, 0.1, 1.0]) * theta_max
    # f, axes = plt.subplots(3, 3, figsize=(10, 10))
    # f.suptitle('Projection of Kmeans clusters pred for p=16 d=1 K=4', fontsize=16)
    kmeans_clusters_by_n = []
    kmeans_subspaces_by_n = []
    elasticnet_clusters_by_n = []
    elasticnet_subspaces_by_n = []
    s = 0
    for n in n_values:
        kmeans_clusters_by_theta = []
        kmeans_subspaces_by_theta = []
        elasticnet_clusters_by_theta = []
        elasticnet_subspaces_by_theta = []
        t = 0
        for theta in theta_values:
            scores = [[], [], [], []]
            for i in range(1):
                print('n', n, 'theta', theta)
                data_points, true_subspaces, true_clusters = generate_data(n, p, d, K, theta)
                subspaces_KMeans, clusters_KMeans = run_KMeans(data_points, K, d)
                subspaces_ElasticNet, clusters_ElasticNet = run_EnSC(data_points, K, d, true_clusters)
                # sub = np.array([true_subspaces[i].T[0] for i in range(K)])
                # temp = np.concatenate([data_points, sub])
                # pca = PCA(n_components=2)
                # pca.fit(temp.T)
                # temp2 = pca.components_[:, :-K]
                # sub_pca = pca.components_[:, -K:]
                # plt.title('EnSC')
                # # plt.scatter(temp2[0], temp2[1], c='black', s=2)
                # for i in range(K):
                #     # plt.scatter(temp2[0], temp2[1], c='black', s=2)
                #     axes[s][t].scatter(temp2[0][clusters_KMeans == i], temp2[1][clusters_KMeans == i], s=2)
                #     # plt.scatter([centers_pca[0][i]], [centers_pca[1][i]], s=4, c='black')
                #     axes[s][t].plot([-sub_pca[0][i], sub_pca[0][i]], [-sub_pca[1][i], sub_pca[1][i]], alpha=0.5)
                scores[0].append(c_clusters(clusters_KMeans, true_clusters, K))
                scores[1].append(c_subspaces(subspaces_KMeans, true_subspaces, K))
                scores[2].append(c_clusters(clusters_ElasticNet, true_clusters, K))
                scores[3].append(c_subspaces(subspaces_ElasticNet, true_subspaces, K))
                # axes[s][t].set_title('score=' + str(np.round(c_clusters(clusters_KMeans, true_clusters, K), 2)))
            t += 1
            kmeans_clusters_by_theta.append(np.mean(scores[0]))
            kmeans_subspaces_by_theta.append(np.mean(scores[1]))
            elasticnet_clusters_by_theta.append(np.mean(scores[2]))
            elasticnet_subspaces_by_theta.append(np.mean(scores[3]))
        s += 1
        kmeans_clusters_by_n.append(kmeans_clusters_by_theta)
        kmeans_subspaces_by_n.append(kmeans_subspaces_by_theta)
        elasticnet_clusters_by_n.append(elasticnet_clusters_by_theta)
        elasticnet_subspaces_by_n.append(elasticnet_subspaces_by_theta)

    # for s in range(3):
    #     for t in range(3):
    #         axes[s][t].set(xlabel=np.array([0.01, 0.1, 1.0])[t], ylabel=n_values[s])
    #         plt.setp(axes[s][t].get_xticklabels(), visible=False)
    #         plt.setp(axes[s][t].get_yticklabels(), visible=False)
    # for ax in axes.flat:
    #     ax.label_outer()
    # # plt.xlabel('theta')
    # # plt.ylabel('n')
    # plt.legend()
    # plt.savefig('out/heatmaps/kmeans_clusters_analysis.png')
    # plt.show()
    # res = np.array([kmeans_clusters_by_n, kmeans_subspaces_by_n, elasticnet_clusters_by_n, elasticnet_subspaces_by_n])
    # titles = np.array(["KMeans Clusters", "KMeans Subspaces", "ElasticNet Clusters", "ElasticNet Subspaces"])
    # f, axes = plt.subplots(2, 2, figsize=(10, 10))
    # f.suptitle('Heat map for p=' + str(p) + ' d=' + str(d) + ' K=' + str(K), fontsize=16)
    # for i in range(4):
    #     g = sns.heatmap(res[i], annot=res[i], ax=axes.flat[i])
    #     g.set_xlabel('theta')
    #     g.set_ylabel('log(n)')
    #     g.set_title(titles[i])
    #     g.set(xticks=np.arange(len(theta_values)), xticklabels=np.round(theta_values, 2),
    #           yticks=np.arange(len(n_values)), yticklabels=np.arange(10, 2, -1))
    # plt.savefig('out/heatmaps/2heatmap_p=' + str(p) + '_d=' + str(d) + '.png')
    # plt.show()
    return elasticnet_clusters_by_n


def plot_all():
    get_n = lambda x: 2 ** x
    n_values = np.array([get_n(x) for x in np.arange(10, 2, -1)])
    theta_values = np.array([0.01, 0.1, 1.0])
    f, axes = plt.subplots(4, 4, figsize=(25, 20))
    f.suptitle('Heat map of cluster scores for EnSC algo', fontsize=30)
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    # p_values = np.array([2**3, 2**4])
    for i in range(len(p_values)):
        p = p_values[i]
        get_d = lambda x: (0.5 ** x) * p
        d_values = np.array([int(get_d(x)) for x in np.arange(4, 0, -1)])
        # d_values = np.array([1, 2])
        for j in range(len(d_values)):
            d = d_values[j]
            elasticnet_clusters_by_n = get_heatmaps(p, d, K)
            g = sns.heatmap(elasticnet_clusters_by_n, annot=elasticnet_clusters_by_n, ax=axes[i][j])
            g.set_xlabel('theta')
            g.set_ylabel('log(n)')
            g.set_title('p=' + str(p) + ' d=' + str(d))
            g.set(xticks=np.arange(len(theta_values)), xticklabels=np.round(theta_values, 2),
                  yticks=np.arange(len(n_values)), yticklabels=np.arange(10, 2, -1))

    plt.savefig('out/heatmaps/heatmap_EnSC_clusters.png')
    # plt.show()


if __name__ == '__main__':
    K = 4
    pool = Pool(processes=4)
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    for p in p_values:
        get_d = lambda x: (0.5 ** x) * p
        d_values = np.array([int(get_d(x)) for x in np.arange(4, 0, -1)])
        for d in d_values:
            pool.apply_async(get_heatmaps, (p, d, K))
    pool.close()
    pool.join()





