import numpy as np
from fashion_mnist.utils import mnist_reader
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from run_EnSC import run_EnSC
from evaluate_algo import c_clusters
from sklearn.cluster import KMeans


def get_fashion_MNIST():
    # get fashion MNIST data
    X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')
    return X_train, y_train


def sub_mean_from_each_class(X_train, y_train):
    # subtract the mean from every class
    train_means = np.array([X_train[y_train == i].mean(axis=0) for i in range(10)])
    X_train = X_train - train_means[y_train]
    return X_train


def plot_projection(X_train, y_train):
    # PCA & plot of projection on the first 2 principal components with different colors for each class.
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['label'] = y_train
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('Projection on first 2 principal components', fontsize = 20)
    labels = np.arange(10)
    for label in labels:
        indicesToKeep = principalDf['label'] == label
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
                   principalDf.loc[indicesToKeep, 'principal component 2']
                   , s=5)
    plt.savefig('out/q2a')
    plt.show()


def q2a(X_train, y_train):
    plot_projection(X_train, y_train)


def compute_angle(vectors):
    return np.arccos(np.clip(np.dot(vectors[0] / np.linalg.norm(vectors[0]), vectors[1] / np.linalg.norm(vectors[1])), -np.pi, np.pi))


def sample(X_train, y_train):
    # sample 5000 pairs of points from same class
    # choose class
    label = np.random.randint(0, 10, 5000)
    num_from_class = [5000 - (np.count_nonzero(label - i)) for i in range(10)]
    # choose points
    idx = [np.random.choice(np.arange(len(X_train[y_train == i])), num_from_class[i] * 2) for i in range(10)]
    one_class_pairs = np.concatenate([X_train[y_train == i][idx[i]].reshape((num_from_class[i], 2, 784)) for i in range(10)])

    # sample 5000 pairs of points from different classes
    idx = np.random.choice(np.arange(len(X_train)), 10000)
    different_classes_pairs = X_train[idx].reshape((5000, 2, 784))
    return one_class_pairs, different_classes_pairs


def plot_angles_dist(one_class_angles, different_classes_angles):
    plt.hist(different_classes_angles, density=True, bins=100, alpha=0.8, color='teal', label='different classes')
    plt.hist(one_class_angles, density=True, bins=100, alpha=0.8, color='darkslategrey', label='one class')
    plt.title('Density function of angles between random points')
    plt.legend(title='angles from:')
    plt.savefig('out/q2b_random')
    plt.show()


def q2b_random_sampling():
    data = np.random.uniform(-225, 225, 784 * 10000).reshape((10000, 784))
    labels = np.random.randint(0, 10, 10000)
    data = sub_mean_from_each_class(data, labels)
    q2b(data, labels)


def q2b(X_train, y_train):
    one_class_pairs, different_classes_pairs = sample(X_train, y_train)

    # compute angles between each pair
    one_class_angles = np.array([compute_angle(pair) for pair in one_class_pairs])
    different_classes_angles = np.array([compute_angle(pair) for pair in different_classes_pairs])

    plot_angles_dist(one_class_angles, different_classes_angles)


def plot_variance_explained(X_train, y_train, cut=784, cumsum=True):
    for i in range(10):
        pca = PCA()
        pca.fit(X_train[y_train == i])
        variance_explained = pca.explained_variance_ratio_[:cut]
        if cumsum:
            plt.plot(np.arange(cut), np.cumsum(variance_explained), label='class ' + str(i))
        else:
            plt.plot(np.arange(cut), variance_explained, label='class ' + str(i))
    pca = PCA()
    pca.fit(X_train)
    variance_explained = pca.explained_variance_ratio_[:cut]
    if cumsum:
        plt.plot(np.arange(cut), np.cumsum(variance_explained), label='all data', color='black', linewidth=2)
    else:
        plt.plot(np.arange(cut), variance_explained, label='all data', color='black', linewidth=2)
    # plt.title('Ratio of variance explained vs. num of components for 400 components')
    plt.title('Ratio of variance explained vs. num of components')
    plt.xlabel('num of components')
    plt.ylabel('variance explained')
    plt.legend()
    plt.savefig('out/q2c_400.png')
    plt.show()


def q2c(X_train, y_train):
    # plot_variance_explained(X_train, y_train)
    plot_variance_explained(X_train, y_train, cut=400)


def q2d_Kmeans(X_train, y_train):
    # run KMeans with K=10
    K = 10
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X_train)
    clusters = kmeans.labels_
    print('kmeans', c_clusters(clusters, y_train, K))


def q2d_PCA_Kmeans(X_train, y_train):
    # run PCA with num_component from (c), then KMeans with K=10 on the projection on the top components
    num_components = 250
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(X_train)
    K = 10
    kmeans = KMeans(n_clusters=K, random_state=0).fit(principal_components)
    clusters = kmeans.labels_
    print('PCA kmeans', c_clusters(clusters, y_train, K))


def q2d_EnSC(X_train, y_train):
    # run with the chosen subspace clustering algorithm
    K = 10
    _, clusters = run_EnSC(X_train, K, 0, return_subspaces=False)
    print('EnSC', c_clusters(clusters, y_train, K))


def plot_q2d_results(c_kmeans, c_pca_kmeans, c_EnSC):
    b = plt.bar(np.arange(3), [c_kmeans, c_pca_kmeans, c_EnSC], width=0.35, tick_label=['Kmeans', 'PCA+Kmeans', 'EnSC'])
    plt.ylabel('C_cluster')
    plt.title('C_cluster for Kmeans, PCA+Kmeans and EnSC on Fashion MNIST')
    plt.yticks(np.arange(0, 1.1, 0.1))

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(round(height,3)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(b)
    plt.tight_layout()
    plt.savefig('out/q2d.png')
    plt.show()


def q2d(X_train, y_train):
    from multiprocessing import Pool
    pool = Pool(processes=3)
    algo = [q2d_Kmeans, q2d_PCA_Kmeans, q2d_EnSC]
    for i in range(3):
        try:
            pool.apply_async(algo[i], (X_train, y_train))
        except:
            print(algo[i].__name__)
    pool.close()
    pool.join()


if __name__ == '__main__':
    X_train, y_train = get_fashion_MNIST()
    X_train = sub_mean_from_each_class(X_train, y_train)
    q2d_PCA_Kmeans(X_train, y_train)
    # q2a(X_train, y_train)
    # q2b(X_train, y_train)
    # q2b_random_sampling()
    # q2c(X_train, y_train)
    # q2d(X_train, y_train)
    # kmeans 0.2582833333333333
    # PCA kmeans 0.264
    # EnSC 0.6088
    plot_q2d_results(0.2582833333333333, 0.264, 0.6088)
    a = 2