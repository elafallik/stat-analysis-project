import numpy as np
from scipy.linalg import subspace_angles
from itertools import permutations
from generate_data import generate_data
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def c_subspaces(pred_subspaces, true_subspaces, K):
    perm = np.array(list(permutations(np.arange(0, K))))
    perm_pred_clusters = pred_subspaces[perm]
    angels = []
    for i in range(len(perm)):
        temp = [subspace_angles(perm_pred_clusters[i][j], true_subspaces[j])[0] for j in range(K)]
        angels.append(np.sum(np.square(np.cos(temp))))

    return max(angels)


def c_clusters(pred_clusters, true_clusters, K):
    # credit: https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
    cm = confusion_matrix(pred_clusters, true_clusters)

    def _make_cost_m(cm):
        s = np.max(cm)
        return - cm + s

    perm = linear_sum_assignment(_make_cost_m(cm))
    perm = [e[1] for e in sorted(zip(perm[0], perm[1]), key=lambda x: x[0])]
    cm_permuted = cm[:, perm]
    return np.trace(cm_permuted) / np.sum(cm_permuted)



if __name__ == '__main__':

    a = np.array([0] * 5 + [1])
    b = np.array([0, 0, 1, 1, 4, 5])

    print(c_clusters(b, a, 6))

    _, subspaces, _ = generate_data(100, 10, 5, 3, 0.5)
    _, subspaces2, _ = generate_data(100, 10, 5, 3, 0.5)
    print(c_subspaces(subspaces, subspaces2, 3))
