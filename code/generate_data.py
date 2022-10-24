import numpy as np
from scipy.stats import special_ortho_group
from scipy.linalg import hadamard, subspace_angles, orth, svd
from scipy.special import binom


def _find_theta(subspaces):
    K = len(subspaces)
    sum_thetas = 0
    for i in range(K):
        for j in range(i):
            sum_thetas += subspace_angles(subspaces[i], subspaces[j])[0]
    return sum_thetas / binom(K, 2)

# def _find_theta(subspaces):
#     K = len(subspaces)
#     sum_thetas = 0
#     # sum_thetas2 = 0
#     for i in range(K):
#         for j in range(i):
#             sum_thetas += subspace_angles(subspaces[i], subspaces[j])[0]
#             # print('*', subspace_angles(subspaces[i], subspaces[j])[0])
#             # s = svd(np.dot(subspaces[i].T, subspaces[j]), compute_uv=False)
#             # print(np.arccos(s)[-1])
#             # sum_thetas2 += np.max(np.arccos(s))
#             # print('**', np.max(np.arccos(s)))
#     print(sum_thetas / binom(K, 2))
#     # print((1 / binom(K, 2)) * sum_thetas2)
#     return sum_thetas / binom(K, 2)


def get_theta_max(p, d, K):
    uniform_subspaces = _generate_uniform_subspaces(p, d, K)
    return _find_theta(uniform_subspaces)


def _get_subspaces_for_theta(theta, p, d, K):  # todo how to find subspaces? not monotonic
    # print('theta', theta)
    subspaces = _generate_uniform_subspaces(p, d, K + 1)
    B_K = subspaces[-1]
    subspaces = subspaces[:K]
    mid_subspaces = subspaces
    mid_theta = _find_theta(subspaces)
    if np.abs(mid_theta - theta) < 0.05:
        return mid_subspaces, mid_theta
    arr_alpha = np.arange(0.00005, 1, 0.00005)
    first = 0
    last = len(arr_alpha) - 1
    while first <= last:
        midpoint = (first + last) // 2
        mid_alpha = arr_alpha[midpoint]
        mid_subspaces = mid_alpha * subspaces + (1 - mid_alpha) * B_K
        mid_theta = _find_theta(mid_subspaces)
        if np.abs(mid_theta - theta) < 0.05:
            return mid_subspaces, mid_theta
        else:
            if theta < mid_theta:
                last = midpoint - 1
            else:
                first = midpoint + 1
    print("problem in _get_subspaces_for_theta")
    return _get_subspaces_for_theta(theta, p, d, K)


def _generate_uniform_subspaces(p, d, K):
    # K matrices p x d, columns = vectors in R^p, orthonormal and defines B_k with dim = d
    return special_ortho_group.rvs(p, K)[:, :, :d]


def complete_to_subspace_uniformly(vectors, missing_dim):
    # we'll find an orthonormal basis to the range of A = [vectors, missing_dim random vectors] = the column space of A.
    random_vectors = np.random.rand(len(vectors[0]) * missing_dim).reshape((missing_dim, len(vectors[0])))
    A = np.concatenate([vectors, random_vectors])
    return orth(A.transpose())


def generate_data(n, p, d, K, theta, sigma2=0):
    z = np.random.randint(0, K, n)
    w = np.random.multivariate_normal(np.array([0] * d), np.identity(d), n)
    subspaces, theta = _get_subspaces_for_theta(theta, p, d, K)
    x = np.array([np.random.multivariate_normal(np.dot(subspaces[z[i]], w[i]), sigma2 * np.identity(p)) for i in range(n)])
    return x, subspaces, z


if __name__ == '__main__':
    # temp = _generate_uniform_subspaces(70, 50, 2)
    # print(np.rad2deg(_find_theta(temp)))

    # H = hadamard(4)
    # print(np.rad2deg(calc_average_subspace_angles([H[:, :2], H[:, 2:]]))) # test (print 90.0)

    x, subspaces, z = generate_data(100, 10, 5, 3, 0.5)
    K=4
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    for p in p_values:
        get_d = lambda x: (0.5 ** x) * p
        d_values = np.array([int(get_d(x)) for x in np.arange(4, 0, -1)])
        for d in d_values:
            theta_max = get_theta_max(p, d, K)

            get_n = lambda x: 2 ** x
            n_values = np.array([get_n(x) for x in np.arange(10, 2, -1)])
            theta_values = np.array([0.01, 0.1, 1.0])
            for n in n_values:
                for theta in theta_values:
                    data_points, true_subspaces, true_clusters = generate_data(n, p, d, K, theta * theta_max)

    a = 2