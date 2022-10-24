import numpy as np
import matplotlib.pyplot as plt
from generate_data import get_theta_max, generate_data
from run_EnSC import run_EnSC
from evaluate_algo import c_clusters
from scipy.optimize import line_search
import pandas as pd


def simulate(n, p, d, K, theta):
    data_points, true_subspaces, true_clusters = generate_data(n, p, d, K, theta)
    _, clusters_ElasticNet = run_EnSC(data_points, K, d, return_subspaces=False)
    c = c_clusters(clusters_ElasticNet, true_clusters, K)
    return c


def c_vs_n(p, d, K, theta, n_values):
    scores = []
    for n in n_values:
        c = simulate(n, p, d, K, theta)
        scores.append(c)

    return scores


def plot_all_c_curves(data):
    n_values = np.array([int(s) for s in data.columns[4:]])
    theta_values = np.array([0.01, 0.1, 1.0])
    f, axes = plt.subplots(4, 3, figsize=(25, 20))
    f.suptitle('C_cluster vs n', fontsize=30)
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    for i in range(len(p_values)):
        p = p_values[i]
        get_d = lambda x: (0.5 ** x) * p
        d_values = np.array([int(get_d(x)) for x in np.arange(4, 0, -1)])
        for j in range(len(theta_values)):
            theta = theta_values[j]
            x = data[data['p'] == p][data['theta'] == theta]
            scores = np.array(x[np.array([str(n) for n in n_values])])
            for t in range(len(d_values)):
                axes[i][j].plot(n_values, scores[t], label=str(d_values[t]))

    for i in range(len(p_values)):
        p = p_values[i]
        for j in range(len(theta_values)):
            theta = theta_values[j]
            axes[i][j].plot(n_values, [0.5] * len(n_values), color='black')
            axes[i][j].legend(title='d')
            axes[i][j].set_xlabel('n')
            axes[i][j].set_ylabel('C_cluster')
            axes[i][j].set_ylim([0, 1.1])
            axes[i][j].set_title('p=' + str(p) + ' theta/theta_max=' + str(round(theta, 3)))

    plt.savefig('out/q1b/c_vs_n_curves.png')
    plt.show()


def find_high_n(p, d, K, theta, q):
    n = K * 2
    c = 0
    while c < q and n < 2 ** 14:
        n = n * 2
        c = simulate(n, p, d, K, theta)
    return n, c


def find_nq(p, d, K, theta, q):
    n_high, c = find_high_n(p, d, K, theta, q)
    while n_high == 2 ** 14:
        n_high, c = find_high_n(p, d, K, theta, q)
    n_mid = 0
    if c == q:
        return n_high, c
    n_low = n_high // 2
    while n_low <= n_high:
        n_mid = (n_high + n_low) // 2
        c = simulate(n_mid, p, d, K, theta)
        if c == q:
            return n_mid, c
        elif c < q:
            n_low = n_mid + 1
        else:
            n_high = n_mid - 1
    return find_nq(p, d, K, theta, q)


def plot_nq_curve(q=0.5):
    n_q_res = []
    n_q_var = []
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    for p in p_values:
        get_d = lambda x: (0.5 ** x) * p
        d_values = np.array([int(get_d(x)) for x in np.arange(4, 0, -1)])
        for d in d_values:
            theta_max = get_theta_max(p, d, K)
            theta_values = np.array([0.01, 0.1, 1.0]) * theta_max
            for theta in theta_values:
                n_q = []
                print(p, d, theta)
                for i in range(10):
                    n_q_temp = find_nq(p, d, K, theta, q)
                    n_q.append(n_q_temp[0])
                print(np.mean(n_q))
                n_q_res.append(np.mean(n_q))
                n_q_var.append(np.std(n_q))
                print('***')
    res = np.array(n_q_res).reshape((4, 4, 1))  # len(p_values) x len(d_values) x len(theta_values)
    var = np.array(n_q_var).reshape((4, 4, 1))
    res = res.transpose((0, 2, 1))
    var = var.transpose((0, 2, 1))

    theta_div_max_values = np.array([1])

    idx = np.array([(0.5 ** x) for x in np.arange(4, 0, -1)])  # d/p
    for i in range(len(p_values)):
        p = p_values[i]
        for j in range(len(theta_div_max_values)):
            theta = theta_div_max_values[j]
            n_q_by_d = res[i][j]
            var_by_d = var[i][j]
            plt.title('n_0.5 as func of d/p for p=' + str(p) + ', theta/theta_max=' + str(theta))
            plt.xlabel('d/p')
            plt.ylabel('n_0.5')
            plt.plot(idx, n_q_by_d)
            plt.errorbar(idx, n_q_by_d, var_by_d, linestyle='None', marker='^')
            plt.legend()
            plt.savefig('out/q1b/n_q_p=' + str(p) + '_theta=' + str(theta) + '.png')
            plt.show()


def find_plot_b(res):
    get_p = lambda x: 2 ** x
    p_values = np.array([get_p(x) for x in np.arange(4, 8, 1)])
    theta_div_max_values = np.array([0.01, 0.1, 1.0])
    curve0 = np.array(res[0][0])
    res_b = np.zeros((len(p_values), len(theta_div_max_values)))
    for i in range(len(p_values)):
        p = p_values[i]
        for j in range(len(theta_div_max_values)):
            curve1 = np.array(res[i][j])

            def find_b(b, curve0=curve0, curve1=curve1):
                return (np.linalg.norm(curve0 - (curve1 / b), ord=1))

            def find_b_grad(b, curve0=curve0, curve1=curve1):
                return np.sum(2 * curve0 * (b * curve1 - curve0)) / (b ** 3)

            b = line_search(find_b, find_b_grad, np.array([0.01]), np.array([-1]))
            print(i, j, 'b=', b, 'f(b)=', find_b(b))
            res_b[i][j] = b

    idx = np.array([(0.5 ** x) for x in np.arange(4, 0, -1)])  # d/p
    for i in range(len(p_values)):
        p = p_values[i]
        for j in range(len(theta_div_max_values)):
            theta = theta_div_max_values[j]
            n_q_by_d = np.array(res[i][j])
            plt.title('n_0.5 as func of d/p for p=' + str(p) + ', theta/theta_max=' + str(theta))
            plt.xlabel('d/p')
            plt.ylabel('n_0.5')
            plt.plot(idx, n_q_by_d / res_b[i][j], label=str(i) + str(j))
    # plt.legend()
    # plt.savefig('out/3n_q_p=' + str(p) + '_theta=' + str(theta) + '.png')
    plt.show()

    for i in range(len(p_values)):
        p = p_values[i]
        for j in range(len(theta_div_max_values)):
            theta = theta_div_max_values[j]
            n_q_by_d = np.array(res[i][j])
            # var_by_d = var[i][j]
            plt.title('n_0.5 as func of d/p for p=' + str(p) + ', theta/theta_max=' + str(theta))
            plt.xlabel('d/p')
            plt.ylabel('n_0.5')
            plt.plot(idx, n_q_by_d)
    # plt.savefig('out/3n_q_p=' + str(p) + '_theta=' + str(theta) + '.png')
    plt.show()


if __name__ == '__main__':
    K = 4
    # plot_nq_curve()
    # plot_all_c_curves()
    data = pd.read_csv('out/q1b/res.csv')
    plot_all_c_curves(data)






