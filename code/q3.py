from nlp.TopicModel import *
from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering

# run 3 methods from q2d on the facebook posts, compare to the algebraic algorithms, with/ without "other"

def run_other_topic_first_with_EnSC(data, algo_step1, m, certainty_prec=30):
    model = algo_step1()
    model.fit(data, 3)
    top_topics, top_topics_relevance = model.predict_topics()
    docs_topic_dist, docs_tags = model.predict_docs_dist(certainty_prec=certainty_prec)

    K = 4
    i = np.argmax([sum(docs_tags == i) for i in np.arange(0, K)])
    # i -> 0, 0 -> i in docs_tags
    temp_pred = np.copy(docs_tags)
    idx0 = np.array(temp_pred == 0)
    temp_pred[temp_pred == i] = 0
    temp_pred[idx0] = i
    # print("accurate zeros =", np.average(temp_pred[topic_tags == 0] == topic_tags[topic_tags == 0]))
    # print("accuracy =", accuracy_score(temp_pred, topic_tags))
    # print("zeros = ", sum(temp_pred == 0))
    # print(confusion_matrix(temp_pred, topic_tags))

    data_new = data[temp_pred != 0]
    model = m

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(remove_stopwords(data_new)).toarray()
    model.fit(vectors)
    docs_tags = model.labels_
    pred = np.zeros(len(data), dtype=int)
    pred[temp_pred == 0] = 0
    pred[temp_pred != 0] = docs_tags + 1
    return pred


def run_iterations_2_steps(data, topic_tags, algo1, num_iter=10):
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    for i in range(4):
        print(len(topic_tags[topic_tags == i]))
    for m in models:
        print(m.__name__)
        scores = []
        accuracies = []
        for j in range(num_iter):
            pred = run_other_topic_first(data, algo_step1=algo1, algo_step2=m)
            score, perm_pred = c_clusters_topicmodel(pred, topic_tags, K=4)
            accuracy = accuracy_score(perm_pred, topic_tags)
            scores.append(score)
            accuracies.append(accuracy)
        print('score', np.average(scores))
        print('accuracy', np.average(accuracies))
        print('average =', (np.average(scores) + np.average(accuracies)) / 2)

    # EnSC
    print('EnSC')
    scores = []
    accuracies = []
    for j in range(num_iter):
        pred = run_other_topic_first_with_EnSC(data, algo_step1=algo1, m=ElasticNetSubspaceClustering(n_clusters=3, algorithm='lasso_cd', gamma=50))
        score, perm_pred = c_clusters_topicmodel(pred, topic_tags, K=4)
        accuracy = accuracy_score(perm_pred, topic_tags)
        scores.append(score)
        accuracies.append(accuracy)
    print('score', np.average(scores))
    print('accuracy', np.average(accuracies))
    print('average =', (np.average(scores) + np.average(accuracies)) / 2)


def run_iterations_3_topics(data, topic_tags, num_iter=10):
    data = data[topic_tags != 0]
    topic_tags = topic_tags[topic_tags != 0] - 1
    models = [SVDTopicModel, RandomSVDTopicModel, PCATopicModel, NMFTopicModel, NMFTFIDFTopicModel]
    for i in range(3):
        print(len(topic_tags[topic_tags == i]))
    for m in models:
        print(m.__name__)
        scores = []
        accuracies = []
        for j in range(num_iter):
            model = m(extra_topic=False)
            model.fit(data, 3)
            top_topics, top_topics_relevance = model.predict_topics()
            docs_topic_dist, pred = model.predict_docs_dist()
            score, perm_pred = c_clusters_topicmodel(pred, topic_tags, K=3, score_without0=False)
            accuracy = accuracy_score(perm_pred, topic_tags)
            accuracies.append(accuracy)
        print('accuracy', np.average(accuracies))

    # EnSC
    print('EnSC')
    accuracies = []
    for j in range(num_iter):
        model = ElasticNetSubspaceClustering(n_clusters=3, algorithm='lasso_cd', gamma=50)
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(remove_stopwords(data)).toarray()
        model.fit(vectors)
        pred = model.labels_
        score, perm_pred = c_clusters_topicmodel(pred, topic_tags, K=3, score_without0=False)
        accuracy = accuracy_score(perm_pred, topic_tags)
        accuracies.append(accuracy)
    print('accuracy', np.average(accuracies))


def compute_angle(vectors):
    return np.arccos(np.clip(np.dot(vectors[0] / np.linalg.norm(vectors[0]), vectors[1] / np.linalg.norm(vectors[1])), -np.pi, np.pi))


def sample(X_train, y_train):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(remove_stopwords(X_train)).toarray()
    # sample 5000 pairs of points from same class
    # choose class
    label = np.random.randint(0, 3, 5000)
    num_from_class = [5000 - (np.count_nonzero(label - i)) for i in range(3)]
    # choose points
    idx = [np.random.choice(np.arange(len(vectors[y_train == i])), num_from_class[i] * 2) for i in range(3)]
    one_class_pairs = np.concatenate(
        [vectors[y_train == i][idx[i]].reshape((num_from_class[i], 2, len(vectors[0]))) for i in range(3)])

    # sample 5000 pairs of points from different classes
    idx = np.random.choice(np.arange(len(vectors)), 10000)
    different_classes_pairs = vectors[idx].reshape((5000, 2, len(vectors[0])))
    return one_class_pairs, different_classes_pairs


def plot_angles_dist(one_class_angles, different_classes_angles):
    plt.hist(different_classes_angles, density=True, bins=100, alpha=0.8, color='teal', label='different classes')
    plt.hist(one_class_angles, density=True, bins=100, alpha=0.8, color='darkred', label='one class')
    plt.title('Density function of angles between points')
    plt.legend(title='angles from:')
    plt.savefig('out/q3')
    plt.show()


def angles_dist(X_train, y_train):
    one_class_pairs, different_classes_pairs = sample(X_train, y_train)

    # compute angles between each pair
    one_class_angles = np.array([compute_angle(pair) for pair in one_class_pairs])
    different_classes_angles = np.array([compute_angle(pair) for pair in different_classes_pairs])

    plot_angles_dist(one_class_angles, different_classes_angles)


if __name__ == '__main__':
    data, topic_tags = get_sentences("sentences_list_shuffled", return_topic_tags=True)
    # run_iterations_3_topics(data, topic_tags, num_iter=10)
    # run_iterations_2_steps(data, topic_tags, algo1=PCATopicModel, num_iter=10)
    data = data[topic_tags != 0]
    topic_tags = topic_tags[topic_tags != 0] - 1
    angles_dist(data, topic_tags)

