import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as compute_nmi
from sklearn.metrics.cluster._supervised import check_clusterings


def get_cluster_result(features, n_clusters):
    km = KMeans(n_clusters=n_clusters, n_init=10)
    pred = km.fit_predict(features)
    return pred


def compute_acc(Y, Y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    from scipy.optimize import linear_sum_assignment

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size * 100


def compute_fscore(labels_true, labels_pred):
    # b3_precision_recall_fscore就是Fscore
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError("input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score


def cluster_one_time(features, labels, n_clusters):
    pred = get_cluster_result(features, n_clusters)
    labels = np.reshape(labels, np.shape(pred))
    if np.min(labels) == 1:
        labels -= 1

    nmi = compute_nmi(labels, pred) * 100
    acc = compute_acc(labels, pred)
    fscore = compute_fscore(labels, pred) * 100
    return round(nmi, 2), round(acc, 2), round(fscore, 2)


def cluster(n_clusters, features, labels, count=1, desc="cluster_mine"):
    nmi_array, acc_array, f1_array = [], [], []
    for _ in range(count):
        nmi, acc, fscore = cluster_one_time(features, labels, n_clusters)
        nmi_array.append(nmi)
        acc_array.append(acc)
        f1_array.append(fscore)
        logging.info(f"kmeans NMI={nmi}, ACC={acc}, Fscore={fscore}")

    nmi_avg, nmi_std = round(np.mean(nmi_array), 2), round(np.std(nmi_array), 2)
    acc_avg, acc_std = round(np.mean(acc_array), 2), round(np.std(acc_array), 2)
    f1_avg, f1_std = round(np.mean(f1_array), 2), round(np.std(f1_array), 2)
    logging.info(
        f"{desc} Kmeans({count} times average) NMI={nmi_avg:.2f}±{nmi_std:.2f}, ACC={acc_avg:.2f}±{acc_std:.2f}, fscore={f1_avg:.2f}±{f1_std:.2f}"
    )
    results=[nmi_avg, nmi_std, acc_avg, acc_std, f1_avg, f1_std]
    return results
