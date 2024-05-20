import random
import logging
import os
import time
import torch
import numpy as np

# from metrics import cluster
# from metrics_old import cluster_mine as cluster
from metrics import cluster as cluster

"""
模型相关的工具函数
"""


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    # top_k是每个样本和它最远邻居的距离
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    # sum_top_k是每个样本和它前num_neighbors个邻居的距离之和
    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))

    sorted_distances = None
    torch.cuda.empty_cache()
    # T[i,j] 表示第 i 个数据点与第 j 个数据点之间的接近程度（权重），离得越近，T值越大
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    # weights相当于对T做归一化
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


def update_graph(embedding_mv, num_neighbors):
    with torch.no_grad():
        weights_mv, raw_weights_mv, laplacian_mv = [], [], []
        for v in range(len(embedding_mv)):
            x = embedding_mv[v]
            weights, raw_weights = cal_weights_via_CAN(x.t(), num_neighbors)
            Laplacian = get_Laplacian_from_weights(weights)
            # Laplacian = Laplacian.to_sparse()
            weights_mv.append(weights)
            raw_weights_mv.append(raw_weights)
            laplacian_mv.append(Laplacian)
        return weights_mv, raw_weights_mv, laplacian_mv


def cluster_by_multi_ways(
    z_list,
    L_list,
    Y,
    n_cluster,
    count=1,
    fusion_kind="pinjiezv_pingjunlv_lxz",
    view_index=0,
):
    if fusion_kind == "pinjiezv":
        z = z_list[view_index].detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=f"X{view_index}")

        z = torch.hstack(z_list).detach().cpu().numpy()
        results = cluster(n_cluster, z, Y, count=count, desc=fusion_kind)
        return results

    elif fusion_kind == "pinjiezv_pingjunlv_lxz":
        new_z = torch.matmul(L_list[view_index], z_list[view_index])
        new_z = new_z.detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=f"X{view_index}")

        z = torch.hstack(z_list)
        L = torch.mean(torch.stack(L_list), dim=0)
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        results = cluster(n_cluster, new_z, Y, count=count, desc=fusion_kind)
        return results

    else:
        raise ValueError("fusion_kind error")


"""
模型无关的工具函数
"""


def train_wrapper(train_func, dataset_name, args, seed=1234, gpu_id=0):
    logging.info(f"set random seed: {seed}, gpu_id: {gpu_id}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()
    try:
        logging.info(
            f"train start: train_func: {train_func.__name__}, dataset: {dataset_name}, args: {args}"
        )
        results = train_func(dataset_name, args)
        logging.info("train end")
    except Exception as e:
        logging.exception(f"Exception occurred: {str(e)}")
    logging.info(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return results


def prepare_log(log_kind=1, log_dir=None, log_file_name=None):
    # log_kind:1表示输出到控制台，2表示输出到文件，3表示同时输出到控制台和文件

    log_format = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    log_level = logging.INFO
    date_format = "%Y-%m-%d %H:%M:%S"

    if log_kind == 1:
        logging.basicConfig(format=log_format, level=log_level, datefmt=date_format)
    elif log_kind == 2:
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log_path = f"{log_dir}/{log_file_name}-{cur_time}.log"
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format=log_format,
            level=log_level,
            datefmt=date_format,
        )
    elif log_kind == 3:
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        logging.basicConfig(format=log_format, level=log_level, datefmt=date_format)
        cur_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
        log_path = f"{log_dir}/{log_file_name}-{cur_time}.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger("").addHandler(file_handler)
        return log_path
