import logging
import os
import torch
import sys
from itertools import chain
from torch import nn

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "code"))

from data import load_dataset
from model import AdaGAEMV
from utils import update_graph, cluster_by_multi_ways


def train_base(
    dataset_name, args, is_finetune=True, is_fusion=True, is_graph=True, fixed_k=None
):
    """基础训练函数，其他训练函数均由该函数修改得到"""
    logging.info(f"dataset: {dataset_name}, args: {args}")
    logging.info(
        f"is_finetune: {is_finetune}, is_fusion: {is_fusion}, is_graph: {is_graph}"
    )

    fusion_kind = "pinjiezv_pingjunlv_lxz" if is_fusion else "pinjiezv"
    lam_tr = args["lam_tr"] if is_graph else 0

    logging.info("load data")
    X, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = [torch.from_numpy(x).float().to(device) for x in X]

    logging.info("compute similarity")
    neighbor_num = args["neighbor_init"] if fixed_k is None else fixed_k
    weights_mv, raw_weights_mv, laplacian_mv = update_graph(X, neighbor_num)

    # cluster_by_multi_ways(X, laplacian_mv, Y, n_cluster, fusion_kind="pinjiezv")

    logging.info("init model and optimizer")
    gae_model = AdaGAEMV(X, args["layers"], device)
    optimizer = torch.optim.Adam(
        chain(*[gae_model.gae_list[v].parameters() for v in range(n_view)]),
        lr=args["learning_rate"],
    )

    logging.info("start pretrain")
    for epoch in range(args["pretrain_epoch"]):
        logging.info(
            f"neighbor_num: {neighbor_num}, neighbor_max: {args['neighbor_max']}"
        )

        for i in range(args["pretrain_iter"]):
            embedding_list, recons_w_list = gae_model.forward(X, laplacian_mv)
            re_loss, tr_loss = gae_model.cal_loss(
                raw_weights_mv,
                recons_w_list,
                weights_mv,
                embedding_list,
                lam_tr,
            )
            loss = re_loss + lam_tr * tr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % args["log_freq"] == 0:
                logging.info(
                    f'Epoch[{epoch+1}/{args["pretrain_epoch"]}], Step [{i+1}/{args["pretrain_iter"]}], L: {loss.item():.4f}, Lre: {re_loss.item():.4f}, Ltr: {tr_loss.item():.8f}'
                )

        weights_mv, raw_weights_mv, laplacian_mv = update_graph(
            embedding_list, neighbor_num
        )
        if fixed_k is None:
            neighbor_num = min(
                neighbor_num + args["neighbor_incr"], args["neighbor_max"]
            )

        # cluster_by_multi_ways(embedding_list, laplacian_mv, Y, n_cluster)

    if is_finetune:
        logging.info("start fine tuning")
        mse_loss_func = nn.MSELoss()
        for epoch in range(args["finetune_epoch"]):
            embedding_list, recons_w_list = gae_model.forward(X, laplacian_mv)
            re_loss, tr_loss = gae_model.cal_loss(
                raw_weights_mv,
                recons_w_list,
                weights_mv,
                embedding_list,
                lam_tr,
            )
            con_loss = 0
            for vi in range(n_view):
                for vj in range(vi + 1, n_view):
                    con_loss += mse_loss_func(embedding_list[vi], embedding_list[vj])
            loss = re_loss + lam_tr * tr_loss + args["lam_con"] * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % args["log_freq"] == 0:
                logging.info(
                    f'Epoch[{epoch+1}/{args["finetune_epoch"]}], L: {loss.item():.4f}, Lre: {re_loss.item():.4f}, Ltr: {tr_loss.item():.4f}, Lcon: {con_loss.item():.4f}'
                )
            # if (epoch + 1) % 10 == 0:
            #     cluster_by_multi_ways(embedding_list, laplacian_mv, Y, n_cluster)

    results = cluster_by_multi_ways(
        embedding_list, laplacian_mv, Y, n_cluster, count=10, fusion_kind=fusion_kind
    )
    return results
