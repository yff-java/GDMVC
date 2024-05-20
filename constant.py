dataset_list = [
    "100leaves",
    "coil20mv",
    "handwritten_2v",
    "msrcv1_6v",
    "orl",
    "yale",
    "mnist_usps",
    "fashion",
]

fixed_args = {
    # 训练无关参数
    "log_freq": 20,
    "cluster_freq": 10,
    # 训练有关，每个数据集一样的参数
    "learning_rate": 10**-3,
    "layers": [256, 64],
    "neighbor_init": 6,
    "neighbor_incr": 5,
    "pretrain_iter": 100,
    "pretrain_epoch": 10,
    "finetune_epoch": 100,
}

# lam_tr lam_con 是网格搜索得到的最佳参数
dataset_args_map = {
    "100leaves": {
        # 训练有关，每个数据集不一样的参数
        "neighbor_max": 16,
        "lam_tr": 0.1,
        "lam_con": 0.1,
        # fixed_k_list
        # 范围1-宽：1 - neighbor_max
        # 范围2-窄：neighbor_init - 自增的最后一个邻居值=min(邻居自增策略的最大邻居值=neighbor_init+(pretrain_epoch-1)*neighbor_incr=51, neighbor_max)
        # 范围3-综合选择(comprehensive)，不一定从neighbor_init开始，但是不超过neighbor_max（选择这种方式做实验）
        # 范围1：1 - 16
        "fixed_k_wide_list": [2, 4, 6, 8, 10, 12, 14, 16],
        # 范围2：6 - 16
        "fixed_k_narrow_list": [6, 8, 10, 12, 14, 16],
        "fixed_k_com_list": [2, 4, 6, 8, 10, 12, 14, 16],
    },
    "coil20mv": {
        "neighbor_max": 72,
        "lam_tr": 1,
        "lam_con": 1,
        # 范围1：1 - 72
        "fixed_k_wide_list": [9, 18, 27, 36, 45, 54, 63, 72],
        # 范围2：6 - 51
        "fixed_k_narrow_list": [6, 11, 24, 33, 42, 51],
        "fixed_k_com_list": [5, 10, 15, 20, 25, 30, 35, 40],
    },
    "handwritten_2v": {
        "neighbor_max": 200,
        "lam_tr": 0.001,
        "lam_con": 0.001,
        # 范围1：1 - 200
        "fixed_k_wide_list": [25, 50, 75, 100, 125, 150, 175, 200],
        # 范围2：6 - 51
        "fixed_k_narrow_list": [6, 15, 24, 33, 42, 51],
        "fixed_k_com_list": [5, 10, 15, 20, 25, 30, 35, 40],
    },
    "msrcv1_6v": {
        "neighbor_max": 30,
        "lam_tr": 0.01,
        "lam_con": 0.001,
        # 范围1：1 - 30
        "fixed_k_wide_list": [2, 6, 10, 14, 18, 22, 26, 30],
        # 范围2：6 - 30
        "fixed_k_narrow_list": [5, 10, 15, 20, 25, 30],
        "fixed_k_com_list": [3, 6, 9, 12, 15, 18, 21, 24],
    },
    "orl": {
        "neighbor_max": 10,
        "lam_tr": 1,
        "lam_con": 0.001,
        # 范围1：1 - 10
        "fixed_k_wide_list": [3, 4, 5, 6, 7, 8, 9, 10],
        # 范围2：6 - 10
        "fixed_k_narrow_list": [5, 6, 7, 8, 9, 10],
        "fixed_k_com_list": [3, 4, 5, 6, 7, 8, 9, 10],
    },
    "yale": {
        "neighbor_max": 11,
        "lam_tr": 1,
        "lam_con": 10,
        # 范围1：1 - 11
        "fixed_k_wide_list": [4, 5, 6, 7, 8, 9, 10, 11],
        # 范围2：6 - 11
        "fixed_k_narrow_list": [6, 7, 8, 9, 10, 11],
        "fixed_k_com_list": [3, 4, 5, 6, 7, 8, 9, 10],
    },
    "mnist_usps": {
        "neighbor_max": 500,
        "lam_tr": 0.01,
        "lam_con": 10,
        # 范围1：1 - 500
        "fixed_k_wide_list": [62, 125, 187, 250, 312, 375, 437, 500],
        # 范围2：6 - 51
        "fixed_k_narrow_list": [6, 15, 24, 33, 42, 51],
        "fixed_k_com_list": [5, 10, 15, 20, 25, 30, 35, 40],
    },
    "fashion": {
        "neighbor_max": 1000,
        "lam_tr": 0.1,
        "lam_con": 10,
        # 范围1：1 - 1000
        "fixed_k_wide_list": [62, 125, 187, 250, 312, 375, 437, 500],
        # 范围2：6 - 51
        "fixed_k_narrow_list": [6, 15, 24, 33, 42, 51],
        "fixed_k_com_list": [5, 10, 15, 20, 25, 30, 35, 40],
    },
}

xiaorong_dataset_args_map = {}
