import logging
import os

from train_base import train_base
from utils import prepare_log, train_wrapper
from constant import dataset_list, fixed_args, dataset_args_map


def train_duibi(dataset_name, args):
    return train_base(dataset_name, args)


if __name__ == "__main__":
    train_kind = "train_duibi"
    view_index = 0

    log_dir = os.path.join(os.path.dirname(__file__), "output-log")
    log_file_name = f"{train_kind}"
    prepare_log(3, log_dir, log_file_name)

    recorder = [["dataset", "results"]]
    for dataset_index in range(len(dataset_list)):
        dataset_name = dataset_list[dataset_index]
        logging.info(
            f"duibi progress: dataset_name[{dataset_index+1}/{len(dataset_list)}]"
        )

        args = dataset_args_map[dataset_name]
        args.update(fixed_args)
        results = train_wrapper(train_duibi, dataset_name, args, gpu_id=1)
        recorder.append([dataset_name, results])
        # break

    for row in recorder:
        logging.info(row)
