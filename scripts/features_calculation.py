import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool, Process, Queue
from pathlib import Path

import dist2patterns_count
import numpy as np
import ripser_count
import stats_count
from tqdm import tqdm, trange

os.chdir("/app")

RIPSER_FEATURES = [
    "h0_s",
    "h0_e",
    "h0_t_d",
    "h0_n_d_m_t0.75",
    "h0_n_d_m_t0.5",
    "h0_n_d_l_t0.25",
    "h1_t_b",
    "h1_n_b_m_t0.25",
    "h1_n_b_l_t0.95",
    "h1_n_b_l_t0.70",
    "h1_s",
    "h1_e",
    "h1_v",
    "h1_nb",
]


def split_data(adj_matrices: list, ntokens_array: list, num_of_workers: int = 20):
    split_adj_matrices = np.array_split(adj_matrices, num_of_workers)
    split_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all(
        [len(m) == len(n) for m, n in zip(split_adj_matrices, split_ntokens)]
    ), "Split is not valid!"
    return zip(split_adj_matrices, split_ntokens)


def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
    queue.close()
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=["stats", "barcodes", "distances"], default="stats"
    )
    parser.add_argument(
        "--data_path", type=str, default="assets/attention_maps/llama-2-7b-chat"
    )
    parser.add_argument("--save_path", type=str, default="assets/tda_features")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_dump", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--lower_bound", type=float, default=0.001)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_name = data_path.name
    attn_mx_filename = lambda i: data_path / f"pt_{i}/attn_matrices.npz"
    ntokens_filename = lambda i: data_path / f"pt_{i}/tokens_count.json"
    num_files = len(os.listdir(data_path))

    save_path = Path(args.save_path) / model_name
    save_path.mkdir(exist_ok=True)

    if args.task == "stats":
        stats_names = "s_e_v_c_b0b1"
        stats_cap = args.cap

        thresholds = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]

        num_of_workers = args.n_workers
        pool = Pool(num_of_workers)

        stats_features, keys = [], []
        for i in trange(num_files):
            attn_matrices = np.load(attn_mx_filename(i))

            with open(ntokens_filename(i), "r") as f:
                ntokens = json.load(f)

            mx_list, ntokens_list = [], []
            keys_pt = list(attn_matrices.keys())
            for key in keys_pt:
                mx_list.append(attn_matrices[key])
                ntokens_list.append(ntokens[key])
            keys += keys_pt

            split = split_data(
                np.asarray(mx_list),
                np.asarray(ntokens_list),
                num_of_workers=min(num_of_workers, len(mx_list)),
            )
            args = [
                (mxs, thresholds, tokens, stats_names.split("_"), stats_cap)
                for mxs, tokens in split
            ]
            stats_features_ = pool.starmap(stats_count.count_top_stats, args)
            stats_features.append(np.concatenate([_ for _ in stats_features_], axis=3))

        stats_features = np.concatenate(stats_features, axis=3)
        stats_features_dict = dict(
            zip(keys, stats_features.transpose(3, 0, 1, 2, 4))
        )  # instance x layer x head x features x thresholds
        np.savez_compressed(save_path / "stats_features", **stats_features_dict)

    elif args.task == "barcodes":
        queue = Queue()
        number_of_splits = 2
        for i in trange(num_files, desc="Barcodes calculation"):
            attn_matrices = np.load(attn_mx_filename(i))

            with open(ntokens_filename(i), "r") as f:
                ntokens = json.load(f)

            mx_list, ntokens_list = [], []

            for key in attn_matrices.keys():
                mx_list.append(attn_matrices[key])
                ntokens_list.append(ntokens[key])

            barcodes = defaultdict(list)
            split = split_data(
                np.asarray(mx_list), np.asarray(ntokens_list), number_of_splits
            )
            for matrices, ntokens in tqdm(split, leave=False):
                p = Process(
                    target=subprocess_wrap,
                    args=(
                        queue,
                        ripser_count.get_only_barcodes,
                        (matrices, ntokens, args.dim, args.lower_bound),
                    ),
                )
                p.start()
                barcodes_part = queue.get()
                p.join()
                p.close()

                barcodes = ripser_count.unite_barcodes(barcodes, barcodes_part)

            ripser_count.save_barcodes(barcodes, save_path / f"barcodes_{i}.json")

        features_array = []

        for i in trange(num_files, desc="Calculating ripser++ features"):
            with open(save_path / f"barcodes_{i}.json", "r") as f:
                barcodes = json.load(f)
            print(
                f"Barcodes loaded from: {args.save_path}/barcodes_{i}.json", flush=True
            )
            features_part = []
            for layer in barcodes:
                features_layer = []
                for head in barcodes[layer]:
                    ref_barcodes = ripser_count.reformat_barcodes(barcodes[layer][head])
                    features = ripser_count.count_ripser_features(
                        ref_barcodes, RIPSER_FEATURES
                    )
                    features_layer.append(features)
                features_part.append(features_layer)
            features_array.append(np.asarray(features_part))

        features = np.concatenate(features_array, axis=2)
        np.save(save_path / "ripser_features", features)

    else:
        pool = Pool(args.n_workers)
        feature_names = ["self", "beginning", "prev", "next", "comma", "dot"]
        features_array, keys = [], []

        for i in trange(num_files):
            attn_matrices = np.load(attn_mx_filename(i))

            with open(ntokens_filename(i), "r") as f:
                ntokens = json.load(f)

            with open(data_path.parent / f"input_ids_{model_name}.json", "r") as f:
                input_ids = json.load(f)

            mx_list, ntokens_list, input_ids_list = [], [], []
            keys_pt = list(attn_matrices.keys())
            for key in attn_matrices.keys():
                mx_list.append(attn_matrices[key])
                ntokens_list.append(ntokens[key])
                input_ids_list.append(input_ids[key])

            n_matrices = len(mx_list)
            split = np.array_split(np.arange(n_matrices), min(args.n_workers, n_matrices))
            ids_split = [np.asarray(input_ids_list)[idx] for idx in split]
            data_split = [np.asarray(mx_list)[idx] for idx in split]
            func_args = [
                (m, feature_names, list_of_ids)
                for m, list_of_ids in zip(data_split, ids_split)
            ]

            features_array_part = pool.starmap(
                dist2patterns_count.calculate_features_t, func_args
            )
            features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))

        features = np.concatenate(features_array, axis=3)
        np.save(save_path / "template_features", features)
