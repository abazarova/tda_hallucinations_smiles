import json

import numpy as np
from ripser_count import count_ripser_features, reformat_barcodes
from tqdm import trange

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

if __name__=="__main__":
    for i in trange(6):
        point_clouds = []

        with open(f"/app/assets/point_clouds/barcodes_{i}.json", "r") as f:
            barcodes = json.load(f)
        
        ref_barcodes = reformat_barcodes(barcodes)
        features = count_ripser_features(ref_barcodes, RIPSER_FEATURES)
        np.save(f"features_{i}", features)
