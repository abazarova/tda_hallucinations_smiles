import json

import numpy as np
import ripserplusplus as rpp
from tqdm import tqdm, trange


def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]


def calculate_barcode(point_cloud: np.ndarray):
    """
    Calculates the persistent homology barcode of a point cloud using Ripser.

    Args:
    point_cloud: A numpy array representing the point cloud.

    Returns:
    The barcode as a dictionary, where keys are homology dimensions and 
    values are lists of persistence intervals.
    """
    barcode = rpp.run(f"--dim 1 --format point-cloud", data=point_cloud)
    return barcode


if __name__=="__main__":
    for i in trange(6):
        point_clouds = np.load(f"point_clouds_{i}.npy")
        barcodes = []
        for point_cloud in tqdm(point_clouds):
            barcode = calculate_barcode(point_cloud)
            barcodes.append(barcode)

        with open(f"barcodes_{i}.json", "w") as f:
            json.dump(format_barcodes(barcodes), f)