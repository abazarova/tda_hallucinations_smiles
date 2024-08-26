import json

import numpy as np
import ripserplusplus as rpp
from tqdm import tqdm, trange


def project_flatten_matrices(matrix1, matrix2):
    """
    Flattens two square matrices, projects the first onto the second, 
    and returns the projection.

    Args:
    matrix1: The first square matrix.
    matrix2: The second square matrix.

    Returns:
    The projection of the flattened matrix1 onto the flattened matrix2.
    """
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    # Calculate the projection
    projection = np.dot(flat_matrix1, flat_matrix2)

    return projection


def project_on_sequence(matrix, matrix_sequence):
    """
    Projects a given matrix onto each matrix in a sequence and returns 
    a sequence of projections.

    Args:
    matrix: The matrix to be projected.
    matrix_sequence: A sequence of matrices to project onto.

    Returns:
    A list of projections of the matrix onto each matrix in the sequence.
    """
    projections = []
    for m in matrix_sequence:
        projections.append(project_flatten_matrices(matrix, m)) 
    return np.array(projections)

def run_ripser_on_matrix(matrix, dim):
    barcode = rpp.run(f"--dim {dim} --format distance", data=matrix)
    return barcode

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


def attention_to_self(matrices: np.ndarray):
    """
    Calculates the distance between input matrices and identity matrix,
    which representes the attention to the same token.
    """
    _, _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return template_matrix


def attention_to_next_token(matrices):
    """
    Calculates the distance between input and E=(i, i+1) matrix,
    which representes the attention to the next token.
    """
    _, _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matrices.dtype), k=1)
    return template_matrix


def attention_to_prev_token(matrices):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to the previous token.
    """
    _, _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matrices.dtype), k=-1)
    return template_matrix


def attention_to_beginning(matrices):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to [CLS] token (beginning).
    """
    _, _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return template_matrix


def attention_to_ids(matrices, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix,
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """

    _, _, n, m = matrices.shape
    EPS = 1e-7
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    #     assert len(list_of_ids) == batch_size, f"List of ids length doesn't match the dimension of the matrix"
    template_matrix = np.zeros((n, n))
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        template_matrix[:, np.array(ids)] = 1.0
    return template_matrix


def all_templates(matrices, input_ids):
    self_pattern = attention_to_self(matrices)
    next_pattern = attention_to_next_token(matrices)
    prev_pattern = attention_to_prev_token(matrices)
    begin_pattern = attention_to_beginning(matrices)
    dot_pattern = attention_to_ids(matrices, input_ids, 869) +  attention_to_ids(matrices, input_ids, 29889)
    comma_pattern = attention_to_ids(matrices, input_ids, 1919) +  attention_to_ids(matrices, input_ids, 29892)

    return self_pattern, next_pattern, prev_pattern, begin_pattern, dot_pattern, comma_pattern

if __name__=="__main__":
    with open("/app/assets/attention_maps/input_ids_llama-2-7b-chat.json", "r") as f:
        input_ids_total = json.load(f)

    for i in trange(6):
        point_clouds = []
        matrices = np.load(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/attn_matrices.npz")
        with open(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/tokens_count.json", "r") as f:
            tokens_count = json.load(f)

        n_layers, n_heads = 2, 32
        for key in tqdm(list(matrices.keys())):
            ntokens = tokens_count[key]
            mxs = matrices[key][:, :, :ntokens, :ntokens]
            point_cloud = []
            templates = all_templates(mxs, input_ids_total[key])
            for l in range(n_layers):
                for j in range(n_heads):
                    mx = mxs[l][j]
                    projection = project_on_sequence(mx, templates)
                    point_cloud.append(projection)
                    
            point_clouds.append(np.stack(point_cloud))

        np.save(f"point_clouds_{i}", point_clouds)