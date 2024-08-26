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


def attention_to_prompt_and_generation(matrices, n_prompt_tokens, n_output_tokens):
    _, _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix_prompt = np.zeros((n, n))
    template_matrix_prompt[n_prompt_tokens : n_output_tokens, :n_prompt_tokens] = 1

    template_matrix_generation = np.zeros((n, n))
    template_matrix_generation[n_prompt_tokens : n_output_tokens, n_prompt_tokens:n_output_tokens] = 1

    return template_matrix_prompt, template_matrix_generation


if __name__=="__main__":
    with open("/app/assets/attention_maps/input_ids_llama-2-7b-chat.json", "r") as f:
        input_ids_total = json.load(f)

    with open("/app/assets/attention_maps/prompt_lengths.json", "r") as f:
        ntokens_prompt = json.load(f)

    for i in trange(6):
        point_clouds = []
        matrices = np.load(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/attn_matrices.npz")
        with open(f"/app/assets/attention_maps/llama-2-7b-chat/pt_{i}/tokens_count.json", "r") as f:
            tokens_count = json.load(f)

        n_layers, n_heads = 2, 32
        for key in tqdm(list(matrices.keys())):
            ntokens = tokens_count[key]
            ntokens_prt = ntokens_prompt[key]
            mxs = matrices[key][:, :, :ntokens, :ntokens]
            point_cloud = []
            templates = attention_to_prompt_and_generation(mxs, ntokens_prt, ntokens)
            for l in range(n_layers):
                for j in range(n_heads):
                    mx = mxs[l][j]
                    projection = project_on_sequence(mx, templates)
                    point_cloud.append(projection)
                    
            point_clouds.append(np.stack(point_cloud))

        np.save(f"point_clouds_{i}_prompt", point_clouds)




