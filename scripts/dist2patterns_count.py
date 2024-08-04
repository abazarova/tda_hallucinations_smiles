import numpy as np


def matrix_distance(matrices: np.ndarray, template: np.ndarray, broadcast: bool = True):
    """
    Calculates the distance between the list of matrices and the template matrix.
    Args:

    -- matrices: np.array of shape (n_matrices, dim, dim)
    -- template: np.array of shape (dim, dim) if broadcast else (n_matrices, dim, dim)

    Returns:
    -- diff: np.array of shape (n_matrices, )
    """
    diff = np.linalg.norm(matrices - template, ord="fro", axis=(1, 2))
    div = np.linalg.norm(matrices, ord="fro", axis=(1, 2)) ** 2
    if broadcast:
        div += np.linalg.norm(template, ord="fro") ** 2
    else:
        div += np.linalg.norm(template, ord="fro", axis=(1, 2)) ** 2
    return diff / np.sqrt(div)


def attention_to_self(matrices: np.ndarray):
    """
    Calculates the distance between input matrices and identity matrix,
    which representes the attention to the same token.
    """
    _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return matrix_distance(matrices, template_matrix)


def attention_to_next_token(matrices):
    """
    Calculates the distance between input and E=(i, i+1) matrix,
    which representes the attention to the next token.
    """
    _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matrices.dtype), k=1)
    return matrix_distance(matrices, template_matrix)


def attention_to_prev_token(matrices):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to the previous token.
    """
    _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matrices.dtype), k=-1)
    return matrix_distance(matrices, template_matrix)


def attention_to_beginning(matrices):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to [CLS] token (beginning).
    """
    _, n, m = matrices.shape
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return matrix_distance(matrices, template_matrix)


def attention_to_ids(matrices, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix,
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """

    _, n, m = matrices.shape
    EPS = 1e-7
    assert (
        n == m
    ), f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    #     assert len(list_of_ids) == batch_size, f"List of ids length doesn't match the dimension of the matrix"
    template_matrix = np.zeros_like(matrices)
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        batch_ids, row_ids = zip(*ids)
        template_matrix[np.array(batch_ids), :, np.array(row_ids)] = 1.0
        template_matrix /= np.sum(template_matrix, axis=-1, keepdims=True) + EPS
    return matrix_distance(matrices, template_matrix, broadcast=False)


def count_template_features(
    matricies,
    feature_list=["self", "beginning", "prev", "next", "comma", "dot"],
    ids=None,
):
    features = []
    comma_id = 1010
    dot_id = 1012
    for feature in feature_list:
        if feature == "self":
            features.append(attention_to_self(matricies))
        elif feature == "beginning":
            features.append(attention_to_beginning(matricies))
        elif feature == "prev":
            features.append(attention_to_prev_token(matricies))
        elif feature == "next":
            features.append(attention_to_next_token(matricies))
        elif feature == "comma":
            features.append(attention_to_ids(matricies, ids, comma_id))
        elif feature == "dot":
            features.append(attention_to_ids(matricies, ids, dot_id))
    return np.array(features)


def calculate_features_t(adj_matricies, template_features, ids=None):
    """Calculate template features for adj_matricies"""
    features = []
    for layer in range(adj_matricies.shape[1]):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            lh_features = count_template_features(
                matricies, template_features, ids
            )  # samples X n_features
            features[-1].append(lh_features)
    return np.asarray(features)  # layer X head X n_features X samples
