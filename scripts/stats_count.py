from functools import partial
from typing import Callable

import networkx as nx
import numpy as np
from tqdm import tqdm, trange


def crop_matrix(matrix: np.ndarray, n_tokens: int) -> np.ndarray:
    """Return normalized submatrix of first n_tokens"""
    matrix = matrix[:n_tokens, :n_tokens]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def get_filtered_mat_list(adj_matrix: np.ndarray, thresholds: list, n_tokens: int):
    """
    Converts adjancency matrix with real weights into list of binary matrices.
    For each threshold, those weights of adjancency matrix, which are less than
    threshold, get "filtered out" (set to 0), remained weights are set to ones.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        n_tokens (int)

    Returns:
        filtered_matrices (list[np.array[int, int]])
    """
    cropped_matrix = crop_matrix(adj_matrix, n_tokens)
    filtered_matrices = [np.where(cropped_matrix >= th, 1, 0) for th in thresholds]

    return filtered_matrices


def adj_m_to_nx_list(adj_matrix: np.ndarray, thresholds_array: list, ntokens: int):
    """
    Converts adjancency matrix into list of unweighted digraphs, using filtering
    process from previous function.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        n_tokens (int)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_list(list[np.array[int, int]])

    """
    filt_mat_list = get_filtered_mat_list(adj_matrix, thresholds_array, ntokens)
    fn = partial(nx.from_numpy_array, create_using=nx.MultiDiGraph())
    nx_graphs_list = [fn(mat) for mat in filt_mat_list]

    return nx_graphs_list, filt_mat_list


def adj_ms_to_nx_lists(
    adj_matrices: list[np.ndarray],
    thresholds_array: list[float],
    n_tokens_list: list[int],
    verbose: bool = True,
):
    """
    Executes adj_m_to_nx_list for each matrix in adj_matrices array, arranges
    the results. If verbose==True, shows progress bar.

    Args:
        adj_matrix (np.array[float, float])
        thresholds_array (iterable[float])
        verbose (bool)

    Returns:
        nx_graphs_list (list[nx.MultiDiGraph])
        filt_mat_lists (list[list[np.array[int,int]]])
    """
    graph_lists, filtrations = [], []

    n_matrices = len(adj_matrices)
    for adj_matrix, n_tokens in tqdm(
        zip(adj_matrices, n_tokens_list),
        desc="Filtration graphs calculation",
        total=n_matrices,
        disable=not verbose,
    ):
        g_list, filt_mat_list = adj_m_to_nx_list(
            adj_matrix,
            thresholds_array,
            n_tokens,
        )
        graph_lists.append(g_list)
        filtrations.append(filt_mat_list)

    return graph_lists, filtrations


def count_stat(graph_list: list[nx.MultiDiGraph], func: Callable, cap: int = 500):
    """
    Calculates stat (topological feature), using the function, which returns a
    generator (for example, generator of simple cycles in the DiGraph).

    Args:
        graph_list (list[nx.MultiDiGraph])
        func (Callable)
        cap (int)

    Returns:
        stat_amount (int)
    """
    gen = func(graph_list)
    return sum(1 for _ in zip(range(cap), gen))


def count_weak_components(graph_list: list[nx.MultiDiGraph], cap: int = 500):
    return count_stat(graph_list, func=nx.weakly_connected_components, cap=cap)


def count_strong_components(graph_list: list[nx.MultiDiGraph], cap: int = 500):
    return count_stat(graph_list, func=nx.strongly_connected_components, cap=cap)


def count_simple_cycles(graph_list: list[nx.MultiDiGraph], cap: int = 500):
    return count_stat(graph_list, func=nx.simple_cycles, cap=cap)


def dim_connected_components(
    graph_lists: list[list[nx.MultiDiGraph]],
    strong: bool = False,
    cap: int = 500,
    verbose: bool = False,
):
    """
    Calculates amount of connected components for each graph in list
    of lists of digraphs. If strong==True, calculates strongly connected
    components, otherwise calculates weakly connected components.
    If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        strong (bool)
        verbose (bool)

    Returns:
        w_lists (list[list[int])
    """
    cc_lists = []

    n_graph_lists = len(graph_lists)

    fn = (
        partial(count_strong_components, cap=cap)
        if strong
        else partial(count_weak_components, cap=cap)
    )
    for graph_list in tqdm(
        graph_lists,
        desc="Connected cmpts calculation",
        total=n_graph_lists,
        disable=not verbose,
    ):
        cc_lists.append([fn(graph) for graph in graph_list])

    return cc_lists


def dim_simple_cycles(
    graph_lists: list[list[nx.MultiDiGraph]], cap: int = 500, verbose: bool = False
):
    """
    Calculates amount of simple cycles for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        c_lists (list[list[int])
    """
    c_lists = []
    n_graph_lists = len(graph_lists)

    for graph_list in tqdm(
        graph_lists,
        desc="Simple cycles calculation",
        total=n_graph_lists,
        disable=not verbose,
    ):
        c_lists.append([count_simple_cycles(graph, cap=cap) for graph in graph_list])
    return c_lists


def b0_b1(graph_lists: list[list[nx.MultiDiGraph]], verbose: bool = False):
    """
    Calculates first two Betti numbers for each graph in list of lists of
    digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        b0_lists (list[list[int])
        b1_lists (list[list[int])
    """
    b0_lists, b1_lists = [], []
    n_graph_lists = len(graph_lists)
    for graph_list in tqdm(
        graph_lists,
        desc="0th and 1st Betti numbers calculation",
        total=n_graph_lists,
        disable=not verbose,
    ):
        b0, b1 = [], []
        for graph in graph_list:
            graph = nx.Graph(graph.to_undirected())
            cc = nx.number_connected_components(g)
            e = graph.number_of_edges()
            v = graph.number_of_nodes()
            b0.append(cc)
            b1.append(e - v + cc)
        b0_lists.append(b0)
        b1_lists.append(b1)

    return b0_lists, b1_lists


def edges_f(graph_lists: list[list[nx.MultiDiGraph]], verbose: bool = False):
    """
    Calculates amount of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        e_lists (list[list[int])
    """
    edges_lists = []  # len == len(pos_w_graph_lists)
    n_graph_lists = len(graph_lists)
    for graph_list in tqdm(
        graph_lists,
        desc="Number of edges calculation",
        total=n_graph_lists,
        disable=not verbose,
    ):
        edges = [graph.number_of_edges() for graph in graph_list]
        edges_lists.append(edges)
    return edges_lists


def v_degree_f(graph_lists: list[list[nx.MultiDiGraph]], verbose: bool = False):
    """
    Calculates amount of edges for each graph in list
    of lists of digraphs. If verbose==True, shows progress bar.

    Args:
        graph_lists (list[list[nx.MultiDiGraph]])
        verbose (bool)

    Returns:
        v_lists (list[list[int])
    """
    v_lists = []  # len == len(pos_w_graph_lists)
    n_graph_lists = len(graph_lists)
    for graph_list in tqdm(
        graph_lists,
        desc="Average vertex degree calculation",
        total=n_graph_lists,
        disable=not verbose,
    ):
        v = []
        for graph in graph_list:
            vertices, degrees = zip(*graph.degree())
            sum_of_edges = sum(degrees) / float(len(vertices))
            v.append(sum_of_edges)
        v_lists.append(v)
    return v_lists


def count_top_stats(
    adj_matrices: np.ndarray,
    thresholds: list[float],
    ntokens_array: np.ndarray,
    stats_to_count={"s", "e", "c", "v", "b0b1"},
    stats_cap: int = 500,
    verbose: bool = True,
):
    """
    The main function for calculating topological invariants. Unites the
    functional of all functions above.

    Args:
        adj_matrices (np.array[float, float, float, float, float])
        thresholds_array (list[float])
        stats_to_count (str)
        function_for_v (function)
        stats_cap (int)
        verbose (bool)

    Returns:
        stats_tuple_lists_array (np.array[float, float, float, float, float])
    """
    stats_tuple_lists_array = []
    _, n_layers, n_heads, _, _ = adj_matrices.shape

    for layer_of_interest in trange(n_layers, disable=not verbose):
        stats_tuple_lists_array.append([])
        for head_of_interest in range(n_heads):
            adj_ms = adj_matrices[:, layer_of_interest, head_of_interest, :, :]
            g_lists, _ = adj_ms_to_nx_lists(
                adj_ms,
                thresholds_array=thresholds,
                ntokens_array=ntokens_array,
                verbose=False,
            )
            feat_lists = []
            if "s" in stats_to_count:
                feat_lists.append(
                    dim_connected_components(
                        g_lists, strong=True, verbose=False, cap=stats_cap
                    )
                )
            if "w" in stats_to_count:
                feat_lists.append(
                    dim_connected_components(
                        g_lists, strong=False, verbose=False, cap=stats_cap
                    )
                )
            if "e" in stats_to_count:
                feat_lists.append(edges_f(g_lists, verbose=False))
            if "v" in stats_to_count:
                feat_lists.append(v_degree_f(g_lists, verbose=False))
            if "c" in stats_to_count:
                feat_lists.append(
                    dim_simple_cycles(g_lists, verbose=False, cap=stats_cap)
                )
            if "b0b1" in stats_to_count:
                b0_lists, b1_lists = b0_b1(g_lists, verbose=False)
                feat_lists.append(b0_lists)
                feat_lists.append(b1_lists)
            stats_tuple_lists_array[-1].append(tuple(feat_lists))

    stats_tuple_lists_array = np.asarray(stats_tuple_lists_array, dtype=np.float16)
    return stats_tuple_lists_array
