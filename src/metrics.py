import numpy as np


def mrr(similarity_matrix):
    """
    Implementation adapted from:
     https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/downstream-tasks/code-search/run.py#L250

    Compute MRR, assuming that for ith text sample the relevant response is ith code sample
    :param similarity_matrix: torch.Tensor or np.ndarray, square matrix, (i, j)th position of which represents
        (possibly not normalized) similarity of ith text sample and jth code sample
    :return: mean reciprocal rank of the similarity matrix
    """
    assert len(similarity_matrix.shape) == 2, 'Only 2D tensors allowed in MRR'
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], 'Non-square matrix provided to MRR'
    if type(similarity_matrix) == np.ndarray:
        found = similarity_matrix.argsort(-1)[:, ::-1]
    else:
        found = similarity_matrix.argsort(-1, descending=True)
    ranks = []
    for i in range(found.shape[0]):
        rank = 1 + np.where(found[i] == i)[0][0]
        ranks.append(1 / rank if rank < 1000 else 0)
    return np.mean(ranks).astype(float)


if __name__ == '__main__':
    import torch

    sim = torch.tensor([[0, 1, 100], [100, 1, 0], [0, 1, 100]])
    # found = [[2, 1, 0], [0, 1, 2], [2, 1, 0]]
    # ranks = [3, 2, 1]
    assert np.allclose(mrr(sim), 11 / 18)

    sim = np.array([[0, 1, 100], [100, 1, 0], [0, 1, 100]])
    assert np.allclose(mrr(sim), 11 / 18)
