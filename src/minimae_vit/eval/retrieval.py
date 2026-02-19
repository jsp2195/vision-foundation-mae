import torch


def knn_retrieval(feats: torch.Tensor, queries: torch.Tensor, k: int = 5) -> torch.Tensor:
    sims = queries @ feats.t()
    return sims.topk(k, dim=1).indices
