import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances(x):
    """
    x: (B, N, D)
    returns: pairwise squared Euclidean distances (B, N, N)
    """
    x_inner = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
    x_square = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, N, 1)
    dists = x_square - 2 * x_inner + x_square.transpose(1, 2)
    return dists

def compute_pairwise_affinities(x, use_student_t=False, eps=1e-8):
    """
    x: (B, N, D)
    returns: (B, N, N) normalized affinities p_ij or q_ij
    """
    dists = pairwise_distances(x)  # (B, N, N)

    if use_student_t:
        # q_ij in t-SNE: 1 / (1 + d^2)
        sim = 1 / (1 + dists)
    else:
        # p_ij in t-SNE: exp(-d^2)
        sim = torch.exp(-dists)

    # zero out diagonal to remove self-affinity
    B, N, _ = sim.shape
    eye = torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    sim = sim * (1 - eye)

    # normalize
    sim_sum = torch.sum(sim, dim=(1, 2), keepdim=True)
    sim = sim / (sim_sum + eps)

    return sim

def kl_divergence_tsne(p, q, eps=1e-8):
    """
    KL(p || q) over batch: (B, N, N)
    """
    ratio = (p + eps) / (q + eps)
    kl = torch.sum(p * torch.log(ratio), dim=(1, 2))  # sum over N x N
    return kl.mean()  # average over batch


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        """
        features: (B, N, D) latent representations
        labels: (B, N) ground-truth labels
        """
        B, N, D = features.size()
        features = features.view(-1, D)        # (B*N, D)
        labels = labels.view(-1).long()        # (B*N,)

        centers_batch = self.centers[labels]   # (B*N, D)
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss

def umap_loss(z, y, a=1.0, b=1.0, repulsion_strength=1.0):
    """
    UMAP-inspired loss function.
    z: (B, N, D) - latent
    y: (B, N) - label
    """
    B, N, D = z.shape
    z = z.view(B * N, D)
    y = y.view(B * N)

    dists = torch.cdist(z, z, p=2)  # (BN, BN)
    same_class = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    diff_class = 1.0 - same_class

    # Similarity function from UMAP paper
    sim = 1 / (1 + a * dists**2)**b

    # Loss terms
    attract = -torch.log(sim + 1e-8) * same_class
    repel = sim * diff_class * repulsion_strength

    return (attract.sum() + repel.sum()) / (B * N)**2
