
import time
import os
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise ValueError(f"{path} looks like an npz. Please load arrays explicitly.")
    return arr


def bow_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize counts per cell to sum to 1 (bag-of-words style)."""
    denom = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / denom


class TwoViewCountDataset(Dataset):
    def __init__(self, gene_counts: np.ndarray, iso_counts: np.ndarray):
        assert gene_counts.shape[0] == iso_counts.shape[0], "gene/iso must have same n_cells"
        self.g = gene_counts.astype(np.float32)
        self.i = iso_counts.astype(np.float32)

    def __len__(self):
        return self.g.shape[0]

    def __getitem__(self, idx):
        return self.g[idx], self.i[idx]


def group_log_softmax(logits: torch.Tensor, group_ids: torch.Tensor, n_groups: int, eps: float = 1e-12):
    """
    logits: (B, I)
    group_ids: (I,) values in [0, n_groups-1]
    returns log_probs: (B, I) where sum_{iso in gene} exp(log_probs)=1
    """
    B, I = logits.shape
    device = logits.device
    gid = group_ids.view(1, I).expand(B, I)  # (B, I)

    group_max = torch.full((B, n_groups), -1e9, device=device, dtype=logits.dtype)
    group_max.scatter_reduce_(dim=1, index=gid, src=logits, reduce="amax", include_self=True)

    centered = logits - group_max.gather(1, gid)
    exp_centered = torch.exp(centered)

    group_sumexp = torch.zeros((B, n_groups), device=device, dtype=logits.dtype)
    group_sumexp.index_add_(dim=1, index=group_ids, source=exp_centered)  # uses I-length index, adds column-wise

    log_denom = torch.log(group_sumexp.clamp_min(eps)).gather(1, gid)
    log_probs = centered - log_denom
    return log_probs


class IsoGeneTopicVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        iso2gene: np.ndarray,
        n_topics: int = 30,
        hid: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_isoforms = n_isoforms
        self.n_topics = n_topics

        iso2gene_t = torch.from_numpy(iso2gene.astype(np.int64))
        assert iso2gene_t.shape[0] == n_isoforms, "iso2gene length must match n_isoforms"
        assert int(iso2gene_t.min()) >= 0 and int(iso2gene_t.max()) < n_genes, \
            "iso2gene must be in [0, n_genes-1]"
        self.register_buffer("iso2gene", iso2gene_t)
        self.n_groups = n_genes  # group per gene

        self.enc_gene = nn.Sequential(
            nn.Linear(n_genes, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.enc_iso = nn.Sequential(
            nn.Linear(n_isoforms, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )


        self.enc_fuse = nn.Sequential(
            nn.Linear(2 * hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


        self.mu = nn.Linear(hid, n_topics)
        self.logvar = nn.Linear(hid, n_topics)

        self.beta_gene = nn.Parameter(torch.zeros(n_topics, n_genes))
        self.bias_gene = nn.Parameter(torch.zeros(n_genes))

        self.beta_iso = nn.Parameter(torch.zeros(n_topics, n_isoforms))
        self.bias_iso = nn.Parameter(torch.zeros(n_isoforms))

        nn.init.xavier_uniform_(self.beta_gene)
        nn.init.xavier_uniform_(self.beta_iso)

    def encode(self, gene_counts: torch.Tensor, iso_counts: torch.Tensor):
        g = bow_normalize(gene_counts)
        i = bow_normalize(iso_counts)
        hg = self.enc_gene(g)
        hi = self.enc_iso(i)
        h = self.enc_fuse(torch.cat([hg, hi], dim=1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, gene_counts: torch.Tensor, iso_counts: torch.Tensor):
        mu, logvar = self.encode(gene_counts, iso_counts)
        z = self.reparam(mu, logvar)
        theta = F.softmax(z, dim=1)  # (B, K) topic proportion

        gene_logits = theta @ self.beta_gene + self.bias_gene  # (B, G)
        logp_gene = F.log_softmax(gene_logits, dim=1)          # (B, G)

        iso_logits = theta @ self.beta_iso + self.bias_iso     # (B, I)
        logp_iso = group_log_softmax(iso_logits, self.iso2gene, self.n_groups)  # (B, I)

        return mu, logvar, theta, logp_gene, logp_iso

    @staticmethod
    def kl_normal(mu: torch.Tensor, logvar: torch.Tensor):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)


def train(
    model: IsoGeneTopicVAE,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    alpha_iso: float = 1.0,
    beta_kl_max: float = 1.0,
    kl_anneal_epochs: int = 50,
    grad_clip: float = 5.0,
    log_every: int = 10,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(1, epochs + 1):
        if kl_anneal_epochs <= 0:
            beta_kl = beta_kl_max
        else:
            beta_kl = beta_kl_max * min(1.0, ep / float(kl_anneal_epochs))

        total_loss = 0.0
        total_llg = 0.0
        total_lli = 0.0
        total_kl = 0.0
        n = 0

        for gene_x, iso_x in loader:
            gene_x = gene_x.to(device)
            iso_x = iso_x.to(device)

            mu, logvar, __, logp_gene, logp_iso = model(gene_x, iso_x)

            ll_gene = torch.sum(gene_x * logp_gene, dim=1)  # (B,)
            ll_iso = torch.sum(iso_x * logp_iso, dim=1)     # (B,)

            kl = model.kl_normal(mu, logvar)                # (B,)

            loss = -(ll_gene + alpha_iso * ll_iso) + beta_kl * kl
            loss = loss.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bs = gene_x.shape[0]
            n += bs
            total_loss += float(loss.item()) * bs
            total_llg += float(ll_gene.mean().item()) * bs
            total_lli += float(ll_iso.mean().item()) * bs
            total_kl += float(kl.mean().item()) * bs

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            print(
                f"[epoch {ep:03d}] "
                f"loss={total_loss/n:.4f}  "
                f"ll_gene={total_llg/n:.4f}  ll_iso={total_lli/n:.4f}  "
                f"kl={total_kl/n:.4f}  beta_kl={beta_kl:.3f}"
            )


@torch.no_grad()
def infer_theta(model: IsoGeneTopicVAE, loader: DataLoader, device: torch.device):
    model.eval()
    mus = []
    thetas=[]
    for gene_x, iso_x in loader:
        gene_x = gene_x.to(device)
        iso_x = iso_x.to(device)
        mu, __ = model.encode(gene_x, iso_x)
        theta_det = torch.softmax(mu, dim=1)
        thetas.append(theta_det.cpu().numpy())
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0),np.concatenate(thetas, axis=0)


def validate_inputs(gene, iso, iso2gene):
    for name, values in (("gene_counts", gene), ("iso_counts", iso)):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real numeric counts")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite, nonnegative counts")
    if gene.shape[0] != iso.shape[0]:
        raise ValueError("gene_counts and iso_counts must have the same number of cells")
    if iso2gene.ndim != 1 or iso2gene.shape[0] != iso.shape[1]:
        raise ValueError("iso2gene must have one entry per isoform")
    if not np.issubdtype(iso2gene.dtype, np.integer):
        raise ValueError("iso2gene must contain integer gene indices")
    if (iso2gene < 0).any() or (iso2gene >= gene.shape[1]).any():
        raise ValueError("iso2gene indices must be in [0, n_genes-1]")

def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)

    gene = load_npy(args.gene_counts)
    iso = load_npy(args.iso_counts)
    iso2gene = load_npy(args.iso2gene)

    validate_inputs(gene, iso, iso2gene)
    os.makedirs(args.outdir, exist_ok=True)

    n_cells, n_genes = gene.shape
    _, n_iso = iso.shape
    print(f"Loaded: cells={n_cells}, genes={n_genes}, isoforms={n_iso}")

    ds = TwoViewCountDataset(gene, iso)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    inf_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested, but CUDA is unavailable")
    print(f"Device: {device}")

    start=time.time()

    model = IsoGeneTopicVAE(
        n_genes=n_genes,
        n_isoforms=n_iso,
        iso2gene=iso2gene,
        n_topics=args.n_topics,
        hid=args.hid,
        dropout=args.dropout,
    ).to(device)

    train(
        model=model,
        loader=loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        alpha_iso=args.alpha_iso,
        beta_kl_max=args.beta_kl_max,
        kl_anneal_epochs=args.kl_anneal_epochs,
        log_every=max(1, args.epochs // 20),
    )

    mu,theta = infer_theta(model, inf_loader, device)
    end=time.time()
    excute_time= end-start

    print("运行时间: ",excute_time)
    np.save(os.path.join(args.outdir, "embedding.npy"), mu)
    np.save(os.path.join(args.outdir, "theta.npy"), theta)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    print(f"Saved theta and model to: {args.outdir}")


    print("Done.")


def main(argv=None):
    from .cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    main()
