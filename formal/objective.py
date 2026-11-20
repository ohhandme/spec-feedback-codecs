"""Batched CE over all legal windows. Projection fitting uses only the train partition."""

import torch
from torch.nn import functional as F

from raw_data import require


def features(batch, table):
    with torch.no_grad():
        local = batch["hidden"][:, 0]
        has_rejection = batch["rejected"].ge(0)[:, None]
        rejected = table(batch["rejected"].clamp_min(0))
        correction = table(batch["correction"])
        # The C repair falls back to local hidden on a full offer. The wire uses zero.
        local_rejected = torch.where(has_rejection, rejected, local)
        wire_rejected = torch.where(has_rejection, rejected, 0.0)
        vector = torch.cat(
            (
                batch["target"].float(),
                wire_rejected.float(),
                correction.float(),
                batch["accepted"][:, None].float() / 8,
            ),
            dim=-1,
        )
        return local, local_rejected, correction, vector


def message_control(codec, vector):
    if codec.codec_type == "grouped_vq8":
        scores = codec.encoder(vector).reshape(-1, 8, 256)
        probs = scores.softmax(-1)
        ids = scores.argmax(-1)
        if codec.training:
            hard = F.one_hot(ids, 256).to(probs)
            return torch.einsum("bgc,gcd->bgd", hard + (probs - probs.detach()), codec.codebook).flatten(1)
        return codec.codebook[torch.arange(8, device=vector.device), ids].flatten(1)
    units = codec.normalized(codec.project(vector)).clamp(-127, 127)
    quantized = units.detach().round() + (units - units.detach())
    return codec.decoder((quantized * codec.scale).to(codec.decoder.weight.dtype))


def logits(head, table, batch):
    local, rejected, correction, vector = features(batch, table)
    args = (local, rejected, correction, batch["accepted"])
    base = head.base
    codes = base.core.local_sample_st(*args) if base.training else base.core.local_encode(*args)
    control = sum(base.core.repair._lookup(codes[:, i], len(local)) for i in range(2))
    local_fp = local.to(base.local_hidden_lift[0].weight.dtype)
    control = control + base.hidden_projection(base.local_hidden_lift(local_fp))
    control = control + base.top32_projection(base.local_top32_surrogate(local_fp))
    control = control + message_control(head.message, vector)
    return base.core.repair.repair_from_control(
        batch["hidden"], control, torch.arange(1, 9, device=local.device)
    )


def losses(head, table, batch):
    z = logits(head, table, batch)
    y = batch["labels"]
    valid = y.ne(-100)
    per_token = F.cross_entropy(z.float().flatten(0, 1), y.flatten(), reduction="none").reshape_as(y)
    values = per_token.sum(-1) / valid.sum(-1)
    require(values.isfinite().all(), "nonfinite CE")
    return values


def take(data, ids, device):
    return {k: v[ids].to(device) for k, v in data.items()}


@torch.no_grad()
def fit_pca(codec, data, table, split, chunk=512):
    require(split == "train" and not codec.fitted.item(), "PCA requires a fresh train-only fit")
    device = codec.mean.device
    count = len(data["accepted"])
    require(count >= 9 and codec.width >= 8, "insufficient PCA data")

    def blocks():
        for off in range(0, count, chunk):
            batch = take(data, slice(off, off + chunk), device)
            yield features(batch, table)[-1]

    mean = torch.zeros(codec.width, dtype=torch.float64, device=device)
    for vector in blocks():
        mean += vector.double().sum(0)
    mean /= count
    gram = mean.new_zeros(codec.width, codec.width)
    for vector in blocks():
        centered = vector.double() - mean
        gram.addmm_(centered.T, centered)
    # Solve the small feature-space symmetric problem. Never allocate an N x N matrix.
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    axes = eigenvectors[:, -8:].T.flip(0)
    require(
        eigenvalues[-8] > eigenvalues[-1] * max(count, codec.width) * torch.finfo(mean.dtype).eps,
        "insufficient numerical PCA rank",
    )
    signs = axes.gather(1, axes.abs().argmax(-1, keepdim=True)).sign()
    codec.mean.copy_(mean)
    codec.projection.copy_(axes * signs)
    peak = torch.zeros_like(codec.scale)
    for vector in blocks():
        peak = torch.maximum(peak, F.linear(vector - codec.mean, codec.projection).abs().amax(0))
    codec.scale.copy_((peak / 127).clamp_min(1e-8))
    codec.fitted.fill_(True)
    return {
        "rows": count,
        "solver": "FP64 centered Gram/eigh, sign by max absolute loading",
        "effective_rank": int(
            (eigenvalues > eigenvalues[-1] * max(count, codec.width) * torch.finfo(mean.dtype).eps).sum()
        ),
        "top8_eigenvalues": eigenvalues[-8:].flip(0).tolist(),
        "scale": codec.scale.tolist(),
    }
