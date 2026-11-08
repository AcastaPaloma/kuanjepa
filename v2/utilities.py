import torch
import torch.nn.functional as F

@torch.no_grad()
def embed_images(images, model, proc, chunk=32, T=2):
    """images: list of raw PIL RGB images (any size) -> (B, N, D) patch tokens, N = 256, D = 1024.

    The processor handles resize/crop to 256 and normalization, so pass the original images,
    not tensors that were already resized or normalized.
    The image is repeated to T frames. V-JEPA 2 groups frames into tubelets of 2, giving T/2
    time steps of 256 patches each; all steps see the same image, so they are averaged.
    Patches are NOT pooled: token n is grid cell (row n // 16, col n % 16)."""
    tubelet = getattr(model.config, "tubelet_size", 2)
    out = []
    for i in range(0, len(images), chunk):
        px = torch.cat([proc(im, return_tensors="pt")["pixel_values_videos"]
                        for im in images[i:i + chunk]])           # (b, 1, 3, 256, 256)
        px = px.repeat(1, T, 1, 1, 1).to(model.device)            # (b, T, 3, 256, 256)
        feats = model.get_vision_features(px)                     # (b, (T/tubelet) * 256, D)
        steps = max(1, T // tubelet)
        b, n, d = feats.shape
        out.append(feats.view(b, steps, n // steps, d).mean(1).cpu())   # (b, 256, D)
    return torch.cat(out)


def pool_2x2(tokens):
    b, n, d = tokens.shape                                        # (b, 256, D)
    return tokens.view(b, 8, 2, 8, 2, d).mean((2, 4)).reshape(b, 64, d)   # (b, 64, D)


@torch.no_grad()
def embed_texts(texts, model, batch_size=256):
    emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                       show_progress_bar=True).float().cpu()      # (M, 768)
    return F.layer_norm(emb, (emb.size(-1),))                     # (M, 768)


def sample_block(grid, scale, aspect):
    area = grid * grid * torch.empty(1).uniform_(*scale).item()
    ratio = torch.empty(1).uniform_(*aspect).item()
    h = max(1, min(grid, round((area * ratio) ** 0.5)))
    w = max(1, min(grid, round((area / ratio) ** 0.5)))
    top = torch.randint(0, grid - h + 1, (1,)).item()
    left = torch.randint(0, grid - w + 1, (1,)).item()
    return top, left, h, w


def sample_keep_ids(batch_size, grid=8, n_blocks=4, scale=(0.15, 0.2), aspect=(0.75, 1.5), min_keep=4):
    keeps = []
    for _ in range(batch_size):
        while True:
            visible = torch.ones(grid, grid, dtype=torch.bool)    # (grid, grid)
            for _ in range(n_blocks):
                top, left, h, w = sample_block(grid, scale, aspect)
                visible[top:top + h, left:left + w] = False
            ids = visible.flatten().nonzero().squeeze(1)          # (n_visible,)
            if len(ids) >= min_keep:
                break
        keeps.append(ids)
    K = max(len(ids) for ids in keeps)
    keep_ids = torch.zeros(batch_size, K, dtype=torch.long)       # (B, K)
    pad = torch.ones(batch_size, K, dtype=torch.bool)             # (B, K)
    for b, ids in enumerate(keeps):
        keep_ids[b, :len(ids)] = ids
        pad[b, :len(ids)] = False
    return keep_ids, pad


def recall_at_k(sims, correct, ks=(1, 5, 10)):
    top = sims.topk(max(ks), dim=1).indices                       # (Q, max_k)
    hit = correct.gather(1, top)                                  # (Q, max_k)
    return {k: hit[:, :k].any(1).float().mean().item() for k in ks}


def retrieval_metrics(img_emb, cap_emb, cap_owner):
    sims = F.normalize(img_emb, dim=-1) @ F.normalize(cap_emb, dim=-1).t()            # (N, M)
    images = torch.arange(img_emb.size(0), device=img_emb.device)                     # (N,)
    correct = cap_owner.to(img_emb.device)[None, :] == images[:, None]                # (N, M)
    i2t = recall_at_k(sims, correct)
    t2i = recall_at_k(sims.t(), correct.t())
    return i2t, t2i
