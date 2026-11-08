import torch
from torchvision.transforms.functional import to_pil_image

@torch.no_grad()
def embed_images(images, model, proc, chunk=32, T=2):
    """images: (B, 3, 128, 128) tensor, uint8 [0,255] or float [0,1] -> (B, D)"""
    out = []
    for i in range(0, len(images), chunk):
        px = torch.cat([proc(to_pil_image(im.cpu()), return_tensors="pt")["pixel_values_videos"]
                        for im in images[i:i + chunk]])           # (b, 1, 3, 256, 256)
        px = px.repeat(1, T, 1, 1, 1).to(model.device)            # (b, T, 3, 256, 256)
        out.append(model.get_vision_features(px).cpu())
    out = torch.cat(out)
    return out.mean(1) if out.ndim == 3 else out

@torch.no_grad()
def embed_captions(captions, model):
    """captions: list of B lists of strings (any count per image) -> (B, D) unit norm,
    the mean of each image's caption embeddings, renormalized."""
    flat = [c for caps in captions for c in caps]
    emb = model.encode(flat, convert_to_tensor=True, normalize_embeddings=True)   # (sum n_i, D)
    per_img = torch.split(emb, [len(caps) for caps in captions])
    return torch.nn.functional.normalize(torch.stack([e.mean(0) for e in per_img]), p=2, dim=-1)
