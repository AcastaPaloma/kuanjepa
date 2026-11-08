import torch
from transformers.image_utils import load_image
from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

device = "mps"
vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(device).eval()
vjepa_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")


image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
pixel_values = vjepa_proc(image, return_tensors="pt").to(vjepa.device)["pixel_values_videos"]
pixel_values = pixel_values.repeat(1, 16, 1, 1, 1) # repeating image 16 times

with torch.no_grad():
    image_embeddings = vjepa.get_vision_features(pixel_values)    

print(image_embeddings.shape)
