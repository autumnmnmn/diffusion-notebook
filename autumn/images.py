import torch
from PIL import Image

def PILify(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.float()
    with torch.no_grad():
        images = vae.decode(latents).sample
    
    images_nrm = (images / 2 + 0.5).clamp(0, 1)
    images_np = images_nrm.detach().cpu().permute(0, 2, 3, 1).numpy()
    images_byte = (images_np * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images_byte]