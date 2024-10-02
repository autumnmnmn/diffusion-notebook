import torch
from PIL import Image
from IPython.display import display

def pilify(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.float()
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = images.detach().mul_(127.5).add_(127.5).clamp_(0,255).round()
    #return [images]
    images = images.permute(0,2,3,1).cpu().numpy().astype("uint8")
    return [Image.fromarray(image) for image in images]
    

def PILify(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.float()
    with torch.no_grad():
        images = vae.decode(latents).sample
    
    images_nrm = (images / 2 + 0.5).clamp(0, 1)
    images_np = images_nrm.detach().cpu().permute(0, 2, 3, 1).numpy()
    images_byte = (images_np * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images_byte]

def mpilify(z):
    _z = torch.clone(z).clamp_(0,1).mul_(255).round()
    z_np = _z.unsqueeze(2).expand(-1, -1, 3).type(torch.uint8).cpu().numpy()
    return Image.fromarray(z_np)

def msave(x, f):
    mpilify(x).save(f"out/{f}.png")

def mshow(x):
    display(mpilify(x))