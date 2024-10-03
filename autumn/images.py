import torch
from PIL import Image
from IPython.display import display
import numpy as np

def pilify(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.to(vae.dtype)
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = images.detach().mul_(127.5).add_(127.5).clamp_(0,255).round()
    #return [images]
    images = images.permute(0,2,3,1).cpu().numpy().astype("uint8")
    return [Image.fromarray(image) for image in images]
    

def PILify(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.to(vae.dtype)
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

def save_raw_latents(latents):
    lmin = latents.min()
    l = latents - lmin
    lmax = latents.max()
    l = latents / lmax
    l = l.float() * 127.5 + 127.5
    l = l.detach().cpu().numpy()
    l = l.round().astype("uint8")
    
    ims = []
    
    for lat in l:
        row1 = np.concatenate([lat[0], lat[1]])
        row2 = np.concatenate([lat[2], lat[3]])
        grid = np.concatenate([row1, row2], axis=1)
        #for channel in lat:
        im = Image.fromarray(grid)
        im = im.resize(size=(grid.shape[1]*4, grid.shape[0]*4), resample=Image.NEAREST)
        ims += [im]
    
    for im in ims:
        im.save("out/tmp_raw_latents.bmp")

approximation_matrix = [
    [0.85, 0.85, 0.6], # seems to be mainly value
    [-0.35, 0.2, 0.5], # mainly blue? maybe a little green, def not red
    [0.15, 0.15, 0], # yellow. but mainly encoding texture not color, i think
    [0.15, -0.35, -0.35] # inverted value? but also red
]

def save_approx_decode(latents, index):
    lmin = latents.min()
    l = latents - lmin
    lmax = latents.max()
    l = latents / lmax
    l = l.float().mul_(0.5).add_(0.5)
    ims = []
    for lat in l:
        apx_mat = torch.tensor(approximation_matrix).to("cuda")
        approx_decode = torch.einsum("...lhw,lr -> ...rhw", lat, apx_mat).mul_(255).round()
        #lat -= lat.min()
        #lat /= lat.max()
        im_data = approx_decode.permute(1,2,0).detach().cpu().numpy().astype("uint8")
        #im_data = im_data.round().astype("uint8")
        im = Image.fromarray(im_data).resize(size=(im_data.shape[1]*8,im_data.shape[0]*8), resample=Image.NEAREST)
        ims += [im]

    #clear_output()
    for im in ims:
        #im.save(f"out/tmp_approx_decode/{index:06d}.bmp")
        im.save(f"out/tmp_approx_decode.bmp")
        #display(im)