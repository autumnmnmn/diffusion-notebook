import torch
import numpy as np
print("math reloaded")

def sigmoid(shift_x, shift_y, rate_h, range_v):
    def f_sigmoid(x):
        exp = np.exp((shift_x - x) / rate_h)
        sig = 1 / (1 + exp)
        return shift_y + range_v * (sig - 0.5)
    return f_sigmoid

def scale_f(f, scale_x, scale_y):
    def _f(x):
        return scale_y * f(x / scale_x)
    return _f

"""
distort: index of singular value [0,N] => strength of singular value (number, usually in [0.0, 1.0])

* makes wild assumptions about shape of tensor
"""
def svd_distort(tensor, distort):
    (U, S, Vh) = torch.linalg.svd(tensor)
    
    svd_mask = torch.ones_like(S)
    for b in range(len(S)):
        for r in range(len(S[b])):
            l = len(S[b][r])
            for i in range(l):
                svd_mask[b][r][i] = distort(i)
    
    return U @ torch.diag_embed(S * svd_mask) @ Vh


def svd_distort_embeddings(tensor, distort):
    out = torch.clone(tensor)
    
    for r in range(len(tensor)):
        (U, S, Vh) = torch.linalg.svd(tensor[r])
        distortion_mask = torch.ones_like(S)

        for i in range(len(distortion_mask)):
            distortion_mask[i] = distort(i)
        
        S_diag_expanded = torch.zeros_like(tensor[r])
        S_diag_expanded[:, :S.shape[0]] = torch.diag(S * distortion_mask)
        
        out[r] = U @ S_diag_expanded @ Vh
    
    return out

        #l = len(S[r])
        #for i in range(l):
            #svd_mask[r][i] = distort(i)
    
    #return U @ torch.diag_embed(S * svd_mask) @ Vh