import torch
from torch.nn.functional import conv2d as convolve

diffuse_kernel = torch.tensor([[[[0,1,0],[1,0,1],[0,1,0]]]], dtype=torch.float, device="cuda")
project_kernel_x = torch.tensor([[[[0,0,0],[-1,0,1],[0,0,0]]]], dtype=torch.float, device="cuda")
project_kernel_y = torch.tensor([[[[0,-1,0],[0,0,0],[0,1,0]]]], dtype=torch.float, device="cuda")

def continuous_boundary(field):
    field[0] = field[1]
    field[-1] = field[-2]
    field[:,0] = field[:,1]
    field[:,-1] = field[:,-2]
    field[(0,0,-1,-1),(0,-1,0,-1)] = 0.5 * (field[(0,0,-2,-2),(1,-2,0,-1)] + field[(1,1,-1,-1),(0,-1,1,-2)])

def opposed_v_boundary(field):
    field[0] = field[1]
    field[-1] = field[-2]
    field[:,0] = -field[:,1]
    field[:,-1] = -field[:,-2]
    field[(0,0,-1,-1),(0,-1,0,-1)] = 0.5 * (field[(0,0,-2,-2),(1,-2,0,-1)] + field[(1,1,-1,-1),(0,-1,1,-2)])

def opposed_h_boundary(field):
    field[0] = -field[1]
    field[-1] = -field[-2]
    field[:,0] = field[:,1]
    field[:,-1] = field[:,-2]
    field[(0,0,-1,-1),(0,-1,0,-1)] = 0.5 * (field[(0,0,-2,-2),(1,-2,0,-1)] + field[(1,1,-1,-1),(0,-1,1,-2)])


def diffuse(field, rate, set_boundary, dt, h, w):
    a = dt * rate
    result = torch.clone(field)
    if field.shape != (h, w):
        print("bad field shape in diffuse")
    for n in range(20):
        convolution = a * convolve(result.unsqueeze(0), diffuse_kernel, bias=None, padding=[0], stride=[1])[0]
        result[1:h-1,1:w-1] = field[1:h-1,1:w-1] + convolution
        result /= 1 + 4 * a
        set_boundary(result)
        #result = result * ~border_mask[0] + field * border_mask[0]
    return result

def advect(field, velocities, dt, h, w):
    dth, dtw = dt, dt
    inds_x = torch.arange(1,w-1).repeat(h-2,1).float()
    inds_y = torch.arange(1,h-1).repeat(w-2,1).t().float()
    inds_x += dtw * velocities[1,1:h-1,1:w-1]
    inds_y += dth * velocities[0,1:h-1,1:w-1]
    inds_x = inds_x.clamp(1.5, w - 2.5)
    inds_y = inds_y.clamp(1.5, h - 2.5)
    inds_x_i = inds_x.int()
    inds_y_i = inds_y.int()
    inds_x -= inds_x_i
    inds_y -= inds_y_i
    inds_x_inv = 1 - inds_x
    inds_y_inv = 1 - inds_y
    inds_x_i_next = inds_x_i + 1
    inds_y_i_next = inds_y_i + 1
    inds_x_all = torch.stack([inds_x_i, inds_x_i_next, inds_x_i, inds_x_i_next])
    inds_y_all = torch.stack([inds_y_i, inds_y_i, inds_y_i_next, inds_y_i_next])
    if field.shape[0] == 1:
        values = torch.cat([field[:,1:h-1,1:w-1] * inds_x_inv * inds_y_inv,
                            field[:,1:h-1,1:w-1] * inds_x * inds_y_inv,
                            field[:,1:h-1,1:w-1] * inds_x_inv * inds_y,
                            field[:,1:h-1,1:w-1] * inds_x * inds_y])
        res = torch.zeros_like(field[0])
        res.index_put_((inds_y_all, inds_x_all), values, accumulate=True)
        continuous_boundary(res)
        return res.unsqueeze(0)
    else:
        values = torch.stack([field[:,1:h-1,1:w-1] * inds_x_inv * inds_y_inv,
                              field[:,1:h-1,1:w-1] * inds_x * inds_y_inv,
                              field[:,1:h-1,1:w-1] * inds_x_inv * inds_y,
                              field[:,1:h-1,1:w-1] * inds_x * inds_y])
        res = torch.zeros_like(field)
        res[0].index_put_((inds_y_all, inds_x_all), values[:,0,:,:], accumulate=True)
        res[1].index_put_((inds_y_all, inds_x_all), values[:,1,:,:], accumulate=True)
        opposed_h_boundary(res[1])
        opposed_v_boundary(res[0])
        return res

def project(field, h, w):
    hx = -1# / w
    hy = -1# / h
    divergence = convolve(field[1].unsqueeze(0), project_kernel_x, bias=None, stride=[1], padding=[0])[0] * hx
    divergence += convolve(field[0].unsqueeze(0), project_kernel_y, bias=None, stride=[1], padding=[0])[0] * hy
    divergence *= 0.5
    continuous_boundary(divergence)
    p = torch.zeros_like(field[0])
    for i in range(40):
        p[1:h-1,1:w-1] = (divergence + convolve(p.unsqueeze(0), diffuse_kernel, bias=None, stride=[1], padding=[0])[0]) / 4
        continuous_boundary(p)
    field[1,1:h-1,1:w-1] += 0.5 * convolve(p.unsqueeze(0), project_kernel_x, bias=None, stride=[1], padding=[0])[0] / hx
    field[0,1:h-1,1:w-1] += 0.5 * convolve(p.unsqueeze(0), project_kernel_y, bias=None, stride=[1], padding=[0])[0] / hy
    opposed_h_boundary(field[1])
    opposed_v_boundary(field[0])
