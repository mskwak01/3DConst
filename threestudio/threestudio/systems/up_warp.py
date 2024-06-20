
import torch
import torch.nn.functional as F
import shutil
import torchvision
import matplotlib.pyplot as plt

import nvdiffrast.torch as dr

def cond_noise_sampling(src_noise, level=3):
    
    B, C, H, W = src_noise.shape
    
    up_factor = 2 ** level
    
    upscaled_means = F.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')
    
    up_H = up_factor * H
    up_W = up_factor * W
    
    """
        1) Unconditionally sample a discrete Nk x Nk Gaussian sample
    """
    
    raw_rand = torch.randn(B, C, up_H, up_W)
    
    """
        2) Remove its mean from it
    """
    
    Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
    mean_removed_rand = raw_rand - Z_mean
    
    """
        3) Add the pixel value to it
    """

    up_noise = upscaled_means / up_factor + mean_removed_rand.to(src_noise.device)
    
    return up_noise


def plot_gaussians(input_values):
    # Convert input values to a tensor
    input_tensor = input_values
    
    # Calculate mean and standard deviation of input values
    mean_input = torch.mean(input_tensor)
    std_input = torch.std(input_tensor)
    
    # Define the range for the x-axis
    x = torch.linspace(-5, 5, 1000).to(input_values.device)
    
    # Calculate the standard Gaussian (mean=0, std=1)
    standard_gaussian = torch.exp(-0.5 * x**2) / torch.sqrt(2 * torch.tensor(torch.pi).to(input_values.device))
    
    # Calculate the Gaussian distribution for the input values
    input_gaussian = torch.exp(-0.5 * ((x - mean_input) / std_input)**2) / (std_input * torch.sqrt(2 * torch.tensor(torch.pi).to(input_values.device)))
    
    # Convert tensors to numpy arrays for plotting
    x_np = x.cpu().detach().numpy()
    standard_gaussian_np = standard_gaussian.cpu().detach().numpy()
    input_gaussian_np = input_gaussian.cpu().detach().numpy()
    
    # Plot the distributions
    plt.plot(x_np, standard_gaussian_np, color='black')
    plt.plot(x_np, input_gaussian_np, color='crimson')
    plt.xlabel('x')
    plt.ylabel('Probability Density') 
    plt.title('Gaussian Distributions')
    plt.ylim(0, 0.63)
    plt.tick_params(axis='y', labelsize=20)  
    plt.legend()
    plt.show()
    plt.savefig('/home/cvlab15/project/ines/3DConst/distribution.png')
    plt.close()


def get_noise_vertices(res=64):
    tr_W = res * 2 + 1
    tr_H = res * 2 + 1

    i, j = torch.meshgrid(
            torch.arange(tr_W, dtype=torch.int32),
            torch.arange(tr_H, dtype=torch.int32),
            indexing="ij",
        )

    mesh_idxs = torch.stack((i,j), dim=-1)
    reshaped_mesh_idxs = mesh_idxs.reshape(-1,2)
    
    front_tri_verts = torch.tensor([[0, 1, 1+tr_H], [0, tr_H, 1+tr_H], [tr_H, 1+tr_H, 1+2*tr_H], [tr_H, 2*tr_H, 1+2*tr_H]])
    per_tri_verts = torch.cat((front_tri_verts, front_tri_verts + 1),dim=0)
    
    width = torch.arange(0, tr_W - 1, 2)
    height = torch.arange(0, tr_H-1, 2) * (tr_W)
    width_l = torch.linspace(0, tr_W-2, tr_W-1)
    
    start_idxs = (width[None,...] + height[...,None]).reshape(-1,1)
    vertices = (start_idxs.repeat(1,8)[...,None] + per_tri_verts[None,...]).reshape(-1,3)
    num_faces = vertices.shape[0]
    
    vert_dict = {
        "vertices": vertices,
        "num_faces": num_faces
    }
    
    return vert_dict


def integrated_warping(up_noise, warped_locs, vert_dict, up_level=3, cons_mask=None, fin_noise_resolution=64):
    
    C = 4
    
    vertices = vert_dict["vertices"]
    num_faces = vert_dict["num_faces"]
    
    resolution= fin_noise_resolution * (2 ** up_level)
    coords_len = (2*fin_noise_resolution + 1) ** 2
    device = warped_locs.device
    use_opengl = True
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    
    warped_vtx_pos =torch.cat((warped_locs, torch.zeros(coords_len, 1).to(device), torch.ones(coords_len, 1).to(device)), dim=-1)
    warped_vtx_pos = warped_vtx_pos[None,...].to(device)
    vertices = vertices.int().to(device)
    
    cons_mask = cons_mask[0].reshape(-1,1)
    empty_idx = torch.nonzero(1 - cons_mask.int())[:,0]
    
    # import pdb; pdb.set_trace()
    
    valid_vtx_list = []
    valid_pix = []
    
    # import pdb; pdb.set_trace()
        
    for i, vtx in enumerate(vertices):
        if vtx[0] not in empty_idx and vtx[1] not in empty_idx and vtx[2] not in empty_idx:
            valid_vtx_list.append(vtx)
            orig_pix = i // 8
            valid_pix.append(orig_pix)
    
    filtered_vertices = torch.stack(valid_vtx_list)
    # filtered_pix = torch.stack(valid_pix)
    
    with torch.no_grad():
        rast_out, _ = dr.rasterize(glctx, warped_vtx_pos, filtered_vertices, resolution=[resolution, resolution])
    
    rast = rast_out[:,:,:,3:].permute(0,3,1,2).int().to(torch.int64)

    uni_idx = torch.unique(rast).to(device)
    
    # import pdb; pdb.set_trace()
    
    if uni_idx[0] == 0:
        uni_idx = uni_idx[1:]
    
    nonzero_list = []

    for idx in uni_idx:
        nonzeros = torch.nonzero(rast == idx).to(device)
        nonzero_list.append(nonzeros)
    
    vert_canvas = torch.zeros((C,fin_noise_resolution**2)).to(device)
    vert_canvas_num_idx = torch.zeros((1,fin_noise_resolution**2)).to(device)
    
    # import pdb; pdb.set_trace()
    
    for value, val_indice in enumerate(nonzero_list):
        id = uni_idx[value]
        pix_loc = valid_pix[id - 1]
    
        for k in val_indice:
            vert_canvas[:,pix_loc] += up_noise[0,:,k[2],k[3]]
            vert_canvas_num_idx[:,pix_loc] += 1

    # reshaped_v = vert_canvas[:,1:].reshape(C,-1,8)
    # reshaped_v_num = vert_canvas_num_idx[:,1:].reshape(C,-1,8)
    
    # fin_v_val = reshaped_v.sum(dim=-1)
    # fin_v_num = reshaped_v_num.sum(dim=-1)
    
    # import pdb; pdb.set_trace()

    final_values = vert_canvas / torch.sqrt(vert_canvas_num_idx)   
    
    warped_noise = final_values.reshape(1,C,fin_noise_resolution,fin_noise_resolution).float()  

    # B, C, H, W = 1, 4, fin_noise_resolution, fin_noise_resolution
    
    # # Assign same index to triangles from the same original pixel, 0 if no index
    # indices = (rast - 1) // 8 + 1 # there is 8 triangles per pixel

    # # Flatten the upsampled noise
    # up_noise_flat = up_noise.reshape(B*C, -1).cpu()

    # # Create a flatten vector of ones for "Cardinality" value i.e. number of contained pixels
    # ones_flat = torch.ones_like(up_noise_flat[:1])

    # # Flatten the indices (and broadcast to batch size)
    # indices_flat = indices.reshape(1, -1).cpu()

    # # Aggregate the noise values and cardinality using scattering operation
    # fin_v_val = torch.zeros(B*C, H*W+1).scatter_add_(1, index=indices_flat.repeat(B*C, 1), src=up_noise_flat)[..., 1:]
    # fin_v_num = torch.zeros(1, H*W+1).scatter_add_(1, index=indices_flat, src=ones_flat)[..., 1:]

    # final_values = fin_v_val / torch.sqrt(fin_v_num)

    # warped_noise_fast = final_values.reshape(B, C, H, W).float()  
    
    return warped_noise
    