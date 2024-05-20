import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from PIL import Image
from tqdm import tqdm as orig_tqdm
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage



def reprojector(pts_locations, pts_feats, target_pose, target_pose_inv, fovy, device, ref_depth=None, img_size=64, background=False, **kwargs):
    """
    Inverse
    Source: Unseen view
    Target: GT view
    
    """ 
    
    # height, width, _ = source_rays.origins.shape
    
    batch_size = target_pose.shape[0]
    
    # source_viewdirs = source_rays.viewdirs.reshape(-1,3)
    # source_distance_flat = source_distance.reshape(-1,1)
    # source_origins = source_rays.origins.reshape(-1,3)
        
    # pts_locations = source_viewdirs * source_distance_flat + source_origins
    pts_sampled = pts_locations
        
    # if data_type == "blender" or data_type == "dtu":
        
    target_origin = target_pose[:,:3,-1]        

    # if data_type == "blender":
    # target_center_viewdir = ( - target_pose[:,:3,2])
    # elif data_type == "dtu":
    target_center_viewdir = (target_pose[:,:3,2])
        
    
    # Raymaker ########################### From Seen to Unseen

    # pts_to_tgt_origin = pts_sampled - target_origin[None, :]
    
    pts_to_tgt_origin = pts_sampled[None,...] - target_origin[:,None,...]
    
    dist_to_tgt_origin = torch.linalg.norm(pts_to_tgt_origin, axis=-1, keepdims=True)
    target_viewdirs = pts_to_tgt_origin / dist_to_tgt_origin

    new_per_view_lengths = (target_viewdirs * target_center_viewdir[:, None, :]).sum(axis = -1)
    target_directions = target_viewdirs / new_per_view_lengths[..., None]
    
    # Reprojector: Given target view, where do the points fall? ###############

    worldtocamera = target_pose_inv
    
    target_cameradir = (target_directions[...,None,:] * worldtocamera[..., None, :3, :3]).sum(-1)

    target_projection = target_cameradir[...,:2]
    
    # Flip the coordinates so that order is [y_coordinate, x_coordinate]
    
    unflip_proj = target_projection / torch.tan(0.5 * fovy[...,None,None]) # Consider focal length / FoV
    projected_loc_norm = unflip_proj.reshape(-1,2).fliplr().reshape(batch_size,-1,2) * torch.tensor([1,-1]).to(device)[None,None,...]
    
    proj_loc = projected_loc_norm * (img_size / 2) + (img_size / 2)
    
    proj_pix = proj_loc.floor()
        
    feature_maps, idx_maps = one_to_one_rasterizer(proj_pix, pts_feats, dist_to_tgt_origin, device, ref_depth, img_size=img_size, background=background, **kwargs)
    
    if idx_maps is None:
        reprojection_results = feature_maps
    else:
        reprojection_results = (feature_maps, proj_pix.int(), idx_maps)
    
    return reprojection_results


def ray_reprojector(batch_size, src_d, src_o, src_depth, target_pose, target_pose_inv, fovy, device, img_size=64, **kwargs):
    """
    Inverse
    Source: Unseen view
    Target: GT view
    
    """ 
    
    # height, width, _ = source_rays.origins.shape
    
    num_multiview = src_d.shape[0] // batch_size
    
    batch_idxs = [*range(len(src_d))]
    
    # source_viewdirs = source_rays.viewdirs.reshape(-1,3)
    # source_distance_flat = source_distance.reshape(-1,1)
    # source_origins = source_rays.origins.reshape(-1,3)
    
    re_dict = {"proj_maps": {}, "depth_masks": {}, "src_ind": None, "tgt_ind": None}
    
    tgt_ind_list = []
    src_ind_list = []
    
    for k in range(batch_size):
        
        src_idx = k * num_multiview
        tgt_idx = batch_idxs[src_idx+1 : (k+1)*num_multiview]
        
        tgt_ind_list += tgt_idx
        src_list = [src_idx for i in range(len(tgt_idx))]
        src_ind_list += src_list
        
        pts_sampled = (src_d[src_idx].reshape(3,-1) * src_depth[src_idx].reshape(1,-1) + src_o[src_idx].reshape(3,-1)).permute(1,0)
        target_origin = target_pose[tgt_idx,:3,-1]        
        target_center_viewdir = (target_pose[tgt_idx,:3,2])
        pts_to_tgt_origin = pts_sampled[None,...] - target_origin[:,None,...]
        
        dist_to_tgt_origin = torch.linalg.norm(pts_to_tgt_origin, axis=-1, keepdims=True)
        target_viewdirs = pts_to_tgt_origin / dist_to_tgt_origin

        new_per_view_lengths = (target_viewdirs * target_center_viewdir[:, None, :]).sum(axis = -1)
        target_directions = target_viewdirs / new_per_view_lengths[..., None]
        
        # Reprojector: Given target view, where do the points fall? ###############

        worldtocamera = target_pose_inv[tgt_idx]
        
        target_cameradir = (target_directions[...,None,:] * worldtocamera[..., None, :3, :3]).sum(-1)

        target_projection = target_cameradir[...,:2]
        
        # Flip the coordinates so that order is [y_coordinate, x_coordinate]
        
        unflip_proj = target_projection / torch.tan(0.5 * fovy[tgt_idx,None,None]) # Consider focal length / FoV
        
        # import pdb; pdb.set_trace()
        # projected_loc_norm = unflip_proj.reshape(-1,2).fliplr().reshape(num_multiview-1,-1,2) * torch.tensor([1,-1]).to(device)[None,None,...]
        
        projected_loc = unflip_proj * torch.tensor([-1,1]).to(device)[None,None,...]
        
        for i, proj in enumerate(projected_loc):
            re_dict["proj_maps"][f"{k}_{i}"] = proj.reshape(img_size, img_size, 2)
        
        # Generate depth mask
        
        with torch.no_grad():
            coord_loc = projected_loc.reshape(-1,2).fliplr().reshape(num_multiview-1,-1,2)
            proj_loc = (coord_loc * (img_size / 2) + (img_size / 2)).clamp(0,img_size-0.0001)
            proj_pix = proj_loc.int()
                        
            for i, tgt in enumerate(tgt_idx):
                tgt_depth = src_depth[tgt, 0]
                corres_depth = tgt_depth[proj_pix[i,:,0],proj_pix[i,:,1]]
                depth_mask = ((dist_to_tgt_origin[i].squeeze().detach() - corres_depth) < 0.07).float()
                
                re_dict["depth_masks"][f"{k}_{i}"] = depth_mask.reshape(1,img_size, img_size)
                            
                # hello = (depth_mask[None,...,None] * proj_pix).int()
                # canvas = torch.zeros(64,64).to(device)
                # img = canvas[hello[i,:,0], hello[i,:,1]]

    re_dict["tgt_ind"] = tgt_ind_list
    re_dict["src_ind"] = src_ind_list
    # import pdb; pdb.set_trace()    
    
    ## test code ###
    # new = (re_dict['depth_masks']['0_0'].reshape(4096,1) * (((re_dict['proj_maps']['0_0'] * torch.tensor([-1,1]).to(device)[None,...]).reshape(-1,2).fliplr()) * 32 + 32).clamp(0,img_size-0.0001)
    # canvas = torch.zeros(64,64)
    # canvas[new[:,0].int(),new[:,1].int()] = 1    
    # save_image(canvas,"canv.png")          

        
    return re_dict

def one_to_one_rasterizer(pts_proj_pix, pts_feats, pts_depth, device, ref_depth=None, img_size=64, pts_per_pix=5000, pts_thresh=0.3, background=True, **kwargs):
        
    batch_size, num_pts = pts_depth.shape[0], pts_depth.shape[1]
    
    noise_channel = kwargs["noise_channel"]
        
    if "reprojection_info" in kwargs:
        return_reproj_info = kwargs["reprojection_info"]
        pix_key = torch.linspace(0, img_size **2 -1, steps=img_size**2).int()
        idx_map_stack = []
        
    else:
        return_reproj_info = False

        
    if background:
        pts_per_pix = 500
        
        # import pdb; pdb.set_trace()

        inview_mask = ((pts_proj_pix[...,0] >= 0) * (pts_proj_pix[...,0] < img_size) * (pts_proj_pix[...,1] >= 0) * (pts_proj_pix[...,1] < img_size)).float()
        
        pts_proj_pix = inview_mask[...,None] * pts_proj_pix
        pts_final_feats = inview_mask[...,None] * pts_feats[None,...].repeat(batch_size,1,1) 
        # pts_final_feats = pts_feats[None,...].repeat(batch_size,1,1)
    
    else:
        pts_final_feats = pts_feats[None,...].repeat(batch_size,1,1)
                    
    idx_num = torch.linspace(0,num_pts-1,steps=num_pts)[None,...,None].repeat(batch_size,1,1).to(device)
    rasterizer_info = torch.cat((pts_depth, idx_num, pts_final_feats),dim=-1)
    rast_bin = torch.linspace(0,pts_per_pix-1,steps=pts_per_pix).repeat(num_pts // pts_per_pix + 1)[:num_pts].int().to(device)
    
    pts_proj_pix = torch.clamp(pts_proj_pix, min=0, max=img_size-0.001).int()
    
    y_coords = pts_proj_pix[...,0]
    x_coords = pts_proj_pix[...,1]
    
    feature_maps = []
        
    # pix_key = torch.linspace(0, img_size **2 -1, steps=img_size**2)[...,None].repeat(1, pts_per_pix).int().flatten().to(device)
    if noise_channel == 4:
        mult = torch.tensor([1, -1, 0, 0, 0, 0])
    elif noise_channel == 3:
        mult = torch.tensor([1, -1, 0, 0, 0])
        
    for i in range(batch_size):
    
        canv = 10 * torch.ones((img_size, img_size, pts_per_pix, 2 + noise_channel)).to(device) * mult[None,None,None,...].to(device)
                
        canv[y_coords[i], x_coords[i], rast_bin] = rasterizer_info[i]            
                
        # Sorting algorithm - not needed for now ----------------------
        
        # depth_key = torch.argsort(canv[:,:,:,0].reshape(-1, pts_per_pix), dim=-1).flatten()
        # img_shaped_values = canv.reshape(-1,pts_per_pix,6)[pix_key, depth_key].reshape(img_size, img_size, pts_per_pix, 6)
        
        ######################## --------------------------------------
        
        if return_reproj_info:
            res_canv = canv.reshape(-1, pts_per_pix, 2 + noise_channel)
            depth_key = torch.min(res_canv[...,0], dim=-1)[1].flatten()
            idx_map = res_canv[pix_key, depth_key][:,:2].reshape(img_size, img_size, 2)
            idx_map_stack.append(idx_map)
        
        # import pdb; pdb.set_trace()
        
        if ref_depth is not None:
        
            depth_mask = (torch.abs(ref_depth[i] - canv[...,0]) < pts_thresh).float()
            pix_num_pts = depth_mask.sum(-1)
            pix_bin_features = depth_mask[...,None] * canv[...,2:]
            
            pix_feat = pix_bin_features.sum(dim=-2)
            fin_feat = pix_feat / torch.sqrt(pix_num_pts[...,None])
            
            feature_maps.append(fin_feat)
        
        else:
            pix_num_pts = (canv[...,3] != 0).float().sum(dim=-1)
            fin_feat = canv[...,2:].sum(dim=-2) / torch.sqrt(pix_num_pts[...,None])
            feature_maps.append(fin_feat)
            
    feat_maps = torch.stack(feature_maps).permute(0,3,1,2)
    
    if return_reproj_info:
        idx_maps = torch.stack(idx_map_stack)
        
        src_view = 2
        tgt_view= 3

        
        visible_pts_idx = torch.unique(idx_maps[src_view,:,:,1].flatten())[1:].int()
        
        depth_canvas = torch.zeros((64,64,2)).to(device)
        src_canvas = torch.zeros((64,64,2)).int().to(device)
        
        depth_canvas[y_coords[tgt_view, visible_pts_idx], x_coords[tgt_view, visible_pts_idx]] = rasterizer_info[tgt_view, visible_pts_idx,:2].squeeze()
        depth_cons_mask = (torch.abs((depth_canvas[:,:,0] - idx_maps[tgt_view,:,:,0])) < 0.05).float()
        
        valid_idx = torch.unique(depth_cons_mask * depth_canvas[:,:,1])[1:].int()
        
        src_canvas[y_coords[src_view, valid_idx], x_coords[src_view, valid_idx]] = torch.stack((y_coords[tgt_view, valid_idx], x_coords[tgt_view, valid_idx]), dim=-1)
        
    else:
        idx_maps = None
    
    # import pdb; pdb.set_trace()
    

    return feat_maps, idx_maps



def pts_noise_upscaler(points, pts_noise, noise_channel, n_upscaling, up_loc_rand, up_feat_rand, device, pts_var=0.04):
    
    # Increase the number of points by N (point location)
    loc_noising = (pts_var * up_loc_rand).to(device)
    upscaled_locs = (points[...,None] + loc_noising).permute(0,2,1).reshape(-1,3)

    # Conditioned upsampling for the noise (how I warped your noise)
    upscaled_means = pts_noise[...,None].repeat(1,1,n_upscaling)
    raw_up_rand = up_feat_rand.to(device)
    noise_means = torch.mean(raw_up_rand, dim=-1)[...,None]
    upscaled_feats = raw_up_rand - noise_means
    
    up_noise = upscaled_means / torch.sqrt(torch.tensor(n_upscaling)) + upscaled_feats
    up_noise = up_noise.permute(0,2,1).reshape(-1,noise_channel)
    
    # import pdb; pdb.set_trace()    
    upscaled_feats = up_noise
    
    return upscaled_locs, up_noise


def sphere_pts_generator(device, noise_channel, radius = 1.9, orig_down_ratio=3, upscale_ratio=4):
    
    # num_pts = 360 * 360 * upscale_ratio ** 2
    
    num_elev_rads = 540
    
    # Locations

    init_horz_rads = torch.deg2rad(torch.linspace(start= 0, end=360-1, steps = 360 // orig_down_ratio)).to(device)
    init_elev_rads = torch.deg2rad(torch.linspace(start= 0, end=180, steps = num_elev_rads // orig_down_ratio)).to(device)
    
    horz_step = init_horz_rads[1] - init_horz_rads[0]
    elev_step = init_elev_rads[1] - init_elev_rads[0]
    
    horz_up = torch.linspace(0, horz_step, steps=upscale_ratio+1)[:-1].to(device)
    elev_up = torch.linspace(0, elev_step, steps=upscale_ratio+1)[:-1].to(device)
    
    horz_rads = (init_horz_rads[...,None] + horz_up[None, ...]).flatten()
    elev_rads = (init_elev_rads[...,None] + elev_up[None, ...]).flatten()
        
    x_coords = (radius * torch.sin(elev_rads)[...,None] * torch.cos(horz_rads)[None,...]).flatten()
    y_coords = (radius * torch.sin(elev_rads)[...,None] * torch.sin(horz_rads)[None,...]).flatten()
    z_coords = (radius * torch.cos(elev_rads)[...,None].repeat(1, 360 // orig_down_ratio * upscale_ratio)).flatten()

    sphere_coords = torch.stack((x_coords, y_coords, z_coords), dim=-1)
    
    # Upscaled Noise

    upscaled_means = torch.randn((noise_channel, num_elev_rads // orig_down_ratio, 360 // orig_down_ratio)).to(device).repeat_interleave(upscale_ratio,dim=1).repeat_interleave(upscale_ratio,dim=2)
    raw_rand = torch.randn_like(upscaled_means)
    raw_rand_means = raw_rand.unfold(1, upscale_ratio, upscale_ratio).unfold(2, upscale_ratio, upscale_ratio).mean((3,4))
    raw_up_rand = raw_rand_means.repeat_interleave(upscale_ratio,dim=1).repeat_interleave(upscale_ratio,dim=2)

    mean_removed_rand = raw_rand - raw_up_rand
    
    sphere_noise = (upscaled_means / upscale_ratio + mean_removed_rand).reshape(noise_channel,-1).permute(1,0)
    
    return sphere_coords, sphere_noise

    
    


def tester(proj, idx = 0):

    test_projection = proj[idx]
    
    resized_proj = test_projection * 256 + 256
    
    empty_canvas = torch.zeros((1,512,512)).to(test_projection.device)

    for pt in resized_proj:
        y_val = int(pt[1])
        x_val = - int(pt[0])
        
        empty_canvas[:,y_val, x_val] = 1
    
    canv = empty_canvas / torch.max(empty_canvas)
    
    return canv
    

###############
                    
                    
def depth_tester(proj, depth, idx = 0):

    test_projection = proj[idx]
    
    resized_proj = test_projection * 256 + 256
    
    empty_canvas = torch.zeros((1,512,512)).to(test_projection.device)

    for i, pt in enumerate(resized_proj):
        y_val = int(pt[1])
        x_val = - int(pt[0])
        
        empty_canvas[:,y_val, x_val] = depth[idx,i]
    
    canv = empty_canvas / torch.max(empty_canvas)
    
    return canv