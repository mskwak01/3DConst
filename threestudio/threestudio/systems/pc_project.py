import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from tqdm import tqdm as orig_tqdm

import raster

from threestudio.systems.pytorch3d.structures import Pointclouds
from threestudio.systems.pytorch3d.renderer.cameras import PerspectiveCameras
from threestudio.systems.pytorch3d.renderer import (
    AlphaCompositor,
    look_at_view_transform,
)
# from threestudio.systems.pytorch3d.renderer import PointsRasterizationSettings


# threestudio.systems.pytorch3d.structures

import torch.nn.functional as F

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

from typing import NamedTuple, Optional, Tuple, Union, List

from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from torchvision import transforms


class PadToSquare:
    def __call__(self, img):
        # Get the original image size
        width, height = img.size
        # Determine the size of the new square image
        max_side = max(width, height)
        
        # Calculate padding
        left = (max_side - width) // 2
        right = max_side - width - left
        top = (max_side - height) // 2
        bottom = max_side - height - top

        # Pad the image and return it
        padding = (left, top, right, bottom)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

# Define the transform pipeline
pad_transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # Resize to a fixed size (optional)
    PadToSquare(),
    # transforms.ToTensor(),
])

def tqdm(*args, **kwargs):
    is_remote = bool(os.environ.get("IS_REMOTE", False))
    if is_remote:
        f = open(os.devnull, "w")
        kwargs.update({"file": f})
    return orig_tqdm(*args, **kwargs)


class PointFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


class PointsRenderer(nn.Module):
    """
    Modified version of Pytorch3D PointsRenderer
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, return_idx=False, equal_weighting=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # import pdb; pdb.set_trace(

        depth_map = fragments[1][0, ..., :1]
        
        if return_idx:
            idx = fragments[0][0, ..., :1]
        
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
                
        if not equal_weighting:
            weights = 1 - dists2 / (r * r)
            
            images = self.compositor(fragments.idx.long().permute(0, 3, 1, 2), weights, point_clouds.features_packed().permute(1, 0),**kwargs,)
        
        else:            
            pts_canvas = fragments.idx.long().permute(0, 3, 1, 2)
            pix_can = (pts_canvas > 0).int()
            num_pix_pts = torch.sum(pix_can,dim=1)            
            weights = ((torch.ones_like(pts_canvas) * pix_can ) / torch.sqrt(num_pix_pts)[:,None,...]).nan_to_num() + (1 - pix_can) * (1 + 1/(r*r))
            # weights = ((torch.ones_like(pts_canvas) * pix_can ) / num_pix_pts[:,None,...]).nan_to_num() + (1 - pix_can) * (1 + 1/(r*r))
            
            B, C, H, W = pts_canvas.shape
            
            # Composition
            # images = self.compositor(fragments.idx.long().permute(0, 3, 1, 2), weights, point_clouds.features_packed().permute(1, 0),**kwargs,)
            features = torch.cat((point_clouds.features_packed(), torch.zeros(1,4).to(point_clouds.device)))
            idxs = pts_canvas.reshape(-1,1).squeeze()
            pix_feat = features[idxs].reshape(B,C,H,W,4).sum(dim=1)
            raw_sum_images = pix_feat.permute(0,3,1,2)
            
            images = raw_sum_images / torch.sqrt(num_pix_pts[:,None,...])        
        # 

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        
        if return_idx:
            return images, depth_map, idx
        else:
            return images, depth_map


def render_depth_from_cloud(
    points, batch, raster_settings, cam_radius, device, cali=0, dynamic_points=False, raw=False,
):
    radius = cam_radius[0]

    horizontal = batch["azimuth"].to(device).type(torch.float32) + cali
    elevation = batch["elevation"].to(device).type(torch.float32)
    FoV = batch["fov"].to(device).type(torch.float32)

    cameras = py3d_camera(radius, elevation, horizontal, FoV, device)

    if dynamic_points:
        point_loc = points
        colors = torch.ones_like(points)        
        # import pdb; pdb.set_trace()
    
    else:
        point_loc = torch.tensor(points.coords, dtype=torch.float32).to(device)
        
        colors = torch.tensor(
            np.stack(
                [points.channels["R"], points.channels["G"], points.channels["B"]], axis=-1
            ),
            dtype=torch.float32,
        ).to(device)
        
        # colors = torch.andn_like(point_loc).to(device)


    matching_rotation = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]], dtype=torch.float32
    ).to(device)

    rot_points = (
        (matching_rotation @ point_loc[..., None])
        .squeeze()
        .to(device)
        .type(torch.float32)
    )

    point_cloud = Pointclouds(points=[rot_points], features=[colors])
    
    depth_maps = []
    
    raw_maps = []
    
    for camera in cameras:
        
        _, raw_depth_map = pointcloud_renderer(point_cloud, camera, raster_settings, device)

        raw_maps.append(raw_depth_map)

        disparity = camera.focal_length[0, 0] / (raw_depth_map + 1e-9)
        
        # import pdb; pdb.set_trace()
        
        max_disp = torch.max(disparity)
        min_disp = torch.min(disparity[disparity > 0])

        norm_disparity = (disparity - min_disp) / (max_disp - min_disp)

        mask = norm_disparity > 0
        norm_disparity = norm_disparity * mask

        depth_map = F.interpolate(
            norm_disparity.permute(2, 0, 1)[None, ...], size=512, mode="bilinear"
        )[0]
        depth_map = depth_map.repeat(3, 1, 1)
        
        depth_maps.append(depth_map)
    
    depth_maps_tensor = torch.stack(depth_maps)
    raw_depth_maps = torch.stack(raw_maps)
    
    # print(horizontal)
    # save_image(depth_maps_tensor[0], "depth_1.png")
    # save_image(depth_maps_tensor[1], "depth_2.png")
    
    # import pdb; pdb.set_trace()
    
    ##############
    # mask = (depth_maps_tensor[4] > 0.4)
    
    if raw:
        maps = raw_depth_maps
    else:
        maps = depth_maps_tensor
        
    return maps


def render_noised_cloud(
    points, batch, noise_tensor, noised_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
    cali = 0, dynamic_points=False, identical_noising=False, id_tensor=None, 
    viewcomp_setting="all_views"
):
    radius = cam_radius[0]
    loc_tensor = None
    inter_dict = None

    horizontal = batch["azimuth"].to(device).type(torch.float32) + cali
    elevation = batch["elevation"].to(device).type(torch.float32)
    FoV = batch["fov"].to(device).type(torch.float32)

    cameras = py3d_camera(radius, elevation, horizontal, FoV, device)

    if dynamic_points:
        point_loc = points
    
    else:
        point_loc = torch.tensor(points.coords, dtype=torch.float32).to(device)

    feature = noise_tensor
    
    matching_rotation = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]], dtype=torch.float32).to(device)

    rot_points = (
        (matching_rotation @ point_loc[..., None])
        .squeeze()
        .to(device)
        .type(torch.float32)
    )
    
    # import pdb; pdb.set_trace()

    point_cloud = Pointclouds(points=[rot_points], features=[feature])
    
    noise_maps = []
    raw_noise_maps = []
    idx_maps = []
    raw_depth_masks = []
    depth_maps = []
    
    raw_depth = []
    # interpolated_mask = True
    
    for camera in cameras:
        
        # import pdb; pdb.set_trace()
        
        noise_map, raw_depth_map, idx_map = pointcloud_renderer(point_cloud, camera, noised_raster_settings, device, return_idx=True)
        _, smooth_depth_map, _ = pointcloud_renderer(point_cloud, camera, surface_raster_settings, device, return_idx=True)
        # Depth map location
        
        # Raw depth
        
        disparity = camera.focal_length[0, 0] / (raw_depth_map + 1e-9)

        max_disp = torch.max(disparity)
        min_disp = torch.min(disparity[disparity > 0])

        norm_disparity = (disparity - min_disp) / (max_disp - min_disp)
        
        ori_mask = (norm_disparity > 0).float()
        # norm_disparity = ori_mask * norm_disparity
        
        raw_depth.append(norm_disparity)

        ###########
        
        smooth_disparity = camera.focal_length[0, 0] / (smooth_depth_map + 1e-9)

        sm_max_disp = torch.max(smooth_disparity)
        sm_min_disp = torch.min(smooth_disparity[smooth_disparity > 0])

        smooth_norm_disparity = (smooth_disparity - sm_min_disp) / (sm_max_disp - sm_min_disp)
        # smooth_mask = (smooth_norm_disparity > 0.)
        # smooth_norm_disparity = smooth_mask * smooth_norm_disparity 
        
        differences = smooth_norm_disparity - norm_disparity
        
        diff_mask = (differences < 0.2).float()
        
        ultimate_mask = ori_mask * diff_mask
        
        mask = ultimate_mask
        
        fin_disparity = norm_disparity * mask
        depth_mask = mask.float()
        
        depth_map = fin_disparity.permute(2,0,1).detach()
                
        depth_maps.append(depth_map)
        
        masked_idx = torch.clamp(idx_map - 1e7 * ( 1 - depth_mask ), min=-1, max=None)
    
        # background_noise = (1 - depth_mask) * torch.randn(64, 64, noise_channel).to(device)
        # final_noise = noise_map[0] * depth_mask + background_noise
        final_noise = noise_map[0] * depth_mask
        
        raw_depth_masks.append(depth_mask.permute(2,0,1).detach())
        noise_maps.append(final_noise[None,...])
        idx_maps.append(masked_idx.long())
    
    # import pdb; pdb.set_trace()
    noise_maps_tensor = torch.stack(noise_maps).squeeze()
    depth_masks = torch.stack(raw_depth_masks).detach()
        
    if identical_noising:
        
        # Making a per-point location matrix
                
        grid_y, grid_x = torch.meshgrid(torch.arange(64),torch.arange(64), indexing='ij')
        coords = torch.stack([grid_y,grid_x],dim=-1).reshape(-1,2)
        
        loc_tensor = torch.zeros([len(idx_maps), point_loc.shape[0], 2]).long().detach()
                
        for num, idx_map in enumerate(idx_maps):
            tgt_idxmap = idx_map.reshape(-1)
            loc_tensor[num, tgt_idxmap] = coords

        # Finding Intersecting Points Indices 
        
        total_intersection = np.array([], dtype=np.int32)
        
        inter_dict = {}
                        
        if viewcomp_setting == "all_views":
        
            for num, tgt_idx_map in enumerate(idx_maps):
                
                other_list = idx_maps[num + 1:]
                idx_one = tgt_idx_map.reshape(-1,1).detach().cpu().numpy()

                for new_num, other_idx_map in enumerate(other_list):
                    
                    idx_two = other_idx_map.reshape(-1,1).detach().cpu().numpy()
                    new_inter = np.intersect1d(idx_one, idx_two)
                    
                    inter_key = str(num) + str(new_num + num + 1)
                    inter_dict[inter_key] = torch.tensor(new_inter).detach()

                    total_intersection = np.union1d(total_intersection, new_inter)
                            
            inter_pts = torch.tensor(total_intersection)
                    
        elif viewcomp_setting == "penta":
            
            # import pdb; pdb.set_trace()
                                    
            keys = [[0,1], [0,2], [1,3], [2,4], [5,3], [5,4]]
            
            for view_pair in keys:
                
                idx_one = idx_maps[view_pair[0]].reshape(-1,1).detach().cpu().numpy()
                idx_two = idx_maps[view_pair[1]].reshape(-1,1).detach().cpu().numpy()
                
                new_inter = np.intersect1d(idx_one, idx_two)
                
                inter_key = str(view_pair[0]) + str(view_pair[1])
                inter_dict[inter_key] = torch.tensor(new_inter).detach()
                
                total_intersection = np.union1d(total_intersection, new_inter)
            
            inter_pts = torch.tensor(total_intersection)
            
            # import pdb; pdb.set_trace()
                        
        noise_canvas = torch.zeros([point_loc.shape[0], noise_channel]).float().to(device)      
        if id_tensor is None:  
            noise_canvas[inter_pts] = torch.randn(inter_pts.shape[0],noise_channel).to(device)
        else:
            noise_canvas[inter_pts] = id_tensor[inter_pts]
        # else:
        #     noise_canvas = torch.zeros([point_loc.shape[0], noise_channel]).float().to(device)
            
        #     noise_canvas[inter_pts] = centerpoint_noise        
        
        # import pdb; pdb.set_trace()    
        
        loc_y = loc_tensor[:,:,0]
        loc_x = loc_tensor[:,:,1]
        
        for i in range(len(idx_maps)):
            noise_maps_tensor[i,loc_y[i],loc_x[i],:] = noise_canvas
            
        # import pdb; pdb.set_trace()
    try:
        fin_noise = noise_maps_tensor.permute(0,3,1,2).detach()
    except:
        fin_noise = noise_maps_tensor[None,...].permute(0,3,1,2).detach()
                   
    return fin_noise, loc_tensor, inter_dict, depth_masks


def render_upscaled_noised_cloud(
    points, batch, noise_tensor, surface_raster_settings, up_noise_raster_settings, noise_channel, cam_radius, device, 
    cali = 0, dynamic_points=False, cut_thresh=0.3, consider_depth=True, **kwargs
):  
    
    # import pdb; pdb.set_trace()
    
    radius = cam_radius[0]
    loc_tensor = None
    inter_dict = None

    horizontal = batch["azimuth"].to(device).type(torch.float32) + cali
    elevation = batch["elevation"].to(device).type(torch.float32)
    FoV = batch["fov"].to(device).type(torch.float32)

    cameras = py3d_camera(radius, elevation, horizontal, FoV, device)

    if dynamic_points:
        point_loc = points
    
    else:
        try:
            point_loc = torch.tensor(points.coords, dtype=torch.float32).to(device)
        except:
            point_loc = points

    feature = noise_tensor
    
    matching_rotation = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]], dtype=torch.float32
    ).to(device)

    rot_points = (
        (matching_rotation @ point_loc[..., None])
        .squeeze()
        .to(device)
        .type(torch.float32)
    )

    point_cloud = Pointclouds(points=[rot_points], features=[feature])
    
    noise_maps = []
    raw_noise_maps = []
    idx_maps = []
    raw_depth_masks = []
    depth_maps = []
    
    raw_depth = []
    
    # debugger = True
    
    # while debugger:
        # n_upscaling = int(input("n_upscaling: "))
        # pts_var = float(input("pts_var: "))
        # feat_var = float(input)
        
    pts_var = 0.03

    n_upscaling = kwargs["n_upscaling"]    
    loc_rand = kwargs["loc_rand"]
    feat_rand = kwargs["feat_rand"]
    
    # Increase the number of points by N (point location)
    
    num_points = rot_points.shape[0]
    
    loc_noising = (pts_var * loc_rand).to(device)
    upscaled_locs = (rot_points[...,None] + loc_noising).permute(0,2,1).reshape(-1,3)
    # upscaled_locs = (rot_points[...,None]).permute(0,2,1).reshape(-1,3)


    # Conditioned upsampling for the noise (how I warped your noise)
    upscaled_means = noise_tensor[...,None].repeat(1,1,n_upscaling)
    raw_up_rand = feat_rand.to(device)
    noise_means = torch.mean(raw_up_rand, dim=-1)[...,None]
    upscaled_feats = raw_up_rand - noise_means
    
    up_noise = upscaled_means / torch.sqrt(torch.tensor(n_upscaling)) + upscaled_feats
    up_noise = up_noise.permute(0,2,1).reshape(-1,4)
    
    # import pdb; pdb.set_trace()    
    upscaled_feats = up_noise
    
    # Rasterizer w/ very small points

    upscaled_point_cloud = Pointclouds(points=[upscaled_locs], features=[upscaled_feats])
    
    ########## CAMERA
    
    # surf_depth_map = []
    
    # camera = cameras[0]
    for camera in cameras:
        
        # import pdb; pdb.set_trace()
        
        noise_map, raw_depth_map, idx_map = pointcloud_renderer(upscaled_point_cloud, camera, up_noise_raster_settings, device, return_idx=True, eq_w=True)
        
        if consider_depth:
            _, smooth_depth_map, _ = pointcloud_renderer(point_cloud, camera, surface_raster_settings, device, return_idx=True, eq_w=True)
        # Depth map location
        
        # surf_depth_map.append(smooth_depth_map)
        
        # Raw depth
        
        disparity = camera.focal_length[0, 0] / (raw_depth_map + 1e-9)

        max_disp = torch.max(disparity)
        min_disp = torch.min(disparity[disparity > 0])

        norm_disparity = (disparity - min_disp) / (max_disp - min_disp)
        
        ori_mask = (norm_disparity > 0).float()
        # norm_disparity = ori_mask * norm_disparity
        
        # save_image(norm_disparity.permute(2,0,1), "hello.png")
        # save_image(noise_map[0,:,:,:3].permute(2,0,1), "noise_1.png")
        # save_image(noise_map[0,:,:,1:].permute(2,0,1), "noise_2.png")
            
        raw_depth.append(norm_disparity)
        
        if consider_depth:

            ###########
        
            smooth_disparity = camera.focal_length[0, 0] / (smooth_depth_map + 1e-9)

            sm_max_disp = torch.max(smooth_disparity)
            sm_min_disp = torch.min(smooth_disparity[smooth_disparity > 0])

            smooth_norm_disparity = (smooth_disparity - sm_min_disp) / (sm_max_disp - sm_min_disp)
            # smooth_mask = (smooth_norm_disparity > 0.)
            # smooth_norm_disparity = smooth_mask * smooth_norm_disparity 
            
            differences = smooth_norm_disparity - norm_disparity
            
            # import pdb; pdb.set_trace()
            
            diff_mask = (differences < cut_thresh).float()
            
            ultimate_mask = ori_mask * diff_mask
            
            mask = ultimate_mask
            
            fin_disparity = norm_disparity * mask
            depth_mask = mask.float()
            
            depth_map = fin_disparity.permute(2,0,1).detach()
                    
            depth_maps.append(depth_map)
            
            masked_idx = torch.clamp(idx_map - 1e7 * ( 1 - depth_mask ), min=-1, max=None)
        
            # background_noise = (1 - depth_mask) * torch.randn(64, 64, noise_channel).to(device)
            # final_noise = noise_map[0] * depth_mask + background_noise
            
            final_noise = (noise_map[0])  * depth_mask
            
            # final_noise = (noise_map[0]/ 3.5 + 0.5) * depth_mask

            raw_depth_masks.append(depth_mask.permute(2,0,1).detach())
            noise_maps.append(final_noise[None,...])
            idx_maps.append(masked_idx.long())
        
        else:
            mask = ori_mask
            depth_mask = ori_mask

            background_noise = (1 - depth_mask) * torch.randn(64, 64, noise_channel).to(device)
            # final_noise = noise_map[0] * depth_mask + background_noise            
            
            final_noise = (noise_map[0]) * depth_mask
            
            # final_noise = (noise_map[0]/ 3.5 + 0.5) * depth_mask


            noise_maps.append(final_noise[None,...])
            raw_depth_masks.append(depth_mask.permute(2,0,1).detach())
            idx_maps = None

                
    noise_maps_tensor = torch.stack(noise_maps).squeeze()
    depth_masks = torch.stack(raw_depth_masks).detach()
    # surf_map = torch.stack(surf_depth_map)
    
    # import pdb; pdb.set_trace()
    
    # noise_maps_tensor = noise_maps_tensor / 5 + 0.5
    
    try:
        fin_noise = noise_maps_tensor.permute(0,3,1,2).detach() 
    except:
        fin_noise = noise_maps_tensor[None,...].permute(0,3,1,2).detach()
                            
    return fin_noise, loc_tensor, inter_dict, depth_masks


def py3d_camera(radius, elevations, horizontals, FoV, device, img_size=800):
    # fov_rad = torch.deg2rad(torch.tensor(FoV))
    fov_rad = FoV
    focals = 1 / torch.tan(fov_rad / 2) * (2.0 / 2)
        
    cameras = []
    
    for focal, elev, horiz in zip(focals, elevations, horizontals):
        
        focal_length = torch.tensor([[focal, focal]]).float()
        image_size = torch.tensor([[img_size, img_size]]).double()

        R, T = look_at_view_transform(
            dist=radius, elev=elev, azim=horiz, degrees=True
        )

        camera = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_length,
            image_size=image_size,
            device=device,
        )
        
        cameras.append(camera)

    return cameras


def pointcloud_renderer(point_cloud, camera, raster_settings, device, return_idx=False, eq_w=False):
    camera = camera.to(device)

    rasterizer = our_PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor()).to(
        device
    )

    image = renderer(point_cloud, return_idx=return_idx, equal_weighting=eq_w)

    return image


def point_e(device, exp_dir=None):
    print("creating base model...")
    base_name = "base1B"  # use base300M or base1 B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 3.0],
    )

    img = Image.open(exp_dir)
    transformed_img = pad_transform(img)

    samples = None
    for x in tqdm(
        sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[transformed_img]))
    ):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]

    return pc


def point_e_gradio(img, device):
    print("creating base model...")
    base_name = "base1B"  # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 3.0],
    )

    samples = None
    for x in tqdm(
        sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))
    ):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]

    return pc


class RasterizePoints(torch.autograd.Function):
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        points,  # (P, 3)
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, torch.Tensor] = 0.01,
        points_per_pixel: int = 8,
        bin_size: int = 0,
        max_points_per_bin: int = 0,
    ):
        # TODO: Add better error handling for when there are more than
        # max_points_per_bin in any bin.

        args = (
            points.type(torch.float32),
            cloud_to_packed_first_idx.type(torch.long),
            num_points_per_cloud.type(torch.long),
            image_size,
            radius.type(torch.float32),
            points_per_pixel,
            bin_size,
            max_points_per_bin,
        )

        # import pdb; pdb.set_trace()
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        idx, zbuf, dists = raster.rasterize_points(*args)

        # import pdb; pdb.set_trace()
        # ctx.save_for_backward(points, idx)
        # ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists


def rasterize_points(
    pointclouds,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    radius: Union[float, List, Tuple, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_points_per_bin: Optional[int] = None,
):
    """
    Each pointcloud is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pix

    Args:
        pointclouds: A Pointclouds object representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds, where each point cloud
            can have a different number of points; the coordinates of each point
            are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        radius (Optional): The radius (in NDC units) of the disk to
            be rasterized. This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius per point
            in the batch.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of points allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.

    Returns:
        3-element tuple containing

        - **idx**: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the indices of the nearest points at each pixel, in ascending
          z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
          closest point (along the z-direction) to pixel (y, x) - note that points
          represents the packed points of shape (P, 3).
          Pixels that are hit by fewer than points_per_pixel are padded with -1.
        - **zbuf**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the z-coordinates of the nearest points at each pixel, sorted in
          z-order. Concretely, if `idx[n, y, x, k] = p` then
          `zbuf[n, y, x, k] = points[n, p, 2]`. Pixels hit by fewer than
          points_per_pixel are padded with -1
        - **dists2**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the squared Euclidean distance (in NDC units) in the x/y plane
          for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
          then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
          and the point `(points[n, p, 0], points[n, p,c 1])`. Pixels hit with fewer
          than points_per_pixel are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    points_packed = pointclouds.points_packed()
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    radius = _format_radius(radius, pointclouds)

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    if bin_size is None:
        if not points_packed.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
            bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (max_image_size - 1) // bin_size
        if points_per_bin >= 22:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (22, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, pointclouds._P / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only
    return RasterizePoints.apply(
        points_packed,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        im_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


def parse_image_size(
    image_size: Union[List[int], Tuple[int, int], int]
) -> Tuple[int, int]:
    """
    Args:
        image_size: A single int (for square images) or a tuple/list of two ints.

    Returns:
        A tuple of two ints.

    Throws:
        ValueError if got more than two ints, any negative numbers or non-ints.
    """
    if not isinstance(image_size, (tuple, list)):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("Image size can only be a tuple/list of (H, W)")
    if not all(i > 0 for i in image_size):
        raise ValueError("Image sizes must be greater than 0; got %d, %d" % image_size)
    if not all(type(i) == int for i in image_size):
        raise ValueError("Image sizes must be integers; got %f, %f" % image_size)
    return tuple(image_size)


def _format_radius(
    radius: Union[float, List, Tuple, torch.Tensor], pointclouds
) -> torch.Tensor:
    """
    Format the radius as a torch tensor of shape (P_packed,)
    where P_packed is the total number of points in the
    batch (i.e. pointclouds.points_packed().shape[0]).

    This will enable support for a different size radius
    for each point in the batch.

    Args:
        radius: can be a float, List, Tuple or tensor of
            shape (N, P_padded) where P_padded is the
            maximum number of points for each pointcloud
            in the batch.

    Returns:
        radius: torch.Tensor of shape (P_packed)
    """
    N, P_padded = pointclouds._N, pointclouds._P
    points_packed = pointclouds.points_packed()
    P_packed = points_packed.shape[0]
    if isinstance(radius, (list, tuple)):
        radius = torch.tensor(radius).type_as(points_packed)
    if isinstance(radius, torch.Tensor):
        if N == 1 and radius.ndim == 1:
            radius = radius[None, ...]
        if radius.shape != (N, P_padded):
            msg = "radius must be of shape (N, P): got %s"
            raise ValueError(msg % (repr(radius.shape)))
        else:
            padded_to_packed_idx = pointclouds.padded_to_packed_idx()
            radius = radius.view(-1)[padded_to_packed_idx]
    elif isinstance(radius, float):
        radius = torch.full((P_packed,), fill_value=radius).type_as(points_packed)
    else:
        msg = "radius must be a float, list, tuple or tensor; got %s"
        raise ValueError(msg % type(radius))
    return radius


class PointsRasterizationSettings:
    """
    Class to store the point rasterization params with defaults

    Members:
        image_size: Either common height and width or (height, width), in pixels.
        radius: The radius (in NDC units) of each disk to be rasterized.
            This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (N, P) giving a radius
            per point in the batch.
        points_per_pixel: (int) Number of points to keep track of per pixel.
            We return the nearest points_per_pixel points along the z-axis.
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts
            to set it heuristically based on the shape of the input. This should
            not affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine
            rasterization (bin_size != 0); this is the maximum number of points
            allowed within each bin. This should not affect the output values,
            but can affect the memory usage in the forward pass.
            Setting max_points_per_bin=None attempts to set with a heuristic.
    """

    image_size: Union[int, Tuple[int, int]] = 256
    radius: Union[float, torch.Tensor] = 0.01
    points_per_pixel: int = 8
    bin_size: Optional[int] = None
    max_points_per_bin: Optional[int] = None


class our_PointsRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras=None, raster_settings=None) -> None:
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, point_clouds, **kwargs) -> Pointclouds:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_proj: the points with positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of PointsRasterizer"
            raise ValueError(msg)

        pts_world = point_clouds.points_padded()
        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        eps = kwargs.get("eps", None)
        pts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            pts_world, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = cameras.get_projection_transform(**kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            pts_ndc = projection_transform.transform_points(pts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            pts_proj = cameras.transform_points(pts_world, eps=eps)
            pts_ndc = to_ndc_transform.transform_points(pts_proj, eps=eps)

        pts_ndc[..., 2] = pts_view[..., 2]
        point_clouds = point_clouds.update_padded(pts_ndc)
        return point_clouds

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def forward(self, point_clouds, **kwargs):
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        points_proj = self.transform(point_clouds, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(
            points_proj,
            image_size=raster_settings.image_size,
            radius=raster_settings.radius,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
        )
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)