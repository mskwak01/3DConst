import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from tqdm import tqdm as orig_tqdm

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras   
from pytorch3d.renderer import (
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,
)

import torch.nn.functional as F

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
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

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        
        # import pdb; pdb.set_trace()
        
        depth_map = fragments[1][0,...,:1]
        
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

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

def pointcloud_renderer(point_cloud, camera, raster_settings, device):

    camera = camera.to(device)

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    ).to(device)
    
    image = renderer(point_cloud)
    
    return image
    
def point_e(device,exp_dir):
    print('creating base model...')
    base_name = 'base1B' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    # img = Image.open(os.path.join(exp_dir))

    img = Image.open(exp_dir)
    transformed_img = pad_transform(img)

    samples = None

    # import pdb; pdb.set_trace()
    # for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
    #     samples = x

    for x in tqdm(
        sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[transformed_img]))
    ):
        samples = x   

    pc = sampler.output_to_point_clouds(samples)[0]
    
    return pc


def point_e_gradio(img,device):
    print('creating base model...')
    base_name = 'base1B' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )


    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
        
    pc = sampler.output_to_point_clouds(samples)[0]
    
    return pc