from dataclasses import dataclass, field

import torch
import numpy as np

import threestudio
import matplotlib.pyplot as plt

from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from torchvision.utils import save_image

from threestudio.systems.pc_project import point_e, render_depth_from_cloud
from pytorch3d.renderer import PointsRasterizationSettings

from threestudio.utils.saving import SaverMixin

@threestudio.register("covariance-system")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        threefuse: bool = True
        image_dir: str = "hello"
        calibration_value: int = 0
        identical_noising: bool = False
        three_noise: bool = False
        pts_radius: float = 0.02
        surf_radius: float = 0.05
        gradient_masking: bool = False
        nearby_fusing: bool = False
        tag: str = "no_tag"
        three_warp_noise: bool = False
        consider_depth: bool = True
        
    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        
        self.calibration_value = self.cfg.calibration_value

        self.threefuse = self.cfg.threefuse
        self.image_dir = self.cfg.image_dir        
        self.three_noise = self.cfg.three_noise
        
        if self.threefuse is True:
            device = self.device
            
            self.cond_pc = point_e(device="cuda", exp_dir=self.image_dir)
                        
            points = self.cond_pc     
            points_loc = torch.tensor(points.coords, dtype=torch.float32).to(device)
            
            deg = torch.deg2rad(torch.tensor([self.calibration_value]))
            rot_z = torch.tensor([[torch.cos(deg), -torch.sin(deg), 0],[torch.sin(deg), torch.cos(deg), 0],[0, 0, 1.]]).to(self.device)
            points_loc = (rot_z[None,...] @ points_loc[...,None]).squeeze()
            
            points.coords = points_loc        
                        
            self.cond_pc = points

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        
        device = self.device
        
        raster_settings = PointsRasterizationSettings(
                image_size= 800,
                radius = 0.01,
                points_per_pixel = 2
            )
        
        cam_radius = batch["camera_distances"]
        
        points = self.cond_pc    
                
        depth_maps = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device)
        
        # import pdb; pdb.set_trace()
        
        return {
            **render_out,
            "pts_depth" : depth_maps
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        prompt_utils = self.prompt_processor()
        noise_channel = 4
        
        if self.threefuse:
            with torch.no_grad():
                points = self.cond_pc
                
                points = self.cond_pc
                points_num = torch.tensor(points.coords, dtype=torch.float32).shape[0]
                
                n_tables = 6
                num_samples = 100
                cov_size = 8
                
                noise_tensor = torch.randn(num_samples, points_num, noise_channel).to(self.device)
                
                device = self.device
                
                if self.cfg.three_warp_noise:
                    pts_per_pix = 20
                    pts_radius = self.cfg.pts_radius
                    surf_radius = self.cfg.surf_radius
                    
                else:
                    pts_per_pix = 2
                    pts_radius = self.cfg.pts_radius
                    surf_radius = 0.08
                    
                                
                noise_raster_settings = PointsRasterizationSettings(
                        image_size= 64,
                        radius = pts_radius,
                        points_per_pixel = pts_per_pix
                    )
                
                up_noise_raster_settings = noise_raster_settings
                                
                surface_raster_settings = PointsRasterizationSettings(
                    image_size= 64,
                    radius = surf_radius,
                    points_per_pixel = 2
                )
    
                cam_radius = batch["camera_distances"]
                depth_maps = out["pts_depth"]
                
                # import pdb; pdb.set_trace()
                               
                # if self.three_noise:
                
                # THIS IS NOISE_COVARIANCE!!!!!
                
                center_y, center_x = torch.randint(25, 39, (2, n_tables))
                
                center_x[0] = 32
                center_y[0] = 32
                
                noise_maps_list = []
                                
                # import pdb; pdb.set_trace()
                
                for i in range(num_samples):
                    with torch.no_grad():
                        if not self.cfg.three_warp_noise:
                            
                            # import pdb; pdb.set_trace()

                            noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor[i], noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
                                        identical_noising=self.cfg.identical_noising)
                            noise_map = noised_maps
                        
                        else:
                            noised_maps, loc_tensor, inter_dict, depth_masks = render_upscaled_noised_cloud(points, batch, noise_tensor[i], noise_raster_settings, surface_raster_settings, up_noise_raster_settings, noise_channel, cam_radius, device, 
                                        identical_noising=self.cfg.identical_noising, consider_depth=self.cfg.consider_depth)
                            noise_map = noised_maps                          
                        
                        # import pdb; pdb.set_trace()
                        
                        noise_maps_list.append(noised_maps[:,0])
                        
                # import pdb; pdb.set_trace()
                
                all_noise_tensors = torch.stack(noise_maps_list).permute(1,0,2,3)
                                
                samples = all_noise_tensors.cpu().detach().numpy()
                
                cov_list = []
                
                for i in range(n_tables):
                    per_view_samples = samples[i, :, center_y[i] - cov_size // 2 : center_y[i] + cov_size // 2, center_y[i] - cov_size // 2 : center_y[i] + cov_size // 2]

                    reshaped_samples = per_view_samples.reshape(100, -1)
                    
                    covariance_matrix = np.cov(reshaped_samples, rowvar=False)
                    
                    cov_list.append(covariance_matrix)
                    
                    horizontal = batch["azimuth"][i].to(device).type(torch.float32) 
                    elevation = batch["elevation"][i].to(device).type(torch.float32)
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(covariance_matrix, cmap='viridis')
                    plt.title(f'Cov. Matrix of {cov_size}x{cov_size} | Pix_x: {center_x[i]} Pix_y: {center_y[i]} | Horz: {horizontal.int()} Elev: {elevation.int()}')
                    plt.colorbar()
                    plt.show()
                    
                    dir = self._save_dir + f"view_{i}.png"
                    
                    plt.savefig(dir)
                    
                    save_image(depth_maps, self._save_dir + f"depth_{i}.png")

                # import pdb; pdb.set_trace()
                
                raise ValueError()
                
                                    
                # self.save_image_grid(

                # Reshaping the samples to a 2D array where each row is a flattened 4x4 matrix
                # reshaped_samples = samples.reshape(100, -1)

                # Computing the covariance matrix of the reshaped samples
                # covariance_matrix = np.cov(reshaped_samples, rowvar=False)
                
                
                # THIS IS NOISE_COVARIANCE!!!!!
                # THIS IS NOISE_COVARIANCE!!!!!
                # THIS IS NOISE_COVARIANCE!!!!!
                # THIS IS NOISE_COVARIANCE!!!!!
                
                
                        
                # else:
                #     noise_map = None
                #     loc_tensor = None
                #     inter_dict = None
                #     depth_masks = None
        
        guidance_out = self.guidance(
            out["comp_rgb"], prompt_utils, **batch, depth_map=depth_maps,  noise_map=noise_map, rgb_as_latents=False, 
                idx_map=loc_tensor, inter_dict=inter_dict, depth_masks=depth_masks,
            )

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
