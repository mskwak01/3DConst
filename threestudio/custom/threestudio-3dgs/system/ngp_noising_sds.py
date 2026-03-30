import math
from dataclasses import dataclass

import torch.nn.functional as F
import numpy as np
import threestudio
import torch

from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from threestudio.systems.pc_project import point_e, render_depth_from_cloud, render_noised_cloud
from threestudio.systems.pytorch3d.renderer import PointsRasterizationSettings

from torchvision.utils import save_image
import open3d as o3d


@threestudio.register("ngp-sds-noising-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        visualize_samples: bool = False
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        threefuse: bool = True
        image_dir: str = "hello"
        calibration_value: int = 0
        three_noise: bool = False
        multidiffusion: bool = False
        identical_noising: bool = False
        gs_noise: bool = False
        obj_only: bool = False
        constant_noising: bool = False
        cons_noise_alter: int = 0
        pts_radius: float = 0.02
        tag: str = "no_tag"
        gradient_masking: bool = False
        nearby_fusing: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        
        self.threefuse = self.cfg.threefuse
        self.image_dir = self.cfg.image_dir
        self.three_noise = self.cfg.three_noise
        self.multidiffusion = self.cfg.multidiffusion
        
        self.gs_noise = self.cfg.gs_noise
        self.obj_only = self.cfg.obj_only
        self.constant_noising = self.cfg.constant_noising

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
            
        self.cond_pc = point_e(device="cuda", exp_dir=self.image_dir)
        self.calibration_value = self.cfg.calibration_value
        
        self.c_noise = None
        self.i_noise = None
        
        # pcd = self.pcb()
    
        # self.geometry.create_from_pcd(pcd, 10)
        # self.geometry.training_setup()


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
        
        return {
            **render_out,
            "pts_depth" : depth_maps
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
            
        out = self(batch)
        noise_channel = 4
                
        if self.threefuse:

            points = self.cond_pc
            points_num = torch.tensor(points.coords, dtype=torch.float32).shape[0]
            noise_tensor = torch.randn(points_num, noise_channel).to(self.device)
            
            # Change it to dynamic
            device = self.device

            noise_raster_settings = PointsRasterizationSettings(
                    image_size= 64,
                    radius = self.cfg.pts_radius,
                    points_per_pixel = 2
                )
            
            surface_raster_settings = PointsRasterizationSettings(
                image_size= 64,
                radius = 0.08,
                points_per_pixel = 2
            )
                            
            cam_radius = batch["camera_distances"]
            
            depth_maps = out["pts_depth"]
                                    
            if self.three_noise:
                with torch.no_grad():
                    noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
                                identical_noising=self.cfg.identical_noising)
                    noise_map = noised_maps
                           
            elif self.gs_noise:
                obj_noise = out["noise_image"].permute(0,3,1,2)
                
                if self.obj_only:
                    noise_map = obj_noise
                else:
                    back_noise = (obj_noise == 0.) * torch.randn_like(obj_noise)
                    noise_map = obj_noise + back_noise
                
                # import pdb; pdb.set_trace()
                loc_tensor = None
                inter_dict = None
                
            else:
                noise_map = None
                loc_tensor = None
                inter_dict = None
                                 
            guidance_inp = out["comp_rgb"]     
            
            # if self.multidiffusion:                           
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, depth_map=depth_maps,  noise_map=noise_map, rgb_as_latents=False, 
                idx_map=loc_tensor, inter_dict=inter_dict
            )
        
        
        else:
            with torch.no_grad():
                
                points = self.geometry._xyz
                                
                if self.constant_noising:
                    if self.c_noise is None or self.global_step % self.cfg.cons_noise_alter == 1:
                        # import pdb; pdb.set_trace()
                        self.c_noise = torch.randn(points.shape[0], noise_channel).to(self.device)
                        self.i_noise = torch.randn(points.shape[0], noise_channel).to(self.device)
                        noise_tensor = self.c_noise
                        id_tensor = self.i_noise
                    else:
                        noise_tensor = self.c_noise
                        id_tensor = self.i_noise
                                        
                else:
                    noise_tensor = torch.randn(points.shape[0], noise_channel).to(self.device)
                    id_tensor = None
                                
                device = self.device

                noise_raster_settings = PointsRasterizationSettings(
                        image_size= 64,
                        radius = self.cfg.pts_radius,
                        points_per_pixel = 2
                    )
                
                surface_raster_settings = PointsRasterizationSettings(
                    image_size= 64,
                    radius = 0.08,
                    points_per_pixel = 2
                )
                
                cam_radius = batch["camera_distances"]
                
                # import pdb; pdb.set_trace()
                
                ############
    
                if self.three_noise:
                    
                    if self.cfg.nearby_fusing:
                        viewcomp_setting = "penta"
                    else:
                        viewcomp_setting = "all_views"
                    
                    noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
                                identical_noising=self.cfg.identical_noising, id_tensor=id_tensor, viewcomp_setting=viewcomp_setting)
                    noise_map = noised_maps
                    # loc_tensor = None
                    # inter_dict = None
                                        
                elif self.cfg.cons_noise_alter != 0:
                    if self.global_step % self.cfg.cons_noise_alter == 1:
                        self.c_noise = torch.randn(6,4,64,64).to(self.device)
                        noise_map = self.c_noise
                    else:
                        noise_map = self.c_noise
                        
                    loc_tensor = None
                    inter_dict = None
                    depth_masks = None
                    # fin_noise = noise_maps_tensor.permute(0,3,1,2)

                else:
                    noise_map = None
                    loc_tensor = None
                    inter_dict = None
                    depth_masks = None
                    
                if self.cfg.gradient_masking is False:
                    depth_masks = None
            
            # import pdb; pdb.set_trace()
                            
            guidance_inp = out["comp_rgb"]  
            
            if self.cfg.nearby_fusing:
                grad_setting = "penta"
            
            else:
                grad_setting = "after"
                                               
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, noise_map=noise_map, rgb_as_latents=False, 
                idx_map=loc_tensor, inter_dict=inter_dict, depth_masks=depth_masks, grad_setting=grad_setting
            )        

        guidance_inp = out["comp_rgb"]

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        # if self.C(self.cfg.loss.lambda_orient) > 0:
        #     if "normal" not in out:
        #         raise ValueError(
        #             "Normal is required for orientation loss, no normal is found in the output."
        #         )
        #     loss_orient = (
        #         out["weights"].detach()
        #         * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
        #     ).sum() / (out["opacity"] > 0).sum()
        #     self.log("train/loss_orient", loss_orient)
        #     loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

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
        
        import pdb; pdb.set_trace()
        
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
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

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
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