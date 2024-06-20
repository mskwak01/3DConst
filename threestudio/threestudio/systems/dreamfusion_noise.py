from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import torch.nn.functional as F


from threestudio.systems.pc_project import point_e, render_depth_from_cloud, render_noised_cloud, render_upscaled_noised_cloud
from threestudio.systems.point_noising import reprojector, pts_noise_upscaler, sphere_pts_generator, ray_reprojector
from threestudio.systems.pytorch3d.renderer import PointsRasterizationSettings

from torchvision.utils import save_image

@threestudio.register("dreamfusion-noise-system")
class DreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        threefuse: bool = True
        image_dir: str = "hello"
        tag: str = "no_tag"
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
        n_pts_upscaling: int = 9
        pytorch_three: bool = False
        noise_alter_interval: int = 10
        background_rand: str = "random"
        consistency_mask: bool = False
        reprojection_info: bool = False
        batch_size: int = 1
        constant_viewpoints: bool = False
        filename: str = "name"
        noise_channel: int = 4
        vis_every_grad: bool = False
        depth_warp: bool = True
        everyview_vis_iter: int = 300
        noise_interval_schedule: bool = True
        
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
            
            deg = torch.deg2rad(torch.tensor([self.calibration_value]) + 90.)
            rot_z = torch.tensor([[torch.cos(deg), -torch.sin(deg), 0],[torch.sin(deg), torch.cos(deg), 0],[0, 0, 1.]]).to(self.device)
            points_loc = (rot_z[None,...] @ points_loc[...,None]).squeeze()
            
            points.coords = points_loc            
            self.cond_pc = points
        
        self.noise_pts = None
        self.noise_vals = None
        self.background_noise_pts = None
        self.background_noise_vals = None
        
        self.noise_map_dict = {"fore": {}, "back": {}}
        self.noise_alter_interval = self.cfg.noise_alter_interval


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
                
        depth_maps = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, cali=90)
        depth_map = depth_maps.permute(0,2,3,1).detach()
        depths = render_out['depth'].permute(0,3,1,2)
        
        return {
            **render_out,
            "comp_depth" : F.interpolate(depths, size=(64,64), mode='bilinear'),
            "pts_depth" : depth_map,
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
        
        back_noise_maps = None
        prompt_utils = self.prompt_processor()
        
        batch_size = self.cfg.batch_size
        noise_channel = self.cfg.noise_channel

        device = self.device
        iteration = self.global_step

        if self.cfg.noise_interval_schedule:
            noise_schedule = [1200, 4000, 50000]
            interval_length = [100, 30, 12]
            
            if iteration == 0:
                self.noise_alter_interval = interval_length[0]
            if iteration == noise_schedule[0]:
                self.noise_alter_interval = interval_length[1]
                self.noise_pts = None
            elif iteration == noise_schedule[1]:
                self.noise_alter_interval = interval_length[2]  
                self.noise_pts = None
            else:
                pass       
        
        with torch.no_grad():
            if self.threefuse:
                points = torch.tensor(self.cond_pc.coords, dtype=torch.float32).clone().detach().to(self.device)
                
                depth_maps = out["pts_depth"].permute(0,3,1,2)  
                
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
                    points_per_pixel = 10
                )
                
                cam_radius = batch["camera_distances"]
                depth_maps = out["pts_depth"].permute(0,3,1,2)                
                
                ############
                if self.three_noise:
                                        
                    if self.noise_pts is None or iteration % self.noise_alter_interval == 0:
                        num_points = points.shape[0]
                        
                        noise_tensor = torch.randn(num_points, noise_channel).to(self.device)
                        loc_rand = torch.randn(num_points, 3, self.cfg.n_pts_upscaling)
                        feat_rand = torch.randn(num_points, noise_channel, self.cfg.n_pts_upscaling)
                        
                        self.noise_pts, self.noise_vals = pts_noise_upscaler(points, noise_tensor, noise_channel, self.cfg.n_pts_upscaling, loc_rand, feat_rand, self.device)
                        
                        if self.cfg.background_rand == "ball":
                            self.background_noise_pts, self.background_noise_vals = sphere_pts_generator(self.device, noise_channel)

                        self.noise_map_dict = {"fore": {}, "back": {}}
        
                    surf_map = render_depth_from_cloud(points, batch, surface_raster_settings, cam_radius, device, dynamic_points=True, cali=90, raw=True)           

                    if self.cfg.constant_viewpoints:
                        key_list = [f"{k[0]}_{k[1]}" for k in batch['idx_keys']]
                        fore_noise_list = [None for i in range(len(key_list))]
                        back_noise_list = [None for i in range(len(key_list))]
                        new_rend = []
                        new_keys = []
                        
                        # if self.global_step == 50:
                        #     import pdb; pdb.set_trace()
                        
                        for i, key in enumerate(key_list):
                            if key in self.noise_map_dict["fore"].keys():
                                fore_noise_list[i] = self.noise_map_dict["fore"][key]
                                
                                if self.cfg.background_rand == "ball":
                                    back_noise_list[i] = self.noise_map_dict["back"][key]
                                    
                            else:
                                new_rend.append(i)
                                new_keys.append(key)
                                                    
                        if len(new_rend) != 0:
                            new_fore_noise_maps = reprojector(self.noise_pts, self.noise_vals, batch['c2w'][new_rend], torch.linalg.inv(batch['c2w'][new_rend]), batch["fovy"][new_rend], self.device, ref_depth=surf_map[new_rend], noise_channel=noise_channel, thresh=0.12).nan_to_num()
                            
                            if self.cfg.background_rand == "ball":
                                new_back_noise_maps = reprojector(self.background_noise_pts, self.background_noise_vals, batch['c2w'][new_rend], torch.linalg.inv(batch['c2w'][new_rend]), batch["fovy"][new_rend], self.device, img_size=64, background=True, noise_channel=noise_channel).nan_to_num()    
                            # import pdb; pdb.set_trace()
                            
                            for k, idx in enumerate(new_rend):
                                self.noise_map_dict["fore"][new_keys[k]] = new_fore_noise_maps[k]
                                fore_noise_list[idx] = new_fore_noise_maps[k]
                                
                                if self.cfg.background_rand == "ball":
                                    self.noise_map_dict["back"][new_keys[k]] = new_back_noise_maps[k]
                                    back_noise_list[idx] = new_back_noise_maps[k]
                                    
                        fore_noise_maps = torch.stack(fore_noise_list)
                        
                        if self.cfg.background_rand == "ball":
                            back_noise_maps = torch.stack(back_noise_list)
                            
                    else:
                        fore_noise_maps = reprojector(self.noise_pts, self.noise_vals, batch['c2w'], torch.linalg.inv(batch['c2w']), batch["fovy"], self.device, ref_depth=surf_map, noise_channel=noise_channel, thresh=0.12).nan_to_num()
                                            
                    # import pdb; pdb.set_trace()
                    
                    # if self.cfg.reprojection_info is False:

                    # else:
                    #     fore_noise_maps, proj_loc, idx_maps = reprojector(self.noise_pts, self.noise_vals, batch['c2w'], torch.linalg.inv(batch['c2w']), 
                    #                                                       batch["fovy"], self.device, ref_depth=surf_map, reprojection_info=self.cfg.reprojection_info)
                    #     fore_noise_maps = fore_noise_maps.nan_to_num()
                    
                    # import pdb; pdb.set_trace()
                        
                    if self.cfg.background_rand == "random":
                        back_mask = (fore_noise_maps == 0.).float()
                        noise_map = back_mask * torch.randn_like(fore_noise_maps) + fore_noise_maps
                        
                    elif self.cfg.background_rand == "ball":
                        if back_noise_maps is None:               
                            back_noise_maps = reprojector(self.background_noise_pts, self.background_noise_vals, batch['c2w'], torch.linalg.inv(batch['c2w']), batch["fovy"], self.device, img_size=64, background=True, noise_channel=noise_channel).nan_to_num()                     
                        back_mask = (fore_noise_maps == 0.).float()
                        noise_map = back_mask * back_noise_maps + fore_noise_maps
                    
                    elif self.cfg.background_rand == "same":
                        back_noise_maps = torch.randn_like(fore_noise_maps[0])[None,...].repeat(fore_noise_maps.shape[0],1,1,1)
                        back_mask = (fore_noise_maps == 0.).float()
                        noise_map = back_mask * back_noise_maps + fore_noise_maps
                    
                    else:
                        print("Background option not implemented yet!!")
                    
                    loc_tensor = None
                    inter_dict = None
                    depth_masks = None
                                            
                else:
                    noise_map = None   
                    depth_masks = None                  
                
        guidance_inp = out["comp_rgb"]  
                    
        if self.cfg.depth_warp:                
            dn_rays_d = F.interpolate(batch["rays_d"].permute(0,3,1,2), size=(64,64), mode='bilinear')
            dn_rays_o = batch["rays_o"][:,0,0][...,None,None].repeat(1,1,64,64)
            
            re_dict = ray_reprojector(batch_size, dn_rays_d, dn_rays_o, out["comp_depth"], batch['c2w'], torch.linalg.inv(batch['c2w']), batch["fovy"], self.device, img_size=64)

        else:
            re_dict = None
        # import pdb; pdb.set_trace()
        
        if self.threefuse:
            pts_depth_maps = out["pts_depth"].permute(0,3,1,2)     
        else:
            pts_depth_maps = None
        
        # import pdb; pdb.set_trace()
    
        guidance_out = self.guidance(
            guidance_inp, prompt_utils, **batch, noise_map=noise_map, rgb_as_latents=False, 
            depth_masks=depth_masks, re_dict=re_dict, iter = iteration, filename = self.cfg.filename,
            pts_depth_maps=pts_depth_maps, depth_maps = out["comp_depth"]
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
                    "img": out["pts_depth"][0, :, :, 0],
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
