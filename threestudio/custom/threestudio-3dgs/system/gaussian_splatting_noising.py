import math
from dataclasses import dataclass

import torch.nn.functional as F
import numpy as np
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast

from ..geometry.gaussian import BasicPointCloud, Camera

from threestudio.systems.pc_project import point_e, render_depth_from_cloud, render_noised_cloud, render_upscaled_noised_cloud, reprojector
from threestudio.systems.pytorch3d.renderer import PointsRasterizationSettings

from torchvision.utils import save_image
import open3d as o3d


@threestudio.register("gaussian-splatting-noising-system")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        threefuse: bool = True
        image_dir: str = "hello"
        gaussian_dynamic : bool = False
        calibration_value: int = 0
        three_noise: bool = False
        multidiffusion: bool = False
        identical_noising: bool = False
        gs_noise: bool = False
        obj_only: bool = False
        constant_noising: bool = False
        cons_noise_alter: int = 0
        pts_radius: float = 0.02
        surf_radius: float = 0.05
        tag: str = "no_tag"
        gradient_masking: bool = False
        nearby_fusing: bool = False
        three_warp_noise: bool = False
        consider_depth: bool = True
        gau_d_cond: bool = True
        n_pts_upscaling: int = 9

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False
        
        self.threefuse = self.cfg.threefuse
        self.image_dir = self.cfg.image_dir
        self.gaussian_dynamic = self.cfg.gaussian_dynamic
        self.three_noise = self.cfg.three_noise
        self.multidiffusion = self.cfg.multidiffusion
        
        self.gs_noise = self.cfg.gs_noise
        self.obj_only = self.cfg.obj_only
        self.constant_noising = self.cfg.constant_noising

        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
        # Point-e initialization
        # if self.threefuse is True:
        
        # canvas = torch.zeros(1,512,512)
        # canvas[0,20,430] = 1.
        # canvas[0,420,230] = 1.
        # canvas[0,154,433] = 1.
        # canvas[0,254,500] = 1.
        
        
        # import pdb; pdb.set_trace()
            
        self.cond_pc = point_e(device="cuda", exp_dir=self.image_dir)
        self.calibration_value = self.cfg.calibration_value
        
        self.c_noise = None
        self.i_noise = None
        
        pcd = self.pcb()
    
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        
        self.noise_tensor = None
            

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            self.optim_num = 2
            return [optim, net_optim]
        self.optim_num = 1
        return [optim]

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.cfg.position_lr_max_steps
        scale_lr_max_steps = self.geometry.cfg.scale_lr_max_steps

        if self.global_step < lr_max_step:
            self.geometry.update_xyz_learning_rate(self.global_step)

        if self.global_step < scale_lr_max_steps:
            self.geometry.update_scale_learning_rate(self.global_step)

        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        normals = []
        depths = []
        masks = []
        noise_image = []
        # back_noise = []
        
        # import pdb; pdb.set_trace()
        
        for batch_idx in range(bs):
            batch["batch_idx"] = batch_idx
            fovy = batch["fovy"][batch_idx]
                        
            w2c, proj, cam_p = get_cam_info_gaussian(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.renderer(
                    viewpoint_cam, self.background_tensor, **batch
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
                if render_pkg.__contains__("normal"):
                    normals.append(render_pkg["normal"])
                if render_pkg.__contains__("depth"):
                    depths.append(render_pkg["depth"])
                if render_pkg.__contains__("mask"):
                    masks.append(render_pkg["mask"])
                if render_pkg.__contains__("noise_image"):
                    noise_image.append(render_pkg["noise_image"])
                # if render_pkg.__contains__("back_noise_image"):
                #     back_noise.append(render_pkg["back_noise_image"])
                
        if self.gaussian_dynamic:
            points = self.geometry._xyz
        else:
            points = self.cond_pc
            
        # Change it to dynamic
                    
        # if self.threefuse:
        device = self.device
        
        raster_settings = PointsRasterizationSettings(
                image_size= 800,
                radius = 0.01,
                points_per_pixel = 2
            )
        
        cam_radius = batch["camera_distances"]
                   
        depth_maps = render_depth_from_cloud(points, batch, raster_settings, cam_radius, device, dynamic_points=self.gaussian_dynamic, cali=90)
        
        # import pdb; pdb.set_trace()
        
        ### Disparity Calculation
        
        if self.cfg.gau_d_cond:
            gau_depths = torch.stack(depths)
            mask = 1 - (gau_depths < 0.8).float() 
            masked_depth = mask * gau_depths

            focal_length = 1.4520
            disparity = focal_length / (-(masked_depth == 0).float() + masked_depth + 1e-9)
            # disparity = disparity * (disparity < 4).float()
            # max_disp = torch.amax(disparity, dim=(2,3))[...,None,None]
    
            min_disp_list = []
            max_disp_list = []
            
            for disp in disparity:
                max = torch.max(disp)
                min = torch.min(disp[disp > 0])
                min_disp_list.append(min)
                max_disp_list.append(max)

            min_disp = torch.stack(min_disp_list)[...,None,None,None]
            max_disp = torch.stack(max_disp_list)[...,None,None,None]
            
            norm_disparity = (disparity - min_disp) / (max_disp - min_disp)
    
            depth_map = norm_disparity.repeat(1,3,1,1).permute(0,2,3,1).detach()
            
            # import pdb; pdb.set_trace()
            # depth_maps = gau_depths / torch.amax(gau_depths, dim=(2,3))[...,None,None]
                    
        else:
            depth_map = depth_maps.permute(0,2,3,1).detach()
        

        ################### NOISE #############
        
        device = self.device
        
        if self.cfg.three_warp_noise:
            pts_per_pix = 50
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
        
        

        noise_channel = 4

        
        id_tensor = None
        
        # import pdb; pdb.set_trace()
        
        # noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
        #                     dynamic_points=self.gaussian_dynamic, cali=90, identical_noising=self.cfg.identical_noising)
        
        if self.noise_tensor is None:
            num_points = points.shape[0]
            
            self.noise_tensor = torch.randn(num_points, noise_channel).to(self.device)
            self.loc_rand = torch.randn(num_points, 3, self.cfg.n_pts_upscaling)
            self.feat_rand = torch.randn(num_points, 4, self.cfg.n_pts_upscaling)
        
        noise_tensor = self.noise_tensor
        loc_rand = self.loc_rand
        feat_rand = self.feat_rand        
            
        # if not self.cfg.three_warp_noise:
        #     noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
        #                                 dynamic_points=self.gaussian_dynamic, cali=90, identical_noising=self.cfg.identical_noising)

        #     noise_map = noised_maps
            
        # else:
        #     noised_maps, loc_tensor, inter_dict, depth_masks = render_upscaled_noised_cloud(points, 
        #                                                                                     batch, 
        #                                                                                     noise_tensor, 
        #                                                                                     surface_raster_settings, 
        #                                                                                     up_noise_raster_settings, 
        #                                                                                     noise_channel, 
        #                                                                                     cam_radius, 
        #                                                                                     device, 
        #                                                                                     cali=90, 
        #                                                                                     dynamic_points=self.gaussian_dynamic, 
        #                                                                                     identical_noising=self.cfg.identical_noising, 
        #                                                                                     consider_depth=self.cfg.consider_depth, 
        #                                                                                     loc_rand=loc_rand, 
        #                                                                                     feat_rand=feat_rand,
        #                                                                                     n_upscaling=self.cfg.n_pts_upscaling)
        #     noise_map = noised_maps         
            
        # F.interpolate
        
        with torch.no_grad():
            surf_map = render_depth_from_cloud(points, batch, surface_raster_settings, cam_radius, device, dynamic_points=self.gaussian_dynamic, cali=90, raw=True)
            
            hey = reprojector(points, self.noise_tensor, batch['c2w'], torch.linalg.inv(batch['c2w']), batch["fovy"], self.device, ref_depth=surf_map)

        
        
        noised_maps = F.interpolate(noised_maps, size=(64,64), mode='nearest')
        init_image = F.interpolate(noised_maps, size=(512,512), mode='nearest')
        
        depth_map = init_image[:,:3].permute(0,2,3,1)
        
        
        #######################################

        # import pdb; pdb.set_trace()

        
        outputs = {
            "comp_rgb": torch.stack(renders, dim=0).permute(0, 2, 3, 1),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
            "pts_depth" : depth_map,
            # "noise_image": noise_img,
            # "back_noise_image": back_noise_img
        }
                
        if len(normals) > 0:
            outputs.update(
                {
                    "comp_normal": torch.stack(normals, dim=0).permute(0, 2, 3, 1),
                    "comp_depth": torch.stack(depths, dim=0).permute(0, 2, 3, 1),
                    "comp_mask": torch.stack(masks, dim=0).permute(0, 2, 3, 1),
                }
            )
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        if self.optim_num == 1:
            opt = self.optimizers()
        else:
            opt, net_opt = self.optimizers()
            
        out = self(batch)
        
        noise_channel = 4
        
        if self.threefuse:
            with torch.no_grad():                
                if self.gaussian_dynamic:
                    points = self.geometry._xyz
                    noise_tensor = torch.randn(points.shape[0], noise_channel).to(self.device)
                else:
                    points = self.cond_pc
                    points_num = torch.tensor(points.coords, dtype=torch.float32).shape[0]
                    noise_tensor = torch.randn(points_num, noise_channel).to(self.device)
                    
                # Change it to dynamic
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
                depth_maps = out["pts_depth"].permute(0,3,1,2)
                
                # import pdb; pdb.set_trace()
                
                if self.three_noise:
                    if not self.cfg.three_warp_noise:
                        noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
                                                    dynamic_points=self.gaussian_dynamic, cali=90, identical_noising=self.cfg.identical_noising)
        
                        noise_map = noised_maps
                        
                    else:                        
                        noise_tensor = self.noise_tensor
                        loc_rand = self.loc_rand
                        feat_rand = self.feat_rand       
                        
                        noised_maps, loc_tensor, inter_dict, depth_masks = render_upscaled_noised_cloud(points, 
                                                                                                        batch, 
                                                                                                        noise_tensor, 
                                                                                                        surface_raster_settings, 
                                                                                                        up_noise_raster_settings, 
                                                                                                        noise_channel, 
                                                                                                        cam_radius, 
                                                                                                        device, 
                                                                                                        cali=90, 
                                                                                                        dynamic_points=self.gaussian_dynamic, 
                                                                                                        identical_noising=self.cfg.identical_noising, 
                                                                                                        consider_depth=self.cfg.consider_depth, 
                                                                                                        loc_rand=loc_rand, 
                                                                                                        feat_rand=feat_rand,
                                                                                                        n_upscaling=self.cfg.n_pts_upscaling)
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
                                 
                if self.cfg.gradient_masking is False:
                    depth_masks = None
                    
            guidance_inp = out["comp_rgb"]     
            
            # if self.multidiffusion:            
            
            # import pdb; pdb.set_trace()
                           
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, depth_map=depth_maps,  noise_map=noise_map, rgb_as_latents=False, 
                idx_map=loc_tensor, inter_dict=inter_dict
            )
            
            guidance_out = self.guidance(guidance_inp, self.prompt_utils, **batch, depth_map=depth_maps,  noise_map=noise_map, rgb_as_latents=False, idx_map=loc_tensor, inter_dict=inter_dict)
        
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
                depth_maps = out["pts_depth"].permute(0,3,1,2)                
                # import pdb; pdb.set_trace()
                
                ############
    
                if self.three_noise:
                    if self.cfg.nearby_fusing:
                        viewcomp_setting = "penta"
                    else:
                        viewcomp_setting = "all_views"
                    
                    if not self.cfg.three_warp_noise:
                        noised_maps, loc_tensor, inter_dict, depth_masks = render_noised_cloud(points, batch, noise_tensor, noise_raster_settings, surface_raster_settings, noise_channel, cam_radius, device, 
                                    dynamic_points=self.gaussian_dynamic, cali=90, identical_noising=self.cfg.identical_noising, id_tensor=id_tensor, viewcomp_setting=viewcomp_setting)
                        noise_map = noised_maps

                    else:                        
                        noise_tensor = self.noise_tensor
                        loc_rand = self.loc_rand
                        feat_rand = self.feat_rand       
                        
                        noised_maps, loc_tensor, inter_dict, depth_masks = render_upscaled_noised_cloud(points, 
                                                                                                        batch, 
                                                                                                        noise_tensor, 
                                                                                                        surface_raster_settings, 
                                                                                                        up_noise_raster_settings, 
                                                                                                        noise_channel, 
                                                                                                        cam_radius, 
                                                                                                        device, 
                                                                                                        cali=90, 
                                                                                                        dynamic_points=self.gaussian_dynamic, 
                                                                                                        identical_noising=self.cfg.identical_noising, 
                                                                                                        consider_depth=self.cfg.consider_depth, 
                                                                                                        loc_rand=loc_rand, 
                                                                                                        feat_rand=feat_rand,
                                                                                                        n_upscaling=self.cfg.n_pts_upscaling)
                        noise_map = noised_maps                                          
                    
                                        
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


        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        viewspace_point_tensor = out["viewspace_points"]

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        xyz_mean = None
        
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_sum = torch.sum(self.geometry.get_scaling)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        # import pdb; pdb.set_trace()

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.global_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        if loss > 0:
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if self.optim_num > 1:
            net_opt.step()
            net_opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds}

    def validation_step(self, batch, batch_idx):
        out = self(batch)        

        self.save_image_grid(
            
            f"it{self.global_step}-{batch['index'][0]}.png",
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
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["pts_depth"][0],
                        "kwargs": {"data_range": (0, 1)},
                    }
                ]
                if "pts_depth" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
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
                        "img": out["pts_depth"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "pts_depth" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
        
        
    def shape(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              
        points = self.cond_pc
        device = self.device
        scale = 0.4
        coords = torch.tensor(points.coords, dtype=torch.float32).to(device)
        
        rgb = torch.tensor(
            np.stack(
                [points.channels["R"], points.channels["G"], points.channels["B"]], axis=-1
            ),
            dtype=torch.float32,
        ).to(device)
        
        n_coords = coords.cpu().numpy()
        n_rgb = rgb.cpu().numpy()

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(n_coords)
        point_cloud.colors = o3d.utility.Vector3dVector(n_rgb)
        self.point_cloud = point_cloud

        return n_coords, n_rgb, scale


    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        
        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 300000
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))

        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        points_inside = []
        color_inside= []
        
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        
        return all_coords, all_rgb


    def pcb(self):
        
        coords,rgb,scale = self.shape()
        all_coords, all_rgb = self.add_points(coords,rgb)
        
        # import pdb; pdb.set_trace()
        
        # matching_rotation = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]], dtype=torch.float32).to(self.device)
        # all_coords = (matching_rotation[0] @ all_coords[...,None]).squeeze()
        
        # all_coords
        
        fin_rgb = 0.4 * torch.ones_like(torch.tensor(all_rgb)).to(self.device)
        
        # calibration_value = 180
        
        deg = torch.deg2rad(torch.tensor([self.calibration_value]))
        rot_z = torch.tensor([[torch.cos(deg), -torch.sin(deg), 0],[torch.sin(deg), torch.cos(deg), 0],[0, 0, 1.]]).to(self.device)
        fin_coords = (rot_z[None,...] @ all_coords[...,None]).squeeze()
                
        pcd = BasicPointCloud(points=fin_coords, colors=fin_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd


    def on_load_checkpoint(self, ckpt_dict) -> None:
        # num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        # pcd = BasicPointCloud(
        #     points=np.zeros((num_pts, 3)),
        #     colors=np.zeros((num_pts, 3)),
        #     normals=np.zeros((num_pts, 3)),
        # )
        
        pcd = self.pcb()
        
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)
