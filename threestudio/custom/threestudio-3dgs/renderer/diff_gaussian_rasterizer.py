import math
from dataclasses import dataclass

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

from torchvision.utils import save_image

@threestudio.register("diff-gaussian-rasterizer")
class DiffGaussian(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        scaling_modifier: float = 1.0

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,        
        back_var = None,
        scaling_modifier=1.0,
        override_color=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if self.training and back_var is None:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        elif self.training and back_var is not None:
            invert_bg_color = True if back_var==0 else False
        else:
            invert_bg_color = False
        # import pdb; pdb.set_trace()

        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)

        pc = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        
        # raster_settings_2 = GaussianRasterizationSettings(
        #     image_height=64,
        #     image_width=64,
        #     tanfovx=tanfovx,
        #     tanfovy=tanfovy,
        #     bg=bg_color,
        #     scale_modifier=0.2,
        #     viewmatrix=viewpoint_camera.world_view_transform,
        #     projmatrix=viewpoint_camera.full_proj_transform,
        #     sh_degree=pc.active_sh_degree,
        #     campos=viewpoint_camera.camera_center,
        #     prefiltered=False,
        #     debug=False,
        # )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # rasterizer_2 =  GaussianRasterizer(raster_settings=raster_settings_2)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        
        ## 
        
        # Override colors
        
        # shs = torch.randn_like(pc.get_features)
        
        ##
        
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color
            
        # import pdb; pdb.set_trace()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        
        # import pdb; pdb.set_trace()
                
        # var_mul = 2.5
        
        # with torch.no_grad():
        
            # colors_1 = torch.randn_like(means3D) * var_mul   
            # colors_2 = torch.randn_like(means3D) * var_mul   
            
            # result_list_2 = rasterizer_2(
            #     means3D=means3D,
            #     means2D=means2D,
            #     shs=None,
            #     colors_precomp=colors_1,
            #     opacities=torch.ones_like(opacity),
            #     scales=scales,
            #     rotations=rotations,
            #     cov3D_precomp=cov3D_precomp,
            # )
                                
            # result_list_3 = rasterizer_2(
            #     means3D=means3D,
            #     means2D=means2D,
            #     shs=None,
            #     colors_precomp=colors_2,
            #     opacities=torch.ones_like(opacity),
            #     scales=scales,
            #     rotations=rotations,
            #     cov3D_precomp=cov3D_precomp,
            # )
            
            # result_2 = result_list_2[0].detach()
            # result_3 = result_list_3[0].detach()
                    
            # mask = (result_list_2[0][0] != 0.)[None,...] * (result_list_2[0][0] != 1.)[None,...]
            # # indexes = torch.nonzero(mask, as_tuple=False)
            # # hello = result_list_2[0][indexes[:,0],indexes[:,1],indexes[:,2]]
            
            # noise_image = mask * torch.cat((result_2, result_3[2:,...]),dim=0)
            
            # back_noise_image = (1 - mask.float()) * torch.randn_like(noise_image)
                
        
        # import pdb; pdb.set_trace()
                
        # pseudo_feat = torch.randn_like(pc.get_rotation).to(self.device)
        
        # result_list_2 = rasterizer(
        #     means3D=means3D,
        #     means2D=means2D,
        #     shs=shs,
        #     colors_precomp=colors_precomp,
        #     opacities=opacity,
        #     scales=scales,
        #     rotations=pseudo_feat,
        #     cov3D_precomp=cov3D_precomp,
        # )
        
        # rendered_image, radii = result_list[0], result_list[1]
        
        noise_image = None
        
        # rendered_noise_image, radii = result_list_2[0], result_list_2[1]
        
        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.clamp(0, 1),
            "depth": rendered_depth,
            # "alpha": rendered_alpha,
            "noise_render": rendered_image.clamp(0, 1),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "noise_image" : noise_image,
            # "back_noise_image": back_noise_image
        }
