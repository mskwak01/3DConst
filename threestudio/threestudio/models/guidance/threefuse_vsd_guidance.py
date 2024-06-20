import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from torchvision.utils import save_image
import matplotlib.pyplot as plt

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype, this_device):
        super().__init__()
        self.module = module.to(this_device)
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("threefuse-vsd-guidance")
class StableDiffusionVSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-1"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"
        multi_textenc: bool = True
        add_loss: str = "no_loss"
        add_loss_stepping: bool = False
        use_normalized_grad: bool = False
        vis_grad: bool = False
        visualization_type: str = "grad"
        backprop_grad: bool = False
        grad_cons_mask: bool = False
        mask_w_timestep: bool = False
        
        debugging: bool = False
        high_timesteps: bool = False
        grad_thresh: float = 2.0
        vis_every_thresh: float = 4.0
        use_disp_loss: bool = False
        use_sim_loss: bool = False
        
        weight_sim_loss: float = 5.0
        weight_disp_loss: float = 0.5
        disp_loss_to_latent: bool = False
        only_geo: bool = False
        only_geo_front_on: bool = False
        geo_start_int: int = 50
        geo_interval: bool = False
        geo_interval_len: int = 400
        geo_re_optimize: bool = False
        geo_interv_different: bool = False
        geo_intr_on: int = 5
        geo_intr_off: int = 10

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionControlNetPipeline
            pipe_lora: StableDiffusionPipeline

        # "/host_files/matthew/matt_threestudio/threestudio/controlnet"
        # breakpoint()

        # "/home/dreamer/host_files/matthew/matt_threestudio/threestudio/controlnet"

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if (
            self.cfg.pretrained_model_name_or_path
            == self.cfg.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = StableDiffusionPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            ).to(self.device)
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        # import pdb; pdb.set_trace()

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        self.controlnet = self.pipe.controlnet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype, self.device
        )
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()
        ###############################################
        lora_attn_procs_ = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs_[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet.set_attn_processor(lora_attn_procs_)

        self.lora_layers_ = AttnProcsLayers(self.unet.attn_processors)
        self.lora_layers_._load_state_dict_pre_hooks.clear()
        self.lora_layers_._state_dict_hooks.clear()
        ############################################
        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.cos_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        
        self.int_counter = 0
        self.sds_on = True
        
        if self.cfg.use_normalized_grad:
            self.mse_loss = nn.MSELoss(reduction="mean")
            self.cos_loss = nn.CosineEmbeddingLoss(reduction="mean")
        else:
            self.mse_loss = nn.MSELoss(reduction="sum")
            self.cos_loss = nn.CosineEmbeddingLoss(reduction="sum")

        threestudio.info(f"Loaded Stable Diffusion!")
        
        
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    def sample_lora(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding


    def visualize_all(self, imgs, depths, azimuth, name = "test", iter=0):
        
        if not os.path.exists(name):
            os.makedirs(name)
        
        for i, img in enumerate(imgs):
            fin = torch.cat((img, depths[i]), dim=0)
            save_image(fin, str(name) + f'/_{azimuth[i]:.4f}_deg' + '.png')
        
        return None
    

    def warp_visualizer(self, rep, projections, depth_masks, re_dict, vis_magnitude=False):
        
        target_rep = rep[re_dict["tgt_ind"]]
        warped_gradients = depth_masks * F.grid_sample(target_rep, projections, mode='nearest')
        
        sim_maps = []
        
        for i, src in enumerate(re_dict["src_ind"]):
            sim = self.cos_similarity(rep[src], warped_gradients[i])[None,...]
            sim_maps.append(sim)
                    
        sims = torch.stack(sim_maps)
        
        valid_sim = (1 - depth_masks) * -1 + sims
        # negative_sim = (1 - depth_masks) * -1 - sims
        
        fin_sim_maps = torch.cat((-torch.ones_like(sim)[None,...], valid_sim[:2], -torch.ones_like(sim)[None,...], valid_sim[2:4]),dim=0).squeeze()
        
        if vis_magnitude:
            normed_rep = rep.norm(dim=1) / rep.norm(dim=1).mean((1,2))[...,None,None]
            fin_mag_maps = normed_rep / 1.4
            
            d_1 = (1 - (1 - depth_masks[0]) * (1 - depth_masks[1]))
            d_2 = (1 - (1 - depth_masks[2]) * (1 - depth_masks[3]))
            
            masks = torch.cat((d_1[None,...], depth_masks[:2], d_2[None,...], depth_masks[2:4]), dim=0).squeeze()
            fin_mag_maps = (1 - masks) * -1 + masks * fin_mag_maps
        
        else:
            fin_mag_maps = None
        
        return fin_sim_maps, fin_mag_maps

    
    def grad_warp(self, re_dict, grad, timestep = [0], name = "test", iter=0, depths=None, azimuth=None, guide_utils = None, vis_grad=False):
    
        src_inds = torch.tensor(re_dict["src_ind"])
        num_multiview = src_inds.shape[0] // torch.unique(src_inds).shape[0]
        center_idx = torch.unique(src_inds)
        num_sets = center_idx.shape[0]
        
        gradient_masker = None
        
        return_dict = {}
    
        # re_dict = kwargs["re_dict"]
        keylist = [key for key in re_dict["proj_maps"].keys()]
        
        projections = []
        depth_masks = []
                    
        for key in keylist:
            projections.append(re_dict["proj_maps"][key])
            depth_masks.append(re_dict["depth_masks"][key])
                    
        projections = torch.stack(projections)
        depth_masks = torch.stack(depth_masks)
        
        if self.cfg.visualization_type == "grad":
            tgt_grads = grad[re_dict["tgt_ind"]]
        elif self.cfg.visualization_type == "e_pos":
            # import pdb; pdb.set_trace()       
            if self.cfg.use_normalized_grad:
                # import pdb; pdb.set_trace()
                grad = guide_utils["e_pos"]
                grad = grad / grad.norm(dim=1).mean()
            else:     
                grad = guide_utils["e_pos"]
            
            tgt_grads = grad[re_dict["tgt_ind"]]
        else:
            print("Not implemented yet")
        
        warped_gradients = depth_masks * F.grid_sample(tgt_grads, projections, mode='nearest')
        
        return_dict["warped_grad"] = warped_gradients
        return_dict["depth_masks"] = depth_masks
        
        similiarty_maps = []
        mse_maps = []
        depth_maps = []
        
        depths = F.interpolate(depths, size=(64,64))[:,0]
        tg = torch.ones_like(grad[0].reshape(4,-1)[0])
        
        add_loss = 0.
        
        dis_grad_1 = None
        dis_grad_2 = None
        
        if self.cfg.debugging:
            # import pdb; pdb.set_trace()
            
            mean_grad_1 = (depth_masks[0] * grad[0] + warped_gradients[0]) / 2
            mean_grad_2 = (depth_masks[2] * grad[3] + warped_gradients[2]) / 2
            
            dis_grad_1 = grad[0] - mean_grad_1
            dis_grad_2 = grad[3] - mean_grad_2
            
            return_dict["d1"] = dis_grad_1
            return_dict["d2"] = dis_grad_2
                
        for i, src in enumerate(re_dict["src_ind"]):
            # sim = self.cos_similarity(grad[src], warped_gradients[i])[None,...]
            # similiarty_maps.append(sim)
            
            # if self.cfg.debugging:
            #     import pdb; pdb.set_trace()
            
            if self.cfg.add_loss == "no_loss":
                add_loss = 0.
            
            elif self.cfg.add_loss == "cosine_sim":
                # if azimuth[src] < -30. or azimuth[src] > 30.:
                add_loss += self.cos_loss(grad[src].reshape(4,-1).permute(1,0), warped_gradients[i].reshape(4,-1).permute(1,0), target=tg)
                
                # if self.cfg.debugging:
                #     import pdb; pdb.set_trace()

            elif self.cfg.add_loss == "cosine_dissim":
                # import pdb; pdb.set_trace()
                add_loss += 10 * self.cos_loss(grad[src].reshape(4,-1).permute(1,0), warped_gradients[i].reshape(4,-1).permute(1,0), target=-tg)
                
            elif self.cfg.add_loss == "mse_loss":
                add_loss += self.mse_loss(grad[src] * depth_masks[i], warped_gradients[i])
            
            mse = ((grad[src] * depth_masks[i] - warped_gradients[i]) ** 2).mean(dim=0)
            mse_maps.append(mse)
            depth_maps.append(depths[src])
        
        mses = torch.stack(mse_maps)
        mse_mask = (mses >= self.cfg.grad_thresh).float()
        
        # if iter > 700:
        # import pdb; pdb.set_trace()
            
        if self.cfg.grad_cons_mask:

            with torch.no_grad():
                
                img_size = projections.shape[1]
                batch_size = grad.shape[0]
                
                coord_loc = projections.reshape(-1,2).fliplr().reshape(len(keylist),-1,2)
                proj_loc = (coord_loc * (img_size / 2) + (img_size / 2)).clamp(0,img_size-0.51)
                proj_pix = torch.round(proj_loc * depth_masks.permute(0,2,3,1).reshape(len(keylist), -1, 1)).int()
                
                mse_mask[:,0,0] = 0.
                
                # Temporary ###############################
                
                ma_1 = (1 - (1 - mse_mask[0]) * (1 - mse_mask[1]))
                ma_2 = (1 - (1 - mse_mask[2]) * (1 - mse_mask[3]))
                
                comb_mse_mask = torch.stack((ma_1, ma_1, ma_2, ma_2))
                mse_val = comb_mse_mask.reshape(len(keylist), -1)
                
                ########################################

                y_c = proj_pix[...,0]
                x_c = proj_pix[...,1]
                
                w_mse_masks = []
                
                tgt = 0
                src = 0
                            
                for k in range(batch_size):
                    
                    if k in re_dict["tgt_ind"]:
                    
                        m_canvas = torch.zeros(img_size, img_size).to(self.device)
                        m_canvas[y_c[tgt], x_c[tgt]] = mse_val[tgt]
                        w_mse_masks.append(m_canvas)
                        tgt += 1
                    
                    else:
                        m_canvas = 1 - mse_mask[src]
            
                        for i in range(num_multiview-1):
                            m_canvas *= 1 - mse_mask[src+i+1]
                        w_mse_masks.append(1 - m_canvas)
                        src += num_multiview
            
                gradient_masker = 1 - torch.stack(w_mse_masks).unsqueeze(1)
                
                return_dict["grad_mask"] = gradient_masker
        
        # if self.cfg.debugging:
        #     if iter > 600:
        #         import pdb; pdb.set_trace()
                
        #         depth_maps = torch.stack(depth_maps)
        #         save_image(depth_maps.unsqueeze(1), "depth_maps.png")
        #         save_image(mse_mask.unsqueeze(1), "mse_masks.png")
        #         save_image((similiarty_maps[3] > 0.9).float(), "sim.png")

                    # m_canvas[y_c[k], x_c[k]] = depth_masks[0].reshape(4096)
              
        # re_dict["src_ind"].shape / torch.unique(re_dict["src_ind"]).shape
        
        ##############
        
        if vis_grad:
            
            # 
                                        
                ##############        
            
                # var_canvas = torch.zeros(num_sets, 1+num_multiview, grad.shape[1], grad.shape[2], grad.shape[3]).to(self.device)        
                # c = 0  
                
                # for i, ct in enumerate(center_idx):
                #     var_canvas[i,0] = grad[ct]
                #     w_mask = torch.ones_like(depth_masks[0])
                    
                #     for k in range(num_multiview):
                #         var_canvas[i, k+1]= warped_gradients[c]
                #         w_mask *= depth_masks[c]
                        
                #         c += 1
                    
                #     var_canvas[i] = var_canvas[i] * w_mask[None,...]

                # visualize_grad_var = torch.var(var_canvas,dim=1).mean(1).repeat_interleave(center_idx.shape[0], dim=0)  
                
                ##############
                
            everything = [depths]
            
            visualize_keys = ["noise_uncond", "noise_text", "noise_pred"]
            
            grad_sim, grad_mag = self.warp_visualizer(grad, projections, depth_masks, re_dict, vis_magnitude=False)
            everything.append(grad_sim)
            
            num_col = grad_sim.shape[0]
            row_names = [""] * num_col + ["grad_sim"] * num_col
                
            for key in visualize_keys:
            
                rep_sim, rep_mag = self.warp_visualizer(guide_utils[key], projections, depth_masks, re_dict, vis_magnitude=True)
                everything.append(rep_sim)
                
                row_names += [key + "cos_sim"] * num_col

                if rep_mag is not None:
                    everything.append(rep_mag)
                    row_names += [key + "_magnitude"] * num_col
                    
            # Visualizer ##########
            
            # import pdb; pdb.set_trace()
            
            num_row = len(everything)
            
            everything = torch.cat(everything, dim=0)
            map = everything.cpu().detach().numpy()
            fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 4, num_row * 4 + 1))

            # Flatten the axes array for easier iteration
            axes = axes.flatten()

            # Loop over the images and display them on the subplots
            for i, ax in enumerate(axes):
                if i < num_col:
                    ax.imshow(map[i], cmap='gray') # Assuming grayscale images
                    ax.axis('off')  
                    ax.set_title(f'azi_({str(int(azimuth[i]))})_step_{str(int(timestep[0]))}')  # Set title for each image`
                
                else:
                    ax.imshow(map[i], cmap='hot')  
                    ax.axis('off')  
                    ax.set_title(f'{row_names[i]}_step_{str(int(timestep[0]))}')  # Set title for each image`
            
            plt.tight_layout()
            plt.show()
            plt.savefig(str(name) + '.png')
            plt.close(fig)
            
        # import pdb; pdb.set_trace()
            
        return add_loss, return_dict


    def compute_threefuse_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        depth: Float[Tensor, "B 3 512 512"],
        camera_condition: Float[Tensor, "B 4 4"],
        text_embeddings_vd_aux: Float[Tensor, "BB 77 1024"],
        text_embeddings_aux: Float[Tensor, "BB 77 1024"],
        noise_map,
        same_timestep=False,
        **kwargs,
    ):
        B = latents.shape[0]

        # import pdb; pdb.set_trace()

        with torch.no_grad():
            # random timestamp
            if same_timestep:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [1],
                    dtype=torch.long,
                    device=self.device,
                ).repeat(B)
            else:
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    [B],
                    dtype=torch.long,
                    device=self.device,
                )            # add noise
                        
            if noise_map is not None:
                noise = noise_map
            else:
                noise = torch.randn_like(latents)
                
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            control_model_input = latent_model_input.to(torch.float16)
            controlnet_prompt_embeds = text_embeddings_vd.to(torch.float16)
            cond_scale = 1

            # import pdb; pdb.set_trace()

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=controlnet_prompt_embeds,
                # controlnet_cond=depth.to(torch.float16),
                # encoder_hidden_states=torch.cat([controlnet_prompt_embeds] * 2),
                controlnet_cond=torch.cat([depth.to(torch.float16)] * 2),
                conditioning_scale=cond_scale,
                guess_mode=False,
                return_dict=False,
            )
            
            # import pdb; pdb.set_trace()

            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings_vd,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            # import pdb; pdb.set_trace()

            # use view-independent text embeddings in LoRA
            text_embeddings_cond, _ = text_embeddings_aux.chunk(2)
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)
        
        guidance_eval_utils = {
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred_pretrain,
            "noise_text" : noise_pred_pretrain_text,
            "noise_uncond" : noise_pred_pretrain_uncond,
            "e_pos": (noise_pred_pretrain - noise_pred_est),
            "weight": w
        }
        
        return grad, guidance_eval_utils

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
        same_timestep = False,
        noise_map = None,
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        if same_timestep:
            t = torch.randint(
                int(self.num_train_timesteps * 0.0),
                int(self.num_train_timesteps * 1.0),
                [1 * self.cfg.lora_n_timestamp_samples],
                dtype=torch.long,
                device=self.device,
            ).repeat(B)
            
        else:
            t = torch.randint(
                int(self.num_train_timesteps * 0.0),
                int(self.num_train_timesteps * 1.0),
                [B * self.cfg.lora_n_timestamp_samples],
                dtype=torch.long,
                device=self.device,
            )

        if noise_map is not None:
            noise = noise_map
        else:
            noise = torch.randn_like(latents)        
        
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:            
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            
            rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
            
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents


    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        pts_depth_maps=None,
        noise_map=None,
        rgb_as_latents=False,
        idx_map=None,
        inter_dict=None,
        depth_masks=None,
        same_timestep=True,
        **kwargs,
    ):
            
        # import pdb; pdb.set_trace()
                
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        
        # import pdb; pdb.set_trace()
        
        # depth_map
        
        # import pdb; pdb.set_trace()
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)
        # import pdb; pdb.set_trace()

        multi_textenc = self.cfg.multi_textenc
        
        if multi_textenc is False:
            # view-dependent text embeddings
            text_embeddings_vd = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                view_dependent_prompting=self.cfg.view_dependent_prompting,
            )

            # input text embeddings, view-independent
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )

            text_embeddings_vd_aux = None
            text_embeddings_aux = None

        else:
            (
                text_embeddings_vd,
                text_embeddings_vd_aux,
            ) = prompt_utils.get_multi_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                view_dependent_prompting=self.cfg.view_dependent_prompting,
            )
            # input text embeddings, view-independent
            (
                text_embeddings,
                text_embeddings_aux,
            ) = prompt_utils.get_multi_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )
        
        if same_timestep:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=self.device,
            ).repeat(batch_size)
        
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

        grad, guidance_eval_utils = self.compute_threefuse_grad_vsd(
            latents,
            text_embeddings_vd,
            text_embeddings,
            pts_depth_maps,
            camera_condition,
            text_embeddings_vd_aux,
            text_embeddings_aux,
            noise_map,
        )
        
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        
        # import pdb; pdb.set_trace()
        
        if kwargs["re_dict"] is not None:
            if noise_map is not None:
                saver = "const"
            else:
                saver = "rand"
            
            foldername = "sim_out/" + saver + "/" + kwargs["filename"]
            
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            
            name = foldername + "/_iter_" + str(kwargs["iter"]) + "_timestep_" + str(str(int(t[0])))
            # warped_gradients, sims, var, add_loss = self.grad_warp(kwargs["re_dict"], grad, timestep = t, name=name, iter=kwargs["iter"])
                        
            if self.cfg.use_sim_loss:
                if self.cfg.backprop_grad:
                    grad = latents - target
                                            
                if self.cfg.use_normalized_grad:
                    grad_mean_norm = grad.norm(dim=1).mean()
                    var_grad = grad / grad_mean_norm
                    # add_loss, grad_mask = self.grad_warp(kwargs["re_dict"], var_grad, timestep = t, name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"],  azimuth = azimuth, guide_utils = guidance_eval_utils)
                    similarity_loss, return_dict = self.grad_warp(kwargs["re_dict"], var_grad, timestep = t, name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"], azimuth = azimuth, guide_utils = guidance_eval_utils)
                    
                    if "grad_mask" in return_dict.keys():
                        grad_mask = return_dict["grad_mask"]
                        
                else:
                    # add_loss, grad_mask = self.grad_warp(kwargs["re_dict"], grad, timestep = t, name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"], azimuth = azimuth, guide_utils = guidance_eval_utils)
                    similarity_loss, return_dict = self.grad_warp(kwargs["re_dict"], grad, timestep = t, name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"], azimuth = azimuth, guide_utils = guidance_eval_utils)
                    
                    if "grad_mask" in return_dict.keys():
                        grad_mask = return_dict["grad_mask"]
                    
            
            if self.cfg.use_disp_loss:
                pred_noise = guidance_eval_utils["noise_pred"]
                _, return_dict = self.grad_warp(kwargs["re_dict"], pred_noise, timestep = t, name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"], azimuth = azimuth, guide_utils = guidance_eval_utils)
                
                pred_noise_warped = return_dict["warped_grad"]
                depth_masks = return_dict["depth_masks"] 
                
                src_ind = kwargs["re_dict"]["src_ind"]
                
                pred_noise_pretrain = depth_masks * pred_noise[src_ind] 
                                
                warp_grad = guidance_eval_utils["weight"][src_ind] * (pred_noise_pretrain - pred_noise_warped)
                
                warp_grad = torch.nan_to_num(warp_grad)
                # clip grad for stable training?
                if self.grad_clip_val is not None:
                    warp_grad = warp_grad.clamp(-self.grad_clip_val, self.grad_clip_val)
                    
                loss_to_latent = self.cfg.disp_loss_to_latent
                        
                if loss_to_latent:

                    warp_target = (latents[src_ind] - warp_grad).detach()
                    
                    f_latents = depth_masks * latents[src_ind]
                    f_target = depth_masks * warp_target
                    
                    disp_loss = 0.5 * F.mse_loss(f_latents, f_target, reduction="sum") / f_target.shape[0]
                
                else:
                    zero_pad = torch.zeros_like(warp_grad).detach()
                    disp_loss = 0.5 * F.mse_loss(warp_grad, zero_pad, reduction="sum") / warp_grad.shape[0]
                # import pdb; pdb.set_trace()


            if self.cfg.vis_grad:
                if kwargs["iter"] % 250 == 0:
                    for k in range(1,6):
                        t = torch.ones_like(t) * k * 160
                        
                        vis_grad, vis_guidance_eval_utils = self.compute_threefuse_grad_vsd(
                                                                    latents,
                                                                    text_embeddings_vd,
                                                                    text_embeddings,
                                                                    pts_depth_maps,
                                                                    camera_condition,
                                                                    text_embeddings_vd_aux,
                                                                    text_embeddings_aux,
                                                                    noise_map,
                                                                )
                        name = foldername + "/_iter_" + str(kwargs["iter"]) + "_timestep_" + str(str(int(t[0])))
                        grad_mean_norm = vis_grad.norm(dim=1).mean()
                        var_grad = vis_grad / grad_mean_norm
                        _, _ = self.grad_warp(kwargs["re_dict"], var_grad, timestep = t, 
                                                name=name, iter=kwargs["iter"], depths = kwargs["depth_maps"], azimuth = azimuth, guide_utils = vis_guidance_eval_utils, vis_grad=True)
                
            
            if self.cfg.grad_cons_mask:
                if self.cfg.mask_w_timestep and t[0] < 400:
                    pass
                    # No masking if t is smaller than 400
                
                else:
                    front_views = ((-30. < azimuth) * (azimuth < 30.)).float()[...,None,None,None].to(self.device)
                    fin_grad_mask = torch.ones_like(grad_mask) * front_views +  grad_mask * (1-front_views)
                    latents = fin_grad_mask * latents
                    target = fin_grad_mask * target
        
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        # import pdb; pdb.set_trace()

        if self.cfg.only_geo:
            if not self.cfg.geo_re_optimize:
                if kwargs["iter"] < self.cfg.geo_start_int :
                    total_loss = loss_sds
                    
                else:
                    if self.cfg.only_geo_front_on:                    
                        azim_mask = (( azimuth < 30.) * (azimuth > -30.)).float()[...,None,None,None]
                        
                        latents = azim_mask * latents
                        target = azim_mask * target
                        
                        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / (torch.count_nonzero(azim_mask) + 1e-9)
                        total_loss = loss_sds
                        
                        
                    else:
                        total_loss = 0. * loss_sds
                        
            if self.cfg.geo_re_optimize:
                
                if self.cfg.geo_interv_different:
                                        
                    if self.sds_on: # SDS on, default state
                        # print("on")
                        total_loss = loss_sds
                        self.int_counter += 1
                        if self.int_counter == self.cfg.geo_intr_on:
                            self.sds_on = False
                            self.int_counter = 0

                    else:
                        if self.cfg.only_geo_front_on:       
                            # print("off")
             
                            azim_mask = (( azimuth < 23.) * (azimuth > -23.)).float()[...,None,None,None]
                            
                            latents = azim_mask * latents
                            target = azim_mask * target
                            
                            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / (torch.count_nonzero(azim_mask) + 1e-9)
                            total_loss = loss_sds
                            
                            # print("geofront_" + str(kwargs["iter"]))
                            
                        else:
                            total_loss = 0. * loss_sds    
                            
                        self.int_counter += 1
                        if self.int_counter == self.cfg.geo_intr_off:
                            self.sds_on = True
                            self.int_counter = 0        
                                    
                elif self.cfg.geo_interval:
                
                    if kwargs["iter"] < self.cfg.geo_start_int or (kwargs["iter"] // self.cfg.geo_interval_len) % 2 == 0:
                        total_loss = loss_sds
                        # print("good_" + str(kwargs["iter"]))

                    else:
                        if self.cfg.only_geo_front_on:                    
                            azim_mask = (( azimuth < 30.) * (azimuth > -30.)).float()[...,None,None,None]
                            
                            latents = azim_mask * latents
                            target = azim_mask * target
                            
                            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / (torch.count_nonzero(azim_mask) + 1e-9)
                            total_loss = loss_sds
                            
                            # print("geofront_" + str(kwargs["iter"]))

                            
                        else:
                            total_loss = 0. * loss_sds
                
                else:
                    resume_interval = 2000
                    
                    if kwargs["iter"] < self.cfg.geo_start_int or kwargs["iter"] > resume_interval:
                        total_loss = loss_sds

                    else:
                        if self.cfg.only_geo_front_on:                    
                            azim_mask = (( azimuth < 30.) * (azimuth > -30.)).float()[...,None,None,None]
                            
                            latents = azim_mask * latents
                            target = azim_mask * target
                            
                            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / (torch.count_nonzero(azim_mask) + 1e-9)
                            total_loss = loss_sds
                            
                        else:
                            total_loss = 0. * loss_sds                    
                    
                    
        else:
            total_loss = loss_vsd
        
        if self.cfg.use_sim_loss:
            if self.cfg.add_loss_stepping:
                if loss_sds < 100.:
                    total_loss += self.cfg.weight_sim_loss * similarity_loss
                else:
                    total_loss += self.cfg.weight_sim_loss * 10 * similarity_loss
                    
            else:
                total_loss += self.cfg.weight_sim_loss * similarity_loss
            
        if self.cfg.use_disp_loss:
            total_loss += disp_loss
        
        # import pdb; pdb.set_trace()

        loss_lora = self.train_lora(latents, text_embeddings_aux, camera_condition)

        
        return {
            "loss_vsd": total_loss,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
