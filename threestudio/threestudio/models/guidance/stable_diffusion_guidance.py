from dataclasses import dataclass, field

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from torchvision.utils import save_image
import matplotlib.pyplot as plt


@threestudio.register("stable-diffusion-guidance")
class StableDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True
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
        cfg_lastup: bool = False
        cfg_change_iter: int = 2300

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

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

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

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

        self.guidance_scale = self.cfg.guidance_scale
        self.initial_guidance_scale = self.guidance_scale
        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
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

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        noise_map
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting         
            )
            with torch.no_grad():
                if noise_map is not None:
                    noise = noise_map
                else:
                    noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                if noise_map is not None:
                    noise = noise_map
                else:
                    noise = torch.randn_like(latents)
                
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            
            e_pos = noise_pred_text - noise_pred_uncond
            
            noise_pred = noise_pred_text + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "noise_text" : noise_pred_text,
            "noise_uncond" : noise_pred_uncond,
            "e_pos": e_pos,
            "weight": w
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils


    
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
    

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        noise_map=None,
        guidance_eval=False,
        # depth_masks=None,
        same_timestep=True,
        **kwargs,
    ):
        # raise NotImplementedError("")
        batch_size = rgb.shape[0]
        
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        
        if self.cfg.cfg_lastup and noise_map is not None: 
            # self.guidance_scale = self.initial_guidance_scale * (self.num_train_timesteps - kwargs["iter"]) / self.num_train_timesteps
            # print(f"guidance_scale for iter {kwargs['iter']} = {self.guidance_scale}")
            if kwargs["iter"] > self.cfg.cfg_change_iter:
                self.guidance_scale = 100
        
        if batch_size > 30:
            with torch.no_grad():
                if rgb_as_latents:
                    latents = F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                else:
                    rgb_BCHW_512 = F.interpolate(
                        rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                    )
                    
                    # import pdb; pdb.set_trace()
                    # encode image into latents with vae
                    
                    hello = []
                    
                    prev = 0
                    
                    for k in range(320 // 32):
                        next = (k + 1) * 32
                        latents = self.encode_images(rgb_BCHW_512[prev:next])
                        prev = next
                        hello.append(latents.detach())
                    
        
        else:
            if rgb_as_latents:
                latents = F.interpolate(
                    rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                )
            else:
                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                )
                # encode image into latents with vae
                latents = self.encode_images(rgb_BCHW_512)
            
        # if self.cfg.high_timesteps:
        #     self.min_step = 600
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level            
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
        
        ###########
        
        if batch_size > 30:
            with torch.no_grad():
                
                # import pdb; pdb.set_trace()
                
                timesteps_list = []
                                
                for i in range(6):
                    
                    t = torch.ones((hello[0].shape[0],)).int().to(self.device) * (i+1) * 150
                    timestep = (i+1) * 150
                    
                    new = []
                    prev = 0
                    for k, latents in enumerate(hello):
                        next = (k + 1) * 32
                        grad, guidance_eval_utils = self.compute_grad_sds(latents, t, prompt_utils, elevation[prev:next], azimuth[prev:next], camera_distances[prev:next], noise_map[prev:next])
                        new.append(grad)
                        prev = next

                    all_of_them = torch.cat(new,dim=0)
                    please = all_of_them.norm(dim=1)
                    grads = please / torch.mean(please)
                    
                    timesteps_list.append(grads)
                    
                # import pdb; pdb.set_trace()
                
                gradients = torch.stack(timesteps_list).permute(1,0,2,3).unsqueeze(dim=2)
                
                thresh_list = [4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5]
                depths = F.interpolate(kwargs["depth_maps"], size=(64,64))[:,:1].unsqueeze(dim=1)
                
                for thresh in thresh_list:
                    
                    new = (gradients > thresh).float()
                    naming = 'all_views/' + kwargs["filename"] +  f'/_thresh_{thresh}/iter_{kwargs["iter"]}'
                        
                    self.visualize_all(new, depths,  azimuth, name = naming)
                    
                raise ValueError()
                
                
        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents, t, prompt_utils, elevation, azimuth, camera_distances, noise_map
            )
        
        # if self.cfg.debugging: 
        #     import pdb; pdb.set_trace()
        #     torch.mean(grad[0].norm(dim=0))                                                                         
            

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        

        
        # import pdb; pdb.set_trace()
        
        # save_image 
        
        # map = sims.cpu().detach().numpy()
        # plt.imshow(map, cmap='hot')
        # plt.colorbar()
        # plt.savefig('new.png')
        # import pdb; pdb.set_trace()

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
                        
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
            else:
                similarity_loss = 0
                    
            
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
            else:
                disp_loss = 0

            if self.cfg.vis_grad:
                if kwargs["iter"] % 250 == 0:
                    for k in range(1,6):
                        t = torch.ones_like(t) * k * 160
                        
                        vis_grad, vis_guidance_eval_utils = self.compute_grad_sds(latents, t, prompt_utils, elevation, azimuth, camera_distances, noise_map)
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
            
            # if self.cfg.debugging:
                
            #     import pdb; pdb.set_trace()
                
            #     grad[0] = d1
            #     grad[3] = d2
                
            #     target = (latents - grad).detach()
            
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

            # import pdb; pdb.set_trace()    
                # if self.cfg.debugging:
                #     import pdb; pdb.set_trace()
            # if self.cfg.debugging:

            
            # if kwargs["iter"] % 50 == 0:
                
            #     with torch.no_grad():
                
            #         for p in range(1, 7):
                        
            #             p_num = 140 * p
            #             test_t = p_num * torch.ones_like(t)
            #             test_grad, _ = self.compute_grad_sds(latents, test_t, prompt_utils, elevation, azimuth, camera_distances, noise_map)
            #             name = foldername + "/_iter_" + str(kwargs["iter"]) + "_timestep_" + str(p_num)
            #             warped_gradients, sims, var = self.grad_warp(kwargs["re_dict"], test_grad, timestep = test_t, name=name, iter=kwargs["iter"])

        
        # import pdb; pdb.set_trace()
                
        # if consistency_mask:
            
        #     import pdb; pdb.set_trace()
            
        #     cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            
        #     mask1 = (cos(grad[0], grad[1]) > 0.6).float()
        #     mask2 = (cos(grad[2], grad[3]) > 0.6).float()
            
        #     masks = torch.stack((mask1, mask1, mask2, mask2))
            
        #     latents = masks * latents
        #     target = masks * target
        
        # import pdb; pdb.set_trace()
            
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad   
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
            total_loss = loss_sds
        
        # if self.cfg.use_sim_loss:
        #     if self.cfg.add_loss_stepping:
        #         if loss_sds < 100.:
        #             total_loss += self.cfg.weight_sim_loss * similarity_loss
        #         else:
        #             total_loss += self.cfg.weight_sim_loss * 10 * similarity_loss
                    
        #     else:
        #         total_loss += self.cfg.weight_sim_loss * similarity_loss
            
        # if self.cfg.use_disp_loss:
        #     total_loss += disp_loss
            
        guidance_out = {
            "loss_sds": loss_sds,
            "loss_sim": similarity_loss,
            "loss_disp": disp_loss,
            # "loss_total": total_loss,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
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
