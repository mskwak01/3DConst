prompts="a goose made out of gold"
img_dirs="/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/goose.png"
cal_vals=0
seed=0

cfg_up_iter=0
max_steps=2000

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "2" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@baseline" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=1 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="random" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &


python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu "5" \
    seed="${seed}" \
    system.tag="0805-1-blur_cause_ablation@+threenoise" \
    system.three_noise=true \
    system.pytorch_three=false \
    data.num_multiview=1 \
    system.prompt_processor.prompt="${prompts}" \
    system.image_dir="${img_dirs}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals}" \
    system.geometry.densification_interval=300 \
    system.geometry.prune_interval=300 \
    system.gau_d_cond=false \
    system.n_pts_upscaling=9 \
    system.background_rand="random" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg=5 \
    data.constant_viewpoints=false \
    data.num_const_views=15 \
    system.reprojection_info=false \
    system.guidance.guidance_scale=7.5 \
    system.guidance.add_loss="cosine_sim" \
    system.guidance.use_normalized_grad=false \
    system.guidance.add_loss_stepping=false \
    system.guidance.grad_cons_mask=false \
    system.guidance.mask_w_timestep=false \
    system.guidance.vis_grad=false \
    system.guidance.use_disp_loss=false \
    system.guidance.use_sim_loss=false \
    system.guidance.weight_sim_loss=5.0 \
    system.guidance.weight_disp_loss=1.0 \
    trainer.max_steps="${max_steps}" \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=false \
    trainer.val_check_interval=250 \
    system.guidance.cfg_lastup=false \
    system.guidance.cfg_change_iter="${cfg_up_iter}" \
    data.n_val_views=10 \
#     &

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "3" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@+threenoise+multiview=2" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="random" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "4" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@+threenoise+multiview=2+ball" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="ball" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "5" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@+disp_loss=1.0" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=1 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="random" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=true \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "6" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@+disp_loss=1.0+threenoise+multiview=2" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="random" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=true \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &

# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "7" \
#     seed="${seed}" \
#     system.tag="0805-1-blur_cause_ablation@+disp_loss=1.0+threenoise+multiview=2+ball" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts}" \
#     system.image_dir="${img_dirs}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="true" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=true \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps="${max_steps}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=250 \
#     system.guidance.cfg_lastup=false \
#     system.guidance.cfg_change_iter="${cfg_up_iter}" \
#     data.n_val_views=10 \
#     &