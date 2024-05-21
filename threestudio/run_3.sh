prompts=(
    "a DSLR photo of an owl"
    "a DSLR photo of an owl"
    "a DSLR photo of an owl"
    "a DSLR photo of an owl"
    "a cute meercat"
    "a cute meercat"
    "a cute meercat"
    "a cute meercat"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
)

cal_vals=(
    45
    45
    45
    45
    135
    135
    135
    135
)

interval=(
    400
    600
    800
    1000
    400
    600
    800
    1000
)


for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 3 \
    system.tag="real_only_geo_cos_front_start_800_out_long_${interval[i]}" \
    system.three_noise=true \
    system.pytorch_three=false \
    data.num_multiview=2 \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals[i]}" \
    system.geometry.densification_interval=300 \
    system.geometry.prune_interval=300 \
    system.gau_d_cond=false \
    system.n_pts_upscaling=9 \
    system.background_rand="ball" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg=15 \
    data.constant_viewpoints=true \
    data.num_const_views=10 \
    system.reprojection_info=false \
    system.guidance.guidance_scale=7.5 \
    system.guidance.add_loss="cosine_sim" \
    system.guidance.use_normalized_grad=false \
    system.guidance.add_loss_stepping=false \
    system.guidance.grad_cons_mask=false \
    system.guidance.mask_w_timestep=false \
    system.guidance.vis_grad=true \
    system.guidance.use_disp_loss=false \
    system.guidance.use_sim_loss=true \
    system.guidance.weight_sim_loss=5.0 \
    system.guidance.weight_disp_loss=1.0 \
    trainer.max_steps=3000 \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.guidance.disp_loss_to_latent=false \
    system.guidance.only_geo=true \
    system.guidance.only_geo_front_on=true \
    system.guidance.geo_re_optimize=true \
    system.guidance.geo_interval=true \
    system.guidance.geo_interval_len="${interval[i]}" \
    system.guidance.geo_start_int=800 \
    trainer.val_check_interval=100 \

done
