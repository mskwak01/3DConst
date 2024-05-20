prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a DSLR photo of an owl"
    "a cute meercat"
    "a rabbit on a pancake"
    "a peacock with a crown"
    "a mysterious LEGO wizard"
    "a product photo of cat-shaped toy"
    "a full body of a cat wearing a hat"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
    "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
    "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
)

cal_vals=(
    90
    45
    135
    90
    90
    90
    75
    135
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 2 \
    system.tag="disp_loss_0_5_immanuel_5_deg" \
    system.three_noise=true \
    system.pytorch_three=false \
    data.num_multiview=2 \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.calibration_value="${cal_vals[i]}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals[i]}" \
    system.geometry.densification_interval=300 \
    system.geometry.prune_interval=300 \
    system.gau_d_cond=false \
    system.n_pts_upscaling=9 \
    system.background_rand="ball" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg=5 \
    data.constant_viewpoints=true \
    data.num_const_views=10 \
    system.reprojection_info=false \
    system.guidance.guidance_scale=7.5 \
    system.guidance.add_loss="no_loss" \
    system.guidance.use_normalized_grad=false \
    system.guidance.add_loss_stepping=false \
    system.guidance.grad_cons_mask=false \
    system.guidance.mask_w_timestep=false \
    system.guidance.vis_grad=true \
    system.guidance.use_disp_loss=true \
    system.guidance.use_sim_loss=false \
    system.guidance.weight_sim_loss=5.0 \
    system.guidance.weight_disp_loss=1.0 \
    trainer.max_steps=3000 \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=true \
    system.guidance.disp_loss_to_latent=true \

done

# prompts=(
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     "a cute meercat"
#     "a cute meercat"
#     "a cute meercat"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
# )

# cal_vals=(
#     45
#     45
#     45
#     45
#     135
#     135
#     135
#     135
# )

# ev_thresh_vals=(
#     4.0
#     3.5
#     3.0
#     2.5
#     4.0
#     3.5
#     3.0
#     2.5
# )



# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 2 \
#     system.tag="fin_all_degrees_visualize_iter_300" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=0 \
#     data.batch_size=322 \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals[i]}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="ball" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=true \
#     data.num_const_views=322 \
#     system.reprojection_info=false \
#     system.depth_warp=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.weight_add_loss=10.0 \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=true \
#     system.guidance.debugging=true \
#     system.guidance.grad_thresh=2.5 \
#     trainer.max_steps=3000 \
#     system.vis_every_grad=true \
#     system.everyview_vis_iter=300 \
#     system.guidance.vis_every_thresh="${ev_thresh_vals[i]}" \

# done

# prompts=(
#     # "a zoomed out DSLR photo of a ceramic lion, white background"
#     # "a peacock with a crown"
#     # "a mysterious LEGO wizard"
#     # "a product photo of cat-shaped toy"
#     "a cute meercat"
#     "a rabbit on a pancake"
#     # "a DSLR photo of an owl"
#     # "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     # "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     # "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     # 90
#     # 90
#     # 90
#     # 75
#     135
#     90
#     # 45
#     # 135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 1 \
#     system.tag="SD_c75_real_dissimilarity_cosine_loss_5_0_stepped" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.calibration_value="${cal_vals[i]}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals[i]}" \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="ball" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=15 \
#     data.constant_viewpoints=true \
#     data.num_const_views=6 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_dissim" \
#     system.guidance.weight_add_loss=5.0 \
#     system.guidance.add_loss_stepping=true \
#     trainer.max_steps=3000 \


# done

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_multi.yaml  --train \
#     --gpu 7 \
#     system.tag="finmask_penta_view_2_2_pts_002" \
#     system.gradient_masking=true \
#     system.prompt_processor.prompt="a zoomed out DSLR photo of a ceramic lion, white background" \
#     system.image_dir="/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png" \
#     system.calibration_value=90 \
#     system.pts_radius=0.02 \
#     system.geometry.split_thresh=0.01 \
#     system.nearby_fusing=false \

# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a DSLR photo of a silver metallic robot tiger"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a DSLR photo of an ironman figure"
# )

# --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \


# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# )

# cal_vals=(
#     0
#     0
#     0
#     0
#     345
#     0    
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 2 \
#     system.tag="noise_pc_pts005" \
#     system.gradient_masking=false \
#     data.num_multiview=1 \
#     system.three_noise=true \
#     system.identical_noising=false \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.pts_radius=0.05 \
#     system.calibration_value="${cal_vals[i]}"
# done

