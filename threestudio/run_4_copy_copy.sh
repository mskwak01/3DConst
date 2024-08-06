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
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/a_ceramic_lion.png"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/meercat.png"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/rabbit-pancake.jpeg"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/peacock.png"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/lego-wizard2.png"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/cat-toy.png"
    "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/cat-hat.png"
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
    --gpu "${gpu_val[i]}" \
    system.tag="sanity-check-${cfg_up_iter[i]}_max_${max_steps[i]}" \
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
    data.multiview_deg=5 \
    data.constant_viewpoints=true \
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
    system.guidance.use_sim_loss=true \
    system.guidance.weight_sim_loss=5.0 \
    system.guidance.weight_disp_loss=1.0 \
    trainer.max_steps="${max_steps[i]}" \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=true \
    trainer.val_check_interval=200 \
    system.guidance.cfg_lastup=true \
    system.guidance.cfg_change_iter="${cfg_up_iter[i]}" \
    data.n_val_views=20 \
    &
done

# prompts=(
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     "a rabbit on a pancake"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/rabbit-pancake.jpeg"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/a_ceramic_lion.png"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/peacock.png"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/lego-wizard2.png"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/cat-toy.png"
#     "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/cat-hat.png"
# )

# cal_vals=(
#     45
#     135
#     90
#     90
#     90
#     90
#     75
#     135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 4 \
#     system.tag="vis_all_grads_deg_25" \
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
#     data.multiview_deg=25 \
#     data.constant_viewpoints=true \
#     data.num_const_views=10 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=true \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=0.23 \

# done

# prompts=(
#     # "a DSLR photo of an owl"
#     # "a DSLR photo of an owl"
#     # "a DSLR photo of an owl"
#     # "a DSLR photo of an owl"
#     "a cute meercat"
#     "a cute meercat"
#     "a cute meercat"
#     "a cute meercat"
# )

# img_dirs=(
#     # "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
#     # "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
#     # "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
#     # "/home/cvlab06/project/donghoon/text-to-3d/3DConst/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
# )

# cal_vals=(
#     # 45
#     # 45
#     # 45
#     # 45
#     135
#     135
#     135
#     135
# )

# thresh_vals=(
#     # 2.5
#     # 2
#     # 1.5
#     # 1
#     2.5
#     2
#     1.5
#     1
# )



# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 4 \
#     system.tag="c_75_mask_visualizer_random_thresh_deg_15_th_${thresh_vals[i]}" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.calibration_value="${cal_vals[i]}" \
#     system.surf_radius=0.05 \
#     system.geometry.densification_interval=300 \
#     system.geometry.prune_interval=300 \
#     system.gau_d_cond=false \
#     system.n_pts_upscaling=9 \
#     system.background_rand="ball" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=15 \
#     data.constant_viewpoints=true \
#     data.num_const_views=10 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.weight_add_loss=10.0 \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.add_loss_stepping=true \
#     system.guidance.debugging=false \
#     system.guidance.vis_grad=true \
#     system.guidance.grad_thresh="${thresh_vals[i]}" \
#     trainer.max_steps=3000 \

# done