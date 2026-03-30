prompts=(
    "a cat wearing a bee costume"
    # "a cat wearing a bee costume"
    # "a cat wearing a bee costume"
    # "a cat wearing a bee costume"
)

img_dirs=(
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
)

cal_vals=(
    90
    # 90
    # 90
    # 90
)

gpu_val=(
    0
    # 1
    # 2
    # 3
)

# pts_var=(
#     0.01
#     0.02
#     0.04
#     0.08
# )

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu "${gpu_val[i]}" \
    system.tag="ours_ptsvar_002_run_1_8_const_view" \
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
    trainer.max_steps=3000 \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=true \
    trainer.val_check_interval=200 \
    system.guidance.cfg_lastup=true \
    system.guidance.cfg_change_iter=1800 \
    data.n_val_views=20 \
    system.pts_var=0.02 \
    # &
    
done

# prompts=(
#     "a DSLR photo of a big elephant"
#     "a DSLR photo of a cute meercat"
#     # "a DSLR photo of a cute teddy-bear"
#     # "a DSLR photo of a wild wolf"
# )

# img_dirs=(
#     /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/big-elephant.png
#     /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png
#     # /home/cvlab15/project/woojeong/naver/images/teddy-bear.png
#     # /home/cvlab15/project/woojeong/naver/images/wolf.png
# )

# cal_vals=(
#     135
#     135
#     # 90
#     # 135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/dreamfusion-sd-noise.yaml \
#     --train \
#     --gpu 0 \
#     system.tag="dreamfusion_final_ours" \
#     data.num_multiview=2 \
#     system.three_noise=true \
#     system.identical_noising=true \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals[i]}" \
#     system.n_pts_upscaling=9 \
#     system.background_rand="ball" \
#     system.noise_alter_interval=30 \
#     system.consistency_mask=false \
#     data.multiview_deg=5 \
#     data.constant_viewpoints=false \
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     data.n_val_views=10 \
#     trainer.val_check_interval=500 \

# done

# prompts=(
#     "an old vintage car"
#     "a DLSR photo of an origami motorcycle"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/car.png"
#     "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" 
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
# )


# cal_vals=(
#     180
#     0
#     90
# )

# cfgs=(
#     7.5
#     7.5
#     100
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 0 \
#     system.tag="results_for_comparison_with_other_baselines_${cfgs[i]}" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
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
#     data.num_const_views=15 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale="${cfgs[i]}" \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=100 \
#     data.n_val_views=20 \
#     system.guidance.cfg_lastup=true \

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
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
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

# interval=(
#     # 8
#     # 15
#     # 25
#     # 32
#     8
#     15
#     25
#     32
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 2 \
#     system.tag="real_geo_only_short_on_5_off_${interval[i]}" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
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
#     data.multiview_deg=15 \
#     data.constant_viewpoints=true \
#     data.num_const_views=10 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=true \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=false \
#     system.guidance.only_geo=true \
#     system.guidance.only_geo_front_on=true \
#     system.guidance.geo_re_optimize=true \
#     system.guidance.geo_interval=false \
#     system.guidance.geo_interv_different=true \
#     system.guidance.geo_intr_on=5 \
#     system.guidance.geo_intr_off="${interval[i]}" \
#     system.guidance.geo_start_int=0 \
#     trainer.val_check_interval=100 \

# done


# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
#     "a DSLR photo of an owl"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
# )

# cal_vals=(
#     90
#     90
#     90
#     90
#     45
#     45
#     45
#     45
# )

# iter=(
#     400
#     800
#     1200
#     1600
#     400
#     800
#     1200
#     1600  
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 0 \
#     system.tag="real_only_geo_both_front_on_intv_30_${iter[i]}" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
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
#     data.multiview_deg=15 \
#     data.constant_viewpoints=true \
#     data.num_const_views=10 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=true \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=false \
#     system.guidance.only_geo=true \
#     system.guidance.only_geo_front_on=true \
#     system.guidance.iter_only_geo="${iter[i]}" \
#     trainer.val_check_interval=100 \

# done


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 0 \
#     system.tag="crrraaaazzzzzzyyyyyyy" \
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
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.weight_add_loss=0.23 \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.debugging=false \
#     trainer.max_steps=3000 \

# done

# prompts=(
#     # "a zoomed out DSLR photo of a ceramic lion, white background"
#     # "a peacock with a crown"
#     # "a mysterious LEGO wizard"
#     # "a product photo of cat-shaped toy"
#     "a cute meercat"
#     # "a rabbit on a pancake"
#     "a DSLR photo of an owl"
#     # "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     # "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     # "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     # "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     # 90
#     # 90
#     # 90
#     # 75
#     135
#     # 90
#     45
#     # 135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 0 \
#     system.tag="final_visualizer_deg_15" \
#     system.three_noise=true \
#     system.pytorch_three=false \
#     data.batch_size=2 \
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
#     system.guidance.add_loss="no_loss" \
#     system.guidance.weight_add_loss=5.0 \
#     system.guidance.add_loss_stepping=true \
#     trainer.max_steps=3000 \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.vis_grad=true \
#     system.guidance.debugging=true \

# done