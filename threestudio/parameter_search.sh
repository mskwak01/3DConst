prompts=(
    "a cat wearing a bee costume"
    "a cat wearing a bee costume"
    "a cat wearing a bee costume"
    "a cat wearing a bee costume"
)

img_dirs=(
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
)

cal_vals=(
    90
    90
    90
    90
)

gpu_val=(
    4
    5
    6
    7
)

multi_deg=(
    3
    6 # 0.02 seems best
    9
    12
)

# cfg_dens_iter=(
#     100
#     150
#     200
#     250
# )


for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu "${gpu_val[i]}" \
    system.tag="ours_run_random_multi_YES_const_${multi_deg[i]}" \
    system.three_noise=true \
    system.pytorch_three=false \
    data.num_multiview=2 \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals[i]}" \
    system.geometry.densification_interval=300\
    system.geometry.prune_interval=300 \
    system.gau_d_cond=false \
    system.n_pts_upscaling=9 \
    system.background_rand="ball" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg="${multi_deg[i]}" \
    data.rand_multi_deg=true \
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
    system.guidance.cfg_change_iter=2500 \
    data.n_val_views=20 \
    system.pts_var=0.02 \
    &
    
done

# prompts=(
#     "a beautiful rainbow fish"
#     "a cat wearing a bee costume"
#     "a cat with wings"
#     # "a chow chow puppy"
#     "a goose made out of gold"
#     # "a pug made out of metal"
#     "a toy robot"
#     "a snail"
#     "a turtle"
#     "a fennec fox"
#     # "a train engine made out of clay"
# )

# cal_vals=(
#     135 #180
#     90
#     90
#     # 90
#     135 #0 + 135
#     # 90
#     270
#     120 #90 + 30
#     315
#     90
#     # 90
# )

# img_dirs=(
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/fish.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-wing.png"
#     # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/chow.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/goose.png"
#     # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/pug.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/robot.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/snail.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images//turtle.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/fox.png"
#     # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/train.png"
# )

# prompts=(
#     "a cat wearing a bee costume"
#     "a cat wearing a bee costume"
#     "a cat wearing a bee costume"
#     "a cat wearing a bee costume"
# )

# img_dirs=(
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
# )

# cal_vals=(
#     90
#     90
#     90
#     90
# )

# gpu_val=(
#     0
#     1
#     2
#     3
# )

# # pts_var=(
# #     0.01
# #     0.02 # 0.02 seems best
# #     0.04
# #     0.08
# # )

# cfg_change_iter=(
#     1700
#     2000
#     2300
#     2500
# )

# max_steps=(
#     3000
#     3000
#     3000
#     3000
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "${gpu_val[i]}" \
#     system.tag="ours_run_OPEN_change_iter_${cfg_change_iter[i]}_${max_steps[i]}" \
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
#     system.guidance.guidance_scale=7.5 \
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
#     trainer.max_steps="${max_steps[i]}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=200 \
#     system.guidance.cfg_lastup=true \
#     system.guidance.cfg_change_iter="${cfg_change_iter[i]}" \
#     data.n_val_views=20 \
#     system.pts_var=0.02 \
#     &
    
# done

# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
# )

# img_dirs=(
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
#     "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
# )

# cal_vals=(
#     90
#     90
#     90
#     90
#     90
#     90
#     90
#     90
# )

# cfg_up_iter=(
#     1200
#     1500
#     1800
#     2100
#     1200
#     1500
#     1800
#     2100
# )

# max_steps=(
#     3000
#     3000
#     3000
#     3000
#     3500
#     3500
#     3500
#     3500
# )

# gpu_val=(
#     0
#     1
#     2
#     3
#     4
#     5
#     6
#     7
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu "${gpu_val[i]}" \
#     system.tag="pointcloud_vis" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=1 \
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
#     system.guidance.guidance_scale=7.5 \
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
#     trainer.max_steps="${max_steps[i]}" \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=200 \
#     system.guidance.cfg_lastup=true \
#     system.guidance.cfg_change_iter="${cfg_up_iter[i]}" \
#     data.n_val_views=20 \
#     &
    
# done


# prompts=(
#     "a beautiful rainbow fish"
#     "a cat wearing a bee costume"
#     "a cat with wings"
#     # "a chow chow puppy"
#     # "a goose made out of gold"
#     # "a pug made out of metal"
#     # "a toy robot"
#     # "a snail"
#     # "a turtle"
#     # "a fennec fox"
#     # "a train engine made out of clay"
# )

# cal_vals=(
#     180
#     90
#     90
#     # 90
#     # 0
#     # 90
#     # 270
#     # 90
#     # 90
#     # 90
#     # 90
# )

# img_dirs=(
#     "/mnt/data3/3DConst/images/fish.png"
#     "/mnt/data3/3DConst/images/cat-bee.png"
#     "/mnt/data3/3DConst/images/cat-wing.png"
#     # "/mnt/data3/3DConst/images/chow.png"
#     # "/mnt/data3/3DConst/images/goose.png"
#     # "/mnt/data3/3DConst/images/pug.png"
#     # "/mnt/data3/3DConst/images/robot.png"
#     # "/mnt/data3/3DConst/images/snail.png"
#     # "/mnt/data3/3DConst/images/turtle.png"
#     # "/mnt/data3/3DConst/images/fox.png"
#     # "/mnt/data3/3DConst/images/train.png"
# )

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
#     # 15
#     # 20
#     # 30
#     # 40
#     30
#     20
#     15
#     40
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 5 \
#     system.tag="real_geo_only_short_on_10_off_${interval[i]}" \
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
#     trainer.max_steps=15000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=false \
#     system.guidance.only_geo=true \
#     system.guidance.only_geo_front_on=true \
#     system.guidance.geo_re_optimize=true \
#     system.guidance.geo_interval=false \
#     system.guidance.geo_interv_different=true \
#     system.guidance.geo_intr_on=10 \
#     system.guidance.geo_intr_off="${interval[i]}" \
#     system.guidance.geo_start_int=0 \
#     trainer.val_check_interval=100 \

# done



# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     "a rabbit on a pancake"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     90
#     45
#     135
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
#     --gpu 0 \
#     system.tag="disp_loss_0_5_immanuel" \
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
#     data.multiview_deg=10 \
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
#     system.guidance.use_disp_loss=true \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=true \

# done