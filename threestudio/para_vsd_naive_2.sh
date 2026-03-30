prompts=(
    # "a DSLR photo of a bear dressed in medieval armor"
    # "a DSLR photo of a blue jay standing on a large basket of rainbow macarons"
    # "a DSLR photo of a knight holding a lance and sitting on an armored horse"
    # "a DSLR photo of a porcelain dragon"
    "a DSLR photo of a robot dinosaur"
    "a zoomed out DSLR photo of a monkey riding a bike" 
    "a zoomed out DSLR photo of a corgi wearing a top hat" 
    "a zoomed out DSLR photo of a squirrel DJing"
)

cal_vals=(
    # 90
    # 90
    # 90
    # 135
    135
    135
    90
    180
)

img_dirs=(
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/bear_2.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/bird_1.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/horse_1.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/porcelain_2.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/dinosaur.jpg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/bicycle_1.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/corgi_hat_3.png"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/deejae.png"
)

gpu_val=(
    # 0
    # 1
    # 2
    # 3
    0
    1
    2
    3
)


for i in "${!prompts[@]}";
do
python launch.py \
    --config configs/prolificdreamer-noise.yaml \
    --train \
    --gpu "${gpu_val[i]}" \
    system.tag="NAIVE_prolific_complex_scenes" \
    data.num_multiview=1 \
    system.three_noise=false \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals[i]}" \
    system.n_pts_upscaling=9 \
    data.rand_multi_deg=false \
    system.background_rand="ball" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg=9 \
    data.constant_viewpoints=false \
    system.reprojection_info=false \
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
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=false \
    trainer.val_check_interval=500 \
    data.n_val_views=20 \
    system.pts_var=0.02 \
    &
    
done

# prompts=(
#     # "a DSLR photo of a big elephant"
#     # "a DSLR photo of a cute meercat"
#     # "a DSLR photo of a blue bird with a beak and feathered wings"
#     "a DSLR photo of a cute teddy-bear"
#     # "a DSLR photo of a wild wolf"
# )

# img_dirs=(
#     # /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/big-elephant.png
#     # /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png
#     # /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/blue-bird.png
#     /home/cvlab15/project/woojeong/naver/images/teddy-bear.png
#     # /home/cvlab15/project/woojeong/naver/images/wolf.png
# )

# cal_vals=(
#     # 135
#     # 135
#     # 135
#     90
#     # 135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 6 \
#     system.tag="prolific_naive" \
#     system.gradient_masking=true \
#     data.num_multiview=1 \
#     system.three_noise=false \
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
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=500 \
#     data.n_val_views=20 \

# done


# # prompts=(
# #     # "a beautiful rainbow fish"
# #     # "a cat wearing a bee costume"
# #     # "a cat with wings"
# #     # "a chow chow puppy"
# #     # "a goose made out of gold"
# #     # "a pug made out of metal"
# #     # "a toy robot"
# #     "a snail"
# #     "a turtle"
# #     "a fennec fox"
# #     "a train engine made out of clay"
# # )

# # cal_vals=(
# #     # 180
# #     # 90
# #     # 90
# #     # 90
# #     # 0
# #     # 90
# #     # 270
# #     90
# #     90
# #     90
# #     90
# # )

# # img_dirs=(
# #     # "/mnt/data3/3DConst/images/fish.png"
# #     # "/mnt/data3/3DConst/images/cat-bee.png"
# #     # "/mnt/data3/3DConst/images/cat-wing.png"
# #     # "/mnt/data3/3DConst/images/chow.png"
# #     # "/mnt/data3/3DConst/images/goose.png"
# #     # "/mnt/data3/3DConst/images/pug.png"
# #     # "/mnt/data3/3DConst/images/robot.png"
# #     "/mnt/data3/3DConst/images/snail.png"
# #     "/mnt/data3/3DConst/images/turtle.png"
# #     "/mnt/data3/3DConst/images/fox.png"
# #     "/mnt/data3/3DConst/images/train.png"
# # )

# # for i in "${!prompts[@]}";
# # do
# # python launch.py \
# #     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
# #     --train \
# #     --gpu 6 \
# #     system.tag="x___results_for_improvement_ours_random_4000_normal" \
# #     system.three_noise=true \
# #     system.pytorch_three=false \
# #     data.num_multiview=2 \
# #     system.prompt_processor.prompt="${prompts[i]}" \
# #     system.image_dir="${img_dirs[i]}" \
# #     system.surf_radius=0.05 \
# #     system.calibration_value="${cal_vals[i]}" \
# #     system.geometry.densification_interval=300 \
# #     system.geometry.prune_interval=300 \
# #     system.gau_d_cond=false \
# #     system.n_pts_upscaling=9 \
# #     system.background_rand="ball" \
# #     system.noise_alter_interval=30 \
# #     system.consistency_mask=false \
# #     data.multiview_deg=5 \
# #     data.constant_viewpoints=false \
# #     data.num_const_views=15 \
# #     system.reprojection_info=false \
# #     system.guidance.guidance_scale=7.5 \
# #     system.guidance.add_loss="cosine_sim" \
# #     system.guidance.use_normalized_grad=false \
# #     system.guidance.add_loss_stepping=false \
# #     system.guidance.grad_cons_mask=false \
# #     system.guidance.mask_w_timestep=false \
# #     system.guidance.vis_grad=false \
# #     system.guidance.use_disp_loss=false \
# #     system.guidance.use_sim_loss=true \
# #     system.guidance.weight_sim_loss=5.0 \
# #     system.guidance.weight_disp_loss=1.0 \
# #     trainer.max_steps=4000 \
# #     system.guidance.backprop_grad=false \
# #     system.guidance.debugging=false \
# #     system.noise_interval_schedule=true \
# #     trainer.val_check_interval=100 \
# #     system.guidance.cfg_lastup=true \
# #     data.n_val_views=20 \

# # done

# # prompts=(
# #     # "a beautiful rainbow fish"
# #     # "a cat wearing a bee costume"
# #     # "a cat with wings"
# #     # "a chow chow puppy"
# #     # "a goose made out of gold"
# #     # "a pug made out of metal"
# #     # "a toy robot"
# #     "a snail"
# #     "a turtle"
# #     "a fennec fox"
# #     "a train engine made out of clay"
# # )

# # cal_vals=(
# #     # 180
# #     # 90
# #     # 90
# #     # 90
# #     # 0
# #     # 90
# #     # 270
# #     90
# #     90
# #     90
# #     90
# # )

# # img_dirs=(
# #     # "/mnt/data3/3DConst/images/fish.png"
# #     # "/mnt/data3/3DConst/images/cat-bee.png"
# #     # "/mnt/data3/3DConst/images/cat-wing.png"
# #     # "/mnt/data3/3DConst/images/chow.png"
# #     # "/mnt/data3/3DConst/images/goose.png"
# #     # "/mnt/data3/3DConst/images/pug.png"
# #     # "/mnt/data3/3DConst/images/robot.png"
# #     "/mnt/data3/3DConst/images/snail.png"
# #     "/mnt/data3/3DConst/images/turtle.png"
# #     "/mnt/data3/3DConst/images/fox.png"
# #     "/mnt/data3/3DConst/images/train.png"
# # )


# # for i in "${!prompts[@]}";
# # do
# # python launch.py \
# #     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
# #     --train \
# #     --gpu 6 \
# #     system.tag="x___results_for_improvement_baseline_cfg100" \
# #     system.three_noise=false \
# #     system.pytorch_three=false \
# #     data.num_multiview=2 \
# #     system.prompt_processor.prompt="${prompts[i]}" \
# #     system.image_dir="${img_dirs[i]}" \
# #     system.surf_radius=0.05 \
# #     system.calibration_value="${cal_vals[i]}" \
# #     system.geometry.densification_interval=300 \
# #     system.geometry.prune_interval=300 \
# #     system.gau_d_cond=false \
# #     system.n_pts_upscaling=9 \
# #     system.background_rand="ball" \
# #     system.noise_alter_interval=30 \
# #     system.consistency_mask=false \
# #     data.multiview_deg=15 \
# #     data.constant_viewpoints=true \
# #     data.num_const_views=10 \
# #     system.reprojection_info=false \
# #     system.guidance.guidance_scale=100 \
# #     system.guidance.add_loss="no_loss" \
# #     system.guidance.use_normalized_grad=false \
# #     system.guidance.add_loss_stepping=false \
# #     system.guidance.grad_cons_mask=false \
# #     system.guidance.mask_w_timestep=false \
# #     system.guidance.vis_grad=false \
# #     system.guidance.use_disp_loss=false \
# #     system.guidance.use_sim_loss=false \
# #     system.guidance.weight_sim_loss=5.0 \
# #     system.guidance.weight_disp_loss=1.0 \
# #     trainer.max_steps=3000 \
# #     system.guidance.backprop_grad=false \
# #     system.guidance.debugging=false \
# #     system.noise_interval_schedule=false \
# #     trainer.val_check_interval=100 \
# #     system.guidance.cfg_lastup=false \
# #     data.n_val_views=20 \

# # done

# # )

# # for i in "${!prompts[@]}";
# # do
# # python launch.py \
# #     --config custom/threestudio-3dgs/configs/gau_threefuse_diffusion.yaml \
# #     --train \
# #     --gpu 6 \
# #     system.tag="pc_gau_depth_conditioned" \
# #     data.num_multiview=1 \
# #     system.gradient_masking=false \
# #     system.threefuse=true \
# #     system.three_noise=true \
# #     system.gau_d_cond=true \
# #     system.gaussian_dynamic=true \
# #     system.identical_noising=false \
# #     system.prompt_processor.prompt="${prompts[i]}" \
# #     system.image_dir="${img_dirs[i]}" \
# #     system.pts_radius=0.003 \
# #     system.calibration_value="${cal_vals[i]}" \
# #     system.geometry.densification_interval=300 \
# #     system.geometry.prune_interval=300 \
# #     system.three_warp_noise=false \
# #     system.consider_depth=false \

# # done

# # # prompts=(
# # #     "a zoomed out DSLR photo of a ceramic lion, white background"
# # #     "a peacock with a crown"
# # #     "a DSLR photo of a silver metallic robot tiger"
# # # )

# # prompts=(
# #     "a mysterious LEGO wizard"
# #     "a product photo of cat-shaped toy"
# #     "a DSLR photo of an ironman figure"
# # )

# # # img_dirs=(
# # #     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
# # #     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
# # #     "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
# # # )

# # img_dirs=(
# #     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
# #     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
# #     "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# # )

# # # cal_vals=(
# # #     0
# # #     0
# # #     0
# # # )

# # cal_vals=(
# #     0
# #     345
# #     0    
# # )

# # for i in "${!prompts[@]}";
# # do
# # python launch.py \
# #     --config configs/dreamfusion-sd-noise.yaml \
# #     --train \
# #     --gpu 6 \
# #     system.tag="noise_pc_jnh_ver" \
# #     data.num_multiview=1 \
# #     system.gradient_masking=false \
# #     system.three_noise=true \
# #     system.identical_noising=false \
# #     system.prompt_processor.prompt="${prompts[i]}" \
# #     system.image_dir="${img_dirs[i]}" \
# #     system.pts_radius=0.003 \
# #     system.calibration_value="${cal_vals[i]}" \
# #     system.three_warp_noise=true \
# #     system.consider_depth=false

# # done

# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     # "a peacock with a crown"
#     # "a DSLR photo of a silver metallic robot tiger"
#     # "a mysterious LEGO wizard"
#     # "a product photo of cat-shaped toy"
#     # "a DSLR photo of an ironman figure"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     # "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     # "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
#     # "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     # "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# )

# cal_vals=(
#     0
#     # 0
#     # 0
#     # 0
#     # 345
#     # 0    
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/dreamfusion-sd-noise.yaml \
#     --train \
#     --gpu 6 \
#     system.tag="dreamfusion_final_ours_fixed" \
#     system.gradient_masking=true \
#     data.num_multiview=1 \
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
#     data.constant_viewpoints=true \
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
#     trainer.val_check_interval=100 \
#     system.guidance.cfg_lastup=true \
#     data.n_val_views=20 \

# done