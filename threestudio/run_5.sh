prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
)

cal_vals=(
    90
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 5 \
    system.tag="zzzzzzzzzz_naive" \
    system.three_noise=false \
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
    system.guidance.guidance_scale=100 \
    system.guidance.add_loss="no_loss" \
    system.guidance.use_normalized_grad=false \
    system.guidance.add_loss_stepping=false \
    system.guidance.grad_cons_mask=false \
    system.guidance.mask_w_timestep=false \
    system.guidance.vis_grad=false \
    system.guidance.use_disp_loss=false \
    system.guidance.use_sim_loss=false \
    system.guidance.weight_sim_loss=5.0 \
    system.guidance.weight_disp_loss=1.0 \
    trainer.max_steps=3000 \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=false \
    trainer.val_check_interval=200 \
    system.guidance.cfg_lastup=false \
    data.n_val_views=20 \

done


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
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 5 \
#     system.tag="prolific_final_ours" \
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
#     data.n_val_views=20 \

# done


# prompts=(
#     # "a beautiful rainbow fish"
#     # "a cat wearing a bee costume"
#     # "a cat with wings"
#     # "a chow chow puppy"
#     "a goose made out of gold"
#     "a pug made out of metal"
#     "a toy robot"
#     # "a snail"
#     # "a turtle"
#     # "a fennec fox"
#     # "a train engine made out of clay"
# )

# cal_vals=(
#     # 180
#     # 90
#     # 90
#     # 90
#     0
#     90
#     270
#     # 90
#     # 90
#     # 90
#     # 90
# )

# img_dirs=(
#     # "/mnt/data3/3DConst/images/fish.png"
#     # "/mnt/data3/3DConst/images/cat-bee.png"
#     # "/mnt/data3/3DConst/images/cat-wing.png"
#     # "/mnt/data3/3DConst/images/chow.png"
#     "/mnt/data3/3DConst/images/goose.png"
#     "/mnt/data3/3DConst/images/pug.png"
#     "/mnt/data3/3DConst/images/robot.png"
#     # "/mnt/data3/3DConst/images/snail.png"
#     # "/mnt/data3/3DConst/images/turtle.png"
#     # "/mnt/data3/3DConst/images/fox.png"
#     # "/mnt/data3/3DConst/images/train.png"
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 5 \
#     system.tag="x___results_for_improvement_ours_random_4000_normal" \
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
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=4000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=100 \
#     system.guidance.cfg_lastup=true \
#     data.n_val_views=20 \

# done

# prompts=(
#     # "a beautiful rainbow fish"
#     # "a cat wearing a bee costume"
#     # "a cat with wings"
#     "a chow chow puppy"
#     "a goose made out of gold"
#     "a pug made out of metal"
#     "a toy robot"
#     # "a snail"
#     # "a turtle"
#     # "a fennec fox"
#     # "a train engine made out of clay"
# )

# cal_vals=(
#     # 180
#     # 90
#     # 90
#     90
#     0
#     90
#     270
#     # 90
#     # 90
#     # 90
#     # 90
# )

# img_dirs=(
#     # "/mnt/data3/3DConst/images/fish.png"
#     # "/mnt/data3/3DConst/images/cat-bee.png"
#     # "/mnt/data3/3DConst/images/cat-wing.png"
#     "/mnt/data3/3DConst/images/chow.png"
#     "/mnt/data3/3DConst/images/goose.png"
#     "/mnt/data3/3DConst/images/pug.png"
#     "/mnt/data3/3DConst/images/robot.png"
#     # "/mnt/data3/3DConst/images/snail.png"
#     # "/mnt/data3/3DConst/images/turtle.png"
#     # "/mnt/data3/3DConst/images/fox.png"
#     # "/mnt/data3/3DConst/images/train.png"
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 5 \
#     system.tag="x___results_for_improvement_baseline_cfg100" \
#     system.three_noise=false \
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
#     system.guidance.guidance_scale=100 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.use_normalized_grad=false \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=false \
#     trainer.val_check_interval=100 \
#     system.guidance.cfg_lastup=false \
#     data.n_val_views=20 \

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
#     300
#     600
#     900
#     1200
#     300
#     600
#     900
#     1200   

# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 5 \
#     system.tag="only_geo_both_only_back_out_${iter[i]}" \
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
#     system.guidance.use_disp_loss=true \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=2000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=false \
#     system.guidance.only_geo=true \
#     system.guidance.only_geo_front_on=true \
#     system.guidance.iter_only_geo="${iter[i]}" \
#     trainer.val_check_interval=100 \

# done