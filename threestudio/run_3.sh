prompts=(
    "a cat wearing a bee costume"
    # "a DSLR photo of a big elephant"
    # "a DSLR photo of a big elephant"
    # "a DSLR photo of a big elephant"
)

img_dirs=(
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/cat-bee.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/elephant2.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/elephant2.png"
    # "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/ext_images/elephant2.png"
)

cal_vals=(
    90
    # 90
    # 90
    # 90
)

gpu_val=(
    7
    # 1
    # 2
    # 3
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu "${gpu_val[i]}" \
    system.tag="trial" \
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
    system.guidance.cfg_change_iter=2500 \
    data.n_val_views=20 \
    system.pts_var=0.02 \
    # &
    
done


# prompts=(
#     # "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     # "a DSLR photo of a silver metallic robot tiger"
#     "a mysterious LEGO wizard"
#     # "a product photo of cat-shaped toy"
#     # "a DSLR photo of an ironman figure"
# )

# img_dirs=(
#     # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     # "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     # "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# )

# cal_vals=(
#     # 0
#     0
#     # 0
#     0
#     # 345
#     # 0    
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 3 \
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
#     # "a goose made out of gold"
#     # "a pug made out of metal"
#     # "a toy robot"
#     "a snail"
#     "a turtle"
#     "a fennec fox"
#     "a train engine made out of clay"
# )

# cal_vals=(
#     # 180
#     # 90
#     # 90
#     # 90
#     # 0
#     # 90
#     # 270
#     90
#     90
#     90
#     90
# )

# img_dirs=(
#     # "/mnt/data3/3DConst/images/fish.png"
#     # "/mnt/data3/3DConst/images/cat-bee.png"
#     # "/mnt/data3/3DConst/images/cat-wing.png"
#     # "/mnt/data3/3DConst/images/chow.png"
#     # "/mnt/data3/3DConst/images/goose.png"
#     # "/mnt/data3/3DConst/images/pug.png"
#     # "/mnt/data3/3DConst/images/robot.png"
#     "/mnt/data3/3DConst/images/snail.png"
#     "/mnt/data3/3DConst/images/turtle.png"
#     "/mnt/data3/3DConst/images/fox.png"
#     "/mnt/data3/3DConst/images/train.png"
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 3 \
#     system.tag="x___results_for_improvement_ours_random" \
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
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=100 \
#     system.guidance.cfg_lastup=true \
#     data.n_val_views=20 \

# done
