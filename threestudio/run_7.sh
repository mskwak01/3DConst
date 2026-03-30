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

for i in "${!prompts[@]}";
do
python launch.py \
    --config configs/prolificdreamer-noise.yaml \
    --train \
    --gpu 7 \
    system.tag="prolific_noised" \
    system.gradient_masking=true \
    data.num_multiview=1 \
    system.three_noise=false \
    system.identical_noising=false \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.surf_radius=0.05 \
    system.calibration_value="${cal_vals[i]}" \
    system.n_pts_upscaling=25 \
    system.background_rand="ball" \
    system.noise_alter_interval=30 \
    system.consistency_mask=false \
    data.multiview_deg=3 \
    data.constant_viewpoints=false \
    data.num_const_views=15 \
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
    system.noise_interval_schedule=true \
    trainer.val_check_interval=250 \
    data.n_val_views=10 \

done



#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a rabbit on a pancake"
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     # "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     90
#     90
#     90
#     75
#     90
#     45
#     135
#     # 135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 7 \
#     system.tag="similarity_tester_grad_backprop_view_12" \
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
#     data.num_const_views=12 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="cosine_sim" \
#     system.guidance.weight_add_loss=5.0 \
#     system.guidance.add_loss_stepping=false \
#     trainer.max_steps=3000 \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.vis_grad=false \
#     system.guidance.visualization_type="grad" \
#     system.guidance.high_timesteps=false \
#     system.guidance.backprop_grad=true \

# done