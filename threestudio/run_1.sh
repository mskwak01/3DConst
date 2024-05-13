

prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a peacock with a crown"
    "a mysterious LEGO wizard"
    "a product photo of cat-shaped toy"
    "a cute zebra" 
    "a cute meercat"
    "a rabbit on a pancake"
    "a DSLR photo of an owl"
    "a full body of a cat wearing a hat"
    "a majestic eagle"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
    "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/zebra.png" 
    "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
    "/home/cvlab15/project/woojeong/naver/images/eagle.jpeg"
)

cal_vals=(
    90
    90
    90
    75
    90
    135
    90
    45
    135
    90
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 1 \
    system.tag="SD_c75_views_10_naive" \
    system.three_noise=false \
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
    data.multiview_deg=15 \
    data.constant_viewpoints=true \
    data.num_const_views=6 \
    system.reprojection_info=false \
    system.guidance.guidance_scale=100 \
    system.guidance.add_loss="cosine_sim" \
    trainer.max_steps=5000 \

done


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 1 \
#     system.tag="noise_pc" \
#     data.num_multiview=1 \
#     system.three_noise=true \
#     system.identical_noising=false \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.calibration_value="${cal_vals[i]}"
# done
