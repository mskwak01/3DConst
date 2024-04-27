prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a peacock with a crown"
    # "a DSLR photo of a silver metallic robot tiger"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    # "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
)

cal_vals=(
    90
    90
    # 90
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_threefuse_diffusion.yaml \
    --train \
    --gpu 6 \
    system.tag="pc_gau_depth_conditioned" \
    data.num_multiview=1 \
    system.gradient_masking=false \
    system.threefuse=true \
    system.three_noise=true \
    system.gau_d_cond=true \
    system.gaussian_dynamic=true \
    system.identical_noising=false \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.pts_radius=0.003 \
    system.calibration_value="${cal_vals[i]}" \
    system.geometry.densification_interval=300 \
    system.geometry.prune_interval=300 \
    system.three_warp_noise=false \
    system.consider_depth=false \

done

# # prompts=(
# #     "a zoomed out DSLR photo of a ceramic lion, white background"
# #     "a peacock with a crown"
# #     "a DSLR photo of a silver metallic robot tiger"
# # )

# prompts=(
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a DSLR photo of an ironman figure"
# )

# # img_dirs=(
# #     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
# #     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
# #     "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
# # )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# )

# # cal_vals=(
# #     0
# #     0
# #     0
# # )

# cal_vals=(
#     0
#     345
#     0    
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/dreamfusion-sd-noise.yaml \
#     --train \
#     --gpu 6 \
#     system.tag="noise_pc_jnh_ver" \
#     data.num_multiview=1 \
#     system.gradient_masking=false \
#     system.three_noise=true \
#     system.identical_noising=false \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.pts_radius=0.003 \
#     system.calibration_value="${cal_vals[i]}" \
#     system.three_warp_noise=true \
#     system.consider_depth=false

# done

# prompts=(
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a DSLR photo of a silver metallic robot tiger"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a DSLR photo of an ironman figure"
# )

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
#     --gpu 6 \
#     system.tag="noise_full_grmask" \
#     system.gradient_masking=true \
#     data.num_multiview=1 \
#     system.three_noise=true \
#     system.identical_noising=true \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.calibration_value="${cal_vals[i]}" \
#     system.three_warp_noise=true 
# done
