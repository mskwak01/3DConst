prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a peacock with a crown"
    "a DSLR photo of a silver metallic robot tiger"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    "/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg"
)

cal_vals=(
    90
    0
    0
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml \
    --train \
    --gpu 2 \
    system.tag="pc_depth" \
    data.num_multiview=1 \
    system.gradient_masking=false \
    system.threefuse=true \
    system.three_noise=true \
    system.identical_noising=false \
    system.prompt_processor.prompt="${prompts[i]}" \
    system.image_dir="${img_dirs[i]}" \
    system.pts_radius=0.003 \
    system.calibration_value="${cal_vals[i]}" \
    # system.three_warp_noise=true \
    # system.consider_depth=false

done