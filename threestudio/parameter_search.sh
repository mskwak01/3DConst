prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a zoomed out DSLR photo of a ceramic lion, white background"
)

img_dirs=(
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
)

cal_vals=(
    90
    90
    90
    90
    90
    90
    90
    90
)

cfg_up_iter=(
    1200
    1500
    1800
    2100
    1200
    1500
    1800
    2100
)

max_steps=(
    3000
    3000
    3000
    3000
    3500
    3500
    3500
    3500
)

gpu_val=(
    0
    1
    2
    3
    4
    5
    6
    7
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu "${gpu_val[i]}" \
    system.tag="pointcloud_vis_cfgup_${cfg_up_iter[i]}_max_${max_steps[i]}" \
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