prompts=(
    # "a zoomed out DSLR photo of a ceramic lion, white background"
    "a DSLR photo of an owl"
    # "a cute meercat"
    # "a rabbit on a pancake"
    # "a peacock with a crown"
    # "a mysterious LEGO wizard"
    # "a product photo of cat-shaped toy"
    # "a full body of a cat wearing a hat"
)

img_dirs=(
    # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
    # "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
    # "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
    # "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    # "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
    # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
    # "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
)

cal_vals=(
    # 90
    # 45
    135
    # 90
    # 90
    # 90
    # 75
    # 135
)

grad_clips=(
    "[0,0.5,2.0,3000]"
    "[0,0.5,1.0,3000]"
    "[0,0.5,0.5,3000]"
    "[0,1.0,2.0,3000]"
    "[0,1.0,1.0,3000]"
)

gpu_ids=(1 2 3 4 5)

for i in "${!prompts[@]}"; do
    for j in "${!gpu_ids[@]}"; do
        gpu_id=${gpu_ids[j]}
        grad_clip=${grad_clips[j % ${#grad_clips[@]}]}
        
        if [ $j -eq $((${#gpu_ids[@]} - 1)) ]; then
            # Do not run in the background for the last GPU
            sleep 5s
            python launch.py \
                --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
                --train \
                --gpu $gpu_id \
                system.tag="dynamic_clip" \
                system.three_noise=true \
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
                data.num_const_views=10 \
                system.reprojection_info=false \
                system.guidance.guidance_scale=7.5 \
                system.guidance.add_loss="cosine_sim" \
                system.guidance.use_normalized_grad=false \
                system.guidance.add_loss_stepping=false \
                system.guidance.grad_cons_mask=false \
                system.guidance.mask_w_timestep=false \
                system.guidance.vis_grad=true \
                system.guidance.use_disp_loss=true \
                system.guidance.use_sim_loss=true \
                system.guidance.weight_sim_loss=5.0 \
                system.guidance.weight_disp_loss=1.0 \
                trainer.max_steps=3000 \
                system.guidance.backprop_grad=false \
                system.guidance.debugging=false \
                system.guidance.disp_loss_to_latent=false \
                system.guidance.only_geo=false \
                system.guidance.grad_clip=$grad_clip
        else
            sleep 5s
            # Run in the background for all other GPUs
            python launch.py \
                --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
                --train \
                --gpu $gpu_id \
                system.tag="dynamic_clip" \
                system.three_noise=true \
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
                data.num_const_views=10 \
                system.reprojection_info=false \
                system.guidance.guidance_scale=7.5 \
                system.guidance.add_loss="cosine_sim" \
                system.guidance.use_normalized_grad=false \
                system.guidance.add_loss_stepping=false \
                system.guidance.grad_cons_mask=false \
                system.guidance.mask_w_timestep=false \
                system.guidance.vis_grad=true \
                system.guidance.use_disp_loss=true \
                system.guidance.use_sim_loss=true \
                system.guidance.weight_sim_loss=5.0 \
                system.guidance.weight_disp_loss=1.0 \
                trainer.max_steps=3000 \
                system.guidance.backprop_grad=false \
                system.guidance.debugging=false \
                system.guidance.disp_loss_to_latent=false \
                system.guidance.only_geo=false \
                system.guidance.grad_clip=$grad_clip &
        fi
    done
done
wait
