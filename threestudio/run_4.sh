prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
)

img_dirs=(
    "/mnt/image-net-full/j1nhwa.kim/interns/minseop.kwak/3DConst/threestudio/images/tiger.jpeg"
)

cal_vals=(
    90
)

for i in "${!prompts[@]}";
do
python launch.py \
    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 1 \
    system.tag="zzzzzzzzzz_final_ours" \
    system.three_noise=true \
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
    trainer.max_steps=3000 \
    system.guidance.backprop_grad=false \
    system.guidance.debugging=false \
    system.noise_interval_schedule=true \
    trainer.val_check_interval=200 \
    system.guidance.cfg_lastup=true \
    system.guidance.cfg_change_iter=2300 \
    data.n_val_views=20 \

done


# # prompts=(
# #     "a product photo of a toy tank"
# #     # "a DSLR photo of a silver metallic robot tiger"
# # )

# # img_dirs=(
# #     /home/cvlab15/project/woojeong/naver/images/tank2.png
# #     # /home/cvlab15/project/woojeong/naver/images/eagle.jpeg
# #     # /home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/mermaid.png
# #     # /home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg
# # )

# # cal_vals=(
# #     90
# #     # 160
# #     # 225
# #     # 90 
# # )

# prompts=(
#     "a DSLR photo of a penguin"
#     # "a DSLR photo of a giraffe"
#     # "a cute rabbit"
# )

# img_dirs=(
#     /home/cvlab15/project/woojeong/naver/images/penguin.jpeg
#     # /home/cvlab15/project/woojeong/naver/images/giraffe.png
#     # /home/cvlab15/project/woojeong/naver/images/rabbit.jpeg
# )

# cal_vals=(
#     90
#     # 90
#     # 100
# )
# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 4 \
#     system.tag="z__prolific_naive" \
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


#     "a cat with wings"
#     "a beautiful rainbow fish"
#     90
#     180
#     "/mnt/data3/3DConst/images/cat-wing.png"
#     "/mnt/data3/3DConst/images/fish.png"

# prompts=(
#     "a beautiful rainbow fish"
#     "a cat wearing a bee costume"
#     "a cat with wings"
#     "a chow chow puppy"
#     "a goose made out of gold"
#     "a pug made out of metal"
#     "a toy robot"
#     "a snail"
#     "a turtle"
#     "a fennec fox"
#     "a train engine made out of clay"
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


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 4 \
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


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 4 \
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
#     # "a rabbit on a pancake"
#     # "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     # "a product photo of cat-shaped toy"
#     # "a DSLR photo of an owl"
#     "a cute meercat"
#     "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     # "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     # "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     # "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     # 90
#     # 90
#     90
#     90
#     # 75
#     # 75
#     45
#     135
# )

# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config configs/prolificdreamer-noise.yaml \
#     --train \
#     --gpu 3 \
#     system.tag="prolificdreamer_cos_sim" \
#     data.num_multiview=1\
#     system.three_noise=true \
#     system.threefuse=true \
#     system.pytorch_three=false \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.surf_radius=0.05 \
#     system.calibration_value="${cal_vals[i]}" \
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
#     system.guidance.vis_grad=false \
#     system.guidance.use_disp_loss=false \
#     system.guidance.use_sim_loss=true \
#     system.guidance.weight_sim_loss=5.0 \
#     system.guidance.weight_disp_loss=1.0 \
#     trainer.max_steps=15000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.noise_interval_schedule=true \
#     trainer.val_check_interval=250 \
#     data.n_val_views=20 \

# done

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
#     --gpu 4 \
#     system.tag="real_geo_only_cfg_20_short_on_10_off_${interval[i]}" \
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
#     system.guidance.guidance_scale=20 \
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
#     trainer.max_steps=3000 \
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
#     400
#     800
#     1200
#     1600
#     400
#     800
#     1200
#     1600  
# )


# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 7 \
#     system.tag="real_only_geo_both_all_out_intv_30_${iter[i]}" \
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
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.disp_loss_to_latent=false \
#     system.guidance.only_geo=true \
#     system.guidance.only_geo_front_on=false \
#     system.guidance.geo_start_in="${iter[i]}" \
#     # trainer.val_check_interval=[100,100] \

# done


# prompts=(
#     # "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     "a rabbit on a pancake"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     # "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     # 90
#     45
#     135
#     90
#     90
#     90
#     75
#     135
# )


# prompts=(
#     "a DSLR photo of an owl"
#     "a cute meercat"
#     "a rabbit on a pancake"
#     "a zoomed out DSLR photo of a ceramic lion, white background"
#     "a peacock with a crown"
#     "a mysterious LEGO wizard"
#     "a product photo of cat-shaped toy"
#     "a full body of a cat wearing a hat"
# )

# img_dirs=(
#     "/home/cvlab15/project/woojeong/naver/images/owl.jpeg"
#     "/home/cvlab15/project/soowon/naver/3DConst/threestudio/load/images/meercat.png"
#     "/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg"
#     "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
#     "/home/cvlab15/project/woojeong/naver/images/peacock.png"
#     "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
#     "/home/cvlab15/project/woojeong/naver/images/cat-hat.png"
# )

# cal_vals=(
#     45
#     135
#     90
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
#     --gpu 4 \
#     system.tag="vis_all_grads_deg_25" \
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
#     data.multiview_deg=25 \
#     data.constant_viewpoints=true \
#     data.num_const_views=10 \
#     system.reprojection_info=false \
#     system.guidance.guidance_scale=7.5 \
#     system.guidance.add_loss="no_loss" \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.add_loss_stepping=false \
#     system.guidance.grad_cons_mask=false \
#     system.guidance.mask_w_timestep=false \
#     system.guidance.vis_grad=true \
#     trainer.max_steps=3000 \
#     system.guidance.backprop_grad=false \
#     system.guidance.debugging=false \
#     system.guidance.use_sim_loss=false \
#     system.guidance.weight_sim_loss=0.23 \

# done

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

# thresh_vals=(
#     # 2.5
#     # 2
#     # 1.5
#     # 1
#     2.5
#     2
#     1.5
#     1
# )



# for i in "${!prompts[@]}";
# do
# python launch.py \
#     --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
#     --train \
#     --gpu 4 \
#     system.tag="c_75_mask_visualizer_random_thresh_deg_15_th_${thresh_vals[i]}" \
#     system.three_noise=false \
#     system.pytorch_three=false \
#     data.num_multiview=2 \
#     system.prompt_processor.prompt="${prompts[i]}" \
#     system.image_dir="${img_dirs[i]}" \
#     system.calibration_value="${cal_vals[i]}" \
#     system.surf_radius=0.05 \
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
#     system.guidance.add_loss="no_loss" \
#     system.guidance.weight_add_loss=10.0 \
#     system.guidance.use_normalized_grad=true \
#     system.guidance.add_loss_stepping=true \
#     system.guidance.debugging=false \
#     system.guidance.vis_grad=true \
#     system.guidance.grad_thresh="${thresh_vals[i]}" \
#     trainer.max_steps=3000 \

# done