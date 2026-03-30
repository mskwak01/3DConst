
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=45 data.front_optimize=true

prompts=(
    "a zoomed out DSLR photo of a ceramic lion, white background"
    "a peacock with a crown"
    "a mysterious LEGO wizard"
    "a product photo of cat-shaped toy"
    # "a DSLR photo of an ironman figure"
)

img_dirs=(
    "/home/cvlab15/project/woojeong/wj_threestudio/images/a_ceramic_lion.png"
    "/home/cvlab15/project/woojeong/naver/images/peacock.png"
    "/home/cvlab15/project/woojeong/naver/images/lego-wizard2.png"
    "/home/cvlab15/project/woojeong/naver/images/cat-toy.png"
    # "/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
)

cal_vals=(
    90
    90
    90
    75
    # 90   
)

for i in "${!prompts[@]}";
do
python launch.py \

    --config custom/threestudio-3dgs/configs/gau_stable_diffusion.yaml \
    --train \
    --gpu 7 \
    system.tag="custom_raster_up9_itv_10_ball" \
    system.pytorch_three=false \
    data.num_multiview=1 \
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
    system.noise_alter_interval=10 \
    data.multiview_deg=20 \

done


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_multi.yaml  --train \
#     --gpu 6 \
#     system.tag="finmask_penta_view_2_2_pts_002" \
#     system.gradient_masking=true \
#     system.prompt_processor.prompt="a rabbit on a pancake" \
#     system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" \
#     system.calibration_value=90 \
#     system.pts_radius=0.02 \
#     system.geometry.split_thresh=0.01 \


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train \
#     --gpu 6 \
#     system.tag="tester" \
#     system.gradient_masking=false \
#     system.prompt_processor.prompt="a rabbit on a pancake" \
#     system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" \
#     data.batch_size=2 \
#     data.num_multiview=2 \
#     data.multiview_deg=20 \
#     data.front_optimize=true \
#     data.batch_uniform_azimuth=false \
#     system.calibration_value=90 \
#     system.pts_radius=0.02 \

    # system.tag="finmask_nograd_view_fr_2_2_pts_002" \

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 6 system.tag="fr_view_2_2_pts_002_masking_interp_250_0008" system.gradient_masking=true system.interpolated_masking=true system.prompt_processor.prompt="a rabbit on a pancake"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=90 data.front_optimize=true system.pts_radius=0.02

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 6 system.tag="all_view_2_2_pts_002_masking_interp_250_0008" system.gradient_masking=true system.interpolated_masking=true system.prompt_processor.prompt="a full body of a cat wearing a hat"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/cat-hat.png" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=135 data.front_optimize=false system.pts_radius=0.02

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 6 system.tag="all_view_2_2_pts_002_masking_interp_250_0008" system.gradient_masking=true system.interpolated_masking=true system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=45 data.front_optimize=false system.pts_radius=0.02


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a rabbit on a pancake"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" data.batch_size=1 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=90 data.front_optimize=true
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a full body of a cat wearing a hat"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/cat-hat.png" data.batch_size=1 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=135 data.front_optimize=true
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=45 data.front_optimize=true


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_const.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=45


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a red Porsche sportscar"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sports.png" data.multiview_deg=30.0 system.calibration_value=0


# python launch.py --config configs/gaussian_splatting.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-fuse.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_noising.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=100 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gs_threefuse_noising.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=30.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
# python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=15.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
# python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=45.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
