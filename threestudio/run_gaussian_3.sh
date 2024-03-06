

python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_multi.yaml  --train --gpu 3 system.prompt_processor.prompt="a rabbit on a pancake"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" data.batch_size=4 data.num_multiview=0 data.multiview_deg=20.0 system.calibration_value=90 data.front_optimize=false
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive_all.yaml  --train --gpu 3 system.prompt_processor.prompt="a full body of a cat wearing a hat"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/cat-hat.png" data.batch_size=4 data.num_multiview=0 data.multiview_deg=20.0 system.calibration_value=135 data.front_optimize=false
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive_all.yaml  --train --gpu 3 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=4 data.num_multiview=0 data.multiview_deg=20.0 system.calibration_value=45 data.front_optimize=false


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 2 system.prompt_processor.prompt="a rabbit on a pancake"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" data.batch_size=6 data.num_multiview=0 data.multiview_deg=20.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 2 system.prompt_processor.prompt="a rabbit on a pancake"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=90


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 3 system.prompt_processor.prompt="a full body of a cat wearing a hat"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/cat-hat.png" data.batch_size=6 data.num_multiview=0 data.multiview_deg=20.0 system.calibration_value=135
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 3 system.prompt_processor.prompt="a full body of a cat wearing a hat"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/cat-hat.png" data.batch_size=2 data.num_multiview=2 data.multiview_deg=20.0 system.calibration_value=135


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.multiview_deg=30.0 system.calibration_value=45


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 3 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=80 data.multiview_deg=30.0 

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_value=160


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_full.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_value=160

# python launch.py --config configs/gaussian_splatting.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-fuse.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_noising.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=100 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gs_threefuse_noising.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=45.0 system.calibration_value=180 system.identical_noising=false system.gaussian_dynamic=true system.three_noise=false
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of a blue ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=15.0 system.calibration_value=180 system.identical_noising=false system.gaussian_dynamic=true system.three_noise=false
