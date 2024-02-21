# python launch.py --config configs/gaussian_splatting.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"
# CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-fuse.yaml --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png"

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gaussian_splatting_noising.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=100 system.gaussian_dynamic=true

# python launch.py --config custom/threestudio-3dgs/configs/gs_threefuse_noising.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=180 system.gaussian_dynamic=true

python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=30.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=15.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
python launch.py --config custom/threestudio-3dgs/configs/gs_naive_threefuse_noising.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" data.multiview_deg=45.0 system.calibration_value=180 system.identical_noising=true system.gaussian_dynamic=true system.three_noise=true
