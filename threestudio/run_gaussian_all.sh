

python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.multiview_deg=10.0 system.calibration_value=45
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 3 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.multiview_deg=20.0 system.calibration_value=45
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.multiview_deg=30.0 system.calibration_value=45

python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=3 data.multiview_deg=10.0 system.calibration_value=45
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 6 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=3 data.multiview_deg=15.0 system.calibration_value=45
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=3 data.multiview_deg=20.0 system.calibration_value=45
python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=3 data.multiview_deg=5.0 system.calibration_value=45


python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an owl"  system.image_dir="/home/cvlab15/project/woojeong/naver/images/owl.jpeg" data.batch_size=1 data.num_multiview=4 data.multiview_deg=15.0 system.calibration_value=45




# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=100 data.multiview_deg=30.0 

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=80 data.multiview_deg=30.0 
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_pts.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=80 data.multiview_deg=30.0 
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=80 data.multiview_deg=30.0 
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_multi.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an ironman figure" system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/matt_threestudio/threestudio/threestudio/images/ironman.png" system.calibration_value=80 data.multiview_deg=30.0 


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_value=160

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_noising_full.yaml  --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of a red Porsche sportscar"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sports.png" data.multiview_deg=30.0 system.calibration_value=0

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_pts.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_pts.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_pts.yaml  --train --gpu 2 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_value=160

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_pts.yaml  --train --gpu 3 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_noising_pts.yaml  --train --gpu 3 system.prompt_processor.prompt="a DSLR photo of a red Porsche sportscar"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sports.png" data.multiview_deg=30.0 system.calibration_value=0


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 4 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_value=160

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_pc_init_naive.yaml  --train --gpu 5 system.prompt_processor.prompt="a DSLR photo of a red Porsche sportscar"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sports.png" data.multiview_deg=30.0 system.calibration_value=0


# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 6 system.prompt_processor.prompt="a DSLR photo of an origami motorcycle"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/motor.png" data.multiview_deg=30.0 system.calibration_value=0
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 6 system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/tiger.jpeg" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 6 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sail.jpeg" data.multiview_deg=30.0 system.calibration_valthreefuse

# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/chimpa.png" data.multiview_deg=30.0 system.calibration_value=90
# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 7 system.prompt_processor.prompt="a DSLR photo of a red Porsche sportscar"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/sports.png" data.multiview_deg=30.0 system.calibration_value=0





# python launch.py --config custom/threestudio-3dgs/configs/gs_sds_threefuse_naive.yaml  --train --gpu 4 system.prompt_processor.prompt="a Michelangelo style statue of an astronaut"  system.image_dir="/home/cvlab15/project/naver_diffusion/matthew/fresh_three/3DConst/threestudio/images/human.png" data.multiview_deg=30.0 system.calibration_value=0

# origami motorcycle
# python launch.py --config configs/prolificdreamer-geometry-fuse.yaml --train --gpu 1 system.img_dir="/home/cvlab16/projects/diffusion/ines/threestudio/images/motor.png" 

# robot tiger
# python launch.py --config configs/prolificdreamer-geometry-fuse.yaml --train --gpu 4 system.img_dir="/home/cvlab16/projects/diffusion/ines/threestudio/images/realtiger.png" system.prompt_processor.prompt="a DSLR photo of a silver metallic robot tiger" 

# Sailboat
# python launch.py --config configs/prolificdreamer-fuse.yaml --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of a majestic sailboat" system.img_dir="/home/cvlab15/project/ines/threestudio/images/sail.jpeg" system.calibration_value=60  

# 
# /home/cvlab15/project/ines/threestudio/images/human.png

# Chimpanzee headphone
# python launch.py --config configs/prolificdreamer-fuse.yaml --train --gpu 1 system.prompt_processor.prompt="a DSLR photo of a chimpanzee wearing headphones" system.img_dir="/home/cvlab16/projects/diffusion/ines/threestudio/images/chimpa.png" system.calibration_value=240 system.pc_check=0

# Ceramic Lion
# python launch.py --config configs/prolificdreamer-geometry.yaml --train --gpu 4 system.prompt_processor.prompt="a ceramic lion" system.geometry_convert_from=/mnt/hdd/ines_studio/prolificdreamer/a_ceramic_lion@20231117-132602/ckpts/last.ckpt   system.renderer.context_type=cuda
