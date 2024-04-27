python launch.py \
    --config configs/prolificdreamer-noise.yaml \
    --train \
    --gpu 0 \
    data.batch_size=2 \
    data.num_multiview=0\
    
    # system.gradient_masking=true \
    # system.prompt_processor.prompt="a rabbit on a pancake" \
    # system.image_dir="/home/cvlab15/project/woojeong/naver/images/rabbit-pancake.jpeg" \
    # system.calibration_value=90 \
    # system.pts_radius=0.02 \
    # system.nearby_fusing=false \
    # data.multiview_deg=20 \
    # data.front_optimize=true \
    # data.batch_uniform_azimuth=false \

