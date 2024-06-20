save_dirs=(
    # a_cat_wearing_a_bee_costume_x___results_for_improvement_baseline_cfg100@20240522-114527
    # a_cat_wearing_a_bee_costume_x___results_for_improvement_ours@20240522-122024
    # a_goose_made_out_of_gold_x___results_for_improvement_baseline_cfg100@20240522-114136
    a_goose_made_out_of_gold_x___results_for_improvement_ours@20240522-122028
    # a_snail_x___results_for_improvement_baseline_cfg100@20240522-100810
    # a_snail_x___results_for_improvement_ours@20240522-100744
    # a_turtle_x___results_for_improvement_baseline_cfg100@20240522-114058
    # a_turtle_x___results_for_improvement_ours@20240522-121654
    
)

for i in "${!save_dirs[@]}";
do
python launch.py \
    --config "/mnt/data4/matthew/outputs/gau_stable_diffusion/${save_dirs[i]}/configs/parsed.yaml" \
    --train \
    --gpu 3 \
    trainer.max_steps=3003 \
    system.restart=true \
    data.num_multiview=2 \
    data.multiview_deg=7 \
    system.visualize_noise=true \
    system.constant_viewpoints=false \
    system.visualize_noise_res=64 \
    system.n_pts_upscaling=25 \
    resume="/mnt/data4/matthew/outputs/gau_stable_diffusion/${save_dirs[i]}/ckpts/last.ckpt" \

done