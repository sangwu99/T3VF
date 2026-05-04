# T3VF evaluation: w/o Perturbed Train setting (OOD)
# Requires: export LIBERO_PLUS_PATH=/path/to/LIBERO-plus
# Each task suite uses its own pretrained model (no fine-tuning on LIBERO-Plus)

for perturbation_category in "Background Textures" "Camera Viewpoints" "Language_instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise"; do
    python T3VF/evaluation_ood.py \
        --task_suite_name libero_spatial \
        --model_family mantis \
        --model_id Yysrc/LIBERO-Spatial \
        --run_id_note t3vf_ood_eval \
        --local_log_dir experiments/libero_ood_eval_logs \
        --norm_file_path config/norm_stats.json \
        --num_trials_per_task 1 \
        --perturbation_category "$perturbation_category" \
        --ttt_enabled True \
        --ttt_lr 1e-4 \
        --ttt_gap 4 \
        --ttt_batch_size 4 \
        --ttt_time_profile True \
        --additional_suffix "_action_variance_ood" \
        --action_variance_enabled True \
        --action_variance_n_samples 5 \
        --action_variance_warmup 10 \
        --action_variance_percentile 0.7

    python T3VF/evaluation_ood.py \
        --task_suite_name libero_object \
        --model_family mantis \
        --model_id Yysrc/LIBERO-Object \
        --run_id_note t3vf_ood_eval \
        --local_log_dir experiments/libero_ood_eval_logs \
        --norm_file_path config/norm_stats.json \
        --num_trials_per_task 1 \
        --perturbation_category "$perturbation_category" \
        --ttt_enabled True \
        --ttt_lr 1e-4 \
        --ttt_gap 4 \
        --ttt_batch_size 4 \
        --ttt_time_profile True \
        --additional_suffix "_action_variance_ood" \
        --action_variance_enabled True \
        --action_variance_n_samples 5 \
        --action_variance_warmup 10 \
        --action_variance_percentile 0.7

    python T3VF/evaluation_ood.py \
        --task_suite_name libero_goal \
        --model_family mantis \
        --model_id Yysrc/LIBERO-Goal \
        --run_id_note t3vf_ood_eval \
        --local_log_dir experiments/libero_ood_eval_logs \
        --norm_file_path config/norm_stats.json \
        --num_trials_per_task 1 \
        --perturbation_category "$perturbation_category" \
        --ttt_enabled True \
        --ttt_lr 1e-4 \
        --ttt_gap 4 \
        --ttt_batch_size 4 \
        --ttt_time_profile True \
        --additional_suffix "_action_variance_ood" \
        --action_variance_enabled True \
        --action_variance_n_samples 5 \
        --action_variance_warmup 10 \
        --action_variance_percentile 0.7

    python T3VF/evaluation_ood.py \
        --task_suite_name libero_10 \
        --model_family mantis \
        --model_id Yysrc/LIBERO-Long \
        --run_id_note t3vf_ood_eval \
        --local_log_dir experiments/libero_ood_eval_logs \
        --norm_file_path config/norm_stats.json \
        --num_trials_per_task 1 \
        --perturbation_category "$perturbation_category" \
        --ttt_enabled True \
        --ttt_lr 1e-4 \
        --ttt_gap 4 \
        --ttt_batch_size 4 \
        --ttt_time_profile True \
        --additional_suffix "_action_variance_ood" \
        --action_variance_enabled True \
        --action_variance_n_samples 5 \
        --action_variance_warmup 10 \
        --action_variance_percentile 0.7
done
