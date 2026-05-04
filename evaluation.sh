# T3VF evaluation: w/ Perturbed Train setting
# Requires: export LIBERO_PLUS_PATH=/path/to/LIBERO-plus

for perturbation_category in "Background Textures" "Camera Viewpoints" "Language_instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise"; do
    for task_suite in libero_spatial libero_object libero_goal libero_10; do
        python T3VF/evaluation.py \
            --task_suite_name ${task_suite} \
            --model_family mantis \
            --model_id Yysrc/Mantis-Base \
            --checkpoints_dir "ckpt" \
            --run_id_note t3vf_eval \
            --local_log_dir experiments/libero_eval_logs \
            --norm_file_path config/norm_stats.json \
            --num_trials_per_task 1 \
            --perturbation_category "$perturbation_category" \
            --ttt_enabled True \
            --ttt_lr 1e-4 \
            --ttt_gap 4 \
            --ttt_batch_size 4 \
            --ttt_time_profile True \
            --additional_suffix "_action_variance" \
            --action_variance_enabled True \
            --action_variance_n_samples 5 \
            --action_variance_warmup 10 \
            --action_variance_percentile 0.7
    done
done
