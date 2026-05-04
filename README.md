# T³VF: Test-Time Training for Visual Foresight Vision-Language-Action Models

The official source code for **Test-Time Training for Visual Foresight Vision-Language-Action Models**.


## Environment

### 1. LIBERO-plus Setup

Follow the [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) repository to set up the benchmark environment.

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git
cd LIBERO-plus
pip install -e .
```

Download `assets.zip` from the [LIBERO-plus HuggingFace](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and extract it to `LIBERO-plus/libero/libero/`.

Then set the environment variable:

```bash
export LIBERO_PLUS_PATH=/path/to/LIBERO-plus
```

### 2. T³VF Setup

```bash
cd T3VF
pip install -r requirements.txt
```

### 3. Checkpoint

For the **w/ Perturbed Train** setting, fine-tune [Mantis](https://github.com/Yysrc/Mantis) on the LIBERO-Plus dataset following its training procedure, and place the resulting `model.pt` under `ckpt/`.

The **w/o Perturbed Train** setting uses the official pretrained Mantis checkpoints directly, so no additional checkpoint is needed.


## Evaluation

### w/ Perturbed Train (fine-tuned on LIBERO-Plus)

```bash
sh evaluation.sh
```

This evaluates the fine-tuned Mantis model with T³VF across all 7 perturbation dimensions and 4 task suites.

### w/o Perturbed Train (OOD, original LIBERO checkpoints)

```bash
sh evaluation_ood.sh
```

This evaluates the original Mantis checkpoints (without LIBERO-Plus fine-tuning) with T³VF. Each task suite uses its corresponding pretrained model (`LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, `LIBERO-Long`).

### Base Model (without T³VF)

To evaluate the base model without test-time training, set `--ttt_enabled False`:

```bash
CUDA_VISIBLE_DEVICES=0 python T3VF/evaluation.py \
    --task_suite_name libero_spatial \
    --model_family mantis \
    --model_id Yysrc/Mantis-Base \
    --checkpoints_dir "ckpt" \
    --norm_file_path config/norm_stats.json \
    --num_trials_per_task 1 \
    --perturbation_category "Robot Initial States" \
    --ttt_enabled False
```

### Results

Results (videos + metadata) are saved to `experiments/rollouts/` (w/ Perturbed Train) or `experiments/rollouts_ood/` (w/o Perturbed Train).
