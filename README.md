# MAIA: Anticipating Interaction Anomalies in Contact-Rich Manipulation via Multimodal World Models

<p align="center">
  <a href="https://joringell.github.io/project/MAIA/"><img src="https://img.shields.io/badge/Project-Page-2563eb.svg" alt="Project Page"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/Joringell/CRIA"><img src="https://img.shields.io/badge/Dataset-CRIA-f59e0b.svg" alt="CRIA Dataset"></a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2610.21686"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="arXiv Paper"></a>
</p>

## Overview

MAIA is a framework for anticipating anomalies before they occur in contact-rich robot manipulation. Given visual observations, a task instruction, planned future actions, and force/torque feedback, MAIA uses a vision-action-force world model to predict future interaction dynamics. A predictive anomaly head then estimates the risk of an upcoming anomaly from the learned multimodal representations.

We evaluate MAIA on [CRIA](https://huggingface.co/datasets/Joringell/CRIA), a real-world multimodal benchmark containing 2,852 trajectories across 18 manipulation tasks with frame-level anomaly annotations.

## Installation

### Environment

Create the provided Conda environment from the repository root:

```bash
conda env create -f maia.yaml
conda activate maia
```

Alternatively, install the dependencies in an existing Python environment:

```bash
pip install -r maia.txt
```

### Weights

MAIA is initialized from [Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner) and uses components from [LTX-Video](https://huggingface.co/Lightricks/LTX-Video). Download both checkpoints into the locations expected by the provided configurations:

```bash
# GE-Base-fast initialization
hf download agibot-world/Genie-Envisioner-v1.0 \
  --include GE_base_fast_v0.1.safetensors \
  --local-dir weights/ge_base

# LTX-Video tokenizer, text encoder, and VAE
hf download Lightricks/LTX-Video \
  --include "tokenizer/*" \
  --include "text_encoder/*" \
  --include "vae/*" \
  --local-dir weights/ltx_video
```

The checkpoints can also be downloaded manually from the [Genie-Envisioner model page](https://huggingface.co/agibot-world/Genie-Envisioner-v1.0/tree/main) and the [LTX-Video model page](https://huggingface.co/Lightricks/LTX-Video/tree/main).

### Dataset

Download CRIA from the [Hugging Face dataset page](https://huggingface.co/datasets/Joringell/CRIA):

```bash
hf download Joringell/CRIA \
  --repo-type dataset \
  --local-dir /root/CRIA
```

The released configurations use `/root/MAIA` as the repository path and `/root/CRIA` as the dataset path. If the repository or dataset is stored elsewhere, update the absolute paths in `configs/` and `scripts/` before running the experiments. The official train/validation/test manifests and normalization statistics are provided in `data_splits/` and `norm_stats/`, respectively.

## Experiments

Run the following commands from the repository root. `CUDA_VISIBLE_DEVICES` and `PYTHON_BIN` may be set to override the default GPU selection and Python executable.

### 1. Main Experiment

The main experiment trains and evaluates **MAIA-VAF**, the complete model conditioned on vision, planned actions, and force/torque feedback. Training consists of three stages.

**Stage 1 — Vision-action-force world model.** The world model is initialized from GE-Base-fast and trained on nominal interaction windows from the 13 training tasks. It observes four historical frames from three camera views (`wrist`, `side_1`, and `side_2`), conditions on 50-step absolute end-effector action chunks and force/torque history, and predicts future visual and force responses. The main settings in `configs/ltx_model/all/video_action_force_wm.yaml` are:

```yaml
return_video: true
return_force: true
return_interaction: false

train_mode: all
label_filter: only_normal
video_loss_scale: 1.0
force_loss_scale: 1.0
force_delta_loss_scale: 0.2
force_smooth_loss_scale: 0.1
interaction_semantic_loss_scale: 0.05

train_steps: 30000
batch_size: 8
lr: 3.0e-5

diffusion_model:
  config:
    use_action_force_adapter: true
    action_dim: 8
    force_dim: 6
```

**Stage 2 — Predictive anomaly head.** The trained world model is frozen and used to precompute visual, interaction, and force features. The anomaly head is then trained on these cached representations to predict window-level anomalies, step-level anomaly probabilities, and continuous risk. The main settings in `configs/ltx_model/all/video_action_force_anomaly.yaml` are:

```yaml
return_video: false
return_force: true
return_interaction: true

train_mode: anomaly_head
anomaly_target: label
anomaly_positive_is: anomaly
anomaly_pos_weight: 5.0
window_loss_scale: 1.0
risk_loss_scale: 0.5
step_loss_scale: 0.5

train_steps: 5000
batch_size: 32
lr: 5.0e-5
```

**Stage 3 — Novel-task adaptation.** The world model and anomaly head learned on the training tasks are jointly adapted using the 10% training subset of the five novel tasks. The script automatically inserts the checkpoints produced by the first two stages into `configs/ltx_model/all/video_action_force_anomaly_joint.yaml`. Its main settings are:

```yaml
train_mode: anomaly_adapter_joint
anomaly_head_checkpoint: <seen-task-anomaly-head>
split_manifest: /root/MAIA/data_splits/novel5_seed42_val20_test30_train10.json

return_video: false
return_force: true
return_interaction: true

train_steps: 1000
batch_size: 8
lr: 1.0e-5
```

**Evaluation.** After Stage 2, MAIA-VAF is evaluated on the held-out episodes of the 13 training tasks. After Stage 3, the adapted model is evaluated on the held-out episodes of the five novel tasks. `analyze/eval_action_force_anomaly_metrics.py` reports window-, and frame-level AUROC, AUPRC, balanced accuracy, precision, recall, F1, risk MAE, and confusion summaries. `analyze/eval_action_force_anomaly_video.py` additionally renders selected episodes with aligned anomaly scores, labels, predicted and measured force/torque curves, and force residuals.

Run the complete three-stage training and evaluation pipeline with:

```bash
bash scripts/maia_vaf.sh
```

Checkpoints, metric files, diagnostic plots, and rendered videos are saved under `outputs/`.

### 2. Ablation Studies

We evaluate two reduced-modality variants under the same seen-task and novel-task protocol:

| Variant | Input modalities | Command |
| --- | --- | --- |
| MAIA-V | Vision only | `bash scripts/maia_v.sh` |
| MAIA-VA | Vision and planned actions | `bash scripts/maia_va.sh` |

The MAIA-V and MAIA-VA scripts train their corresponding world models and anomaly heads, evaluate the seen-task split, adapt to the novel tasks, and evaluate the novel-task split.

### 3. Baselines

The repository includes reproducible implementations of PCA-KMeans, LogPZO, and SAFE.

**PCA-KMeans and LogPZO.** These are evaluated separately on each supported task. Set `TASK` to `insert`, `press`, `path`, `wipe`, or `articulated`:

```bash
TASK=insert bash scripts/pca.sh
TASK=insert bash scripts/logpzo.sh
```

PCA-KMeans evaluates both observed-feature and world-model-feature branches with pointwise and global calibration. LogPZO trains and evaluates both observed-feature and world-model-feature variants. These pipelines require the corresponding MAIA-VAF feature cache.

**SAFE.** The SAFE pipeline trains and evaluates both LSTM and MLP variants using pi0.5 policy representations on the seen and novel task splits:

```bash
bash scripts/safe.sh
```

SAFE additionally requires an [OpenPI](https://github.com/Physical-Intelligence/openpi) checkout and a pi0.5 checkpoint. Update the OpenPI and checkpoint paths in `scripts/safe.sh` if they differ from the defaults.

For the remaining published baselines, please use their official implementations: [WCM](https://github.com/sylvestf/WCM), [Robometer](https://github.com/robometer/robometer), and [Robo-Dopamine](https://github.com/FlagOpen/Robo-Dopamine).

## License

This project is released under the MIT License. Third-party models, datasets, and code are subject to their respective licenses.


## Citation

If you find MAIA or CRIA useful in your research, please cite our paper:

```bibtex
@misc{qiao2026maia,
  title={MAIA: Anticipating Interaction Anomalies in Contact-Rich Manipulation via Multimodal World Models},
  author={Yi Qiao and Maolin Zheng and Haoran Sun and Lujia Dong and Wenzhao Lian},
  year={2026},
  eprint={2610.21686},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.21686}
}
```
