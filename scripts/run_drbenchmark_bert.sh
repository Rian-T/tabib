#!/bin/bash
# DrBenchmark BERT training campaign
# Run all classification datasets with 3 BERT models on 2 GPUs

set -e

WANDB_PROJECT="tabib-drbenchmark"

# Classification configs (fast: ~10-30 min each)
CLS_CONFIGS=(
    # DiaMED (22 classes)
    "cls_diamed_moderncamembert.yaml"
    "cls_diamed_camembert_bio.yaml"
    "cls_diamed_camembert.yaml"
    # ESSAI (4 classes)
    "cls_essai_moderncamembert.yaml"
    "cls_essai_camembert_bio.yaml"
    "cls_essai_camembert.yaml"
    # MORFITT (12 classes)
    "cls_morfitt_moderncamembert.yaml"
    "cls_morfitt_camembert_bio.yaml"
    "cls_morfitt_camembert.yaml"
)

# Similarity configs
SIM_CONFIGS=(
    "sim_clister_moderncamembert.yaml"
    "sim_clister_camembert_bio.yaml"
    "sim_clister_camembert.yaml"
)

echo "=== DrBenchmark BERT Training Campaign ==="
echo "Total configs: ${#CLS_CONFIGS[@]} classification + ${#SIM_CONFIGS[@]} similarity"
echo ""

# Run classification on GPU 0
echo "=== Classification on GPU 0 ==="
for config in "${CLS_CONFIGS[@]}"; do
    echo "Running $config..."
    WANDB_PROJECT=$WANDB_PROJECT CUDA_VISIBLE_DEVICES=0 poetry run tabib train configs/$config 2>&1 | tee -a results/drbenchmark_cls.log
    echo "Completed $config"
    echo ""
done

# Run similarity on GPU 1 (can run in parallel with classification)
echo "=== Similarity on GPU 1 ==="
for config in "${SIM_CONFIGS[@]}"; do
    echo "Running $config..."
    WANDB_PROJECT=$WANDB_PROJECT CUDA_VISIBLE_DEVICES=1 poetry run tabib train configs/$config 2>&1 | tee -a results/drbenchmark_sim.log
    echo "Completed $config"
    echo ""
done

echo "=== All done! ==="
