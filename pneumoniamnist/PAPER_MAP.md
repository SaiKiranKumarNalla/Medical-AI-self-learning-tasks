# Paper map — read with the learner

## Sources
- Yang et al., **MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification**, Scientific Data 10, 41 (2023): https://doi.org/10.1038/s41597-022-01721-8
- Dataset/API: https://medmnist.com/ and https://github.com/MedMNIST/MedMNIST
- Authors' experiments: https://github.com/MedMNIST/experiments
- Architecture: https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/models.py
- Training: https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/train_and_eval_pytorch.py

Links to main were inspected on 2026-10-25; main is mutable. Pin a commit before implementation. This is a reading guide, not a substitute for the full methods or a frozen reproduction specification.

## What the paper contributes
It standardizes diverse biomedical classification datasets and benchmarks methods. Our subset is pediatric chest X-ray normal/pneumonia classification. The 28-pixel images are centre-cropped/downsampled versions of source images. This is an educational algorithm benchmark, not validation of a clinical diagnostic system. Check the package's actual label map; do not assume which numeric label is positive.

Read in this order: PneumoniaMNIST dataset subsection → preprocessing/splits → neural-network benchmarking methodology → relevant results table → limitations. For each, explain: what decision was made, why it matters, where the code implements it, and what evidence could contradict our assumptions.

## Current authors' code: preliminary replication settings
The inspected script uses Adam (learning rate 0.001), a multistep schedule with factor 0.1 at half and three-quarters of the epoch budget, default 100 epochs and batch size 128. For this binary-class task it uses two logits and cross-entropy, not the multilabel BCE branch. Normalization uses mean/std 0.5. The non-resize branch selects a custom small-image ResNet; it is not an ImageNet-pretrained model. Checkpoint selection uses validation AUC. The script also logs test performance during training; our teaching harness will withhold test metrics until decisions are frozen and record that reporting deviation.

The model has a 3×3 stride-1 stem, residual stage widths 64/128/256/512, two BasicBlocks per stage, projection shortcuts when dimensions change, global average pooling and a classifier. For 28-pixel input, stage spatial sizes are 28 → 28 → 14 → 7 → 4 → 1. Check these by running actual shape probes.

## Resolve before claiming replication
- Pin paper version, upstream commit, MedMNIST package/data version and artifact checksum.
- Confirm exact paper settings against repository defaults and launch commands: grayscale versus RGB, resize flag, loss, epochs, optimizer/scheduler, batch size and any augmentation.
- Extract the exact target result/table row, number of repeats, seed policy and aggregation rule with source locations. Do not use a remembered AUC or assume a default invocation reproduces the table.
- Confirm evaluation package definitions and prediction-column mapping, especially positive-class AUC.
- Record split provenance and what patient independence can/cannot be verified from supplied metadata.
- Document our MPS device adaptation; upstream inspected script selects CPU/CUDA, not MPS.
- Keep a deviations table: source setting | our setting | reason | expected impact.

## Our additions
Linear/MLP/tiny-CNN learning exercises precede the replication. New architectures, altered resolution, augmentation, transfer learning and hyperparameter searches are extensions. Report them in separate tables. Stronger results on this test set do not establish clinical superiority or external generalization.
