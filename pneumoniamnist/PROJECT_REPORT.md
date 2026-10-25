# Reviewed self-learning study report — 2026-10-25

## Scope
This replaces the previous overconfident report. Saved checkpoints were evaluated on CPU without retraining. The README and results/verified_metrics.json are the corrected source of truth. Earlier claims of clinical superiority and exact replication are withdrawn.

| Saved model | AUC | Accuracy | Sensitivity | Specificity | TN / FP / FN / TP | Parameters |
|---|---:|---:|---:|---:|---|---:|
| best_resnet18_val_auc | 0.9480 | 84.62% | 98.97% | 60.68% | 142 / 92 / 4 / 386 | 11,168,706 |
| best_resnet18_weighted | 0.9551 | 87.98% | 99.23% | 69.23% | 162 / 72 / 3 / 387 | 11,168,706 |
| best_resnet18_focal | 0.9511 | 89.10% | 94.87% | 79.49% | 186 / 48 / 20 / 370 | 11,168,706 |
| best_plainnet18 | 0.9675 | 89.90% | 98.46% | 75.64% | 177 / 57 / 6 / 384 | 10,994,882 |

## Corrections
1. README described a torchvision 7×7 stem. Actual saved models use the 3×3 small-image stem. Corrected parameter counts, including PlainNet's missing projection shortcuts.
2. Corrected weighted and focal AUC and PlainNet confusion counts. Earlier README pruning tables and epoch counts conflicted with the report; unverified entries have been removed rather than guessed.
3. Replaced the Grad-CAM figure. The old purported TN was labelled pneumonia/predicted pneumonia. New examples are selected programmatically from measured targets and predictions, with a consistent pneumonia-logit attribution target. Code is now included. Anatomical/causal claims cannot be established from these heatmaps.
4. Confirmed eight train/validation shared image hashes. Withheld claims of no leakage or independent patient counts. No train/test or val/test exact hash overlaps were found; this does not exclude near duplicates or shared patients.
5. Fixed CUDA selection, portable output paths and data acquisition. New run directories protect old weights. Corrected epoch loss aggregation for weighted CE using sum of target weights. Checkpoint selection based on AUC is unaffected by that loss-display repair.
6. Added RNG state to future baseline checkpoints and explicit resume. Legacy checkpoints cannot recover missing historical RNG state. Extension scripts explicitly reject resume rather than pretending to support it.

## What the results support
These saved networks discriminate the two supplied labels at the reported thresholds on this benchmark. In this set, focal loss trades lower sensitivity for higher specificity relative to the baseline. This does not identify a clinically optimal threshold or loss. PlainNet achieved the highest observed AUC among these saved runs, without establishing that residual networks are inferior.

## What remains scientifically unresolved
The test set informed later experiments. It is no longer an untouched final evaluation. A clean future study must freeze hypotheses and use validation only; final confirmation needs new untouched data with suitable labels and grouping. The known duplicated validation examples should be removed for a separately labelled development protocol, without rewriting historical results. Keep manifests, patient grouping where available, multiple seeds, fixed stopping/tuning budgets and uncertainty estimates. Do not present a repeat run on the already-seen test set as fresh confirmation.

A low first-epoch AUC does not demonstrate vanishing gradients; measure gradients to test that hypothesis. A small pruning gain does not establish noise removal or hardware acceleration. Grad-CAM does not prove diagnostic reasoning. No clinical treatment or screening recommendations follow from this tutorial.

## Publication state
Corrected self-learning materials prepared locally. No GitHub repository or remote commit verified. Private conversation and pre-review snapshots are excluded. Original data/checkpoints remain local; do not delete unique artifacts merely because code is pushed. Any later cleanup requires a verified remote copy of each artifact being removed, or explicit agreement to discard unrecoverable outputs.

Future-run safeguards added: exact train-overlap removal from validation by default, explicit --official-validation override, test evaluation opt-in via --evaluate-test, and matching validation-protocol checks on resume. These safeguards do not alter historical checkpoints/results.
