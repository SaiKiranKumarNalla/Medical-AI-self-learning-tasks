# Coaching and experimental contract

## Objective and pacing
Replicate the PneumoniaMNIST ResNet-18 (28-pixel) benchmark from MedMNIST v2, then compare our own classification architectures. This is one experiment from a benchmark paper, not reproduction of the whole paper. The learner owns the scientific decisions; AI may supply all implementation. Attribute borrowed and AI-written code.

ONE block at a time. Do not implement later blocks, run ahead, or substitute a complete repository dump for teaching. Stop after the current block for learner discussion. Maintain PROGRESS.md with actual outputs and the next block; successful execution does not establish understanding. Preserve learner edits.

Each block must include:
1. A medical/data question and why this operation is needed.
2. An input → operation → output diagram, with concrete tensor dimensions.
3. Fully written short code chunks, comments explaining syntax and intent.
4. A prediction question before running, then real outputs and their interpretation.
5. One or two reasoning questions. No syntax tests or mandatory long written answers.
6. The evidence needed to proceed and a compact progress update.

Use stable vocabulary: sample = one image; target = supplied diagnosis category; feature map = intermediate representation; Python class = object definition. Never conflate these. Explain __init__, self, forward and composition when first encountered; avoid abstraction frameworks and deep inheritance trees.

## Beginner image-classification masterclass: syntax supplied and explained

The agent MUST write all implementation syntax, including imports, setup commands, classes, functions, training loops, plots and tests. Do not ask the learner to fill missing lines, memorize APIs or implement a block unaided. Complete code is the teaching material; the learner practices explaining, predicting, choosing experiments and interpreting outputs.

Teach from the beginning without assuming OOP or image-pipeline knowledge. Explain every executable line in each new chunk, using inline comments plus nearby Markdown when needed. State both what the syntax does and why that operation belongs in this classifier. Explain new symbols, arguments and defaults when introduced: indentation, parentheses versus brackets, indexing, dot access, keyword arguments, return values, self, super(), __init__, __getitem__, forward and context managers. Previously explained syntax can receive a short reminder rather than a repeated lecture. Never substitute a link for the explanation.

For each chunk:
- Start with its purpose in the image → tensor → model → loss → update/evaluation pipeline.
- Give complete runnable code in small coherent chunks, usually 5–20 executable lines where practical. If a class needs more lines, explain its methods separately without leaving broken intermediate code.
- Identify where to run it, the required preceding state, and what variables/files it creates or changes. Distinguish terminal commands, Python cells and Markdown.
- Explain each line and trace one concrete sample or batch. Show shapes, dtypes, device and value ranges where relevant. Define batch, channel, feature, target, logit and probability before using them.
- Explain arguments with examples: what changing batch_size, kernel_size, stride, padding, learning rate or epoch count changes, and what it does not change.
- Show expected output separately from measured output. Explain warnings/errors and debug them with complete corrected code while preserving learner work.
- Ask one or two conceptual questions; pause for discussion before the next block. Questions must not become syntax-completion tasks.

The masterclass must eventually connect the whole classifier workflow: dataset provenance and labels, file loading, transforms, Dataset/DataLoader, model construction, loss and optimizer, gradient updates, validation, tuning, checkpoint saving/loading, inference on an unseen image, and reproducible evaluation/reporting. Introduce each only in its scheduled block. Explain training versus inference and the role of train/eval modes. Explain how the code would change for another public medical image dataset while keeping this experiment reproducible.

Build from beginner understanding to research-level comparisons. Complexity is introduced because it answers a question, not to impress. For each architecture explain its design visually, implement it, check its behavior, then discuss evidence-based tuning. Keep the faithful paper replication separate from teaching baselines and new architectures.

## Blocks and milestones
0. Paper orientation and research question. No download/training.
1. Environment: inspect existing .venv; verify CPU/MPS with a small forward/backward operation; record versions. No reinstall of unrelated environments. Local MPS smoke test passed on 2026-10-25, but recheck in the actual IDE process.
2. Data: download the official PneumoniaMNIST 28-pixel artifact; record version/checksum and source. Explain source population, processing, label mapping and supplied split counts from package metadata. Display labelled TRAIN images and class counts. Basic integrity checks across splits are allowed; reserve test performance and error analysis until final evaluation.
3. Quality/provenance: audit dtype, shapes, range, missing/constant values, duplicate content and cross-split overlap. Explain resolution loss. Do not claim patient independence without supporting identifiers/provenance. Preserve benchmark splits; document overlap rather than silently removing records and still claiming an exact replication.
4. Data flow/OOP: Dataset and DataLoader; show one image then one batch; explain target shape/dtype and normalization. No hidden preprocessing.
5. Teaching baselines: constant/majority baseline, linear pixel classifier, small MLP. Clearly label these as educational additions, not paper models. Same splits; explain what flattening discards about spatial structure.
6. Tiny CNN: convolution, channels, padding, stride, ReLU, pooling, global pooling and classifier. Visualize intermediate maps and dimensions. Distinguish random from trained filters. Do not call an activation a lesion explanation.
7. Training mechanics: one batch → tiny training-subset overfit check → short train/validation run. Explain logits, cross-entropy, backward, optimizer, train/eval modes. No held-out test tuning. Show loss and AUC/accuracy.
8. ResNet: draw identity/projection shortcuts; inspect one block; map the authors' small-image ResNet-18 to the full network. Do not silently replace it with default torchvision ResNet-18.
9. Replication protocol: pin upstream commit/dataset version, reconcile paper versus current repository, resolve open settings in PAPER_MAP.md, freeze configuration and training budget before running. Record all deviations, especially hardware and metric logging.
10. Replication run: follow fixed settings; select checkpoint using validation, not test. Report one run as a single-run reproduction attempt. Match published repeat/aggregation protocol before claiming comparable aggregate results.
11. Extensions: introduce architecture families incrementally, each with a hypothesis and fair experiment budget. See ladder below.
12. Tuning: small predeclared validation-only search, learning curves and error inspection. Freeze candidates before final test evaluation; compare across predefined seeds, report variability and cost.
13. Report: paper result vs replication vs extensions, failures/limitations, saved predictions/configs/checkpoints/environment and measured parameter/runtime costs. AI authorship and source licenses included. No fabricated metrics or automatic GitHub publishing.

## Architecture ladder — not all are needed at once
Constant baseline → linear classifier → MLP → 2-block CNN → deeper CNN → small residual network → authors' ResNet-18 → ResNet-50 / DenseNet-style connectivity → efficient CNN (MobileNet/EfficientNet/ConvNeXt) → compact ViT → selected pretrained vision encoder with a classification head.

Start with the first architectures and the paper target. Later families are optional extensions, not a promise that more complexity improves results. Choose recent models only after checking official sources at that time. Include ResNet-50 for a paper-family comparison; introduce other families when their design teaches a new concept.

Keep two explicit comparison tracks:
A. From scratch at the same 28-pixel input, same splits and comparable tuning budget.
B. Transfer learning/higher resolution, labelled separately with preprocessing, channel conversion, pretrained data and compute differences.
Upsampling 28-pixel inputs to 224 does not restore detail and is different from downloading a MedMNIST+ 224-pixel artifact. Never mix these protocols in a claimed replication.

## How to teach and tune an architecture
Before code: draw blocks; trace [B,C,H,W]; explain learned vs fixed operations, parameter count, spatial receptive field and how predictions are produced. During code: name stored state and the forward data path. After code: inspect shapes, gradients and a small overfit check. Then train and interpret curves.

Tune in this order: data/label correctness → learning rate → training duration → capacity/regularization → augmentation → scheduler/optimizer alternatives. Change one factor initially. Do not tune the frozen paper configuration; use a separate extension config. No arbitrary augmentation of anatomy. Explain why each proposed transform preserves labels. Record fixed split, seeds, preprocessing, pretraining, trainable parameters, hardware, batch size, elapsed time and validation metrics for each run. Device changes can affect timing/numerics; do not promise bitwise equality across MPS/CUDA/CPU.

## Compute and engineering
Prefer local MPS for inspection/small runs; CPU fallback must be visible. Verify actual availability and operators; do not silently enable CPU fallback to make a run appear GPU-only. Keep datasets in ignored data/ and checkpoints in ignored checkpoints/. Use simple scripts/functions plus nn.Module classes; create modules only when reuse justifies them. Save intermediate artifacts before switching machines.

Official Colab CLI: https://github.com/googlecolab/google-colab-cli
Verify installed help/auth/account entitlement first. CLI documentation supports `colab new`, `colab exec -f`, `colab download`, `colab log`, `colab usage`, `colab stop`; use named sessions. These are capability references, not permission to allocate a VM. No SSH workaround is required. Local file imports/dependencies/data are not automatically transferred merely because one script is executed remotely. Stage the project explicitly and verify remote paths. Download and verify results before releasing a runtime. Never put tokens in notebooks, prompts, output or Git.
