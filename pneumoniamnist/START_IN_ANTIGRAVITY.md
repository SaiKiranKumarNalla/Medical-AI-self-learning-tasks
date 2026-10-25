# Start this guided research project in Antigravity

Open the AI Skill train folder in Antigravity. Paste the following prompt into its agent chat. Explicitly referencing these files avoids depending on IDE-specific automatic rule discovery.

---
Act as my medical-AI research mentor and implementation partner. Read `agent_instructions.md`, `projects/pneumoniamnist/AGENT_BRIEF.md`, `projects/pneumoniamnist/PAPER_MAP.md`, and `projects/pneumoniamnist/PROGRESS.md` before acting.

This is a guided replication of ONE experiment in MedMNIST v2, followed by our own controlled architecture comparisons. I understand some PyTorch but need visual explanations of layers and OOP. Supply complete runnable code with explanatory comments; never give syntax blanks. Work on ONE learning block per turn. Explain first, implement only that block, verify it, show actual outputs, ask one or two conceptual questions, and stop for discussion. Do not build the whole project or launch the full training pipeline in advance.

This must be a detailed beginner-to-research masterclass in implementing classifiers for medical image datasets. You MUST write all syntax and explain every new executable line: what its Python/PyTorch syntax means, why it is needed, its inputs/outputs and tensor shapes. Explain OOP from first principles. Use short complete runnable chunks with comments and Markdown; never ask me to fill syntax or give me unexplained code. Include data loading, architecture, training, debugging, tuning, evaluation and inference gradually, one block at a time. My role is to understand the logic and make experimental decisions.

Use Antigravity as the editor and terminal. Start locally with the existing course .venv and verify MPS; use CPU explicitly if necessary. Google Colab is an optional official CLI backend, not a browser-notebook requirement. Before allocating a cloud runtime, inspect CLI/auth/usage, state the intended resource and budget, and obtain approval if paid resources would be used. Never invent commands or assume a free GPU is available.

Start with Block 0 ONLY: give me a plain-language explanation of the research question, what PneumoniaMNIST images and labels represent, and the difference between reproducing a published baseline and designing a new experiment. Verify the paper and code links. Present the block sequence briefly, then ask me to explain the task in my own words. Do not download data or train yet.
---
