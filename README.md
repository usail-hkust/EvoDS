# EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management

🎉 Accepted by the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)

---

## 📖 Overview

**EvoDS** is a self-evolving autonomous data science agent designed to address two fundamental challenges in LLM-based data science systems:

1. **Reusable Skill Learning** — enabling agents to synthesize, validate, and internalize reusable tool-usage skills from experience.
2. **Adaptive Context Management** — dynamically compressing long-horizon interaction history to maintain stable reasoning under limited context budgets.

EvoDS formulates autonomous data science as a sequential decision-making process over an evolving action space under bounded context constraints. The framework integrates three key components:

### 🔧 Autonomous Skill Acquisition (ASA)

ASA treats tools as learnable capabilities.
The agent can autonomously synthesize, validate, cache, and reuse executable skills during task solving.

### 🧠 Adaptive Context Compression (ACC)

ACC dynamically determines when and how to compress interaction history during long-horizon multi-step reasoning, improving context efficiency and reasoning stability.

### 🔁 Agentic Reinforcement Learning

EvoDS jointly optimizes:

* task completion quality,
* skill acquisition behavior,
* and context regulation policies

through joint reinforcement learning over multiple agent roles.

---

## 📦 Checkpoints

We provide pretrained checkpoints for EvoDS.

### EvoDS Checkpoint

Download the checkpoint from:

* [Hugging Face - EvoDS](https://huggingface.co/yangzhr/EvoDS)

Place the downloaded files under:

```bash
checkpoints/EvoDS
```

---

## 🚀 Quick Start

### 🔑 Step 1: Configure API Key

Edit:

```bash
config/config.yaml
```

Replace:

```yaml
<your_openai_key>
```

with your OpenAI API key.

---

### 📚 Step 2: Prepare Benchmarks

Due to storage constraints, benchmark datasets are not included in this repository.

Please manually download the following benchmarks before evaluation:

* DABench
* DA-Code
* ScienceAgentBench
* MLE-Dojo

After downloading, place the datasets in the corresponding benchmark directories specified in the configuration files.

---

### ▶️ Step 3: Launch EvoDS

Run:

```bash
bash run.sh
```

This script will:

1. Deploy `EvoDS` using **vLLM**
2. Sequentially evaluate the agent on:

* DABench
* DA-Code
* ScienceAgentBench
* MLE-Dojo

---

## 🖥 Multi-GPU Configuration

By default, `run.sh` uses a single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

To use multiple GPUs:

1. Modify:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,...
```

2. Update the GPU count in:

```bash
start_llm "checkpoints/EvoDS" "EvoDS" 1234 1 "nohup.log"
```

Replace `1` with the number of GPUs you intend to use.

---

## 🏋️ Training

We use **VERL** for both supervised fine-tuning (SFT) and reinforcement learning (RL).

---

### 🔧 Before Training

Edit:

```bash
verl/verl/workers/reward_manager/datascience.py
```

Replace:

```python
your_api_key
```

with your OpenAI API key.

---

### 📂 Enter the Training Directory

```bash
cd verl
```

Download **Qwen3-8B** and place it under:

```bash
Qwen/Qwen3-8B
```

---

## 🧠 Supervised Fine-Tuning (SFT)

Run:

```bash
bash examples/sft/multiturn/run_evods_qwen3_8b_multi_turn.sh 4 checkpoints/evods_sft_8b
```

Arguments:

* `4` — number of GPUs
* `checkpoints/evods_sft_8b` — output checkpoint directory

The trained checkpoint will be saved to:

```bash
checkpoints/evods_sft_8b
```

---

## 🔁 Reinforcement Learning (RL)

Run:

```bash
bash examples/sglang_multiturn/run_evods_qwen3_8b.sh
```

This stage performs multi-agent reinforcement learning to jointly optimize:

* task completion quality
* tool scheduling efficiency
* context length control
* skill acquisition behavior

---
