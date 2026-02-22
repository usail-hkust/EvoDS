# EvoDS: Self-Evolving Autonomous Data Science Agent with Capability Learning and Context Management

---

## 📖 Overview

**EvoDS** is a self-evolving autonomous data science agent designed to address two fundamental challenges in LLM-based data science systems:

1. **Reusable Capability Learning** – enabling agents to synthesize, validate, and internalize tool-usage skills from experience.
2. **Adaptive Context Management** – dynamically compressing long-horizon interaction context to maintain reasoning stability under token constraints.

EvoDS formulates autonomous data science as a decision-making process over an evolving action space under bounded context budgets. It integrates:

* **Autonomous Capability Acquisition Mechanism (ACAM)**
  Treats tools as learnable capabilities that can be synthesized and reused.

* **Adaptive Context Compression Strategy (ACCS)**
  Learns when and how to compress interaction history during multi-step reasoning.

* **Multi-Agent Reinforcement Learning Framework**
  Jointly optimizes task execution, capability acquisition, and context regulation.

---

## 📦 Checkpoints

We provide two pretrained checkpoints:

* **`checkpoints/EvoDS-sft-8B`**
  Supervised fine-tuned model based on **Qwen3-8B**.

* **`checkpoints/EvoDS-rl-8B`**
  Further trained from `EvoDS-sft-8B` using multi-agent reinforcement learning.

---

## 🚀 Quick Start

### 🔑 Step 1: Configure API Key

Edit:

```
config/config.yaml
```

Replace:

```
<your openai key>
```

with your OpenAI API key.

---

### ▶️ Step 2: Running the Agent

You can directly launch EvoDS with:

```bash
bash run.sh
```

This script will:

1. Deploy `EvoDS-rl-8B` using **vLLM**
2. Sequentially evaluate the agent on:

   * DABench
   * DA-Code
   * ScienceAgentBench
   * MLE-Dojo

---

### 🖥 Multi-GPU Configuration

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
   start_llm "checkpoints/EvoDS-rl-8B" "evods-rl-8b" 1234 1 "nohup.log"
   ```

   Change `1` to the number of GPUs you are using.

---

## 🏋️ Training

We use **VERL** for both SFT and RL training.

### 🔧 Before Training

Replace the API key in:

```
verl/verl/workers/reward_manager/datascience.py
```

Change:

```
your_api_key
```

to your OpenAI API key.

---

### 📂 Enter Training Directory

```bash
cd verl
```

---

### 🧠 Supervised Fine-Tuning (SFT)

```bash
bash examples/sft/multiturn/run_evods_qwen3_8b_multi_turn.sh 4 checkpoints/evods_sft_8b
```

* `4` indicates the number of GPUs.
* Output checkpoint will be saved to `checkpoints/evods_sft_8b`.

---

### 🔁 Reinforcement Learning (RL)

```bash
bash examples/sglang_multiturn/run_evods_qwen3_8b.sh
```

This stage performs multi-agent reinforcement learning to jointly optimize:

* Task completion quality
* Tool scheduling efficiency
* Context length control

---

## ⚠️ Note on Checkpoints and Benchmarks

Due to storage constraints, the pretrained checkpoints and benchmark datasets are not provided in this repository. They will be publicly released on Hugging Face after the paper is accepted.
