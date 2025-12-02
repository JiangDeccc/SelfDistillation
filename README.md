Here's the code and data of the self distillation framework.

## Instructions 

This repository implements a full pipeline for data preprocessing, candidate generation, natural-language prompt construction, GRPO-based reasoning, and joint self-distillation for reranking models. The workflow consists of the following major stages:

### 1. Data Preparation

We first preprocess the raw data into **three separate files**, each serving a distinct purpose for the downstream models:

- **`item.pkl`**  
  A dictionary mapping each `iid` to its attributes:

  ```python
  { iid: { "category": ..., "title": ... } }
  ```

- **`user.txt`**  
  The sequential interaction history for each user (one user per line).  
  This is typically referred to as `user_sequence.txt`.

- **`user-item.pkl`**  
  A nested dictionary storing user–item interactions with ratings:

  ```python
  { uid: { iid: { "rating": ... } } }
  ```

These files act as the unified data source for both Recall and Rerank stages.

---

### 2. Recall Stage (SASRec via ReChorus)

We use **ReChorus** to run **SASRec** and perform the Recall task.  
This step produces the candidate items that will later be reranked.

- Output file: **`candidates.pkl`**  
  The format is:

  ```python
  { uid: { "iid": [candidate_item_ids] } }
  ```

These candidate lists are subsequently transformed into natural-language prompts for the reasoner and the reranker.

---

### 3. Converting Data into Natural-Language Prompts

To enable LLM-based reasoning and reranking, we convert the structured data and candidate lists into textual prompts.

Two scripts are used:

- **`explanation_prompt_generate.py`**  
  Generates the **reasoner** input prompts.

- **`train_prompt_generate.py`**  
  Generates the **teacher/student reranker** input prompts.

The outputs are stored in two folders:

- **`reranking_explanation/`** — prompts for the reasoner  
- **`reranking_prompts/`** — prompts for the reranker models

---

### 4. GRPO Training for the Reasoner

The reasoner is trained using **GRPO** (Generalized Repetitive Preference Optimization).

- Main script: **`explan.py`**  
- Reward function: **`explan_reward.py`**

During this step, the model learns to generate high-quality rationales that improve reranking performance.

---

### 5. Joint Self-Distillation for Rerankers

After preparing all prompts for both **teacher** and **student** rerankers, we run joint training via:

- **`joint_all_adaptive.py`**

This script performs **self-distillation**, where the teacher reranker guides the student through a dynamic curriculum that balances supervised loss and distillation loss.

---

### 6. Evaluation

Finally, to evaluate performance on the test set, run:

- **`test_character.py`**

This script produces the final experimental metrics and evaluation results.



## Parameter Analysis

### Global Hyperparameters

Across all three datasets (MovieLens, Clothing, Electronics), we use the same values for the following global hyperparameters:

- **α_min** = 0.001  
- **α_max** = 0.9  
- **β** = 0.1  
- **c** = 5.5  

These four hyperparameters are **not** tuned per dataset and remain fixed throughout all experiments.

---

### Dataset-Specific Hyperparameters

Three additional hyperparameters are tuned for each dataset:

- **τ (tau)** – temperature coefficient for the reasoner
- **γ (gamma)** – temperature coefficient for the reranker
- **α_base** – base weight for the supervised loss in the dynamic α schedule

The final chosen values are:

| Dataset     | τ    | γ    | α_base |
| ----------- | ---- | ---- | ------ |
| MovieLens   | 2.0  | 1.5  | 0.5    |
| Clothing    | 1.0  | 1.5  | 0.9    |
| Electronics | 2.0  | 1.0  | 0.6    |

We observe that the two temperature coefficients (**τ** and **γ**) have relatively minor impact on performance as long as they lie in the range **[0.5, 2.0]**. In contrast, **α_base** is more sensitive and typically performs best when set between **0.5 and 1.0**.

---

### Sensitivity Analysis on the Clothing Dataset

We further study the effect of **τ** and **α_base** on the Clothing dataset using SelfRR.

Tab.1 Effect of τ (with fixed α_base = 0.9)

| τ    | HR@1   | NDCG@3 | NDCG@5 |
| ---- | ------ | ------ | ------ |
| 0.5  | 0.2110 | 0.3596 | 0.4111 |
| 1.0  | 0.2115 | 0.3605 | 0.4154 |
| 1.5  | 0.2101 | 0.3566 | 0.4099 |
| 2.0  | 0.2087 | 0.3563 | 0.4100 |

Performance is relatively stable when τ varies between 0.5 and 2.0, with a slight peak around **τ = 1.0**.

Tab.2 Effect of α_base (with fixed τ = 1.0)

| α_base | HR@1   | NDCG@3 | NDCG@5 |
| ------ | ------ | ------ | ------ |
| 0.2    | 0.1855 | 0.3402 | 0.3978 |
| 0.5    | 0.2032 | 0.3504 | 0.4030 |
| 0.6    | 0.2004 | 0.3480 | 0.4027 |
| 0.7    | 0.2045 | 0.3613 | 0.4132 |
| 0.8    | 0.2083 | 0.3621 | 0.4149 |
| 0.9    | 0.2115 | 0.3605 | 0.4154 |
| 1.0    | 0.2120 | 0.3582 | 0.4116 |
| 1.5    | 0.2008 | 0.3558 | 0.4061 |

SelfRR typically performs best within 1.5-2.0: values in the range **0.7–0.9** offer a good trade-off, with **α_base ≈ 0.8–0.9** yielding the strongest overall performance on the Clothing dataset.



## Training Resources

We report the computational resources using the **Clothing** dataset as an example. All experiments are conducted on NVIDIA A100-40G GPUs.

**Instruction-Tuning Stage**

For the instruction-tuning stage, a **single A100-40G** GPU is sufficient.

- Dataset: Clothing  
- GPU: 1 × A100-40G  
- Epochs: 3  
- Wall-clock time: ≈ 24.6 hours

**GRPO Stage (Reasoner)**

For the GRPO training of the reasoner, a single A100-40G GPU also works, but using multiple GPUs with `accelerate` can significantly speed up training. In our main experiments, we adopt a **2-GPU** setting.

- Dataset: Clothing  
- GPUs: 2 × A100-40G (data-parallel via `accelerate`)  
- Training time until reward convergence: ≈ 21.6 hours

The **instruction-tuning stage** and the **GRPO stage** are independent and can be run **in parallel** to reduce overall wall-clock time.

**Distillation Stage (Self-Distillation)**

For the joint self-distillation of the teacher and student rerankers, we use **2 × A100-40G** GPUs.

- Dataset: Clothing  
- GPUs: 2 × A100-40G  
- Training epochs: early stopped at 1.5 epochs  
- Wall-clock time: ≈ 15 hours



## Usage

./codes: dynamic self distillation framework (reasoner training & distillation)

./raw_data: original & preprocessed dataset

./reranking_prompts: prompts of the reranking task

./reranking_explanation: reasoner-generated explanation

- https://www.dropbox.com/scl/fo/bpi0tvgcn66jdrdg2aae4/ABEWv3GXQP085Pfe3RLh2ws?rlkey=d4ucbe6u5nvxghwwszgzwkj3j&st=ed2xrcjf&dl=0

