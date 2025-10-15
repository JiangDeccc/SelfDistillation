import json
import torch

import logging
from explan_rewards import compute_score, replace_target

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # Set to your desired GPU device
# print(os.environ["CUDA_VISIBLE_DEVICES"])

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# # ml-1m
# SYSTEM_PROMPT = """You are a helpful assistant. Given a user’s past movie-watching history and a new movie recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously interacted movies; there is no need to predict the rating of the new item. 

# Please follow the format below: 
# ### Reason ###
# Your Explanation Here

# **Important:** 
# - In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
# - Always refer to it only as "the new item". 
# - Do not output anything else outside this format."""
# amazon
SYSTEM_PROMPT = """You are a helpful assistant. Given a user’s past purchase history and a new product recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously interacted products; there is no need to predict the rating of the new item. 

Please follow the format below: 
### Reason ###
Your Explanation Here

**Important:** 
- In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
- Always refer to it only as "the new item". 
- Do not output anything else outside this format."""

def get_my_dataset(file_path):
    """
    Loads and prepares the custom JSON dataset with prompts, ranking, and answer.
    
    Args:
        file_path (str): Path to the JSON file.
        tokenizer: Tokenizer object with `apply_chat_template`.
        system_prompt (str): System prompt to prepend.
    
    Returns:
        Dataset: A HuggingFace Dataset with processed examples.
    """
    data = load_json(file_path)

    def format_example(entry):
        # Prepare input as a chat template
        input_text = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry['prompt']}
        ]
        return {
            'prompt': input_text,
            'user_id': entry['user_id'],
            'item_id': entry['item_id'],
        }

    processed_data = [format_example(entry) for entry in data]
    return Dataset.from_list(processed_data)

mydataset = get_my_dataset(
    "../reranking_explanation/electronics/reranking_explanation_his50_2-puser.json",
)

print(f"Dataset loaded with {len(mydataset)} examples.")


# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

model_path = "../../LLM/Meta-Llama-3.1-8B-Instruct"
output_path = "outputs/GRPO_explan_electronics_llama_train_reward0"

# Reward functions
def correctness_reward_func(completions, user_id, item_id, **kwargs) -> list[float]:
    rewards_list = []
    for i in range(len(completions)):
        # logger.info(f"Completion: {completions[i]}")
        # logger.info(f"Ranking: {ranking[i]}")
        # logger.info(f"Answer: {answer[i]}")
        rewards_list.append(compute_score(completions[i][0]['content'], user_id[i], item_id[i], model_path))
    print(f"Completions sample: {completions[0][0]['content']}")
    print(f"Rewards: {rewards_list}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return rewards_list

MODEL_NAME = os.getenv("MODEL_NAME", model_path)
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set.")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", output_path)
if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR environment variable is not set.")

RUN_NAME = os.getenv("RUN_NAME", "default-GRPO-explan-train")
if not RUN_NAME:
    raise ValueError("RUN_NAME environment variable is not set.")

# Model setup
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # device_map="auto"
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# PEFT config (optional)
peft_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)


# Training config
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name=RUN_NAME,
    learning_rate=5e-6,
    # adam_beta1=0.9,
    # adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,  # Increased from 1
    gradient_accumulation_steps=16,  # Reduced from 4
    num_generations=8,  # Reduced from 16
    max_prompt_length=2048,
    max_completion_length=500,
    num_train_epochs=3,
    save_strategy="steps",  # Save by steps
    save_steps=100,        # Save every 30 steps
    save_total_limit=100,
    max_grad_norm=0.1,
    report_to=['tensorboard'],
    # use_vllm=True,
    # vllm_mode="colocate",
    # vllm_tensor_parallel_size=1,
    # vllm_gpu_memory_utilization=0.5,
    # log_on_each_node=False,
)

# Trainer setup
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=mydataset,
    peft_config=peft_config, 
)

# print(trainer.processing_class)

# Train the model
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    raise