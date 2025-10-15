import re
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
)

from utils import load_user_data, load_pickle, generate_ranking_texts_w_explan
import os

input_path = '../raw_data'
dataset = 'electronics'
user_data = load_user_data(os.path.join(input_path, dataset, 'user.txt'))
item_meta = load_pickle(os.path.join(input_path, dataset, 'item.pkl'))
user_item_interaction = load_pickle(os.path.join(input_path, dataset, 'user-item.pkl'))
candidates = load_pickle(os.path.join(input_path, dataset, 'candidates.pkl'))
# 将candidates中的item_id转换为和user_data中一致的type
for k, v in candidates.items():
    sample_list = candidates[k]
    for item in sample_list:
        sample_value = item
        break
    break
if isinstance(sample_value, int):
    target_type = int
elif isinstance(sample_value, str):
    target_type = str
else:
    raise ValueError(f"Unsupported item_id type: {type(sample_value)}")

def cal_llama_to_get_solution(explanation, user_id, item_id, model_name):
    system_prompt = "You are a helpful assistant. Now you are to perform a recommendation task. **Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words."
    total_prompt = generate_ranking_texts_w_explan(
        user_data, item_meta, user_item_interaction, candidates[int(user_id)][target_type(item_id)], 
        user_id, item_id, explanation, dataset_name = dataset
    )
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": total_prompt},
    ]
    )
    # print('one reward calculation:', completion.choices[0].message.content)
    return completion.choices[0].message.content

def cal_llama_to_verify_format(explanation, target, model_name):
    prompt = ( # f"Check whether the given title appears in any form within the explanation text. This includes exact matches, partial matches, or slight variations. Return only True if it appears, otherwise return False. Do not output anything else."
        f"Check whether the given title appears **explicitly** in any form within the explanation text. "
        f"This includes exact matches, or slightly paraphrased versions that clearly refer to the **same item**, "
        f"but not other items from the same series or franchise.\n"    
        f"Do NOT return True for related but different items (e.g., different movies in the same series).\n"
        f"Return only True if the exact item is mentioned or clearly paraphrased. Otherwise return False. "
        f"Do not output anything else.\n"
        f"Here is the input:\n"
        f"Title: {target}\n"
        f"Explanation: {explanation}")
    completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "Output True or False ONLY."},
        {"role": "user", "content": prompt},
    ]
    )
    format_correct = completion.choices[0].message.content
    # print('format check input:', prompt)
    # print('format check:', format_correct)
    if format_correct == 'True':
        return True
    else:
        return False

def strict_match(answer, ground_truth):
    answer_clean = re.sub(r'[^a-zA-Z0-9]', '', answer).lower()
    ground_truth_clean = re.sub(r'[^a-zA-Z0-9]', '', ground_truth).lower()
    if answer_clean == ground_truth_clean:
        return True
    else:
        return False
    
def replace_target(solution_str, target_str):
    pattern = re.compile(re.escape(target_str), re.I)
    norm = pattern.sub('the new item', solution_str)
    # if '(' in target_str:
    #     target_process = target_str[:target_str.rfind('(')].strip()
    #     pattern = re.compile(re.escape(target_process), re.I)
    #     norm = pattern.sub('the new item', norm)
    # if ',' in target_str:
    #     target_process = target_str[:target_str.rfind(',')].strip()
    #     pattern = re.compile(re.escape(target_process), re.I)
    #     norm = pattern.sub('the new item', norm)            
    return norm


def compute_score(solution_str, user_id, item_id, model='llama'):
    if '### Reason ###' not in solution_str:
        return 0
    else:
        # processed_solution_str = solution_str
        cand_set = list(candidates[int(user_id)][target_type(item_id)])
        # should not be a candidate set withoud item_id
        if target_type(item_id) not in cand_set:
            return 0.0
        ground_truth = cand_set.index(target_type(item_id))
        gt_title = item_meta[int(item_id)]['title']
        processed_solution_str = replace_target(solution_str, gt_title)
        # print(user_id, item_id)
        model_name = model
        rank_result = cal_llama_to_get_solution(processed_solution_str, user_id, item_id, model_name)
        # print(f"Rank result: {rank_result}, Ground truth index: {chr(ord('A') + ground_truth)}")
        if chr(ord('A') + ground_truth) == rank_result.strip():
            return 1.0
        else:
            return 0.0
            
if __name__ == "__main__":
    # Example usage
    solution_str = "The user prefers action movies. ### Reason ### The user has watched 'Die Hard' and 'Mad Max'."
    ground_truth = "Die Hard"
    # ranking_prompt = "Please select the top 1 movie based on the user's viewing history."
    cal_llama_to_verify_format(solution_str, ground_truth, "../LLaMA-Factory/output/qwen2_lora_ml1m_rerank_character")

    # score = compute_score(solution_str, '5010', '3366')
    # print(f"Score: {score}")

