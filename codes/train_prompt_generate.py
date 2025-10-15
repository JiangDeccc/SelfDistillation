from utils import load_json, load_user_data, load_pickle, generate_ranking_texts, generate_ranking_texts_w_explan, generate_ranking_texts_title
import os
import random
import json
from tqdm import tqdm
import re

WITH_EXPLAN = True  # 是否使用解释文本

def replace_target(solution_str, target_str):
    pattern = re.compile(re.escape(target_str), re.I)
    norm = pattern.sub('the new movie', solution_str)
    if '(' in target_str:
        target_process = target_str[:target_str.rfind('(')].strip()
        pattern = re.compile(re.escape(target_process), re.I)
        norm = pattern.sub('the new movie', norm)
    if ',' in target_str:
        target_process = target_str[:target_str.rfind(',')].strip()
        pattern = re.compile(re.escape(target_process), re.I)
        norm = pattern.sub('the new movie', norm)            
    return norm

# 主流程
if __name__ == "__main__":
    dataset = 'clothing'
    input_path = './raw_data'
    user_data = load_user_data(os.path.join(input_path, dataset, 'user.txt'))
    item_meta = load_pickle(os.path.join(input_path, dataset, 'item.pkl'))
    if dataset in ['lfm2b']:
        user_item_interaction = {}
    else:
        user_item_interaction = load_pickle(os.path.join(input_path, dataset, 'user-item.pkl'))
    candidates = load_pickle(os.path.join(input_path, dataset, 'candidates.pkl'))
    candidate_num = 20
    if WITH_EXPLAN:
        explan_file = load_json(f'./reranking_explanation/{dataset}/reranking_explanation_his50_2puser_with_gpt-4o-mini-2024-07-18_results.json')
        explan_dict = dict()
        for entry in explan_file:
            user_id = entry['user_id']
            item_id = entry['item_id']
            if user_id not in explan_dict:
                explan_dict[user_id] = dict()
            explan_dict[user_id][item_id] = entry['result']

    num_history = 50
    train_begin_data_len = 0
    train_end_data_len = 2

    prompts = []
    results = []
    for user, interactions in tqdm(user_data.items()):
        cnt = 0
        ptr = len(interactions) - 1
        while cnt < train_end_data_len and ptr > 1:
            target_item = interactions[ptr-1]
            cand_set = list(candidates[int(user)][int(target_item)][:candidate_num])
            if int(target_item) not in cand_set:
                ptr -= 1
                continue
            if cnt >= train_begin_data_len:
                label = cand_set.index(int(target_item))
                # generate character identifier
                if WITH_EXPLAN:
                    try:
                        generated_explan =  explan_dict[user][target_item]
                        target_item_title = item_meta[int(target_item)]['title']
                        generated_explan = replace_target(generated_explan, target_item_title)
                        prompt = generate_ranking_texts_w_explan(
                            user_data, item_meta, user_item_interaction, cand_set, user, target_item, generated_explan, rating_2_str=False, dataset_name=dataset)
                    except KeyError:
                        print(f"Warning: No explanation found for user {user} and item {target_item}. Skipping this entry.")
                        ptr -= 1
                        continue
                else:
                    prompt = generate_ranking_texts(user_data, item_meta, user_item_interaction, cand_set, user, target_item, rating_2_str=False, dataset_name=dataset)
                prompts.append({
                    'user_id': user,
                    'item_id': target_item,
                    'label': label,
                    # # generate titles
                    # 'label': item_meta[int(target_item)]['title'],
                    'prompt': prompt
                })
                results.append({
                    'instruction': "Now you are to perform a recommendation task. **Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words.",
                    'input': prompt,
                    # generate titles
                    # 'output': item_meta[int(target_item)]['title'],
                    'output': chr(ord('A') + label),
                    'user_id': user,
                    'item_id': target_item,
                })
                
            cnt += 1
            ptr -= 1

    # 同时shuffle prompts and results
    # assert len(prompts) == len(results)
    print("Shuffling prompts and results...")
    random.seed(42)
    random.shuffle(results)
    random.seed(42)
    random.shuffle(prompts)
    print(len(prompts))
    with open(f'./reranking_prompts/{dataset}/reranking_explan_gpt_his50_2puser_mask.json', 'w', encoding='utf-8') as f_out:
        json.dump(prompts, f_out, indent=4, ensure_ascii=False)
    print(prompts[0])
    # with open(f"./reranking_prompts/{dataset}/alpaca_reranking_finetune_his50_2puser.json", "w", encoding="utf-8") as file:
    #     for message in results:
    #         file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    print("Done! The training prompts have been generated in 'finetune.json'.")
