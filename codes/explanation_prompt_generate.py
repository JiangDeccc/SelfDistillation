from utils import load_user_data, load_pickle, generate_explanation_texts
import os
import random
import json
from tqdm import tqdm

if __name__ == "__main__":
    dataset = 'electronics'
    input_path = './raw_data'
    user_data = load_user_data(os.path.join(input_path, dataset, 'user.txt'))
    item_meta = load_pickle(os.path.join(input_path, dataset, 'item.pkl'))
    user_item_interaction = load_pickle(os.path.join(input_path, dataset, 'user-item.pkl'))
    candidates = load_pickle(os.path.join(input_path, dataset, 'candidates.pkl'))
    candidate_num = 20

    num_history = 50
    train_begin_data_len = 2
    train_end_data_len = 500

    prompts = []
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
                prompt = generate_explanation_texts(user_data, item_meta, user_item_interaction, user, target_item, rating2str=False, dataset_name=dataset)
                # # generate titles
                # prompt = generate_ranking_texts_title(user_data, item_meta, cand_set, user, target_item)
            
                prompts.append({
                    'user_id': user,
                    'item_id': target_item,
                    'prompt': prompt
                })
            cnt += 1
            ptr -= 1

    # 同时shuffle prompts and results
    print("Shuffling prompts and results...")
    random.seed(42)
    random.shuffle(prompts)
    print(len(prompts))
    # with open(f'./reranking_explanation/{dataset}/reranking_explanation_his50_2-puser.json', 'w', encoding='utf-8') as f_out:
    #     json.dump(prompts, f_out, indent=4, ensure_ascii=False)
    # print(prompts[0])
    
    print("Done! The training prompts have been generated in 'explanation.json'.")
