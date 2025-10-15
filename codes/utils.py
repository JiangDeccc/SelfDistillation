import json
import pickle

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 读取 user.txt 文件，获取用户历史交互数据
def load_user_data(file_path):
    user_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user_id = parts[0]
            item_ids = parts[1:]
            user_data[user_id] = item_ids
    return user_data

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        load_dict = pickle.load(f)
    return load_dict

def load_user_history(user_data: dict, user_id: str, item_id: str, max_history_length=50):
    history = user_data[user_id]
    # item_id在history中的位置
    if item_id in history:
        idx = history.index(item_id)
        # 截取最近的max_history_length个历史记录
        start_idx = max(0, idx - max_history_length + 1)
        return history[start_idx:idx]
    else:
        raise ValueError(f"Item ID {item_id} not found in user history for user {user_id}.")

def generate_ranking_texts(user_data, item_meta, interactions, candidates, user_id, item_id, max_history_length=50, rating_2_str=False, dataset_name="ml-1m"):
    history = load_user_history(user_data, user_id, item_id, max_history_length)
    # cands = candidates[int(user_id)][item_id][:candidate_num]
    def generate_his_item(item_id):
        if dataset_name == "ml-1m":
            rating_2_str_dict = {
                1: "Hated",
                2: "Disliked",
                3: "Neutral",
                4: "Liked",
                5: "Loved",
            }
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            rating = interactions[user_id][item_id]['rating']
            # return f"{title} : Categories({categories})"
            if rating_2_str:
                return f"{title} : Categories({categories}); Rating({rating_2_str_dict.get(rating, 'Unknown')})"
            else:
                return f"{title} : Categories({categories}); Rating({rating})"
        elif dataset_name == "toys" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "home" or dataset_name == "CDs" or dataset_name == "sports":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category'][0]
            rating = interactions[str(user_id)][str(item_id)]['rating']
            summary = interactions[str(user_id)][str(item_id)]['summary']
            return f"{title} : Categories({categories}); Rating({rating}); User Review({summary})"
        elif dataset_name == 'lfm2b':
            title = item_meta[int(item_id)]['track_name']
            # album = item_meta[int(item_id)]['album_name']
            # artist = item_meta[int(item_id)]['artist_name']
            categories = item_meta[int(item_id)]['tags']
            categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            categories = [cat for cat in categories if cat[1] >= 5]
            return f"{title} : Genres({categories})"
            # return f"{title} : Artist({artist}); Album({album}); Genres({categories})"
        elif dataset_name == 'yelp':
            title = item_meta[int(item_id)]['title']
            category = item_meta[int(item_id)]['category']
            # item_stars = item_meta[int(item_id)]['stars']
            # city = item_meta[int(item_id)]['city']
            rating = interactions[str(user_id)][str(item_id)]['rating']
            # summary = interactions[str(user_id)][str(item_id)]['review']
            # return f"{title} : User Rating({rating})"
            return f"{title} : Categories({category}); User Rating({rating})"
            # return f"{title} : Categories({category}); Average Historical Stars({item_stars}); User Rating({rating})"
            # return f"{title} : Categories({category}); Average Historical Stars({item_stars}); City({city}); User Rating({rating}))"
        else:
            raise NotImplementedError("Invalid dataset name.")

    def generate_cand_item(item_id):
        if dataset_name == "ml-1m":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            return f"{title} : Categories({categories})"
        elif dataset_name == "toys" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "home" or dataset_name == "CDs" or dataset_name == "sports":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category'][0]
            return f"{title} : Categories({categories})"
        elif dataset_name == 'lfm2b':
            title = item_meta[int(item_id)]['track_name']
            # album = item_meta[int(item_id)]['album_name']
            # artist = item_meta[int(item_id)]['artist_name']
            categories = item_meta[int(item_id)]['tags']
            categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            categories = [cat for cat in categories if cat[1] >= 5]
            return f"{title} : Genres({categories})"
            # return f"{title} : Artist({artist}); Album({album}); Genres({categories})"
        elif dataset_name == 'yelp':
            title = item_meta[int(item_id)]['title']
            category = item_meta[int(item_id)]['category']
            # item_stars = item_meta[int(item_id)]['stars']
            # city = item_meta[int(item_id)]['city']
            # return f"{title}"
            return f"{title} : Categories({category})"
            # return f"{title} : Categories({category}); Average Historical Stars({item_stars})"
            # return f"{title} : Categories({category}); Average Historical Stars({item_stars}); City({city})"
        else:
            raise NotImplementedError("Invalid dataset name.")

    candidate_str = [f"{chr(65 + i)}. {generate_cand_item(s)}" for i, s in enumerate(candidates)]
    candidate_str = '\n'.join(candidate_str)        
    history_str = [generate_his_item(s) for s in history]
    history_str = '\n'.join(history_str)
    
    if dataset_name == "ml-1m":
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order, recommend an item from the candidate pool. "
            # f"Each item in the user history and candidate pool has its categories after the colon (:). "
            f"Each item in the user history has its categories and the user's rating after the colon (:); each item in the candidate pool has its category after the colon (:). "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    elif dataset_name == "toys" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "home" or dataset_name == "CDs" or dataset_name == "sports": 
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order, recommend an item from the candidate pool. "
            # f"Each item in the user history and candidate pool has its categories after the colon (:). "
            f"Each item in the user history has its categories, the user's rating, and a brief review after the colon (:); each item in the candidate pool has its category after the colon (:). "
            f"Ratings range from 1 to 5, where 5 indicates strong preference and 1 indicates strong dislike. "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    elif dataset_name == 'lfm2b':
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order, recommend an item from the candidate pool. "
            # f"Each item in the user history and candidate pool has its categories after the colon (:). "
            f"Each item in the user history and the candidate pool has its genres after the colon (:). "
            # f"Each item in the user history and the candidate pool has its artist, album, and genres after the colon (:). "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    elif dataset_name == 'yelp':
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order, recommend an item from the candidate pool. "
            # f"Each item in the user history has the user's rating after the colon (:). "
            f"Each item in the user history has its categories and the user's rating after the colon (:); each item in the candidate pool has its categories after the colon (:). "
            # f"Each item in the user history has its categories, average historical stars and the user's rating after the colon (:); each item in the candidate pool has its categories and average historical stars after the colon (:). "
            # f"Each item in the user history has its categories, average historical stars, city and the user's rating after the colon (:); each item in the candidate pool has its categories, average historical stars and city after the colon (:). "
            f"Ratings range from 1 to 5, where 5 indicates strong preference and 1 indicates strong dislike. "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    else:
        raise NotImplementedError("Invalid dataset name.")
    return prompt

def generate_ranking_texts_w_explan(user_data, item_meta, interactions, candidates, user_id, item_id, preference, max_history_length=50, rating_2_str=False, dataset_name="ml-1m"):
    history = load_user_history(user_data, user_id, item_id, max_history_length)
    # cands = candidates[int(user_id)][item_id][:candidate_num]
    def generate_his_item(item_id):
        if dataset_name == "ml-1m":
            rating_2_str_dict = {
                1: "Hated",
                2: "Disliked",
                3: "Neutral",
                4: "Liked",
                5: "Loved",
            }
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            rating = interactions[user_id][item_id]['rating']
            # return f"{title} : Categories({categories})"
            if rating_2_str:
                return f"{title} : Categories({categories}); Rating({rating_2_str_dict.get(rating, 'Unknown')})"
            else:
                return f"{title} : Categories({categories}); Rating({rating})"
        elif dataset_name == "office" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "toys":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category'][0]
            rating = interactions[str(user_id)][str(item_id)]['rating']
            summary = interactions[str(user_id)][str(item_id)]['summary']
            return f"{title} : Categories({categories}); Rating({rating}); User Review({summary})"
        else:
            raise NotImplementedError("Invalid dataset name.")

    def generate_cand_item(item_id):
        if dataset_name == "ml-1m":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            return f"{title} : Categories({categories})"
        elif dataset_name == "office" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "toys":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category'][0]
            return f"{title} : Categories({categories})"
        else:
            raise NotImplementedError("Invalid dataset name.")

    candidate_str = [f"{chr(65 + i)}. {generate_cand_item(s)}" for i, s in enumerate(candidates)]
    candidate_str = '\n'.join(candidate_str)        
    history_str = [generate_his_item(s) for s in history]
    history_str = '\n'.join(history_str)
    preference_str = preference.replace('### Reason ###', '').strip()
    
    if dataset_name == "ml-1m":
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order and a brief analysis of the user's preferences, recommend an item from the candidate pool. "
            # f"Each item in the user history and candidate pool has its categories after the colon (:). "
            f"Each item in the user history has its categories and the user's rating after the colon (:); each item in the candidate pool has its category after the colon (:). "
            f"Ratings range from 1 to 5, where 5 indicates strong preference and 1 indicates strong dislike. "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**User preferences analysis:**\n"
            f"{preference_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    elif dataset_name == "office" or dataset_name == "electronics" or dataset_name == "games" or dataset_name == "clothing" or dataset_name == "toys": 
        prompt = (
            f"### Instruction\n"
            f"Given user history in chronological order and a brief analysis of the user's preferences, recommend an item from the candidate pool. "
            # f"Each item in the user history and candidate pool has its categories after the colon (:). "
            f"Each item in the user history has its categories, the user's rating, and a brief review after the colon (:); each item in the candidate pool has its category after the colon (:). "
            f"Ratings range from 1 to 5, where 5 indicates strong preference and 1 indicates strong dislike. "
            # f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
            # f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
            f"**Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"### Input\n"
            f"**User history:**\n"
            f"{history_str}\n"
            f"**User preferences analysis:**\n"
            f"{preference_str}\n"
            f"**Candidate pool:**\n"
            f"{candidate_str}\n\n"
        )
    else:
        raise NotImplementedError("Invalid dataset name.")

    return prompt

def generate_history_items_prompt(meta, item_type_str, user_data = None, prompt_type=1, rating2str=False):
    """
    根据用户历史的10个 item，构造历史 item 的 prompt 文本。
    
    :param history_items: 这次要放入历史的 item 列表（长度 10 ）
    :param item_meta_dict: item.pkl 的加载结果 (item_id -> meta)
    """
    if prompt_type == 1:
        title = meta.get("title", "N/A")
        categories = meta.get("category", [])
        rating = user_data.get("rating", "N/A")
    
        if rating2str:
            rating_2_str_dict = {
                1: "Hated",
                2: "Disliked",
                3: "Neutral",
                4: "Liked",
                5: "Loved",
            }
            single_item_str = (
                f"{item_type_str.capitalize()} Title: {title}\n"
                f"Categories: {categories}\n"
                f"User Rating: {rating_2_str_dict[rating]}\n"
            )
        else:
            single_item_str = (
                f"{item_type_str.capitalize()} Title: {title}\n"
                f"Categories: {categories}\n"
                f"User Rating: {rating}\n"
            )
    elif prompt_type == 2:
        title = meta.get("title", "N/A")
        categories = meta.get("category", [[]])[0]
        rating = user_data.get("rating", "N/A")
        summary = user_data.get("summary", "N/A")
        
        single_item_str = (
            f"{item_type_str.capitalize()} Title: {title}\n"
            f"Categories: {categories}\n"
            f"User Rating: {rating}\n"
            f"User Review: {summary}\n"
        )
    elif prompt_type == 3:
        title = meta.get('title', 'N/A')
        category = meta.get('category', 'N/A')
        rating = user_data.get('rating', 'N/A')
        single_item_str = (
            f"{item_type_str.capitalize()} Title: {title}\n"
            f"Categories: {category}\n"
            f"User Rating: {rating}\n"
        )
    else:
        raise NotImplementedError("Invalid prompt type.")
    return single_item_str

def generate_new_item_prompt(new_meta, item_type_str, new_user_data = None, prompt_type=1):
    if prompt_type == 1 or prompt_type == 2:
        new_title = new_meta.get("title", "N/A")
        new_categories = new_meta.get("category", [[]])[0] if prompt_type == 2 else new_meta.get("category", [])
        
        new_item_block = (
            "### New Item Information: ###\n"
            f"New {item_type_str.capitalize()}\n"
            f"{item_type_str.capitalize()} Title: {new_title}\n"
            f"Categories: {new_categories}\n"
        )
    elif prompt_type == 3:
        new_title = new_meta.get('title', 'N/A')
        new_category = new_meta.get('category', 'N/A')
        
        new_item_block = (
            "### New Item Information: ###\n"
            f"New {item_type_str.capitalize()}\n"
            f"{item_type_str.capitalize()} Title: {new_title}\n"
            f"Categories: {new_category}\n"
        )
    else:
        raise NotImplementedError("Invalid prompt type.")
    return new_item_block

def generate_explanation_texts(user_data, item_meta, interactions, user_id, item_id, max_history_length=50, rating2str=False, dataset_name='ml-1m'):
    
    history = load_user_history(user_data, user_id, item_id, max_history_length)
    if dataset_name == 'ml-1m':
        item_type_str = "movie"
        prompt_type = 1
    elif dataset_name in ['office', 'electronics', 'games', 'clothing', 'toys']:
        item_type_str = "product"
        prompt_type = 2
    elif dataset_name == 'yelp':
        item_type_str = "business"
        prompt_type = 3
    else:
        raise NotImplementedError("Invalid dataset name.")
    
    history_str_list = []
    for item in history:
        single_item_str = generate_history_items_prompt(item_meta[int(item)], item_type_str, interactions[user_id][item], prompt_type=prompt_type, rating2str=rating2str)
        history_str_list.append(single_item_str)
    history_block = "### Past User History: ###\n" + "\n".join(history_str_list)
    
    # 构造 new item 信息
    new_meta = item_meta.get(int(item_id), {})
    new_item_block = generate_new_item_prompt(new_meta, item_type_str, interactions[user_id][item], prompt_type=prompt_type)

    if dataset_name == 'ml-1m':
        if rating2str:
            prompt_text = f"""Here is information about a user and a new movie being recommended to the user. For the user, we have their past movie-watching history and corresponding ratings. For the new item being recommended, we have the item information.

{history_block}
{new_item_block}
######
Given the user’s past movie-watching history and the new movie information, what information can you infer about the user’s preferences and the reason for their next choice of this movie? 
Your reasoning explanation should be based on any commonalities among the user's previously rated movies and inferred tastes or preferences. You do not need to analyze how the user would rate the new item. Just explain why the user is likely to choose it, based on preference patterns.

Please follow the format below:

### Reason ###
Write your reasoning explanation here. You can have line breaks. Note that do not explicitly mention the title of the new item in your reasoning explanation.
"""    
        else:
            prompt_text = f"""Here is information about a user and a new movie being recommended to the user. For the user, we have their past movie-watching history and corresponding ratings. User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item information.

{history_block}
{new_item_block}
######
Given the user’s past movie-watching history and the new movie information, what information can you infer about the user’s preferences and the reason for their next choice of this movie? 
Your reasoning explanation should be based on any commonalities among the user's previously rated movies and inferred tastes or preferences. You do not need to analyze how the user would rate the new item. Just explain why the user is likely to choose it, based on preference patterns.

Please follow the format below:

### Reason ###
Write your reasoning explanation here. You can have line breaks. Note that do not explicitly mention the title of the new item in your reasoning explanation.
"""
    elif dataset_name in ['office', 'electronics', 'games', 'clothing', 'toys']:
        prompt_text = f"""Here is information about a user and a new product being recommended to the user. For the user, we have their past purchase history and corresponding ratings. User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item information.

{history_block}
{new_item_block}
######
Given the user’s past purchase history and the new product information, what information can you infer about the user’s preferences and the reason for their next choice of this product? 
Your reasoning explanation should be based on any commonalities among the user's previously rated products and inferred tastes or preferences. You do not need to analyze how the user would rate the new item. Just explain why the user is likely to choose it, based on preference patterns.

Please follow the format below:

### Reason ###
Write your reasoning explanation here. You can have line breaks. Note that do not explicitly mention the title of the new item in your reasoning explanation.
"""
    elif dataset_name == 'yelp':
        prompt_text = f"""Here is information about a user and a new merchant being recommended to the user. For the user, we have their past merchant visit history and corresponding ratings. User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new merchant being recommended, we have the merchant information.

{history_block}
{new_item_block}
######
Given the user’s past merchant visit history and the new merchant information, what information can you infer about the user’s preferences and the reason for their next choice of this merchant? 
Your reasoning explanation should be based on any commonalities among the user's previously rated merchants and inferred tastes or preferences. You do not need to analyze how the user would rate the new merchant. Just explain why the user is likely to visit it, based on preference patterns.

Please follow the format below:

### Reason ###
Write your reasoning explanation here. You can have line breaks. Note that do not explicitly mention the name of the new merchant in your reasoning explanation.
"""  
    else:
        raise NotImplementedError("Invalid dataset name.")
        
    return prompt_text


def generate_recsaver_texts(user_data, item_meta, interactions, candidates, user_id, item_id, max_history_length=50, dataset='ml-1m'):

    history = load_user_history(user_data, user_id, item_id, max_history_length)
    if dataset == 'ml-1m':
        item_type_str = "movie"
        prompt_type = 1
    elif dataset in ['office', 'electronics', 'games', 'clothing']:
        item_type_str = "product"
        prompt_type = 2
    elif dataset == 'lfm2b':
        item_type_str = "track"
        prompt_type = 3
    else:
        raise NotImplementedError("Invalid dataset name.")
    
    history_str_list = []
    for item in history:
        single_item_str = generate_history_items_prompt(item_meta[int(item)], item_type_str, interactions[user_id][item], prompt_type=prompt_type)
        history_str_list.append(single_item_str)
    history_block = "### Past User History: ###\n" + "\n".join(history_str_list)

    def generate_cand_item(item_id):
        if prompt_type == 1:
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            return (            
                f"{item_type_str.capitalize()} Title: {title}\n"
                f"Categories: {categories}\n" 
                )
        elif prompt_type == 2:
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category'][0]
            return (            
                f"{item_type_str.capitalize()} Title: {title}\n"
                f"Categories: {categories}\n" 
                )
        elif prompt_type == 3:
            title = item_meta[int(item_id)]['track_name']
            album = item_meta[int(item_id)]['album_name']
            artist = item_meta[int(item_id)]['artist_name']
            categories = item_meta[int(item_id)]['tags']
            categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            categories = [cat for cat in categories if cat[1] >= 5]
            return (            
                f"{item_type_str.capitalize()} Title: {title}\n"
                f"Artist: {artist}\n"
                f"Album: {album}\n"
                f"Genres: {categories}\n" 
                )
        else:
            raise NotImplementedError("Invalid prompt type.")
    candidate_str = [f"{chr(65 + i)}. {generate_cand_item(s)}" for i, s in enumerate(candidates)]
    candidate_block = "### Candidate Pool: ###\n" + "\n".join(candidate_str)

    if prompt_type == 1:
        prompt = (
            f"Here is information about a user and a set of candidate movies being recommended to the user. "
            f"For the user, we have the user’s movie watch history in chronological order along with the corresponding ratings. User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. " 
            f"For each candidate movie, we provide its title and categories.\n"  
            f"Based on the user’s watch history and ratings, analyze the user’s movie preferences and rank the candidates. Select and output the top 1 movie that the user is most likely to watch next.\n"
            f"{str(history_block)}\n"  
            f"{str(candidate_block)}\n"
            # f"The user has watched movies (from the earliest to the latest):\n {str(history_list)}. "
            # f"Based on this watch history, analyze the user's movie preferences and select the top 1 movie from the candidates that the user is most likely to watch next. "
            # f"The candidates are: {str(candidate_titles)}.\n"
            f"First, provide a reasoning analysis of the user's preference based on their watch history. Your reasoning explanation should be based on any commonalities between the user’s watch history and the inferred user tastes or preferences, to support the reranking decision.\n"
            f"Then, select the top 1 movie from the candidates that the user is most likely to watch next. **Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"Please strictly follow this format for your response:\n"
            f"### Reasoning: ###\n<analysis of the user's preference and why they are likely to watch a particular type of movie>\n"
            f"### Selected Movie: ###\n<the index letter of the candidate item>\n"
        )
    elif prompt_type == 2:
        prompt = (
            f"Here is information about a user and a set of candidate products being recommended to the user. "
            f"For the user, we provide their purchase history in chronological order. Each purchase record contains the product title, its categories, the user’s rating (1 to 5, where 1 is the lowest and 5 is the highest), and a brief review. "
            f"For each candidate product, we provide its title and categories.\n"  
            f"Based on the user’s purchase history, ratings, and reviews, analyze the user’s shopping preferences and rank the candidates. Select and output the top 1 product that the user is most likely to buy next.\n"
            f"{str(history_block)}\n"  
            f"{str(candidate_block)}\n"
            f"First, provide a reasoning analysis of the user's preferences based on their purchase history. Your reasoning explanation should be based on commonalities between the user’s past purchases and the inferred user tastes or preferences, to support the reranking decision.\n"
            f"Then, select the top 1 product from the candidates that the user is most likely to buy next. **Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"Please strictly follow this format for your response:\n"
            f"### Reasoning: ###\n<analysis of the user's preferences and why they are likely to buy a particular type of product>\n"
            f"### Selected Product: ###\n<the index letter of the candidate item>\n"
        )
    elif prompt_type == 3:
        prompt = (
            f"Here is information about a user and a set of candidate tracks being recommended to the user. "
            f"For the user, we provide their listening history in chronological order. Each track in the history contains its title, album, artist, and categories. "
            f"For each candidate track, we provide its title, album, artist, and genres.\n"  
            f"Based on the user’s listening history, analyze the user’s music preferences and rank the candidates. Select and output the top 1 track that the user is most likely to listen to next.\n"
            f"{str(history_block)}\n"  
            f"{str(candidate_block)}\n"
            f"First, provide a reasoning analysis of the user's preferences based on their listening history. Your reasoning explanation should be based on commonalities between the user’s past tracks and the inferred user tastes or preferences, to support the reranking decision.\n"
            f"Then, select the top 1 track from the candidates that the user is most likely to listen to next. **Only** output the index letter of the candidate item (one of A-T).\n\n"
            f"Please strictly follow this format for your response:\n"
            f"### Reasoning: ###\n<analysis of the user's preferences and why they are likely to listen to a particular type of track>\n"
            f"### Selected Track: ###\n<the index letter of the candidate item>\n"
        )
    else:
        raise NotImplementedError("Invalid dataset name.")

    return prompt


def generate_ranking_texts_title(user_data, item_meta, candidates, user_id, item_id, max_history_length=50):
    history = load_user_history(user_data, user_id, item_id, max_history_length)
    history_str = [item_meta[int(s)]['title'] for s in history]
    candidate_titles = [item_meta[int(s)]['title'] for s in candidates]
    # cands = candidates[int(user_id)][item_id][:candidate_num]
    prompt = (
                f"The user has watched movies (from the earlier to the later): {str(history_str)}. "
                f"Please select the top 1 movie from the list of candidates below that the user is most likely to watch next. "
                f"The candidates are: {str(candidate_titles)}."            
            )
    return prompt


def generate_explanation_text_llama(prompt, dataset='ml-1m'):
    if dataset == 'ml-1m':
        instruction = "Please strictly follow this format for your response:\n\n### Reasoning: <analysis of the user's preference and why they are likely to watch a particular type of movie>\n### Selected Movie: <the index letter of the candidate item>\n"
    elif dataset in ['office', 'electronics', 'games', 'clothing']:
        instruction = "Please strictly follow this format for your response:\n\n### Reasoning: <analysis of the user's preference and why they are likely to buy a particular type of product>\n### Selected Product: <the index letter of the candidate item>\n"
    template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template

def generate_explanation_text_qwen(prompt):
    instruction = "Please strictly follow this format for your response:\n\n### Reasoning: <analysis of the user's preference and why they are likely to watch a particular type of movie>\n### Selected Movie: <the index letter of the candidate item>\n"
    template = f"<|im_start|>system\n{instruction}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    return template

def generate_explanation_llama(prompt, dataset):
    if dataset == 'ml-1m':
        instruction = """You are a helpful assistant. Given a user’s past movie-watching history and a new movie recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously interacted movies; there is no need to predict the rating of the new item. 

    Please follow the format below: 
    ### Reason ###
    Your Explanation Here

    **Important:** 
    - In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
    - Always refer to it only as "the new item". 
    - Do not output anything else outside this format."""
    elif dataset in ['office', 'electronics', 'games', 'clothing']:
        instruction = """You are a helpful assistant. Given a user’s past purchase history and a new product recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously purchased products; there is no need to predict the rating of the new item. 

    Please follow the format below: 
    ### Reason ###
    Your Explanation Here

    **Important:** 
    - In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
    - Always refer to it only as "the new item". 
    - Do not output anything else outside this format."""
    elif dataset == 'lfm2b':
        instruction = """You are a helpful assistant. Given a user’s past listening history and a new track recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously listened tracks; there is no need to predict the rating of the new item. 

    Please follow the format below: 
    ### Reason ###
    Your Explanation Here

    **Important:** 
    - In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
    - Always refer to it only as "the new item". 
    - Do not output anything else outside this format."""
    else:
        raise NotImplementedError("Invalid dataset name.")
    
    template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template

def generate_explanation_qwen(prompt):
    instruction = """You are a helpful assistant. Given a user’s past movie-watching history and a new movie recommendation, explain why the user might choose the new item. Infer the user’s preferences by identifying common traits among their previously interacted movies; there is no need to predict the rating of the new item. 

Please follow the format below: 
### Reason ###
Your Explanation Here

**Important:** 
- In the explanation, do NOT include the title of the new item in any form — not partially, not paraphrased, and not quoted. 
- Always refer to it only as "the new item". 
- Do not output anything else outside this format."""
    template = f"<|im_start|>system\n{instruction}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    return template

def generate_text_chat(prompt):
    instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    return messages

def generate_text_llama(prompt):
    instruction = "You are a helpful assistant. Now you are to perform a recommendation task. **Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words."
    # instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction)


def generate_think_text_llama(prompt):
    instruction = """
You are a helpful assistant. Now you are to perform a recommendation task. 
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
**Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words.
</answer>
"""    # instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction)


def generate_think_text_qwen(prompt):
    instruction = """
You are a helpful assistant. Now you are to perform a recommendation task. 
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
**Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words.
</answer>
"""    # instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    template = "<|im_start|>system\n{instruction}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction)


def generate_text_qwen(prompt):
    instruction = "You are a helpful assistant. Now you are to perform a recommendation task. **Only** output the index letter of the candidate item (one of A-T). Do not explain the reason or include any other words."
    # instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    template = "<|im_start|>system\n{instruction}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction)

def generate_reflection_text(prompt, result):
    instruction = "Requirements: given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. Just give your answers directly. Do not explain the reason or include any other words."
    reflection_template = (
        f"Carefully evaluate your previous recommendation. "
        f"Reflect on whether it is the best choice given the available candidate list. "
        f"Consider factors such as genre, storyline, director, and user preferences. "
        f"If the previous choice is indeed the best, confirm it. Otherwise, provide a revised recommendation. "
        f"Output format: The title of the chosen item. The title must be exactly consistent with the one in the candidate list. "
        f"Just give your answer directly. Do not explain the reason or include any other words."
    )
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>{result}<|eot_id|><|start_header_id|>user<|end_header_id|>{feedback_prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction).replace("{feedback_prompt}", reflection_template).replace("{result}", result)

def generate_idx_text_llama(prompt):
    instruction = "Requirements: Given the viewing history of the user, predict the next movie this user will watch from the candidate movie set. Candidate movies are listed, each with a unique number from 1 to 20. Output format: Output only the number (1–20) corresponding to the chosen movie. Do not output the movie title, explanation, or any other text."
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instruction}\n<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template.replace("{prompt}", prompt).replace("{instruction}", instruction)

if __name__ == "__main__":
    # Example usage
    user_data = load_user_data('raw_data/ml-1m/user.txt')
    item_meta = load_pickle('raw_data/ml-1m/item.pkl')
    interactions = load_pickle('raw_data/ml-1m/user-item.pkl')
    candidates = load_pickle('raw_data/ml-1m/candidates.pkl')
    # print(user_data)
    # print(item_meta)

    history = load_user_history(user_data, '6040', '593')
    print(history)

    prompt = generate_ranking_texts(user_data, item_meta, interactions, candidates[6040]['593'][:20], '6040', '593')
    print(prompt)
    
    # prompt = "User history: Movie A, Movie B, Movie C. Candidate movies: Movie D, Movie E."
    # print(generate_text_llama(prompt))
    # print(generate_text_qwen(prompt))
    # print(generate_explanation_text_llama(prompt))
    # print(generate_explanation_text_qwen(prompt))