
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import os
import argparse
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from vllm.lora.request import LoRARequest
import json
from evaluate import calMetrics, calSampleMetrics

from utils import load_json, load_user_data, load_pickle, generate_ranking_texts, generate_text_llama, generate_text_qwen

def parse_global_args(parser):
    parser.add_argument('--input_path', type=str, default='./raw_data',
                        help='Input file path.')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset Name.')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Log directory.')    
    parser.add_argument('--log_file', type=str, default='sft_logits.json',
                        help='Log file name.')
    parser.add_argument('--lora_path', type=str, default='',
                        help='Lora checkpoint path.')
    parser.add_argument('--model_path', type=str, default='../../LLaMA-Factory/output/llama3_lora_ml1m_rerank_character',
                        help='Model path.')
    parser.add_argument('--input_file_path', type=str, default='',
                        help='Input file path.')
    parser.add_argument('--generate', type=int, default=1,
                        help='Input file or generate prompts.')
    parser.add_argument('--inf_correct', type=int, default=0,
                        help='Only do the inference of samples that include positive items.')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    args, extras = parser.parse_known_args()

    input_path = args.input_path
    dataset = args.dataset
    log_dir = args.log_dir
    log_file = args.log_file
    lora_path = args.lora_path
    model_path = args.model_path
    input_file_path = args.input_file_path
    generate = bool(args.generate)
    inf_correct = bool(args.inf_correct)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, dataset)):
        os.makedirs(os.path.join(log_dir, dataset))
    output_path = os.path.join(log_dir, dataset, log_file)
    print('begin exp:', output_path)

    if generate:

        user_data = load_user_data(os.path.join(input_path, dataset, 'user.txt'))
        item_meta = load_pickle(os.path.join(input_path, dataset, 'item.pkl'))
        if dataset != 'lfm2b':
            user_item_interaction = load_pickle(os.path.join(input_path, dataset, 'user-item.pkl'))
        else:
            user_item_interaction = dict()
        candidates = load_pickle(os.path.join(input_path, dataset, 'candidates.pkl'))
        candidate_num = 20

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
        print(f"Item ID type: {target_type}")

        # generate character identifier
        test_prompts = []
        for k, v in user_data.items():
            cand_set = list(candidates[int(k)][target_type(v[-1])][:candidate_num])
            prompt = generate_ranking_texts(user_data, item_meta, user_item_interaction, cand_set, k, v[-1], dataset_name=dataset)
            if target_type(v[-1]) in cand_set:
                label = cand_set.index(target_type(v[-1]))
            else:
                label = -1
            test_prompts.append({
                'user_id': k,
                'item_id': v[-1],
                'prompt': prompt,
                'label': label,
            })
    else:
        test_prompts = load_json(input_file_path)
        print('loaded {} prompts'.format(len(test_prompts)))

    print('model_path:', model_path)
    if 'qwen' in model_path or 'Qwen' in model_path:
        prompts = [generate_text_qwen(test_prompts[i]['prompt']) for i in range(len(test_prompts))]
    elif 'llama' in model_path or 'Llama' in model_path:
        prompts = [generate_text_llama(test_prompts[i]['prompt']) for i in range(len(test_prompts))]
    else:
        raise NotImplementedError('Model not supported!')
    print('generated {} prompts'.format(len(prompts)))
    print(prompts[0])

    # model_name = "facebook/opt-1.3b"  # 或者你的本地路径
    llm = LLM(model=model_path, tokenizer=model_path, dtype="auto", max_model_len=10240, gpu_memory_utilization=0.8, enable_lora=True) # , tensor_parallel_size=2

    # 分词器（用于找出A-T的token id）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token  # 防止部分模型报错
    # 准备输入
    # prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=10, max_tokens=10, logprobs=20, n=5)
    # sampling_params = SamplingParams(
    #     temperature=1.0,
    #     top_p=1.0,
    #     max_tokens=1,
    #     logprobs=1  # 开启返回 token-level 概率
    # )

    # 推理
    # outputs = llm.generate([prompts[0]], sampling_params)

    target_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:20]
    target_token_ids = []
    for char in target_chars:
        target_token_ids.extend(tokenizer.encode(char, add_special_tokens=False))
    # print("Target token IDs:", target_token_ids)

    if inf_correct:
        inf_prompts = []
        inf_test_prompts = []
        inf_idx = []
        for i in range(len(test_prompts)):
            if test_prompts[i]['label'] != -1:
                inf_prompts.append(prompts[i])
                inf_idx.append(i)
        print(f"Filtered prompts to {len(inf_prompts)} samples that include positive items.")
    else:
        inf_prompts = prompts
        inf_idx = list(range(len(test_prompts)))

    batch_size = 10000
    for batch in range((len(inf_prompts)-1)//batch_size+1):
        if (batch+1)*batch_size >= len(inf_prompts):
            batch_prompts = inf_prompts[batch*batch_size:]
        else:
            batch_prompts = inf_prompts[batch*batch_size:(batch+1)*batch_size]

        if lora_path:
            outputs = llm.generate(batch_prompts, sampling_params, lora_request=LoRARequest("kd_adapter", 1, lora_path))
        else:
            outputs = llm.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            # prompt = output.prompt
            char_to_prob = {}
            generated_text = []
            for j in range(5):
                generated_text.append(output.outputs[j].text)
            test_prompts[inf_idx[batch*batch_size+i]]['result'] = generated_text
            # 确保token_ids和logprobs长度匹配
            token_count = min(len(output.outputs[0].token_ids), len(output.outputs[0].logprobs))
            for j in range(token_count):
                logprob_info = output.outputs[0].logprobs[j]
                if len(set(target_token_ids).intersection(set(list(logprob_info.keys())))) > 10:
                    for target_token_id in target_token_ids:
                        try:
                            char_to_prob[tokenizer.decode(target_token_id)] = logprob_info.get(target_token_id).logprob
                        except:
                            char_to_prob[tokenizer.decode(target_token_id)] = -math.inf
                    break
            sorted_probs = sorted(char_to_prob.items(), key=lambda x: x[1], reverse=True)
            try:
                top_20_chars, top_20_probs = zip(*sorted_probs[:20])
                test_prompts[inf_idx[batch*batch_size+i]]['top_10_chars'] = top_20_chars
                test_prompts[inf_idx[batch*batch_size+i]]['top_10_probs'] = top_20_probs
            except:
                print(f"Error processing prompt {batch*batch_size+i}: {output.outputs[0].text}")
                print(f"Logprobs: {output.outputs[0].logprobs}")
                test_prompts[inf_idx[batch*batch_size+i]]['top_10_chars'] = []
                test_prompts[inf_idx[batch*batch_size+i]]['top_10_probs'] = []

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(test_prompts, f, indent=4, ensure_ascii=False)

    print(f"The results have been saved to '{output_path}'.")


    with open(os.path.join(log_dir, dataset, 'metrics.txt'), "a", encoding="utf-8") as f:
        f.write(f"Exp: {log_file}\n")
        if inf_correct:
            f.write(f"Only inference samples that include positive items.\n")
            labels = [item['label'] for item in test_prompts if item['label'] != -1]
            results = [item.get('top_10_chars', {}) for item in test_prompts if item['label'] != -1]
            for k in [1,3,5]:
                hr_k, ndcg_k = calMetrics(results, labels, k)
                print(f"Logits HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}")
                f.write(f"Logits HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}\n")

            results = [item.get('result', '') for item in test_prompts if item['label'] != -1]
            for k in [1,3,5]:
                hr_k, ndcg_k = calSampleMetrics(results, labels, k)
                print(f"Sample HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}")
                f.write(f"Sample HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}\n")
        else:
            labels = [item['label'] for item in test_prompts]
            results = [item['top_10_chars'] for item in test_prompts]
            for k in [1,3,5]:
                hr_k, ndcg_k = calMetrics(results, labels, k)
                print(f"Logits HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}")
                f.write(f"Logits HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}\n")

            results = [item['result'] for item in test_prompts]
            for k in [1,3,5]:
                hr_k, ndcg_k = calSampleMetrics(results, labels, k)
                print(f"Sample HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}")
                f.write(f"Sample HR@{k}: {hr_k:.8f}, NDCG@{k}: {ndcg_k:.8f}\n")
