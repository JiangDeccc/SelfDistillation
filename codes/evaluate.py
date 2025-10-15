from utils import load_json
import os
import numpy as np

def calMetrics(pred, label, k):
    assert len(pred) == len(label)

    # cal HR
    hr = 0
    hit = []
    for i in range(len(pred)):
        if chr(65+label[i]) in pred[i][:k]:
            hr += 1
            hit.append(1)
        else:
            hit.append(0)
    hr_k = hr / len(pred)

    # cal NDCG
    dcg = 0
    for i in range(len(pred)):
        if hit[i] == 1:
            dcg += 1 / np.log2(pred[i][:k].index(chr(65+label[i])) + 2)
    ndcg = dcg / len(pred)
    return hr_k, ndcg

def calSampleMetrics(pred, label, k):
    assert len(pred) == len(label)

    hit = []
    rank = []
    for i, entry in enumerate(pred):
        flag = False
        for j, character in enumerate(entry[:k]):
            if character.strip() == chr(65 + label[i]):
                hit.append(1)
                rank.append(j + 1)
                flag = True
                break
        if not flag:
            hit.append(0)
            rank.append(k + 1)
    hit = np.array(hit)
    rank = np.array(rank)
    hr_k = np.mean(hit)
    ndcg = np.mean(hit / np.log2(rank + 1))
    return hr_k, ndcg


def evaluate(file_path):
    """
    Evaluate the results in the given file path.
    
    Args:
        file_path (str): The path to the JSON file containing the results.
        
    Returns:
        float: The strict match score of the results.
    """
    test_prompts = load_json(file_path)

    correct = 0
    for item in test_prompts:
        if item['label'] == -1:
            continue
        if 'result' not in item:
            continue
        generated_text = item['result'].strip()
        # # generate titles
        # if strict_match(item['label'], generated_text):
        #     correct += 1
        # generate character identifier
        if chr(65 + item['label']) == generated_text:
            correct += 1
    accuracy = correct / len(test_prompts)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_prompts)})")

    # # Calculate the accuracy of the generated texts.
    # correct = 0
    # for item in test_prompts:
    #     if item['label'] == -1:
    #         continue
    #     if 'result' not in item:
    #         continue
    #     # 提取 item['result'] 在 "### Selected Movie:" 之后的文本
    #     try:
    #         generated_text = item['result'].strip().split("### Selected Movie: ###")[-1].strip()
    #     except:
    #         print(f"Invalid Format Error processing item {item['user_id']}: {item['result']}")
    #         continue
    #     # # generate titles
    #     # if strict_match(item['label'], generated_text):
    #     #     correct += 1
    #     # generate character identifier
    #     if chr(65 + item['label']) == generated_text:
    #         correct += 1
    # accuracy = correct / len(test_prompts)
    # print(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_prompts)})")

    return accuracy

if __name__ == "__main__":
    file_path = './train_log/clothing/train_sft_logits_explan_deepseek_mask.json'
    # evaluate(file_path)
    # evaluate(file_path)
    test_prompts = load_json(file_path)
    print(len(test_prompts))
    # test_prompts = test_prompts[:10]
    labels = [item['label'] for item in test_prompts]
    results = [item['top_10_chars'] for item in test_prompts]
    for k in [1,3,5]:
        hr_k, ndcg_k = calMetrics(results, labels, k)
        print(f"HR@{k}: {hr_k:.4f}, NDCG@{k}: {ndcg_k:.4f}")
    
    results = [item['result'] for item in test_prompts]
    for k in [1,3,5]:
        hr_k, ndcg_k = calSampleMetrics(results, labels, k)
        print(f"Sample HR@{k}: {hr_k:.4f}, Sample NDCG@{k}: {ndcg_k:.4f}")