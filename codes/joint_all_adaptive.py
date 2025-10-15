import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,4,6' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import generate_text_llama, generate_text_qwen

LOG_INTERVAL = 200
BATCH_SIZE = 1
SAVE_INTERVAL = 5000
DEBUG = False

import argparse

def parse_global_args(parser):
    parser.add_argument('--student_data', type=str, default='./data/ml-1m/generated_prompts_last5_correct.json',
                        help='Student input file path.')
    parser.add_argument('--teacher_data', type=str, default='./data/ml-1m/generated_prompts_with_candidates_gpt3.5_last5_correct.json',
                        help='Teacher input file path.')
    parser.add_argument('--student_model_path', type=str, default='../llama-3.1-8b-instruct',
                        help='Student model path.')
    parser.add_argument('--teacher_model_path', type=str, default='../llama-3.1-8b-instruct',
                        help='Teacher model path.')
    parser.add_argument('--epoch', type=int, default=10,
                        help='Total training epochs.')
    parser.add_argument('--lora_save_path', type=str, default='./lora/',
                        help='Lora adapter save path.')  
    parser.add_argument('--kl_type', type=str, default='fkl',
                        help='Forward KL loss or Revverse KL loss.') 
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for distillation loss.')   
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Penalty parameter.')
    parser.add_argument('--gamma', type=float, default=1.5,
                        help='Temperature parameter.')
    parser.add_argument('--tau', type=float, default=2.0,
                        help='Temperature parameter.')   
    parser.add_argument('--base_alpha', type=float, default=0.5,
                        help='Alpha Base.')  
    parser.add_argument('--logits', type=int, default=0,
                        help='Filter data according to sample results or logits.')   
    parser.add_argument('--logit_bar', type=int, default=1,
                        help='Filter data according to @k.')     
    parser.add_argument('--early_stop_step', type=int, default=5,
                        help='Steps to early stop.')                   
    return parser

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 数据集示例（与之前一致）
class CustomDataset(Dataset):
    def __init__(self, texts, labels, flags, tokenizer, model, max_len=512):
        self.texts = texts
        self.labels = labels
        self.flags = flags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model = model

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if 'qwen' in self.model:
            prompt = generate_text_qwen(text)
            label += "<|im_end|>"
        elif 'llama' in self.model or 'Llama' in self.model:
            prompt = generate_text_llama(text)
            label += "<|eot_id|>"
        else:
            raise NotImplementedError
        prompt_inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        target_inputs = self.tokenizer(label, return_tensors='pt', add_special_tokens=False)
        return {
            'input': prompt_inputs,
            'target': target_inputs,
            'flag': self.flags[idx],
        }


class PairedDataset(Dataset):
    def __init__(self, teacher_dataset, student_dataset):
        self.teacher_dataset = teacher_dataset
        self.student_dataset = student_dataset
        assert len(teacher_dataset) == len(student_dataset)

    def __len__(self):
        return len(self.teacher_dataset)

    def __getitem__(self, idx):
        return self.teacher_dataset[idx], self.student_dataset[idx]
    

# 蒸馏损失函数（与之前一致）
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, label_ids):
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        shift_logits = student_logits[..., :-1, :].contiguous()  # 去掉最后一个token
        shift_labels = label_ids.contiguous()  # 去掉第一个token
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        label_loss = self.ce_loss(shift_logits, shift_labels)
        return self.alpha * distillation_loss + (1-self.alpha) * label_loss

# def compute_dynamic_alpha(student_rank, teacher_rank, *,
#                           N=20,
#                           alpha_low=0.15, alpha_high=0.85,
#                           alpha_min=0.0, alpha_max=0.9,
#                           tau_t=2.0, tau_delta=1.2,
#                           beta_up=0.5, beta_down=0.5):
#     """
#     student_rank, teacher_rank: [B]，正确答案的排名 (1..N 或 N+1 表示不在子集)
#     返回: alpha [B]
#     """
#     device = teacher_rank.device
#     rt = teacher_rank.float()
#     rs = student_rank.float()
#     delta = rs - rt

#     # ---------- 基础部分: rt 越小，alpha 越大 ----------
#     # 以 c=5.5 为中心的 sigmoid 过渡
#     c = 5.5
#     alpha_base = alpha_low + (alpha_high - alpha_low) * torch.sigmoid((c - rt) / tau_t)

#     # ---------- 比较部分: 根据 rt vs rs 调整 ----------
#     sig_delta = torch.sigmoid(delta / tau_delta)        # Δ>=0 时大，Δ<0 时小
#     sig_abs   = torch.sigmoid(torch.abs(delta) / tau_delta)

#     # 初始化
#     alpha = alpha_base.clone()

#     # teacher 更好 (rt < rs) → 上调
#     mask_up = (rt <= rs)
#     alpha[mask_up] = alpha_base[mask_up] * (0.5 + beta_up * sig_delta[mask_up])

#     # teacher 不如/相等 (rt >= rs) → 下调
#     mask_down = ~mask_up
#     alpha[mask_down] = alpha_base[mask_down] * (0.5 - beta_down * sig_abs[mask_down])

#     # ---------- 限制在 [alpha_min, alpha_max] ----------
#     alpha = torch.clamp(alpha, alpha_min, alpha_max)

#     return alpha

def compute_dynamic_alpha(student_rank, teacher_rank, *,
                   alpha_base=0.5,
                   alpha_min=0.001, alpha_max=0.9,
                   beta_penalty=0.1,
                   tau=2.0, gamma=1.5, c=5.5):
    """
    根据公式：
      α'(rt, rs) = α_base * f((rt - c)/γ) * σ((rs - rt)/τ),    rt <= rs
                 = α_base * f((rt - c)/γ) * β_penalty * σ((rs - rt)/τ),  rt > rs
      α(rt, rs)  = clip(α'(rt, rs), α_min, α_max)

    参数：
      student_rank, teacher_rank: [B] 张量，名次（1..20 或其他）；数值越小越好
      alpha_base, alpha_min, alpha_max, beta_penalty, tau, gamma, c: 与图中超参一致
    返回：
      alpha: [B]，每样本自适应 α
    """
    rt = teacher_rank.to(dtype=torch.float32)
    rs = student_rank.to(dtype=torch.float32)

    # f((rt - c)/γ) = 1 + tanh((rt - c)/γ)，rt 越小 → f 越大（∈ (0, 2)）
    f_rt = 1.0 + torch.tanh((c - rt) / gamma)

    # σ((rs - rt)/τ)
    sig = torch.sigmoid((rs - rt) / tau)

    # α' 基础项
    alpha_prime = alpha_base * f_rt * sig

    # 分支：rt > rs 需要乘以 β_penalty
    mask_down = (rt > rs)
    if mask_down.any():
        alpha_prime = torch.where(mask_down, alpha_prime * beta_penalty, alpha_prime)

    # clip 到 [α_min, α_max]
    alpha = torch.clamp(alpha_prime, min=alpha_min, max=alpha_max)
    return alpha


class ReverseDistillationLoss(nn.Module):
    def __init__(self, target_ids, device='cuda', temperature=2.0, alpha=0.5, base_alpha=0.5, gamma=1.5, tau=2.0, beta=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.base_alpha = base_alpha
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.target_ids = torch.tensor(target_ids, device=device)
        self.target_sorted = torch.sort(self.target_ids).values  # 仅用于定位答案时间步

    @torch.no_grad()
    def _find_answer_pos(self, logits):
        """
        logits: [B, T, V]
        返回每个样本答案所在的时间步 pos: [B]（int）
        规则：优先找“该步 top-K 的 token 集合 == target_ids（集合）”的最后一次出现；
             若找不到，则回退到“对 target_ids 的 logsumexp 最大”的时间步。
        """
        B, T, V = logits.shape
        K = self.target_ids.numel()

        # 取每步 top-K 的 token id，并按 id 升序以便与 target_ids 集合比较
        topk_ids = torch.topk(logits, k=K, dim=-1).indices             # [B, T, K]（按 logit 降序）
        topk_sorted = torch.sort(topk_ids, dim=-1).values              # [B, T, K]
        # 比较集合是否完全相同
        equal_mask = (topk_sorted == self.target_sorted.view(1, 1, K)).all(dim=-1)   # [B, T] bool

        # 选“最后一次出现”的时间步
        idxs = torch.arange(T, device=logits.device).view(1, T).expand(B, T)
        masked = torch.where(equal_mask, idxs, torch.full_like(idxs, -1))
        pos_equal = masked.max(dim=1).values                            # [B], -1 表示没匹配到
        has_equal = (pos_equal >= 0)

        # 回退：选对 target_ids 的 logsumexp 最大的时间步（用 teacher / logits 本身都可）
        lse = torch.logsumexp(logits.index_select(dim=-1, index=self.target_ids), dim=-1)  # [B, T]
        pos_fallback = lse.argmax(dim=1)                                                   # [B]

        # 合成最终答案时间步
        answer_pos = torch.where(has_equal, pos_equal, pos_fallback)                       # [B]
        return answer_pos

    @torch.no_grad()
    def _rank_at_pos(self, logits, pos, correct_ids):
        """
        在给定时间步 pos（每个样本一个），计算正确 token 在 target_ids 子集中按 logit 降序的名次（1..K）。
        若 correct_id 不在 target_ids 中，返回 K+1。
        """
        B, T, V = logits.shape
        K = self.target_ids.numel()
        arangeB = torch.arange(B, device=logits.device)

        # 取该时间步的 logits，并只保留 target_ids 的列
        step_logits = logits[arangeB, pos]                                   # [B, V]
        choice_logits = step_logits.index_select(dim=-1, index=self.target_ids)  # [B, K]

        # 对 K 个候选降序排序，建立“token在子集中的名次”的逆映射
        order = torch.argsort(choice_logits, dim=1, descending=True)         # [B, K]
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(K, device=logits.device).view(1, -1).expand_as(order))  # [B, K], 0-based

        # 找到正确 token 在 target_ids 子集中的列号
        eq = (self.target_ids.view(1, -1) == correct_ids.view(-1, 1))        # [B, K] bool
        has = eq.any(dim=1)                                                  # [B]
        corr_k = torch.argmax(eq.int(), dim=1)                               # [B]（若全 False 也会给 0，但我们用 has 过滤）

        # 若不在子集中，rank = K+1；否则 rank = inv+1
        rank = torch.full((B,), K + 1, device=logits.device, dtype=torch.long)
        rank[has] = inv[arangeB[has], corr_k[has]] + 1
        return rank  # 1..K, or K+1(=不在子集)
    
    def forward(self, student_logits, teacher_logits, label_ids, flag=False):
        label_ids = label_ids.view(label_ids.size(1), -1)  # [B, T]
        answer_pos = self._find_answer_pos(teacher_logits[..., :-1, :])        # [B]
        correct_ids = label_ids[torch.arange(label_ids.size(0), device=label_ids.device), answer_pos.view(-1)]  # [B]

        # 分别计算 student / teacher 的名次（1..K；K+1 表示不在 target_ids 子集中）
        rank_student = self._rank_at_pos(student_logits, answer_pos, correct_ids)
        rank_teacher = self._rank_at_pos(teacher_logits, answer_pos, correct_ids)

        teacher_log_probs = torch.log_softmax(teacher_logits[..., :-1, :] / self.temperature, dim=-1)
        student_probs = torch.softmax(student_logits[..., :-1, :] / self.temperature, dim=-1)
        distillation_loss = self.kl_loss(teacher_log_probs, student_probs) * (self.temperature ** 2)
    
        shift_logits = student_logits[..., :-1, :].contiguous()  # 去掉最后一个token
        shift_labels = label_ids.contiguous() 
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        label_loss = self.ce_loss(shift_logits, shift_labels)

        alpha = compute_dynamic_alpha(rank_student, rank_teacher, beta_penalty=self.beta, alpha_base=self.base_alpha, gamma=self.gamma, tau=self.tau)
        # print(f"Student rank: {rank_student.tolist()}, Teacher rank: {rank_teacher.tolist()}, Alpha: {alpha.tolist()}")

        return alpha * distillation_loss + (1-alpha) * label_loss, distillation_loss, label_loss


# 主训练逻辑（支持 LoRA）
def train_teacher_student(teacher_model, student_model, tokenizer, paired_loader, output_dir, gradient_accumulation_steps=8, alpha=0.5, base_alpha=0.5, gamma=1.5, tau=2.0, beta=0.1, epochs=3, lr=5e-5, early_stop_step=2):
    teacher_model.eval()  # 冻结教师模型
    student_model.train()  # 训练学生模型（仅更新 LoRA 层）

    # 获取 A-T 字符对应的 token id（注意这里是单个字符的 token）
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:20]  # A-T
    letter_token_ids = []
    for ch in letters:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            letter_token_ids.append(ids[0])
        else:
            print(f"Warning: {ch} is tokenized into multiple tokens: {ids}")

    print("Experimental Parameters:")
    print(f"  kl_type = {kl_type}, alpha = {alpha}, base_alpha = {base_alpha}, gamma = {gamma}, tau = {tau}, beta = {beta}")

    optimizer = optim.AdamW(student_model.parameters(), lr=lr)
    if kl_type == 'fkl':
        criterion = DistillationLoss(alpha=alpha)
    elif kl_type == 'rkl':
        criterion = ReverseDistillationLoss(letter_token_ids, device=student_model.device, alpha=alpha, base_alpha=base_alpha, gamma=gamma, tau=tau, beta=beta)
    else:
        raise NotImplementedError

    step_losses = []  # 用于记录每个步骤的 loss
    step_ave_losses = []
    step_distillation_losses = []  # 用于记录每个步骤的 distillation loss
    step_ave_distillation_losses = []
    step_ce_losses = []  # 用于记录每个步骤的 ce loss
    step_ave_ce_losses = []
    step_count = 0    # 用于记录步骤数
    step_total_loss = 0.0
    step_log_loss = 0.0
    step_distill_loss = 0.0
    step_ce_loss = 0.0
    cnt = 0
    accumulation_step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        best_train_loss = 1e9
        early_stop = False

        # 同时迭代 teacher_loader 和 student_loader
        for teacher_batch, student_batch in tqdm(
                    paired_loader, 
                    total=len(paired_loader), 
                    desc=f"Epoch {epoch + 1}/{epochs}"):
            step_count += 1
            accumulation_step += 1
            
            # 获取 teacher_loader 的输入数据
            teacher_original_input_ids = teacher_batch['input']['input_ids'].to(teacher_model.device)
            teacher_target_input_ids = teacher_batch['target']['input_ids'].to(teacher_model.device)
            teacher_input_ids = torch.cat([teacher_original_input_ids, teacher_target_input_ids], dim=2)
            teacher_attention_mask = torch.ones_like(teacher_input_ids)

            # 获取 student_loader 的输入数据
            student_original_input_ids = student_batch['input']['input_ids'].to(student_model.device)
            student_target_input_ids = student_batch['target']['input_ids'].to(student_model.device)
            student_input_ids = torch.cat([student_original_input_ids, student_target_input_ids], dim=2)
            student_attention_mask = torch.ones_like(student_input_ids)

            # 运行 teacher_model，获取 teacher_logits
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=teacher_input_ids[0], attention_mask=teacher_attention_mask[0])
                teacher_logits = teacher_outputs.logits

            # 运行 student_model，获取 student_logits
            student_outputs = student_model(input_ids=student_input_ids[0], attention_mask=student_attention_mask[0])
            student_logits = student_outputs.logits

            # 确保 teacher_logits 和 student_logits 在同一设备上
            teacher_logits = teacher_logits.to(student_logits.device)

            # 选取 teacher_logits 和 student_logits 中 target 的部分
            target_length = teacher_target_input_ids.size(-1)
            teacher_logits_selected = teacher_logits[:, -target_length-1:, :].to(student_logits.device)
            student_logits_selected = student_logits[:, -target_length-1:, :].to(student_logits.device)

            if DEBUG:
                topk = 20
                batch_idx = 0  # 固定为第一个样本
                seq_len = teacher_logits_selected.size(1)

                # 提取 topk token ids
                teacher_topk_values, teacher_topk_indices = torch.topk(teacher_logits_selected, k=topk, dim=-1)
                student_topk_values, student_topk_indices = torch.topk(student_logits_selected, k=topk, dim=-1)

                # 遍历每个位置
                for pos_idx in range(seq_len):
                    teacher_ids = teacher_topk_indices[batch_idx, pos_idx].tolist()
                    student_ids = student_topk_indices[batch_idx, pos_idx].tolist()

                    teacher_tokens = tokenizer.convert_ids_to_tokens(teacher_ids)
                    student_tokens = tokenizer.convert_ids_to_tokens(student_ids)

                    print(f"\nPosition {pos_idx}:")
                    print("Teacher Top-20 Tokens:", teacher_tokens)
                    print("Student Top-20 Tokens:", student_tokens)

            # 计算损失
            loss, distillation_loss, ce_loss = criterion(student_logits_selected, teacher_logits_selected, student_target_input_ids, False)
    
            # normalize loss for accumulation
            loss = loss / gradient_accumulation_steps
            distillation_loss = distillation_loss / gradient_accumulation_steps
            ce_loss = ce_loss / gradient_accumulation_steps

            loss.backward()

            # === GRADIENT ACCUMULATION STEP ===
            if accumulation_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                accumulation_step = 0

            # 累积 loss
            total_loss += loss.item() * gradient_accumulation_steps
            step_total_loss += loss.item() * gradient_accumulation_steps
            step_log_loss += loss.item() * gradient_accumulation_steps
            step_distill_loss += distillation_loss.item() * gradient_accumulation_steps
            step_ce_loss += ce_loss.item() * gradient_accumulation_steps
    
            # 每隔 log_interval 步记录一次 loss
            if step_count % LOG_INTERVAL == 0:
                step_losses.append(loss.item())
                step_ave_losses.append(step_log_loss / LOG_INTERVAL)
                step_distillation_losses.append(distillation_loss.item())
                step_ave_distillation_losses.append(step_distill_loss / LOG_INTERVAL)
                step_ce_losses.append(ce_loss.item())
                step_ave_ce_losses.append(step_ce_loss / LOG_INTERVAL)
                step_log_loss = 0.0
                step_distill_loss = 0.0
                step_ce_loss = 0.0
                # print(f"Epoch {epoch + 1}, Step {step_count}, Loss: {loss.item():.4f}")
            if step_count % SAVE_INTERVAL == 0 and step_total_loss < best_train_loss:
                print(f"{epoch}: {step_count}, Average Loss: {step_total_loss / SAVE_INTERVAL:.4f}")
                best_train_loss = step_total_loss
                step_total_loss = 0.0
                # print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(student_loader):.4f}")
                student_model.save_pretrained(os.path.join('lora', output_dir, f"checkpoint_steps{step_count}"))
            elif step_count % SAVE_INTERVAL == 0 and (cnt < early_stop_step or epoch == 0):
                print(f"{epoch}: {step_count}, Average Loss: {step_total_loss / SAVE_INTERVAL:.4f}")
                step_total_loss = 0.0
                cnt += 1
                student_model.save_pretrained(os.path.join('lora', output_dir, f"checkpoint_steps{step_count}"))
            elif step_count % SAVE_INTERVAL == 0:
                early_stop = True
                print(f"{epoch}: {step_count}, Average Loss: {step_total_loss / SAVE_INTERVAL:.4f}")
                break

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(paired_loader):.4f}")

        # 绘制 Loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(step_ave_losses, label="Loss")
        plt.xlabel(f"Steps (every {LOG_INTERVAL} steps)")
        plt.ylabel("Loss")
        plt.title("Training Average Loss Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join('lora', output_dir, 'training_average_loss.jpg'), dpi=200)
        plt.close()

        # 绘制 Loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(step_ave_distillation_losses, label="Loss")
        plt.xlabel(f"Steps (every {LOG_INTERVAL} steps)")
        plt.ylabel("Loss")
        plt.title("Distillation Training Average Loss Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join('lora', output_dir, 'distillation_average_loss.jpg'), dpi=200)
        plt.close()

        # 绘制 Loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(step_ave_ce_losses, label="Loss")
        plt.xlabel(f"Steps (every {LOG_INTERVAL} steps)")
        plt.ylabel("Loss")
        plt.title("Label Training Average Loss Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join('lora', output_dir, 'label_average_loss.jpg'), dpi=200)
        plt.close()

        if early_stop:
            break

    # # 绘制 Loss 曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(step_losses, label="Loss")
    # plt.xlabel(f"Steps (every {LOG_INTERVAL} steps)")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Curve")
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join('lora', output_dir, 'training_loss.jpg'), dpi=200)

# 示例调用（支持 LoRA）
if __name__ == "__main__":
    # 模型和数据准备
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    args, extras = parser.parse_known_args()

    teacher_model_name = args.teacher_model_path
    student_model_name = args.student_model_path
    teacher_json_file = args.teacher_data
    student_json_file = args.student_data
    output_dir = args.lora_save_path
    kl_type = args.kl_type
    alpha = args.alpha
    early_stop_step = args.early_stop_step
    LOGITS = bool(args.logits)
    logit_bar = args.logit_bar
    EPOCHS = args.epoch 
    beta = args.beta
    gamma = args.gamma
    tau = args.tau
    base_alpha = args.base_alpha
    output_dir = output_dir+f"_{kl_type}_beta{beta}_gamma{gamma}_tau{tau}_base{base_alpha}"

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name,
                                                         device_map='auto',
                                                         torch_dtype='bfloat16',
                                                         attn_implementation="flash_attention_2",
                                                         )
                                                        #  max_memory=teacher_max_memory)

    # 配置 LoRA
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name,
                                                         device_map='auto',
                                                         torch_dtype='bfloat16',
                                                         attn_implementation="flash_attention_2",
                                                         )

    lora_config = LoraConfig(
        r=8,  # 矩阵秩，控制可训练参数量
        lora_alpha=32,  # LoRA 缩放因子
        # target_modules=["q_proj", "v_proj"],  # 指定 LoRA 作用的层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,  # LoRA dropout
        bias="none",  # 是否训练偏置
        task_type="CAUSAL_LM",  # 自回归语言模型任务
    )
    student_model = get_peft_model(student_model, lora_config)
    # student_model = student_model.to('cuda:1')

    # 打印可训练参数数目
    student_model.print_trainable_parameters()
    teacher_data = load_json(teacher_json_file)
    student_data = load_json(student_json_file)

    correct_idx = []
    for id, entry in enumerate(teacher_data):
        if entry['label'] == -1:
            raise ValueError("Teacher data contains invalid label -1")
        target_char = chr(65 + entry['label'])
        entry['target'] = target_char
        student_data[id]['target'] = target_char
    print(f"Number of correct entries in teacher data: {len(teacher_data)}")
    print(teacher_data[0])
    print(student_data[0]) 

    # teacher_json_file = "./data/ml-1m/generated_prompts_with_candidates_gpt3.5_last5_correct.json"
    teacher_texts = [x['prompt'] for x in teacher_data]
    teacher_labels = [x['target'] for x in teacher_data]
    teacher_flags = [x.get('flag', False) for x in teacher_data]
    teacher_dataset = CustomDataset(teacher_texts, teacher_labels, teacher_flags, tokenizer, teacher_model_name)

    # student_json_file = "./data/ml-1m/generated_prompts_last5_correct.json"
    student_texts = [x['prompt'] for x in student_data]
    student_labels = [x['target'] for x in student_data]
    student_flags = [x.get('flag', False) for x in student_data]
    student_dataset = CustomDataset(student_texts, student_labels, student_flags, tokenizer, student_model_name)

    # teacher_labels和student_labels必须完全一致
    assert teacher_labels == student_labels, "Teacher and student labels do not match!"

    paired_dataset = PairedDataset(teacher_dataset, student_dataset)

    loader = DataLoader(paired_dataset, batch_size=BATCH_SIZE, shuffle=True) # sampler=sampler)   

    train_teacher_student(teacher_model, student_model, tokenizer, loader, output_dir, alpha=alpha, base_alpha=base_alpha, gamma=gamma, tau=tau, beta=beta, epochs=EPOCHS, lr=2e-5, early_stop_step=early_stop_step)

    # 保存 LoRA 的权重
    # student_model.save_pretrained(output_dir)
    print(f"LoRA adapter weights saved to {output_dir}")