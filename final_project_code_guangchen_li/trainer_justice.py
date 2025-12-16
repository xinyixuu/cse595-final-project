import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from model import QwenClassifier
from data_io import PromptDataset
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).float()

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out["loss"]
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)

def run_finetune_head(
    data_root_train: str,
    data_root_val: str = None,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 3,
    max_length: int = 256
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "/data/guangchen_li/personal/final_project/qwen_0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PromptDataset(data_root_train, tokenizer, max_length=max_length, ICL=False, deon=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if data_root_val is not None:
        val_ds = PromptDataset(data_root_val, tokenizer, max_length=max_length, ICL=False, deon=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    base = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True).to(device)
    hidden_size = getattr(base.config, "hidden_size")

    # 只训练分类头
    model = QwenClassifier(base, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.cls.parameters(), lr=lr, weight_decay=0.01)

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f}")
    os.makedirs("/data/guangchen_li/personal/final_project/head_deontology", exist_ok=True)
    torch.save(model.cls.state_dict(), "/data/guangchen_li/personal/final_project/head_deontology/classifier.pt")
    tokenizer.save_pretrained("/data/guangchen_li/personal/final_project/head_deontology")
    print("Saved classifier head")

    return model, tokenizer

if __name__ == "__main__":
    model, tok = run_finetune_head(
        data_root_train="/data/guangchen_li/personal/ethics/deontology/deontology_train.csv",
        data_root_val=None,
        batch_size=16,
        lr=1e-3,  # 头部较大学习率，收敛快
        epochs=3,
        max_length=64
    )