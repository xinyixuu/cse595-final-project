from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
from tqdm import tqdm
from data_io import prepare_promt, PromptDataset
from model import QwenClassifier
from torch.utils.data import DataLoader
def load_model_for_eval(base_model_name="Qwen/Qwen2.5-0.5B",
                        save_dir="./qwen05b_head_only",):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    hidden_size = getattr(base.config, "hidden_size", 1024)
    model = QwenClassifier(base, hidden_size).to(device)
    model.cls.load_state_dict(torch.load(f"{save_dir}/classifier.pt", map_location=device))
    model.eval()
    return tokenizer, model, device

@torch.no_grad()
def evaluate(model, loader, device, k):
    model.eval()
    correct, n = 0, 0
    for batch in tqdm(loader, desc=f"eval, k={k}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).float()

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        probs = torch.sigmoid(out["logits"])
        preds = (probs > 0.5).float()
        #print(correct, n)
        correct += (preds == labels).sum().item()
        n += labels.size(0)
    return correct / max(1, n)


if __name__ == "__main__":
    base_root = "/data/guangchen_li/personal/final_project/qwen_0.5b"
    cls_root = "/data/guangchen_li/personal/final_project/head_deontology"
    tokenizer, model, device = load_model_for_eval(base_root, cls_root)
    test_csv = "/data/guangchen_li/personal/ethics/deontology/deontology_test.csv"
    
    
    test_dataset = PromptDataset(test_csv, tokenizer, max_length=512, ICL=False, deon=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    acc = evaluate(model, test_loader, device, k="baseline")
    print(f"Test accuracy: {acc:.4f}")

    ks=[1, 2, 4, 8, 16, 32]
    for k_ in ks:
        test_dataset = PromptDataset(test_csv, tokenizer, max_length=512, ICL=True, deon=True, k=k_)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        acc = evaluate(model, test_loader, device, k=k_)
        print(f"Test accuracy: {acc:.4f}")
