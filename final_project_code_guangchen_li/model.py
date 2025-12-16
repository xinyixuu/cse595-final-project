from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
class QwenClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, dropout=0.1,freeze_base=True, freeze_head = False):
        super().__init__()
        self.base = base_model
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.freeze_base(freeze_base)
        self.freeze_head(freeze_head)
        self._freeze_base = None
        self._freeze_head = None
    def freeze_base(self, freeze: bool = True):
        self._freeze_base = freeze
        for p in self.base.parameters():
            p.requires_grad = (not freeze)

    def freeze_head(self, freeze: bool = False):
        self._freeze_head = freeze
        for p in self.cls.parameters():
            p.requires_grad = (not freeze)
    def forward(self, input_ids, attention_mask, labels=None):
        if self._freeze_base:
            with torch.no_grad():
                out = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
        else:
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
        h_last_layer = out.hidden_states[-1]  # [B, T, H]

        lengths = attention_mask.sum(dim=1) - 1  # [B]
        h_last_token = h_last_layer[torch.arange(h_last_layer.size(0)), lengths]

        logit = self.cls(h_last_token).squeeze(-1)    # [B]
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logit, labels)
        return {"loss": loss, "logits": logit}
