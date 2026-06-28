"""
text_encoder.py — 冻结的 BERT 文本编码器（用于 txt2shape 的 cross-attention 条件）

参考 SDFusion 的做法：用一个预训练语言模型把文本编码成
token 级别的序列特征 (B, L, D)，再作为 UNet cross-attention 的 context。

这里用 HuggingFace 的 bert-base-uncased，输出维度 D=768。
模型权重全程冻结，不参与训练。
"""

import torch
import torch.nn as nn


class BERTTextEncoder(nn.Module):
    """
    把一个 batch 的字符串编码成 (B, L, 768) 的 token 特征。

    用法：
        enc = BERTTextEncoder(device="cuda")
        context = enc(["a wooden chair", "an office chair with armrests"])
        # context.shape == (2, max_length, 768)
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 max_length: int = 77,
                 device: str = "cuda"):
        super().__init__()
        from transformers import BertTokenizer, BertModel

        self.max_length = max_length
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(device)

        # 冻结所有参数
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False

        self.output_dim = self.bert.config.hidden_size  # 768

    @torch.no_grad()
    def forward(self, text_list):
        """
        text_list : list[str]，长度为 B
        return    : (B, L, 768) 的 float tensor，在 self.device 上
        """
        tokens = self.tokenizer(
            text_list,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        # last_hidden_state: (B, L, 768)
        return outputs.last_hidden_state

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # 记录新的 device
        for a in args:
            if isinstance(a, (str, torch.device)):
                self.device = a
        return self
