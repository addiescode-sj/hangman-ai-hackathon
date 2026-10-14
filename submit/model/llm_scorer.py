"""LoRA-merged causal LM scorer: log P(Hangul name | English catalog name) per character (T9.3).

Prompt, name normalization and length normalization match plans/scratch/p9/llm/lora_run.py (fwd direction).
Each query row is scored in batches built from that row's candidates only, so predictions stay row-independent.
"""

from __future__ import annotations

import re

import numpy as np


def ko_norm(name: str) -> str:
    s = re.sub(r"^제\s*(?=\d)", "", name.strip())
    return re.sub(r"\s+", "", re.sub(r"호$", "", s.strip()))


def en_name(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


class LLMScorer:
    def __init__(self, model_dir: str, batch_size: int = 64) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.prompt_cache: dict[str, list[int]] = {}

    def _encode(self, en: str, ko: str) -> tuple[list[int], int]:
        prompt = f"영문 선박명: {en}\n한글 표기: "
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = self.tok(prompt).input_ids
        p = self.prompt_cache[prompt]
        t = self.tok(ko, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        return p + t, len(p)

    def __call__(self, query_name: str, catalog_names: list[object]) -> np.ndarray:
        torch = self.torch
        ko = ko_norm(query_name)
        items = [self._encode(en_name(en), ko) for en in catalog_names]
        out = np.empty(len(items), np.float32)
        with torch.no_grad():
            for b in range(0, len(items), self.batch_size):
                chunk = items[b : b + self.batch_size]
                width = max(len(ids) for ids, _ in chunk)
                ids = torch.full((len(chunk), width), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                labels = torch.full_like(ids, -100)
                for r, (x, start) in enumerate(chunk):
                    ids[r, : len(x)] = torch.tensor(x)
                    att[r, : len(x)] = 1
                    labels[r, start : len(x)] = torch.tensor(x[start:])
                ids, att, labels = ids.to(self.device), att.to(self.device), labels.to(self.device)
                h = self.model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                tgt = labels[:, 1:]
                mask = tgt != -100
                lp = torch.log_softmax(self.model.lm_head(h[mask]).float(), -1).gather(-1, tgt[mask].unsqueeze(-1))
                total = torch.zeros(mask.shape, device=self.device).masked_scatter(mask, lp.squeeze(-1)).sum(-1)
                out[b : b + len(chunk)] = total.cpu().numpy()
        return out / max(1, len(ko))
