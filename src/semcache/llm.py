"""Local LLM helpers (Hugging Face transformers): an equivalence verifier and a generator.

The verifier asks a small instruct model whether two user requests can share
the same answer and reads P(yes) from the next-token logits, so one forward
pass is enough (no sampling, deterministic).
"""

from __future__ import annotations

import time

import numpy as np

PROMPT = (
    "Deux demandes d'utilisateurs sont données. Réponds uniquement par Oui ou Non : "
    "ont-elles exactement le même sens, de sorte que la même réponse convienne aux deux ?\n"
    "Attention aux mots inversés, aux négations, aux dates et aux noms qui changent le sens.\n\n"
    "Demande A : {a}\nDemande B : {b}"
)


class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name.split("/")[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self._torch = torch
        self._yes = [
            self.tok.encode(w, add_special_tokens=False)[0] for w in ("Oui", " Oui", "oui")
        ]
        self._no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Non", " Non", "non")]

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def p_same(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                prompts = [
                    self._chat(PROMPT.format(a=a, b=b)) for a, b in pairs[i : i + batch_size]
                ]
                enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                yes = torch.logsumexp(lp[:, self._yes], dim=-1)
                no = torch.logsumexp(lp[:, self._no], dim=-1)
                out.append(torch.sigmoid(yes - no).cpu().numpy())
        return np.concatenate(out)

    def verifier(self, threshold: float = 0.5):
        return lambda a, b: bool(self.p_same([(a, b)])[0] >= threshold)

    def generate(self, question: str, max_new_tokens: int = 160) -> tuple[str, float, int]:
        """Answer a question; returns (text, seconds, generated tokens)."""
        torch = self._torch
        enc = self.tok([self._chat(question)], return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = ids[0, enc["input_ids"].shape[1] :]
        return self.tok.decode(new, skip_special_tokens=True), dt, int(new.shape[0])
