"""Multi-model routing evaluation harness (NB27).

Routes each query to one of several open LLMs (treated as text-in / text-out
black boxes) and scores the answer with domain-specific, source-agnostic graders.

Unified item schema (same shape for hand-curated seeds and benchmark subsets):
    code: {"domain": "code", "question": str, "entry_point": str, "test": str}
          - `test` defines `check(candidate)` and asserts behaviour (HumanEval style)
    math: {"domain": "math", "question": str, "answer": float}
    qa:   {"domain": "qa",   "question": str, "options": {"A":..,"B":..,..}, "answer": "B"}

Each grader returns {"correct": bool, "parse_ok": bool, ...}. `parse_ok` is recorded
separately so we can tell "model was wrong" apart from "grader could not parse" --
a low parse-ok rate means the grader/instruction is weak, not the model.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Output-format instructions. The motherboard attaches these based on the
# routed domain, so every model answers under the same instruction (fair).
# ---------------------------------------------------------------------------
INSTRUCTIONS = {
    "code": "Return ONLY the function `{entry_point}` inside a single ```python code block. No explanation.",
    "math": "Solve step by step, then give the final answer as \\boxed{{number}}.",
    "qa": "Answer with the single letter (A, B, C, or D) only.",
}


def build_prompt(item: dict) -> str:
    """Wrap a question with its domain-appropriate answer-format instruction."""
    domain = item["domain"]
    instr = INSTRUCTIONS[domain]
    if domain == "code":
        instr = instr.format(entry_point=item["entry_point"])
    return f"{instr}\n\n{item['question']}"


# ---------------------------------------------------------------------------
# Domain graders
# ---------------------------------------------------------------------------
def extract_code(resp: str) -> str:
    """Prefer the last fenced python block; fall back to the whole response."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", resp, re.DOTALL)
    return (blocks[-1] if blocks else resp).strip()


def grade_code(resp: str, item: dict, timeout: int = 8) -> dict:
    """Execute generated code against `item['test']` in an isolated subprocess.

    Safety: model-generated code is run in a separate process with a timeout.
    Acceptable for a teaching/Colab context; harden further (seccomp, container)
    before any untrusted/production use.
    """
    code = extract_code(resp)
    if not code:
        return {"correct": False, "parse_ok": False, "err": "no code extracted"}
    program = f"{code}\n{item['test']}\ncheck({item['entry_point']})\n"
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        proc = subprocess.run(
            ["python3", "-I", path], capture_output=True, text=True, timeout=timeout
        )
        return {
            "correct": proc.returncode == 0,
            "parse_ok": True,
            "err": proc.stderr[-300:] if proc.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"correct": False, "parse_ok": True, "err": "timeout"}
    finally:
        if path:
            os.unlink(path)


def extract_number(resp: str):
    r"""Prefer \boxed{...} (Qwen2.5-Math style); else the last number (GSM8K style)."""
    m = re.search(r"\\boxed\{([^}]*)\}", resp)
    cand = m.group(1) if m else None
    if cand is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", resp)
        if not nums:
            return None
        cand = nums[-1]
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade_math(resp: str, item: dict, tol: float = 1e-6) -> dict:
    pred = extract_number(resp)
    if pred is None:
        return {"correct": False, "parse_ok": False}
    return {"correct": abs(pred - float(item["answer"])) < tol, "parse_ok": True}


def extract_choice(resp: str, options: dict):
    """Multi-tier extraction: explicit 'answer is X' -> '(X)'/'X)' -> option-text -> lone letter."""
    if m := re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", resp, re.IGNORECASE):
        return m.group(1).upper()
    if m := re.search(r"\(?([A-D])[\).:]", resp):
        return m.group(1).upper()
    hits = [k for k, v in options.items() if v and v.lower() in resp.lower()]
    if len(hits) == 1:
        return hits[0]
    if letters := re.findall(r"\b([A-D])\b", resp):
        return letters[-1].upper()
    return None


def grade_qa(resp: str, item: dict) -> dict:
    pred = extract_choice(resp, item["options"])
    if pred is None:
        return {"correct": False, "parse_ok": False}
    return {"correct": pred == item["answer"], "parse_ok": True}


GRADERS = {"code": grade_code, "math": grade_math, "qa": grade_qa}


def grade(item: dict, resp: str) -> dict:
    res = GRADERS[item["domain"]](resp, item)
    res["domain"] = item["domain"]
    return res


# ---------------------------------------------------------------------------
# Zero-shot semantic router (routing space decoupled from model internals)
# ---------------------------------------------------------------------------
class SemanticRouter:
    """Routes a query to the synapse whose specialty description is most similar.

    `encoder` is a callable: list[str] -> np.ndarray of L2-normalized row vectors.
    Inject a SentenceTransformer-backed encoder in the notebook, or a stub in tests.
    """

    def __init__(self, synapse_specialties: dict, encoder):
        self.names = list(synapse_specialties)
        self.encoder = encoder
        self.spec_vecs = np.asarray(encoder([synapse_specialties[n] for n in self.names]))

    def route(self, query: str):
        q = np.asarray(self.encoder([query])[0])
        sims = self.spec_vecs @ q
        best = int(sims.argmax())
        return self.names[best], dict(zip(self.names, sims.tolist()))


def make_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Default encoder backend (lazy import so the module loads without the dep)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts):
        return model.encode(list(texts), normalize_embeddings=True)

    return encode


# ---------------------------------------------------------------------------
# Driver + reporting
# ---------------------------------------------------------------------------
def run_eval(items: list, answer_fn) -> list:
    """answer_fn(item) -> response string. Pass a real LLM in the notebook,
    or a crafted-response stub in tests."""
    records = []
    for item in items:
        records.append(grade(item, answer_fn(item)))
    return records


def summarize(records: list) -> dict:
    """Overall + per-domain accuracy and parse-ok rate."""
    def agg(rs):
        n = len(rs)
        return {
            "n": n,
            "accuracy": sum(r["correct"] for r in rs) / n if n else 0.0,
            "parse_ok_rate": sum(r["parse_ok"] for r in rs) / n if n else 0.0,
        }

    out = {"overall": agg(records)}
    for dom in sorted({r["domain"] for r in records}):
        out[dom] = agg([r for r in records if r["domain"] == dom])
    return out


# ---------------------------------------------------------------------------
# Phase-1 hand-curated seed set (validate plumbing before benchmark swap)
# ---------------------------------------------------------------------------
SEED = [
    {
        "domain": "code",
        "entry_point": "add",
        "question": "Implement `add(a, b)` returning the sum of two integers.",
        "test": "def check(c):\n    assert c(2, 3) == 5\n    assert c(-1, 1) == 0\n",
    },
    {
        "domain": "code",
        "entry_point": "is_even",
        "question": "Implement `is_even(n)` returning True iff n is even.",
        "test": "def check(c):\n    assert c(4) is True\n    assert c(7) is False\n",
    },
    {
        "domain": "code",
        "entry_point": "reverse_str",
        "question": "Implement `reverse_str(s)` returning the reversed string.",
        "test": "def check(c):\n    assert c('abc') == 'cba'\n    assert c('') == ''\n",
    },
    {"domain": "math", "answer": 18, "question": "A box has 3 rows of 6 apples. How many apples in total?"},
    {"domain": "math", "answer": 7, "question": "What is 1 + 2 * 3?"},
    {"domain": "math", "answer": 12, "question": "Tom had 20 marbles and gave away 8. How many remain?"},
    {
        "domain": "qa",
        "answer": "B",
        "question": "What is the capital of France?\n(A) Berlin (B) Paris (C) Rome (D) Madrid",
        "options": {"A": "Berlin", "B": "Paris", "C": "Rome", "D": "Madrid"},
    },
    {
        "domain": "qa",
        "answer": "C",
        "question": "Which planet is known as the Red Planet?\n(A) Venus (B) Jupiter (C) Mars (D) Saturn",
        "options": {"A": "Venus", "B": "Jupiter", "C": "Mars", "D": "Saturn"},
    },
    {
        "domain": "qa",
        "answer": "A",
        "question": "Who wrote 'Romeo and Juliet'?\n(A) Shakespeare (B) Dickens (C) Tolstoy (D) Homer",
        "options": {"A": "Shakespeare", "B": "Dickens", "C": "Tolstoy", "D": "Homer"},
    },
]

SYNAPSE_SPECIALTIES = {
    "coder": "writing and fixing program code, Python functions, algorithms, implementation",
    "math": "solving arithmetic and math word problems, numeric reasoning, calculation",
    "general": "general knowledge questions, facts, history, geography, everyday conversation",
}


# ---------------------------------------------------------------------------
# Phase-2 benchmark loaders -> same unified schema (graders unchanged).
# All three datasets are public (no gating). `datasets` is imported lazily so
# the module stays importable offline.
# ---------------------------------------------------------------------------
def load_humaneval(n: int = 20) -> list:
    """HumanEval -> code items. `question` is the function prompt; `test` defines check()."""
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    n = min(n, len(ds))
    return [
        {
            "domain": "code",
            "question": r["prompt"],
            "entry_point": r["entry_point"],
            "test": r["test"],
        }
        for r in ds.select(range(n))
    ]


def load_gsm8k(n: int = 20) -> list:
    """GSM8K (main/test) -> math items. Gold answer follows '#### '."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))
    items = []
    for r in ds.select(range(n)):
        gold = r["answer"].split("####")[-1].replace(",", "").strip()
        items.append({"domain": "math", "question": r["question"], "answer": float(gold)})
    return items


def load_mmlu(n: int = 20, subject: str = "high_school_geography") -> list:
    """MMLU (a single subject) -> qa items. `choices` is a 4-list; `answer` is index 0-3."""
    from datasets import load_dataset

    letters = "ABCD"
    ds = load_dataset("cais/mmlu", subject, split="test")
    n = min(n, len(ds))
    items = []
    for r in ds.select(range(n)):
        choices = r["choices"]
        options = {letters[i]: c for i, c in enumerate(choices)}
        question = r["question"] + "\n" + " ".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        items.append(
            {"domain": "qa", "question": question, "options": options, "answer": letters[r["answer"]]}
        )
    return items


def load_mixed_benchmark(n_per_domain: int = 20, mmlu_subject: str = "high_school_geography") -> list:
    """Combined mixed-workload eval set across the three domains."""
    return (
        load_humaneval(n_per_domain)
        + load_gsm8k(n_per_domain)
        + load_mmlu(n_per_domain, subject=mmlu_subject)
    )
