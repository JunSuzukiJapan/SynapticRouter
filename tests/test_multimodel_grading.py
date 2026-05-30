"""Phase-1 plumbing validation for the NB27 multi-model routing harness.

Validates the risky parts (answer extraction / grading / router mechanics)
deterministically with crafted responses -- no GPU, no model downloads.
The quality of the real MiniLM router and live LLM generation is validated
later on Colab.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath("src"))

from src.experiments.multimodel_eval import (
    SEED,
    SemanticRouter,
    build_prompt,
    extract_choice,
    extract_number,
    grade,
    grade_code,
    grade_math,
    grade_qa,
    run_eval,
    summarize,
)


# --- code grading ----------------------------------------------------------
ADD_ITEM = {
    "domain": "code",
    "entry_point": "add",
    "question": "...",
    "test": "def check(c):\n    assert c(2, 3) == 5\n",
}


def test_code_correct_in_fenced_block():
    resp = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert grade_code(resp, ADD_ITEM) == {"correct": True, "parse_ok": True, "err": ""}


def test_code_wrong_implementation_parses_but_fails():
    resp = "```python\ndef add(a, b):\n    return a - b\n```"
    res = grade_code(resp, ADD_ITEM)
    assert res["parse_ok"] is True
    assert res["correct"] is False


def test_code_no_block_falls_back_to_whole_response():
    resp = "def add(a, b):\n    return a + b\n"
    assert grade_code(resp, ADD_ITEM)["correct"] is True


def test_code_empty_response_is_parse_failure():
    res = grade_code("", ADD_ITEM)
    assert res["parse_ok"] is False and res["correct"] is False


def test_code_infinite_loop_times_out_not_hangs():
    resp = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    res = grade_code(resp, ADD_ITEM, timeout=3)
    assert res["correct"] is False and res["parse_ok"] is True


# --- math extraction / grading ---------------------------------------------
def test_math_prefers_boxed():
    assert extract_number("messy 999 ... final \\boxed{42}") == 42.0


def test_math_falls_back_to_last_number():
    assert extract_number("first 3 then the answer is 18") == 18.0


def test_math_handles_thousands_separator():
    assert extract_number("\\boxed{1,024}") == 1024.0


def test_math_no_number_is_parse_failure():
    assert grade_math("I am not sure", {"answer": 5}) == {"correct": False, "parse_ok": False}


def test_math_correct_and_wrong():
    assert grade_math("\\boxed{18}", {"answer": 18})["correct"] is True
    assert grade_math("\\boxed{17}", {"answer": 18})["correct"] is False


# --- qa extraction / grading -----------------------------------------------
OPTS = {"A": "Berlin", "B": "Paris", "C": "Rome", "D": "Madrid"}


def test_qa_explicit_answer_phrase():
    assert extract_choice("The answer is B.", OPTS) == "B"


def test_qa_letter_with_paren():
    assert extract_choice("(C) looks right to me", OPTS) == "C"


def test_qa_content_fallback_when_no_letter():
    # model names the city only -- recovered via option text match
    assert extract_choice("It is Paris, the capital of France.", OPTS) == "B"


def test_qa_ambiguous_content_is_not_falsely_matched():
    # two option texts present and no letter -> cannot decide via content tier;
    # falls through to lone-letter tier, which finds none here -> None
    assert extract_choice("Either Paris or Rome could be argued.", OPTS) is None


def test_qa_grades_against_gold():
    item = {"domain": "qa", "options": OPTS, "answer": "B"}
    assert grade_qa("B", item)["correct"] is True
    assert grade_qa("A", item)["correct"] is False


# --- prompt construction ----------------------------------------------------
def test_build_prompt_injects_entry_point():
    assert "`add`" in build_prompt(ADD_ITEM)


def test_build_prompt_math_and_qa_instructions():
    assert "boxed" in build_prompt({"domain": "math", "question": "q", "answer": 1})
    assert "single letter" in build_prompt(
        {"domain": "qa", "question": "q", "options": OPTS, "answer": "A"}
    )


# --- router mechanics (stub encoder, deterministic) ------------------------
def _stub_encoder(keymap):
    """Returns one-hot rows so cosine == exact keyword bucket; validates argmax."""
    def encode(texts):
        vecs = []
        for t in texts:
            v = np.zeros(len(keymap))
            for i, kw in enumerate(keymap):
                if kw in t.lower():
                    v[i] = 1.0
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        return np.array(vecs)

    return encode


def test_router_selects_highest_similarity_synapse():
    specialties = {"coder": "code python", "math": "math number", "general": "history fact"}
    router = SemanticRouter(specialties, _stub_encoder(["code", "number", "history"]))
    assert router.route("write python code")[0] == "coder"
    assert router.route("compute this number")[0] == "math"
    assert router.route("a history question")[0] == "general"


def test_router_returns_score_map():
    specialties = {"coder": "code", "math": "number"}
    router = SemanticRouter(specialties, _stub_encoder(["code", "number"]))
    name, scores = router.route("code please")
    assert name == "coder" and set(scores) == {"coder", "math"}


# --- end-to-end driver over the seed set with an oracle stub ---------------
def test_run_eval_and_summarize_on_seed_oracle():
    """Feed each seed item a known-good answer; everything should pass & parse."""
    def oracle(item):
        d = item["domain"]
        if d == "code":
            sols = {
                "add": "```python\ndef add(a, b):\n    return a + b\n```",
                "is_even": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
                "reverse_str": "```python\ndef reverse_str(s):\n    return s[::-1]\n```",
            }
            return sols[item["entry_point"]]
        if d == "math":
            return f"\\boxed{{{int(item['answer'])}}}"
        return item["answer"]  # qa: the gold letter

    records = run_eval(SEED, oracle)
    summary = summarize(records)
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["overall"]["parse_ok_rate"] == 1.0
    for dom in ("code", "math", "qa"):
        assert summary[dom]["accuracy"] == 1.0


def test_summarize_separates_wrong_from_unparseable():
    records = [
        {"domain": "math", "correct": False, "parse_ok": True},   # wrong
        {"domain": "math", "correct": False, "parse_ok": False},  # unparseable
    ]
    s = summarize(records)["math"]
    assert s["accuracy"] == 0.0 and s["parse_ok_rate"] == 0.5
