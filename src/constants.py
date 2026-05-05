PAD = 0
BOS = 1
EOS = 2
SEP = 3
TOKENS = {
    "0": 4, "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13,
    "(": 14, ")": 15, "Y": 16, "N": 17,
    "<TASK_COPY>": 18, "<TASK_REVERSE>": 19, "<TASK_PAREN>": 20, "<TASK_ADDMOD>": 21,
}
ID2TOK = {v: k for k, v in TOKENS.items()}
ID2TOK.update({PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", SEP: "<sep>"})
VOCAB_SIZE = max(ID2TOK) + 1

TASK_ORDER = ["copy", "reverse", "paren", "addmod"]
TASK_DEFAULT_STEPS = {
    "copy": 1000,
    "reverse": 2000,
    "paren": 3000,
    "addmod": 1000,
}

def encode_symbols(symbols):
    return [TOKENS[s] for s in symbols]
