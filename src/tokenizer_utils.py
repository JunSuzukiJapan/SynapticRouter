from dataclasses import dataclass


@dataclass
class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(int(t) for t in tokens).decode("utf-8", errors="ignore")


class TiktokenTokenizer:
    def __init__(self, encoding_name: str):
        try:
            import tiktoken
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "The `tiktoken` package is required for tokenizer_type=tiktoken. Install it with `python3 -m pip install tiktoken`."
            ) from exc
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(list(int(t) for t in tokens))


def build_tokenizer(tokenizer_type: str, tokenizer_name: str):
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "tiktoken":
        return TiktokenTokenizer(tokenizer_name)
    raise SystemExit(f"Unsupported tokenizer_type: {tokenizer_type}")
