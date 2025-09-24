import pickle
import regex as re
from collections.abc import Iterable

class BPETokenizer:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_rev = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_order = {merge: idx for idx, merge in enumerate(merges)}
        self.INF = len(self.merges) + 1e5
        if special_tokens is None:
            self.special_tokens = special_tokens
        else:
            # sort the special tokens to prioritize longer tokens
            # such that when splitting, if two speicial tokens overlap,
            # we keep the longer one
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return BPETokenizer(vocab, merges, special_tokens)
    
    def _merge(self, word: tuple[bytes], loc: int):
        assert loc >= 0 and loc + 1 < len(word), "invalid merge location"
        prefix = word[:loc]
        suffix = word[loc+2:]
        new_word = prefix + (b"".join((word[loc], word[loc+1])),) + suffix
        return new_word        


    def _encode_word(self, word: str) -> list[int]:
        # preserve the special tokens
        if self.special_tokens is not None and word in self.special_tokens:
            return [self.vocab_rev[word.encode("utf-8")]]
        
        word = word.encode("utf-8")
        word = tuple([word[i:i+1] for i in range(len(word))])
        while len(word) > 1:
            (merge_loc, (token1, token2)) = min(
                [(idx, (word[idx], word[idx+1])) for idx in range(len(word)-1)],
                key = (
                    lambda pair: 
                        self.merges_order[pair[1]] if pair[1] in self.merges_order
                        else self.INF
                )
            )
            if (token1, token2) not in self.merges_order:
                break
            word = self._merge(word, merge_loc)
        return [self.vocab_rev[token] for token in word]


    def _pre_tokenization(self, text: str) -> list[str]:
        if self.special_tokens is not None and text in self.special_tokens:
            return [text]
        
        words = list[str]()
        for match in re.finditer(self.GPT2_PAT, text):
            word = match.group()
            words.append(word)
        return words
    
    def _split_special_tokens(self, text: str) -> list[str]:
        if self.special_tokens is None:
            return [text]
        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        chunks = re.split(f"({pattern})", text)
        return chunks
        
    def encode(self, text: str) -> list[int]:
        chunks = self._split_special_tokens(text)
        words = list[str]()
        for chunk in chunks:
            words.extend(self._pre_tokenization(chunk))
        
        token_ids = list[int]()
        for word in words:
            token_ids.extend(self._encode_word(word))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id            

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_files(
        "data/TinyStoriesV2-GPT4-valid-vocab.pkl",
        "data/TinyStoriesV2-GPT4-valid-merges.pkl",
    )

    with open("tests/fixtures/tinystories_sample_5M.txt") as f:
        ids = []
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
    print(ids)