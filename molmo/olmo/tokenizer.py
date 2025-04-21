# import ujson as json
import logging
from os import environ
from typing import List

from transformers import AutoTokenizer

from .torch_util import get_local_rank, barrier
from .util import is_url

try:
    from functools import cache
except ImportError:
    from functools import lru_cache as cache

# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)


class HfTokenizerWrapper:
    """Tokenizer wrapper

    This exists mostly for legacy reasons since we used to support other kinds of tokenizers
    with different API
    """
    def __init__(self, tokenizer, bos_token_id=None, adds_space=False):
        self.adds_space = adds_space
        self.tokenizer = tokenizer
        if bos_token_id is None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            self.bos_token_id = bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_id = -1

    def encode(self, x: str):
        return self.tokenizer.encode(x, add_special_tokens=False)

    def decode(self, x: List[int], truncate_at_eos=True):
        x = [int(t) for t in x]

        if self.eos_token_id == self.bos_token_id and (len(x) > 0 and x[0] == self.eos_token_id):
            # Assume an EOS at the start is functioning as BOS
            x = x[1:]

        if truncate_at_eos:
            # Follow seqio and automatically cut off at EOS
            try:
                eos_ix = x.index(self.eos_token_id)
                x = x[:eos_ix]
            except ValueError:
                pass
        else:
            # Keep our special tokens, but skip BOS/EOS
            x = [t for t in x if t != self.eos_token_id and t != self.bos_token_id]
        return self.tokenizer.decode(x)

    def vocab_size(self):
        return len(self.tokenizer)


def build_tokenizer(
    tokenizer_type, has_extra_token=True,
    tokenizer_dir="gs://mm-olmo/tokenizer",
    pad_tokenizer_to=None,
    memory_cache={}
) -> HfTokenizerWrapper:
    cache_key = (tokenizer_type, has_extra_token, pad_tokenizer_to)
    if cache_key in memory_cache:
        return memory_cache[cache_key]

    cache_dir = None if tokenizer_dir is None or is_url(tokenizer_dir) else tokenizer_dir

    # Stop multiple processes on one node trying to download and cache the tokenizer
    # files, which seems to rarely cause an error
    if get_local_rank() == 0:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_type,
            token=environ.get("HF_ACCESS_TOKEN"),
            cache_dir=cache_dir,
        )
    barrier()

    extra_tokens = list(EXTRA_TOKENS)
    if pad_tokenizer_to is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, token=environ.get("HF_ACCESS_TOKEN"), cache_dir=cache_dir)
        assert len(tokenizer) <= pad_tokenizer_to
        n_extra_tokens = pad_tokenizer_to - len(tokenizer)
        # This handles a case where the LLM embedding matrix is larger than the vocab size
        # We need the extra tokens in `EXTRA_TOKENS` to be assigned id's higher than the embedding
        # matrix size, not the vocab size, since we will concat the embedding and matrix with
        # the special token embedding matrix, so we pad the vocab with additional special tokens
        if n_extra_tokens > 0:
            logging.info(f"Padding tokenizer with {n_extra_tokens} tokens")
            extra_tokens = [f"|<EXTRA_TOKENS_{i}>|" for i in range(n_extra_tokens)] + extra_tokens

    bos_token_id = None

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_type, additional_special_tokens=extra_tokens,
        token=environ.get("HF_ACCESS_TOKEN"),
        cache_dir=cache_dir,
    )
    if ("qwen2" in tokenizer_type.lower()) or ("olmo" in tokenizer_type.lower()):
        # These tokenizers do not have a BOS, and instead use EOS as a generic seperator token.
        # In this case we will use EOS as BOS
        assert tokenizer.bos_token_id is None
        bos_token_id = tokenizer.eos_token_id

    if pad_tokenizer_to is not None:
        for ix, tok in enumerate(EXTRA_TOKENS):
            ids = tokenizer.encode(tok, add_special_tokens=False)
            assert ids == [pad_tokenizer_to + ix]

    tok = HfTokenizerWrapper(tokenizer, bos_token_id=bos_token_id, adds_space=False)
    memory_cache[cache_key] = tok
    return tok


def get_special_token_ids(tokenizer):
    if isinstance(tokenizer, HfTokenizerWrapper):
        ids = tokenizer.encode("".join(EXTRA_TOKENS))
        if len(ids) == len(EXTRA_TOKENS) + 1:
            ids = ids[1:]
    else:
        ids = tokenizer.encode(" ".join(EXTRA_TOKENS))

    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}
