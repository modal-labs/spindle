from types import SimpleNamespace

from spindle.backends.miles_runtime import mamba_packing


class Params:
    seq_idx = None
    total_tokens = None

    def __post_init__(self):
        self.seq_idx = f"rows of {self.total_tokens} tokens"


class Tokens:
    def size(self, dim):
        return 384


def patched(original):
    model = SimpleNamespace(get_packed_seq_params=original)
    mamba_packing.install(model)
    return model.get_packed_seq_params


def test_packed_rows_get_sequence_indices_over_the_whole_micro_batch():
    get_packed_seq_params = patched(lambda batch, args: Params())
    batch = {"tokens": Tokens()}

    one_rank = SimpleNamespace(context_parallel_size=1)
    assert get_packed_seq_params(batch, one_rank).seq_idx == "rows of 384 tokens"
    four_ranks = SimpleNamespace(context_parallel_size=4)
    assert get_packed_seq_params(batch, four_ranks).seq_idx == "rows of 1536 tokens"
    assert patched(lambda batch, args: None)({}, one_rank) is None
