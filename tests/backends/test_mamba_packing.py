from types import SimpleNamespace

from spindle.backends.miles_runtime import mamba_packing


class Params:
    seq_idx = None
    total_tokens = None

    def __post_init__(self):
        self.seq_idx = self.total_tokens and f"rows of {self.total_tokens} tokens"


class Tokens:
    def size(self, dim):
        return 384


def test_packed_rows_get_sequence_indices_without_context_parallelism(monkeypatch):
    model = SimpleNamespace(get_packed_seq_params=lambda batch, args: Params())
    mamba_packing.install(model)
    batch = {"tokens": Tokens()}
    whole_rows = SimpleNamespace(context_parallel_size=1)

    assert model.get_packed_seq_params(batch, whole_rows).seq_idx is None

    monkeypatch.setenv(mamba_packing.ENV_FLAG, "1")
    assert (
        model.get_packed_seq_params(batch, whole_rows).seq_idx == "rows of 384 tokens"
    )
    sliced_rows = SimpleNamespace(context_parallel_size=2)
    assert model.get_packed_seq_params(batch, sliced_rows).seq_idx is None
