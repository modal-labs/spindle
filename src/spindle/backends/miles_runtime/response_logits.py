"""Take every row's response logits from one split of the micro-batch logits.

Miles slices each row's response out of the packed ``[tokens, vocab]`` logits.
In backward, each slice becomes a full-size zero tensor with the row's gradient
copied in, and autograd adds those tensors together: one full-size fill and
add per row. A micro-batch of 30 short rows spends more time there than in
the language-model head. A single ``split`` builds the gradient once.
"""

from __future__ import annotations


def install(logit_processors) -> None:
    original = logit_processors._iter_response_chunks
    if getattr(original, "__spindle_split__", False):
        return

    def _iter_response_chunks(
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        max_seq_lens=None,
        include_response_indices,
    ):
        layout = None
        if (
            args.qkv_format == "thd"
            and not args.true_on_policy_mode
            and logits.dim() == 3
            and logits.size(0) == 1
            and logit_processors.get_parallel_state().cp.size == 1
        ):
            layout = split_layout(total_lengths, response_lengths, logits.size(1))
        if layout is None:
            yield from original(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
                include_response_indices=include_response_indices,
            )
            return
        sizes, pieces_of_rows = layout
        pieces = logits.squeeze(0).split(sizes)
        for piece, tokens, response_length in zip(
            pieces_of_rows, unconcat_tokens, response_lengths, strict=False
        ):
            response_tokens = tokens[-response_length:] if response_length else tokens[0:0]
            indices = range(response_length) if include_response_indices else ()
            yield pieces[piece], response_tokens, indices

    _iter_response_chunks.__spindle_split__ = True
    logit_processors._iter_response_chunks = _iter_response_chunks


def split_layout(total_lengths, response_lengths, rows):
    """Split sizes covering ``[0, rows)`` and the piece holding each row's response.

    Row ``i``'s response logits are ``[end_i - response_i - 1, end_i - 1)``, the
    same window Miles slices. Returns None when the windows are not in order
    inside the logits, so the caller keeps Miles' own slicing.
    """
    sizes, pieces_of_rows, cursor, end = [], [], 0, 0
    for total, response in zip(total_lengths, response_lengths, strict=False):
        end += total
        start, stop = end - response - 1, end - 1
        if response == 0:
            start = stop = max(cursor, min(stop, rows))
        if start < cursor or stop > rows or start > stop:
            return None
        if start > cursor:
            sizes.append(start - cursor)
        pieces_of_rows.append(len(sizes))
        sizes.append(stop - start)
        cursor = stop
    if cursor < rows:
        sizes.append(rows - cursor)
    return sizes, pieces_of_rows
