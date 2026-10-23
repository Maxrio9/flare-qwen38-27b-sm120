"""Unused KV rows may contain continuation bytes from native cache aliases."""

import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires admitted CUDA"
)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize("batch", (1, 4))
@torch.no_grad()
def test_decode_ignores_uncommitted_kv_rows_without_sanitizing_valid_values(dtype, batch):
    device = torch.device("cuda:0")
    query = torch.randn((batch, 24, 256), device=device, dtype=torch.bfloat16) * .1
    key = torch.zeros((batch * 4, 128, 4, 256), device=device, dtype=dtype)
    value = torch.zeros_like(key)
    seeds = [(torch.randn(key.shape, device=device) * .1).to(dtype) for _ in range(2)]
    table = torch.arange(batch * 4, device=device, dtype=torch.int32).view(batch, 4)
    live = batch if batch == 1 else batch - 1
    lengths = torch.full((batch,), 129, device=device, dtype=torch.int32)
    lengths[live:] = 0
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32).clamp_max(live)
    output = torch.empty_like(query)

    def fill(poison):
        for field, cache, seed in zip(("key", "value"), (key, value), seeds):
            if poison in (field, "both"):
                if dtype == torch.float8_e4m3fn:
                    cache.view(torch.uint8).fill_(0x7f)
                else:
                    cache.fill_(float("nan"))
            else:
                cache.zero_()
            for row in range(live):
                count = int(lengths[row])
                cache.view(-1, 4, 256)[row * 512:row * 512 + count].copy_(
                    seed.view(-1, 4, 256)[row * 512:row * 512 + count]
                )

    with RecordedPreparationSession(device=device, checkpoint="dirty-kv-tail",
            workload_schema=f"decode-{batch}-{dtype}", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(query, key, value, table, lengths, cu, output,
            mode="decode", session=session, workspace=arena, name="attention")
        session.tune()
        compiled = compile_forward(attention)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for count in (1, 33, 127, 128, 129, 257):
            lengths[:live].fill_(count)
            baseline = None
            for poison in ("zero", "key", "value", "both"):
                fill(poison)
                before = [cache.view(torch.uint8).clone() for cache in (key, value)]
                graph.replay()
                torch.cuda.synchronize()
                actual = output[:live].clone()
                assert torch.isfinite(actual).all(), (count, poison)
                if baseline is None:
                    baseline = actual
                    for row in range(live):
                        keys = key.view(-1, 4, 256)[row * 512:row * 512 + count].float().repeat_interleave(6, dim=1)
                        values = value.view(-1, 4, 256)[row * 512:row * 512 + count].float().repeat_interleave(6, dim=1)
                        scores = torch.einsum("hd,thd->ht", query[row].float(), keys) / 16
                        expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
                        # Existing prepared-GQA component bounds, not fitted to this regression.
                        torch.testing.assert_close(actual[row].float(), expected, atol=.002, rtol=.02)
                else:
                    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
                for cache, saved in zip((key, value), before):
                    assert torch.equal(cache.view(torch.uint8), saved), "attention mutated cache storage"
        fill("zero")
        if dtype == torch.float8_e4m3fn:
            value.view(-1, 4, 256)[0].view(torch.uint8).fill_(0x7f)
        else:
            value.view(-1, 4, 256)[0].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.isfinite(output[0]).all(), "valid cache NaNs must remain observable"
        graph.reset()
