"""C8: rows with different numbers of image tokens (routing helpers + LLaVA DPVR-LF)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpvr.models.routing import causal_padding_mask, gather_rows, pack_index, scatter_rows  # noqa: E402
from dpvr.models.token_diversion_x3_fusion import (  # noqa: E402
    TokenDiversionX3FusionLlava,
    _build_text_only_causal_mask,
)


def test_pack_equals_old_view_for_uniform_rows():
    torch.manual_seed(0)
    B, S, H = 3, 10, 4
    x = torch.randn(B, S, H)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[0, 1:5] = mask[1, 3:7] = mask[2, 6:10] = True
    plan = pack_index(mask)
    assert plan.valid.all()
    assert torch.equal(gather_rows(x, plan), x[mask].view(B, 4, H))
    assert torch.equal(gather_rows(x, pack_index(~mask)), x[~mask].view(B, 6, H))


def test_pack_scatter_roundtrip_ragged():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 5)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[0, 2:4] = True        # 2 tokens
    mask[1, 1:7] = True        # 6 tokens
    plan = pack_index(mask)    # row 2 selects nothing
    assert plan.length == 6
    assert plan.valid.sum(1).tolist() == [2, 6, 0]
    packed = gather_rows(x, plan)
    for b in range(3):
        n = int(mask[b].sum())
        assert torch.equal(packed[b, :n], x[b][mask[b]])
    out = scatter_rows(torch.zeros_like(x), packed, plan)
    assert torch.equal(out[mask], x[mask]) and out[~mask].abs().sum() == 0


def test_gather_mrope_positions():
    pos = torch.arange(3 * 2 * 6).view(3, 2, 6)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    plan = pack_index(mask)
    got = gather_rows(pos, plan, seq_dim=2)
    assert got.shape == (3, 2, 3)
    assert torch.equal(got[:, 1], pos[:, 1, 1:4])
    assert torch.equal(got[:, 0, :2], pos[:, 0, [0, 2]])


def test_causal_padding_mask_matches_old_mask_when_full():
    valid = torch.ones(2, 5, dtype=torch.bool)
    assert torch.equal(causal_padding_mask(valid, torch.float32),
                       _build_text_only_causal_mask(2, 5, torch.device("cpu"), torch.float32))
    m = causal_padding_mask(torch.tensor([[True, True, False]]), torch.float32)
    assert m[0, 0, 2, 2] < -1e30 and m[0, 0, 1, 0] == 0 and torch.isfinite(m).all()


def _llava_with_counts(counts: list[int], S: int = 16):
    """LLaVA DPVR-LF test instance whose mock marks counts[b] image tokens in row b."""
    m = TokenDiversionX3FusionLlava._build_test_instance(hidden_size=64, n_layers=5, n_heads=4,
                                                          split_layer=2, fusion_count=2)
    embed = torch.nn.Embedding(1000, 64)

    def prepare(input_ids, pixel_values=None):
        B, S_ = input_ids.shape
        emb = embed(input_ids).float()
        if pixel_values is None:
            return emb, None
        mask = torch.zeros(B, S_, dtype=torch.bool)
        for b in range(B):
            n = int(pixel_values[b, 0].item())
            mask[b, 2:2 + n] = True
        return emb, mask.unsqueeze(-1).expand(-1, -1, 64)

    m.llava_model.model = types.SimpleNamespace(prepare_multimodal_input=prepare)
    m.eval()
    return m


def test_llava_dpvr_rows_with_different_image_counts():
    counts = [4, 9, 6]
    m = _llava_with_counts(counts)
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (3, 16))
    pix = torch.tensor(counts, dtype=torch.float32).view(3, 1)
    with torch.no_grad():
        batched = m(input_ids=ids, pixel_values=pix).logits
        for b in range(3):
            alone = m(input_ids=ids[b:b + 1], pixel_values=pix[b:b + 1]).logits
            torch.testing.assert_close(batched[b:b + 1], alone, rtol=1e-5, atol=1e-5)
