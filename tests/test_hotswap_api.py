import os
import sys

import torch

sys.path.append(os.path.abspath("src"))

from src.sra_language_models import MoESRALanguageModel
from src.sra_reference import Router


def test_router_allowed_mask_blocks_disallowed_synapses():
    torch.manual_seed(0)
    router = Router(dim=8, num_synapses=4, k=2)
    h = torch.randn(2, 3, 8)
    allowed = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )

    idx, _, logits = router(h, allowed_mask=allowed)

    for batch_idx in range(h.size(0)):
        allowed_ids = allowed[batch_idx].nonzero().flatten().tolist()
        assert set(idx[batch_idx].flatten().tolist()).issubset(set(allowed_ids))
        assert torch.isneginf(logits[batch_idx, :, ~allowed[batch_idx]]).all()


def test_router_clear_synapses_masks_zeroed_slots():
    torch.manual_seed(1)
    router = Router(dim=8, num_synapses=4, k=2)
    router.clear_synapses([1])
    h = torch.randn(2, 3, 8)

    idx, _, logits = router(h)

    assert torch.isneginf(logits[:, :, 1]).all()
    assert 1 not in idx.flatten().tolist()


def test_language_model_add_clear_pop_synapses_shapes_and_forward():
    torch.manual_seed(2)
    model = MoESRALanguageModel(
        vocab_size=20,
        dim=8,
        layers=1,
        num_synapses=4,
        k=2,
        syn_hidden=16,
        max_seq_len=16,
    )

    model.add_synapses(2, freeze_base=True)
    block = model.blocks[0]

    assert block.num_synapses == 6
    assert block.router.num_synapses == 6
    assert block.router.get_full_emb().shape == (6, 8)
    assert block.get_full_param("w1").shape == (6, 8, 16)
    assert not model.embed.weight.requires_grad
    assert not model.out.weight.requires_grad

    x = torch.randint(0, 20, (2, 5))
    allowed = torch.ones(2, 6, dtype=torch.bool)
    logits, router_logits = model(x, allowed_synapses_mask=allowed)
    assert logits.shape == (2, 5, 20)
    assert router_logits[0].shape == (2, 5, 6)

    model.clear_synapses([1])
    _, router_logits = model(x, allowed_synapses_mask=allowed)
    assert torch.isneginf(router_logits[0][:, :, 1]).all()

    model.pop_synapses(2)
    block = model.blocks[0]
    assert block.num_synapses == 4
    assert block.router.num_synapses == 4
    assert block.router.get_full_emb().shape == (4, 8)
    assert block.get_full_param("w1").shape == (4, 8, 16)
