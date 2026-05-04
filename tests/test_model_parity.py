import torch
import pytest
import sys
import os
sys.path.append(os.path.abspath('src'))

from src.sra_reference import SRAModel
from src.sra_gpu_models import BatchedSRAModel
from src.constants import VOCAB_SIZE

def copy_weights(ref_model, batched_model, num_synapses):
    # Copy embeddings and output layer
    batched_model.embed.weight.data.copy_(ref_model.embed.weight.data)
    batched_model.pos.data.copy_(ref_model.pos.data)
    batched_model.rel_pos.weight.data.copy_(ref_model.rel_pos.weight.data)
    batched_model.seg.weight.data.copy_(ref_model.seg.weight.data)
    batched_model.out.weight.data.copy_(ref_model.out.weight.data)
    batched_model.out.bias.data.copy_(ref_model.out.bias.data)
    
    # Copy block parameters
    for i in range(len(ref_model.blocks)):
        ref_b = ref_model.blocks[i]
        bat_b = batched_model.blocks[i]
        
        bat_b.norm.weight.data.copy_(ref_b.norm.weight.data)
        bat_b.norm.bias.data.copy_(ref_b.norm.bias.data)
        bat_b.router.synapse_emb.data.copy_(ref_b.router.synapse_emb.data)
        
        for s_idx, ref_syn in enumerate(ref_b.synapses):
            bat_syn = bat_b.synapses
            
            # MHA: qkv
            # ref_syn.attn.in_proj_weight is (3*D, D) -> bat_syn.qkv_w is (N, D, 3*D)
            bat_syn.qkv_w.data[s_idx].copy_(ref_syn.attn.in_proj_weight.data.T)
            if ref_syn.attn.in_proj_bias is not None:
                bat_syn.qkv_b.data[s_idx].copy_(ref_syn.attn.in_proj_bias.data)
                
            bat_syn.out_w.data[s_idx].copy_(ref_syn.attn.out_proj.weight.data.T)
            bat_syn.out_b.data[s_idx].copy_(ref_syn.attn.out_proj.bias.data)
            
            bat_syn.norm1_w.data[s_idx].copy_(ref_syn.norm1.weight.data)
            bat_syn.norm1_b.data[s_idx].copy_(ref_syn.norm1.bias.data)
            
            bat_syn.norm2_w.data[s_idx].copy_(ref_syn.norm2.weight.data)
            bat_syn.norm2_b.data[s_idx].copy_(ref_syn.norm2.bias.data)
            
            bat_syn.w1.data[s_idx].copy_(ref_syn.net[0].weight.data.T)
            bat_syn.b1.data[s_idx].copy_(ref_syn.net[0].bias.data)
            
            bat_syn.w2.data[s_idx].copy_(ref_syn.net[2].weight.data.T)
            bat_syn.b2.data[s_idx].copy_(ref_syn.net[2].bias.data)
            
            bat_syn.state.data[s_idx].copy_(ref_syn.state.data)


def test_batched_sra_parity():
    # Model parameters
    dim = 64
    layers = 2
    num_synapses = 16
    k = 2
    syn_hidden = 128
    
    # Initialize both models
    torch.manual_seed(42)
    ref_model = SRAModel(VOCAB_SIZE, dim, layers, num_synapses, k, syn_hidden)
    
    torch.manual_seed(42)
    bat_model = BatchedSRAModel(VOCAB_SIZE, dim, layers, num_synapses, k, syn_hidden)
    
    # Put in eval mode to disable dropout if any (though there isn't any currently)
    ref_model.eval()
    bat_model.eval()
    
    # Copy weights
    copy_weights(ref_model, bat_model, num_synapses)
    
    # Create random input sequences
    batch_size = 4
    x_len = 10
    y_len = 5
    
    x = torch.randint(0, VOCAB_SIZE, (batch_size, x_len))
    y_in = torch.randint(0, VOCAB_SIZE, (batch_size, y_len))
    
    # Forward pass
    with torch.no_grad():
        out_ref, logits_ref, syn_outs_ref = ref_model(x, y_in)
        out_bat, logits_bat, syn_outs_bat = bat_model(x, y_in)
        
    # Check outputs are close
    assert torch.allclose(out_ref, out_bat, atol=1e-5), "Final outputs do not match"
    
    # Check router logits are close
    for lr, lb in zip(logits_ref, logits_bat):
        assert torch.allclose(lr, lb, atol=1e-5), "Router logits do not match"
        
    # Check individual synapse outputs match
    for layer_ref, layer_bat in zip(syn_outs_ref, syn_outs_bat):
        for sr, sb in zip(layer_ref, layer_bat):
            assert torch.allclose(sr, sb, atol=1e-5), "Synapse outputs do not match"
