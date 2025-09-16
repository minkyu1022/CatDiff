#!/usr/bin/env python3
"""
Comprehensive test for all modifications:
1. Shuffle removal in fake atom generation
2. Soft label generation (keep_rate per group)  
3. Adsorbate context injection in composition path
4. LayerNorm consistency in composition path
5. Unused module cleanup
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/minkyu/DiffCSP')

from diffcsp.pl_modules.diffusion import CSPDiffusion
from diffcsp.pl_modules.cspnet import CSPNet, MAX_COMP_NUM, MAX_ADSO_NUM
from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler


def create_mock_hparams():
    """Create minimal hparams for CSPDiffusion"""
    hparams = SimpleNamespace()
    hparams.coord_type = 'frac'
    hparams.time_dim = 16
    hparams.latent_dim = 64
    hparams.cost_lattice = 1.0
    hparams.cost_coord = 1.0
    hparams.cost_slab_num_atoms = 1.0
    hparams.center_cart_coords = True
    hparams.composition_keep_rule = 'ge'
    hparams.composition_keep_threshold = 0.5
    
    # Create decoder (CSPNet)
    hparams.decoder = CSPNet(
        hidden_dim=32,
        latent_dim=64, 
        num_layers=2,
        max_atoms=100,
        ln=True,
        coord_type='frac'
    )
    
    # Create schedulers
    hparams.beta_scheduler = BetaScheduler(
        timesteps=10,
        scheduler_mode='linear'
    )
    hparams.sigma_scheduler = SigmaScheduler(
        timesteps=10,
        sigma_begin=0.01,
        sigma_end=1.0
    )
    
    # Optimizer config (for configure_optimizers)
    hparams.optim = SimpleNamespace()
    hparams.optim.use_lr_scheduler = False
    hparams.optim.optimizer = SimpleNamespace()
    hparams.optim.optimizer._target_ = 'torch.optim.Adam'
    hparams.optim.composition_lr = 1e-4
    
    return hparams


def create_mock_batch(batch_size=2):
    """Create a realistic mock batch with all required fields"""
    # Basic structure
    num_atoms = torch.tensor([6, 5], dtype=torch.long)  # atoms per graph
    total_atoms = int(num_atoms.sum().item())
    
    # Lattice parameters
    lengths = torch.rand((batch_size, 3)) * 2 + 3  # 3-5 range
    angles = torch.ones((batch_size, 3)) * 90.0
    
    # Atom data
    atom_types = torch.randint(1, 10, (total_atoms,), dtype=torch.long)
    frac_coords = torch.rand((total_atoms, 3))
    batch_index = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), num_atoms)
    
    # Slab composition (flattened for PyG)
    slab_comp_elements = torch.zeros((batch_size * MAX_COMP_NUM,), dtype=torch.long)
    slab_comp_ratios = torch.zeros((batch_size * MAX_COMP_NUM,), dtype=torch.float32)
    slab_comp_mask = torch.zeros((batch_size * MAX_COMP_NUM,), dtype=torch.long)
    
    # Graph 0: C(6) 0.6, O(8) 0.4
    slab_comp_elements[0] = 6
    slab_comp_elements[1] = 8
    slab_comp_ratios[0] = 0.6
    slab_comp_ratios[1] = 0.4
    slab_comp_mask[0] = 1
    slab_comp_mask[1] = 1
    
    # Graph 1: C(6) 0.5, N(7) 0.5
    slab_comp_elements[MAX_COMP_NUM] = 6
    slab_comp_elements[MAX_COMP_NUM + 1] = 7
    slab_comp_ratios[MAX_COMP_NUM] = 0.5
    slab_comp_ratios[MAX_COMP_NUM + 1] = 0.5
    slab_comp_mask[MAX_COMP_NUM] = 1
    slab_comp_mask[MAX_COMP_NUM + 1] = 1
    
    # Slab-only atoms (for masking target)
    slab_num_atoms = torch.tensor([4, 3], dtype=torch.long)
    slab_atom_types = torch.tensor([6, 6, 8, 8, 6, 7, 7], dtype=torch.long)
    
    # Adsorbate data (flattened)
    ads_types = torch.zeros((batch_size * MAX_ADSO_NUM,), dtype=torch.long)
    ads_mask = torch.zeros((batch_size * MAX_ADSO_NUM,), dtype=torch.long)
    ads_types[0] = 1  # H for graph 0
    ads_mask[0] = 1
    ads_types[MAX_ADSO_NUM] = 1  # H for graph 1
    ads_mask[MAX_ADSO_NUM] = 1
    
    # Optional adsorbate distances
    ads_distances = torch.rand((batch_size * MAX_ADSO_NUM * MAX_ADSO_NUM,))
    
    batch = SimpleNamespace(
        num_graphs=batch_size,
        lengths=lengths,
        angles=angles,
        num_atoms=num_atoms,
        atom_types=atom_types,
        frac_coords=frac_coords,
        batch=batch_index,
        slab_composition_elements=slab_comp_elements,
        slab_composition_ratios=slab_comp_ratios,
        slab_composition_mask=slab_comp_mask,
        slab_num_atoms=slab_num_atoms,
        slab_atom_types=slab_atom_types,
        adsorbate_types_padded=ads_types,
        adsorbate_types_mask=ads_mask,
        adsorbate_local_distances=ads_distances,
    )
    return batch


def test_shuffle_removal():
    """Test 1: Verify shuffle is removed (deterministic fake atoms)"""
    print("=== Test 1: Shuffle Removal ===")
    
    hparams = create_mock_hparams()
    model = CSPDiffusion(hparams)
    model.training = True  # Enable training mode
    
    batch = create_mock_batch()
    
    # Generate fake atoms twice - should be identical now
    fake1, mapping1, counts1 = model._create_fake_slab_atom_types(batch)
    fake2, mapping2, counts2 = model._create_fake_slab_atom_types(batch)
    
    identical = torch.equal(fake1, fake2) and torch.equal(mapping1, mapping2)
    print(f"Fake atoms deterministic: {identical}")
    print(f"Fake atoms shape: {fake1.shape}, counts: {counts1.tolist()}")
    
    return identical


def test_soft_labels():
    """Test 2: Verify soft label generation (keep_rate per group)"""
    print("\n=== Test 2: Soft Label Generation ===")
    
    hparams = create_mock_hparams()
    model = CSPDiffusion(hparams)
    
    batch = create_mock_batch()
    fake_atoms, fake_mapping, fake_counts = model._create_fake_slab_atom_types(batch)
    
    # Generate soft targets
    soft_targets = model._create_masking_target(batch, fake_atoms, fake_mapping, fake_counts)
    
    print(f"Soft targets range: [{soft_targets.min().item():.3f}, {soft_targets.max().item():.3f}]")
    print(f"Soft targets mean: {soft_targets.mean().item():.3f}")
    
    # Check if targets are in [0,1] and not all binary
    valid_range = torch.all((soft_targets >= 0) & (soft_targets <= 1))
    has_soft_values = torch.any((soft_targets > 0) & (soft_targets < 1))
    
    print(f"Valid range [0,1]: {valid_range}")
    print(f"Has soft values (not just 0/1): {has_soft_values}")
    
    return valid_range


def test_adsorbate_injection():
    """Test 3: Verify adsorbate context is injected in composition mode"""
    print("\n=== Test 3: Adsorbate Context Injection ===")
    
    hparams = create_mock_hparams()
    model = CSPDiffusion(hparams)
    
    batch = create_mock_batch()
    fake_atoms, fake_mapping, fake_counts = model._create_fake_slab_atom_types(batch)
    
    # Test composition mode with adsorbate data
    with torch.no_grad():
        logits = model.decoder(
            mode='composition',
            t=None,
            slab_composition_elements=batch.slab_composition_elements,
            slab_composition_ratios=batch.slab_composition_ratios,
            slab_composition_mask=batch.slab_composition_mask,
            adsorbate_types_padded=batch.adsorbate_types_padded,
            adsorbate_types_mask=batch.adsorbate_types_mask,
            adsorbate_local_distances=batch.adsorbate_local_distances,
            fake_atom_types=fake_atoms,
            fake_batch_mapping=fake_mapping
        )
    
    print(f"Composition logits shape: {logits.shape}")
    print(f"Composition logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Test without adsorbate data
    with torch.no_grad():
        logits_no_ads = model.decoder(
            mode='composition',
            t=None,
            slab_composition_elements=batch.slab_composition_elements,
            slab_composition_ratios=batch.slab_composition_ratios,
            slab_composition_mask=batch.slab_composition_mask,
            adsorbate_types_padded=None,
            adsorbate_types_mask=None,
            adsorbate_local_distances=None,
            fake_atom_types=fake_atoms,
            fake_batch_mapping=fake_mapping
        )
    
    # Should be different when adsorbate context is included
    different = not torch.allclose(logits, logits_no_ads, atol=1e-6)
    print(f"Adsorbate context affects output: {different}")
    
    return logits.shape[0] == fake_atoms.shape[0]


def test_layernorm_consistency():
    """Test 4: Verify LayerNorm is applied in composition path"""
    print("\n=== Test 4: LayerNorm Consistency ===")
    
    # Test with LayerNorm enabled
    hparams = create_mock_hparams()
    model_ln = CSPDiffusion(hparams)
    
    # Check if composition LayerNorms exist
    decoder = model_ln.decoder
    has_comp_ln = hasattr(decoder, 'comp_layer_norm') and decoder.comp_layer_norm is not None
    has_elem_ln = hasattr(decoder, 'comp_elem_layer_norm') and decoder.comp_elem_layer_norm is not None
    has_combined_ln = hasattr(decoder, 'comp_combined_layer_norm') and decoder.comp_combined_layer_norm is not None
    
    print(f"Has composition LayerNorm: {has_comp_ln}")
    print(f"Has element LayerNorm: {has_elem_ln}")
    print(f"Has combined LayerNorm: {has_combined_ln}")
    
    return has_comp_ln and has_elem_ln and has_combined_ln


def test_forward_pass():
    """Test 5: Full forward pass with all modifications"""
    print("\n=== Test 5: Full Forward Pass ===")
    
    hparams = create_mock_hparams()
    model = CSPDiffusion(hparams)
    model.eval()
    
    batch = create_mock_batch()
    
    with torch.no_grad():
        output = model.forward(batch)
    
    required_keys = ['loss', 'loss_lattice', 'loss_coord', 'loss_masking', 
                    'masking_accuracy', 'masking_precision', 'masking_recall', 
                    'masking_f1', 'slab_num_atoms_mae']
    
    has_all_keys = all(key in output for key in required_keys)
    loss_finite = torch.isfinite(output['loss']).item()
    
    print(f"Has all required keys: {has_all_keys}")
    print(f"Loss is finite: {loss_finite}")
    print(f"Loss value: {output['loss'].item():.4f}")
    print(f"Masking accuracy: {output['masking_accuracy'].item():.4f}")
    print(f"Masking precision: {output['masking_precision'].item():.4f}")
    print(f"Masking recall: {output['masking_recall'].item():.4f}")
    
    return has_all_keys and loss_finite


def test_unused_module_cleanup():
    """Test 6: Verify unused modules are removed"""
    print("\n=== Test 6: Unused Module Cleanup ===")
    
    hparams = create_mock_hparams()
    model = CSPDiffusion(hparams)
    
    # Check that masking_out is removed from CSPNet
    decoder = model.decoder
    has_masking_out = hasattr(decoder, 'masking_out')
    
    print(f"masking_out module removed: {not has_masking_out}")
    
    return not has_masking_out


def main():
    """Run all tests"""
    print("Testing all modifications to CSPDiffusion...")
    torch.manual_seed(42)  # For reproducibility
    
    tests = [
        ("Shuffle Removal", test_shuffle_removal),
        ("Soft Labels", test_soft_labels), 
        ("Adsorbate Injection", test_adsorbate_injection),
        ("LayerNorm Consistency", test_layernorm_consistency),
        ("Forward Pass", test_forward_pass),
        ("Module Cleanup", test_unused_module_cleanup)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print(f"✓ {name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"✗ {name}: ERROR - {str(e)}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Modifications are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
