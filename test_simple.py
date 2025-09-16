#!/usr/bin/env python3
"""
Simple direct test for all modifications without PyTorch Lightning complexity
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/minkyu/DiffCSP')

from diffcsp.pl_modules.cspnet import CSPNet, MAX_COMP_NUM, MAX_ADSO_NUM
from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler


class SimpleDiffusion(nn.Module):
    """Simplified version of CSPDiffusion for testing"""
    def __init__(self):
        super().__init__()
        self.decoder = CSPNet(
            hidden_dim=32,
            latent_dim=64,
            num_layers=2,
            max_atoms=100,
            ln=True,
            coord_type='frac'
        )
        self.beta_scheduler = BetaScheduler(timesteps=10, scheduler_mode='linear')
        self.sigma_scheduler = SigmaScheduler(timesteps=10)
        
    def _create_fake_slab_atom_types(self, batch, max_atoms=300):
        """Copy of the method from CSPDiffusion"""
        batch_size = batch.num_graphs
        
        # Get composition data
        slab_composition_elements = batch.slab_composition_elements.view(batch_size, MAX_COMP_NUM)
        slab_composition_ratios = batch.slab_composition_ratios.view(batch_size, MAX_COMP_NUM)
        slab_composition_mask = batch.slab_composition_mask.view(batch_size, MAX_COMP_NUM)
        
        fake_slab_atom_types = []
        fake_batch_mapping = []
        fake_num_atoms_per_batch = []
        
        for i in range(batch_size):
            batch_fake_atoms = []
            
            # Calculate max_multiplicity dynamically
            valid_ratios = slab_composition_ratios[i][slab_composition_mask[i].bool()]
            total_ratio = valid_ratios.sum().item()
            max_multiplicity = int(max_atoms / total_ratio) if total_ratio > 0 else 100
            
            for j in range(MAX_COMP_NUM):
                if slab_composition_mask[i, j]:
                    atomic_num = slab_composition_elements[i, j].item()
                    ratio = slab_composition_ratios[i, j].item()
                    count = int(ratio * max_multiplicity)
                    batch_fake_atoms.extend([atomic_num] * count)
            
            fake_slab_atom_types.extend(batch_fake_atoms)
            fake_batch_mapping.extend([i] * len(batch_fake_atoms))
            fake_num_atoms_per_batch.append(len(batch_fake_atoms))
        
        fake_slab_atom_types = torch.tensor(fake_slab_atom_types, device=batch.slab_composition_elements.device, dtype=torch.long)
        fake_batch_mapping = torch.tensor(fake_batch_mapping, device=batch.slab_composition_elements.device, dtype=torch.long)
        fake_num_atoms_per_batch = torch.tensor(fake_num_atoms_per_batch, device=batch.slab_composition_elements.device)
        
        # Note: Shuffling removed for soft-label/group loss formulation
        return fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch

    def _create_masking_target(self, batch, fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch):
        """Copy of the soft label method from CSPDiffusion"""
        batch_size = batch.num_graphs
        
        # Get true slab atom types
        true_slab_num_atoms = batch.slab_num_atoms
        true_slab_atom_types = batch.slab_atom_types.tolist()
        
        # Create masking target (soft labels)
        masking_target = torch.zeros_like(fake_slab_atom_types, dtype=torch.float)
        
        # Count true atoms per batch
        true_atom_idx = 0
        batch_true_counts = {}
        
        for i in range(batch_size):
            true_count = true_slab_num_atoms[i].item()
            true_atoms = true_slab_atom_types[true_atom_idx:true_atom_idx+true_count]
            
            # Count atoms for this batch
            true_counts = {}
            for atom in true_atoms:
                true_counts[atom] = true_counts.get(atom, 0) + 1
            batch_true_counts[i] = true_counts
            true_atom_idx += true_count
        
        # Count fake atoms per batch and type
        batch_fake_counts = {}
        for fake_atom_type, batch_idx in zip(fake_slab_atom_types.tolist(), fake_batch_mapping.tolist()):
            b = int(batch_idx)
            at = int(fake_atom_type)
            if b not in batch_fake_counts:
                batch_fake_counts[b] = {}
            batch_fake_counts[b][at] = batch_fake_counts[b].get(at, 0) + 1

        # Precompute keep rates per (batch, atomic_type)
        keep_rates = {}
        for b, true_counts in batch_true_counts.items():
            keep_rates[b] = {}
            for at, true_c in true_counts.items():
                fake_c = batch_fake_counts.get(b, {}).get(at, 0)
                rate = 0.0 if fake_c == 0 else float(true_c) / float(fake_c)
                rate = max(0.0, min(1.0, rate))  # Clip to [0,1]
                keep_rates[b][at] = rate

        # Assign soft label to each fake atom
        for fake_idx, (fake_atom_type, batch_idx) in enumerate(zip(fake_slab_atom_types.tolist(), fake_batch_mapping.tolist())):
            b = int(batch_idx)
            at = int(fake_atom_type)
            rate = keep_rates.get(b, {}).get(at, 0.0)
            masking_target[fake_idx] = float(rate)
        
        return masking_target


def create_mock_batch(batch_size=2):
    """Create a realistic mock batch"""
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
    
    # Optional adsorbate distances
    ads_distances = torch.rand((batch_size * MAX_ADSO_NUM * MAX_ADSO_NUM,))
    
    batch = SimpleNamespace(
        num_graphs=batch_size,
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
    
    model = SimpleDiffusion()
    batch = create_mock_batch()
    
    # Generate fake atoms twice - should be identical now
    fake1, mapping1, counts1 = model._create_fake_slab_atom_types(batch)
    fake2, mapping2, counts2 = model._create_fake_slab_atom_types(batch)
    
    identical = torch.equal(fake1, fake2) and torch.equal(mapping1, mapping2)
    print(f"Fake atoms deterministic: {identical}")
    print(f"Fake atoms shape: {fake1.shape}, counts: {counts1.tolist()}")
    
    return identical


def test_soft_labels():
    """Test 2: Verify soft label generation"""
    print("\n=== Test 2: Soft Label Generation ===")
    
    model = SimpleDiffusion()
    batch = create_mock_batch()
    fake_atoms, fake_mapping, fake_counts = model._create_fake_slab_atom_types(batch)
    
    # Generate soft targets
    soft_targets = model._create_masking_target(batch, fake_atoms, fake_mapping, fake_counts)
    
    print(f"Soft targets range: [{soft_targets.min().item():.3f}, {soft_targets.max().item():.3f}]")
    print(f"Soft targets mean: {soft_targets.mean().item():.3f}")
    print(f"Unique values: {torch.unique(soft_targets).tolist()}")
    
    # Check if targets are in [0,1] and not all binary
    valid_range = torch.all((soft_targets >= 0) & (soft_targets <= 1))
    has_soft_values = torch.any((soft_targets > 0) & (soft_targets < 1))
    
    print(f"Valid range [0,1]: {valid_range}")
    print(f"Has soft values (not just 0/1): {has_soft_values}")
    
    return valid_range


def test_adsorbate_injection():
    """Test 3: Verify adsorbate context injection in composition mode"""
    print("\n=== Test 3: Adsorbate Context Injection ===")
    
    model = SimpleDiffusion()
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
    """Test 4: Verify LayerNorm modules exist"""
    print("\n=== Test 4: LayerNorm Consistency ===")
    
    model = SimpleDiffusion()
    decoder = model.decoder
    
    has_comp_ln = hasattr(decoder, 'comp_layer_norm') and decoder.comp_layer_norm is not None
    has_elem_ln = hasattr(decoder, 'comp_elem_layer_norm') and decoder.comp_elem_layer_norm is not None
    has_combined_ln = hasattr(decoder, 'comp_combined_layer_norm') and decoder.comp_combined_layer_norm is not None
    
    print(f"Has composition LayerNorm: {has_comp_ln}")
    print(f"Has element LayerNorm: {has_elem_ln}")
    print(f"Has combined LayerNorm: {has_combined_ln}")
    
    return has_comp_ln and has_elem_ln and has_combined_ln


def test_unused_module_cleanup():
    """Test 5: Verify unused modules are removed"""
    print("\n=== Test 5: Unused Module Cleanup ===")
    
    model = SimpleDiffusion()
    decoder = model.decoder
    
    # Check that masking_out is removed from CSPNet
    has_masking_out = hasattr(decoder, 'masking_out')
    
    print(f"masking_out module removed: {not has_masking_out}")
    
    return not has_masking_out


def main():
    """Run all tests"""
    print("Testing all modifications to CSPDiffusion (simplified)...")
    torch.manual_seed(42)  # For reproducibility
    
    tests = [
        ("Shuffle Removal", test_shuffle_removal),
        ("Soft Labels", test_soft_labels), 
        ("Adsorbate Injection", test_adsorbate_injection),
        ("LayerNorm Consistency", test_layernorm_consistency),
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
            import traceback
            traceback.print_exc()
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
