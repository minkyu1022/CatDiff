import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT

# Constants
MAX_ATOMIC_NUM = 100
MAX_COMP_NUM = 3
MAX_ADSO_NUM = 11
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "train_loss_epoch"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Coordinate type option: 'frac' or 'cart'
        self.coord_type = getattr(self.hparams, 'coord_type', 'frac')
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, coord_type = self.coord_type, _recursive_=False)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        # Whether to center cartesian coordinates per-graph before model input
        self.center_cart_coords = getattr(self.hparams, 'center_cart_coords', True)
        # Sampling-time rule for composition masking: 'le' -> keep if prob<=0.5 (default), 'ge' -> keep if prob>=0.5
        # self.composition_keep_rule = getattr(self.hparams, 'composition_keep_rule', 'le')
        self.composition_keep_rule = getattr(self.hparams, 'composition_keep_rule', 'ge')
        # Threshold for keep decision at sampling time
        self.composition_keep_threshold = getattr(self.hparams, 'composition_keep_threshold', 0.5)
        # Sampling strategy: 'threshold' (default) or 'topk' (per-type top-K by expected counts)
        self.composition_sampling = getattr(self.hparams, 'composition_sampling', 'threshold')

    def configure_optimizers(self):
        """
        파라미터 그룹을 분리하여 composition 경로 전용 모듈에는 더 작은 lr을 적용.
        - main 그룹: 기본 옵티마이저 lr (예: 1e-3)
        - composition 그룹: 더 작은 lr (hparams.optim.composition_lr, 기본 1e-4)
        스케줄러는 동일하게 옵티마이저 전체를 관리하며, 각 그룹의 lr을 개별적으로 갱신합니다.
        """
        decoder = self.decoder

        # composition 전용/주로 사용되는 모듈을 수집
        comp_modules = []
        if hasattr(decoder, 'atom_to_latent'):
            comp_modules.append(decoder.atom_to_latent)
        if hasattr(decoder, 'atomic_num_embedding'):
            comp_modules.append(decoder.atomic_num_embedding)
        if hasattr(decoder, 'ratio_embedding'):
            comp_modules.append(decoder.ratio_embedding)
        if hasattr(decoder, 'element_context_projection'):
            comp_modules.append(decoder.element_context_projection)
        if hasattr(decoder, 'unified_masking_head'):
            comp_modules.append(decoder.unified_masking_head)
        if hasattr(decoder, 'comp_mlp_layers'):
            # ModuleList: 각 서브 모듈의 파라미터 포함
            comp_modules.extend(list(decoder.comp_mlp_layers))

        comp_param_ids = set()
        comp_params = []
        for module in comp_modules:
            for p in module.parameters(recurse=True):
                pid = id(p)
                if not p.requires_grad:
                    continue
                if pid not in comp_param_ids:
                    comp_param_ids.add(pid)
                    comp_params.append(p)

        # 나머지 파라미터는 main 그룹으로
        main_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in comp_param_ids
        ]

        # 그룹 구성: main은 기본 lr(옵티마이저 설정값), composition은 별도 lr
        comp_lr = getattr(self.hparams.optim, 'composition_lr', 1e-4)
        param_groups = [
            { 'params': main_params },
            { 'params': comp_params, 'lr': comp_lr } if len(comp_params) > 0 else { 'params': [] }
        ]

        # 옵티마이저와 스케줄러 생성 (기존 hydra 설정 재사용)
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=param_groups, _convert_='partial'
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return { 'optimizer': opt, 'lr_scheduler': scheduler, 'monitor': 'train_loss_epoch' }

    def _create_fake_slab_atom_types(self, batch, max_atoms=300):
        """
        Create fake slab atom types by multiplying composition ratios by max_multiplicity.
        max_multiplicity is calculated as max_atoms / sum_of_ratios for each batch.
        Returns fake_slab_atom_types and corresponding batch mapping.
        """
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
            
            # Calculate max_multiplicity dynamically: max_atoms / sum of valid ratios
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
        """
        Create masking target based on true slab atom types vs fake slab atom types.
        Now works with shuffled fake atoms by using batch mapping.
        Returns binary mask indicating which atoms should be kept (0) or removed (1).
        """
        batch_size = batch.num_graphs
        
        # Get true slab atom types from the dedicated field
        true_slab_num_atoms = batch.slab_num_atoms
        true_slab_atom_types = batch.slab_atom_types.tolist()
        
        # Create masking target (soft labels)
        # keep label in [0,1] as per-group keep rate: true_count / fake_count
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
                # Clip to [0,1] for safety
                rate = max(0.0, min(1.0, rate))
                keep_rates[b][at] = rate

        # Assign soft label to each fake atom
        for fake_idx, (fake_atom_type, batch_idx) in enumerate(zip(fake_slab_atom_types.tolist(), fake_batch_mapping.tolist())):
            b = int(batch_idx)
            at = int(fake_atom_type)
            rate = keep_rates.get(b, {}).get(at, 0.0)
            masking_target[fake_idx] = float(rate)
        
        return masking_target

    def _apply_masking_to_generate_system_atom_types(self, batch, fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch, pred_masking_logits):
        """
        Apply predicted masking to fake slab atom types to generate final system atom types.
        """
        batch_size = batch.num_graphs
        total_fake_atoms = fake_slab_atom_types.size(0)
        
        # pred_masking_logits is now directly [num_fake_atoms]
        # Apply masking with configurable rule
        probs = torch.sigmoid(pred_masking_logits)
        thr = float(getattr(self, 'composition_keep_threshold', 0.5))
        sampling_mode = getattr(self, 'composition_sampling', 'threshold')
        sampling_mode = 'topk'
        if sampling_mode == 'topk':
            # Per-type top-K selection based on expected counts (sum of probs)
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            # Build per-batch, per-type indices
            for b in range(batch_size):
                b_mask = (fake_batch_mapping == b)
                if not torch.any(b_mask):
                    continue
                types_b = fake_slab_atom_types[b_mask]
                probs_b = probs[b_mask]
                # True counts per type from batch.slab_num_atoms/slab_atom_types
                # Build true counts map for this batch
                # Note: we rely on true_slab counts provided in forward()
                # For sampling, approximate target K using expected counts per type: sum of probs per type
                unique_types = torch.unique(types_b)
                for at in unique_types.tolist():
                    at_mask = b_mask & (fake_slab_atom_types == at)
                    probs_at = probs[at_mask]
                    if probs_at.numel() == 0:
                        continue
                    # expected count rounded to nearest int, clipped to group size
                    k = int(torch.clamp(probs_at.sum().round(), min=0, max=probs_at.numel()).item())
                    if k > 0:
                        topk_vals, topk_idx = torch.topk(probs_at, k)
                        # Map subgroup indices to global indices
                        global_idx = at_mask.nonzero(as_tuple=False).squeeze(1)
                        keep_mask[global_idx[topk_idx]] = True
            try:
                keep_total = int(keep_mask.sum().item())
                keep_per_graph = [int(keep_mask[fake_batch_mapping == i].sum().item()) for i in range(batch_size)]
                print(f"DEBUG[mask]: sampling=topk | keep_total={keep_total} | keep_per_graph={keep_per_graph}")
            except Exception:
                pass
        else:
            # Threshold-based selection (original)
            if getattr(self, 'composition_keep_rule', 'le') == 'ge':
                keep_mask = probs >= thr
                try:
                    # Debug: keep counts after rule applied
                    keep_total = int(keep_mask.sum().item())
                    keep_per_graph = []
                    for i in range(batch_size):
                        keep_per_graph.append(int((keep_mask & (fake_batch_mapping == i)).sum().item()))
                    print(f"DEBUG[mask]: rule=ge thr={thr:.3f} | keep_total={keep_total} | keep_per_graph={keep_per_graph}")
                except Exception:
                    pass
            else:
                keep_mask = probs <= thr
                try:
                    keep_total = int(keep_mask.sum().item())
                    keep_per_graph = []
                    for i in range(batch_size):
                        keep_per_graph.append(int((keep_mask & (fake_batch_mapping == i)).sum().item()))
                    print(f"DEBUG[mask]: rule=le thr={thr:.3f} | keep_total={keep_total} | keep_per_graph={keep_per_graph}")
                except Exception:
                    pass
        kept_slab_atom_types = fake_slab_atom_types[keep_mask]
        kept_batch_mapping = fake_batch_mapping[keep_mask]
        
        # Add adsorbate atoms
        adsorbate_types_padded = batch.adsorbate_types_padded.view(batch_size, MAX_ADSO_NUM)
        adsorbate_types_mask = batch.adsorbate_types_mask.view(batch_size, MAX_ADSO_NUM)
        
        # Generate final system atom types
        predicted_system_atom_types = []
        predicted_num_atoms = []
        
        for i in range(batch_size):
            # Get slab atoms for this batch
            batch_mask = kept_batch_mapping == i
            batch_slab_atoms = kept_slab_atom_types[batch_mask].tolist()
            
            # Get adsorbate atoms for this batch
            batch_adsorbate_atoms = []
            for j in range(MAX_ADSO_NUM):
                if adsorbate_types_mask[i, j]:
                    atomic_num = adsorbate_types_padded[i, j].item()
                    batch_adsorbate_atoms.append(atomic_num)
            
            # Combine slab + adsorbate
            system_atom_types = batch_slab_atoms + batch_adsorbate_atoms
            predicted_system_atom_types.extend(system_atom_types)
            predicted_num_atoms.append(len(system_atom_types))
        
        predicted_system_atom_types = torch.tensor(
            predicted_system_atom_types, 
            device=batch.slab_composition_elements.device, 
            dtype=torch.long
        )
        predicted_num_atoms = torch.tensor(predicted_num_atoms, device=batch.slab_composition_elements.device)
        
        return predicted_system_atom_types, predicted_num_atoms

    def _generate_system_atom_types_vectorized(self, batch, pred_slab_num_atoms):
        """
        Fully vectorized implementation using padded data and valid masks
        """
        batch_size = pred_slab_num_atoms.size(0)
        
        # DEBUG: Print batch information
        print(f"DEBUG: batch_size = {batch_size}")
        print(f"DEBUG: pred_slab_num_atoms = {pred_slab_num_atoms}")
        print(f"DEBUG: pred_slab_num_atoms.shape = {pred_slab_num_atoms.shape}")
        
        # Get padded composition data
        # PyTorch Geometric flattens these during batching, so we need to reshape
        slab_composition_elements = batch.slab_composition_elements.view(batch_size, MAX_COMP_NUM)  # [batch_size, MAX_COMP_NUM]
        slab_composition_ratios = batch.slab_composition_ratios.view(batch_size, MAX_COMP_NUM)      # [batch_size, MAX_COMP_NUM]
        slab_composition_mask = batch.slab_composition_mask.view(batch_size, MAX_COMP_NUM)          # [batch_size, MAX_COMP_NUM]
        
        # DEBUG: Print slab composition data shapes
        print(f"DEBUG: slab_composition_elements = {slab_composition_elements}")
        print(f"DEBUG: slab_composition_elements.shape = {slab_composition_elements.shape}")
        print(f"DEBUG: slab_composition_elements.size() = {slab_composition_elements.size()}")
        print(f"DEBUG: slab_composition_ratios.shape = {slab_composition_ratios.shape}")
        print(f"DEBUG: slab_composition_mask.shape = {slab_composition_mask.shape}")
        
        # Get padded adsorbate data
        # PyTorch Geometric flattens these during batching, so we need to reshape
        adsorbate_types_padded = batch.adsorbate_types_padded.view(batch_size, MAX_ADSO_NUM)        # [batch_size, MAX_ADSO_NUM]
        adsorbate_types_mask = batch.adsorbate_types_mask.view(batch_size, MAX_ADSO_NUM)            # [batch_size, MAX_ADSO_NUM]
        
        # DEBUG: Print adsorbate data shapes
        print(f"DEBUG: adsorbate_types_padded.shape = {adsorbate_types_padded.shape}")
        print(f"DEBUG: adsorbate_types_mask.shape = {adsorbate_types_mask.shape}")
        
        # Vectorized atom count calculation
        pred_slab_num_atoms_expanded = pred_slab_num_atoms.unsqueeze(-1)  # [batch_size, 1]
        atom_counts_float = slab_composition_ratios * pred_slab_num_atoms_expanded  # [batch_size, MAX_COMP_NUM]
        atom_counts = atom_counts_float.round().long()  # [batch_size, MAX_COMP_NUM]
        
        # Apply valid mask (zero out padded elements)
        atom_counts = atom_counts * slab_composition_mask  # [batch_size, MAX_COMP_NUM]
        
        # Handle rounding errors vectorized
        total_counts = atom_counts.sum(dim=1)  # [batch_size]
        diff = pred_slab_num_atoms - total_counts  # [batch_size]
        
        # For each batch, adjust the most abundant valid element
        masked_counts = atom_counts * slab_composition_mask + (1 - slab_composition_mask) * (-1000)
        max_indices = torch.argmax(masked_counts, dim=1)  # [batch_size]
        atom_counts[torch.arange(batch_size), max_indices] += diff
        atom_counts = torch.clamp(atom_counts, min=0)
        
        # Generate system atom types
        predicted_system_atom_types = []
        predicted_num_atoms = []
        
        for i in range(batch_size):
            # Generate slab atom types using valid mask
            slab_atom_types = []
            for j in range(slab_composition_elements.size(1)):  # MAX_COMP_NUM
                if slab_composition_mask[i, j]:  # Only process valid elements
                    atomic_num = slab_composition_elements[i, j].item()
                    count = atom_counts[i, j].item()
                    slab_atom_types.extend([atomic_num] * count)
            
            # Generate adsorbate atom types using valid mask
            adsorbate_atom_types = []
            for j in range(adsorbate_types_padded.size(1)):  # MAX_ADSO_NUM
                if adsorbate_types_mask[i, j]:  # Only process valid elements
                    atomic_num = adsorbate_types_padded[i, j].item()
                    adsorbate_atom_types.append(atomic_num)
            
            # Combine slab + adsorbate
            system_atom_types = slab_atom_types + adsorbate_atom_types
            predicted_system_atom_types.extend(system_atom_types)
            predicted_num_atoms.append(len(system_atom_types))
        
        predicted_system_atom_types = torch.tensor(
            predicted_system_atom_types, 
            device=batch.slab_composition_elements.device, 
            dtype=torch.long
        )
        predicted_num_atoms = torch.tensor(predicted_num_atoms, device=pred_slab_num_atoms.device)
        
        return predicted_system_atom_types, predicted_num_atoms

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        
        if self.coord_type == 'cart':
            # Convert fractional coordinates to cartesian coordinates
            coords = frac_to_cart_coords(frac_coords, batch.lengths, batch.angles, batch.num_atoms, lattices=lattices)
        else:
            # Use fractional coordinates directly
            coords = frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        
        if self.coord_type == 'cart':
            input_coords = coords + sigmas_per_atom * rand_x
        else:
            input_coords = (coords + sigmas_per_atom * rand_x) % 1.

        if self.keep_coords:
            input_coords = coords

        if self.keep_lattice:
            input_lattice = lattices

        # Center cartesian coordinates per-graph (translation invariance)
        if self.coord_type == 'cart' and self.center_cart_coords:
            graph_means = scatter(input_coords, batch.batch, dim=0, reduce='mean', dim_size=batch.num_graphs)
            input_coords = input_coords - graph_means[batch.batch]

        # 1. Create fake slab atom types and masking target
        fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch = self._create_fake_slab_atom_types(batch)
        masking_target = self._create_masking_target(batch, fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch)
        
        # 2. Composition prediction - now predicts masking probabilities per fake atom
        pred_masking_logits = self.decoder(
            mode='composition',
            t=None,
            slab_composition_elements=batch.slab_composition_elements,
            slab_composition_ratios=batch.slab_composition_ratios,
            slab_composition_mask=batch.slab_composition_mask,
            slab_num_atoms=batch.slab_num_atoms,
            adsorbate_types_padded=batch.adsorbate_types_padded,
            adsorbate_types_mask=batch.adsorbate_types_mask,
            adsorbate_local_distances=getattr(batch, 'adsorbate_local_distances', None),
            fake_atom_types=fake_slab_atom_types,
            fake_batch_mapping=fake_batch_mapping
        )

        # 3. Diffusion prediction
        pred_l, pred_x = self.decoder(
            mode='diffusion',
            t=time_emb,
            atom_types=batch.atom_types,
            coords=input_coords,
            lattices=input_lattice,
            num_atoms=batch.num_atoms,
            slab_composition_elements=batch.slab_composition_elements,
            slab_composition_ratios=batch.slab_composition_ratios,
            slab_composition_mask=batch.slab_composition_mask,
            slab_num_atoms=batch.slab_num_atoms,
            adsorbate_types_padded=batch.adsorbate_types_padded,
            adsorbate_types_mask=batch.adsorbate_types_mask,
            adsorbate_local_distances=getattr(batch, 'adsorbate_local_distances', None),
            node2graph=batch.batch
        )

        # Calculate target based on coordinate type
        if self.coord_type == 'cart':
            # For cartesian coordinates, use the actual cartesian noise
            # The model predicts the score of the cartesian coordinate distribution
            tar_x = rand_x / torch.sqrt(sigmas_norm_per_atom)
        else:
            # Use wrapped normal for fractional coordinates
            tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        # Calculate losses
        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        
        # Masking loss - direct per-atom prediction
        # pred_masking_logits is now [num_fake_atoms] instead of [batch_size, max_atoms]
        
        # Binary cross entropy loss for masking with soft labels (no pos_weight)
        loss_masking = F.binary_cross_entropy_with_logits(
            pred_masking_logits, masking_target
        )
        
        # Metrics adapted for soft targets (probabilistic precision/recall/F1)
        probs = torch.sigmoid(pred_masking_logits)
        # accuracy proxy: 1 - MAE between probs and soft targets
        masking_mae = torch.abs(probs - masking_target).mean()
        masking_accuracy = 1.0 - masking_mae
        # distribution ratios (soft)
        masking_target_keep_ratio = masking_target.mean()
        masking_target_remove_ratio = 1.0 - masking_target.mean()
        # soft tp/fp/fn
        tp_soft = (probs * masking_target).sum()
        fp_soft = (probs * (1.0 - masking_target)).sum()
        fn_soft = ((1.0 - probs) * masking_target).sum()
        masking_precision = tp_soft / (tp_soft + fp_soft + EPSILON)
        masking_recall = tp_soft / (tp_soft + fn_soft + EPSILON)
        masking_f1 = 2.0 * masking_precision * masking_recall / (masking_precision + masking_recall + EPSILON)
        
        # Calculate predicted slab_num_atoms from masking using expected counts (sum of probabilities)
        batch_size = batch.num_graphs
        pred_slab_num_atoms_from_masking = torch.zeros(batch_size, device=batch.slab_composition_elements.device)
        for i in range(batch_size):
            batch_mask = fake_batch_mapping == i
            pred_slab_num_atoms_from_masking[i] = probs[batch_mask].sum()
        
        slab_num_atoms_mae = torch.abs(pred_slab_num_atoms_from_masking - batch.slab_num_atoms.float()).mean()

        # Per-type metrics (disabled by request)
        # true_slab_num_atoms = batch.slab_num_atoms
        # true_slab_atom_types_list = batch.slab_atom_types.tolist()
        # batch_true_counts = {}
        # true_atom_idx = 0
        # for i in range(batch_size):
        #     true_count = int(true_slab_num_atoms[i].item())
        #     atoms_i = true_slab_atom_types_list[true_atom_idx:true_atom_idx + true_count]
        #     true_atom_idx += true_count
        #     count_map = {}
        #     for at in atoms_i:
        #         count_map[at] = count_map.get(at, 0) + 1
        #     batch_true_counts[i] = count_map
        # per_type_count_abs_errors = []
        # topk_overlaps = []
        # for b in range(batch_size):
        #     b_mask = (fake_batch_mapping == b)
        #     if not torch.any(b_mask):
        #         continue
        #     b_types = fake_slab_atom_types[b_mask]
        #     b_probs = probs[b_mask]
        #     unique_fake_types = set(b_types.tolist())
        #     unique_true_types = set(batch_true_counts.get(b, {}).keys())
        #     all_types = unique_fake_types.union(unique_true_types)
        #     for at in all_types:
        #         true_c = int(batch_true_counts.get(b, {}).get(at, 0))
        #         at_mask = (b_types == at)
        #         fake_c = int(at_mask.sum().item())
        #         expected_c = float(b_probs[at_mask].sum().item()) if fake_c > 0 else 0.0
        #         per_type_count_abs_errors.append(abs(expected_c - float(true_c)))
        #         if true_c > 0 and fake_c > 0:
        #             k = min(true_c, fake_c)
        #             b_probs_at = b_probs[at_mask]
        #             topk_vals, topk_idx = torch.topk(b_probs_at, k)
        #             target_idx = torch.arange(k, device=b_probs_at.device)
        #             overlap = len(set(topk_idx.tolist()).intersection(set(target_idx.tolist())))
        #             topk_overlaps.append(overlap / float(k))
        # per_type_count_mae = torch.tensor(per_type_count_abs_errors, device=masking_target.device).mean() if len(per_type_count_abs_errors) > 0 else torch.tensor(0.0, device=masking_target.device)
        # per_type_topk_acc = torch.tensor(topk_overlaps, device=masking_target.device).mean() if len(topk_overlaps) > 0 else torch.tensor(0.0, device=masking_target.device)

        # Total loss with masking weight
        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord +
            self.hparams.cost_slab_num_atoms * loss_masking)  # Use masking loss instead

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_masking' : loss_masking,  # Changed from loss_slab_num_atoms
            'masking_accuracy' : masking_accuracy,  # Changed from slab_num_atoms_accuracy
            'masking_precision' : masking_precision,
            'masking_recall' : masking_recall,
            'masking_f1' : masking_f1,
            'masking_target_keep_ratio' : masking_target_keep_ratio,
            'masking_target_remove_ratio' : masking_target_remove_ratio,
            'slab_num_atoms_mae' : slab_num_atoms_mae  # Keep for comparison
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        # Step 1: Create fake slab atom types and predict masking
        fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch = self._create_fake_slab_atom_types(batch)
        # Debug: composition mask and fake atom stats
        try:
            mask_reshaped = batch.slab_composition_mask.view(batch_size, MAX_COMP_NUM)
            mask_sum_per_graph = mask_reshaped.sum(dim=1)
            print(
                f"DEBUG[sample]: comp_mask_sums={mask_sum_per_graph.tolist()} | "
                f"num_fake_atoms={int(fake_slab_atom_types.numel())} | "
                f"fake_atoms_per_graph={fake_num_atoms_per_batch.tolist()}"
            )
        except Exception as e:
            print(f"DEBUG[sample]: failed to compute comp-mask/fake stats due to: {e}")
        
        pred_masking_logits = self.decoder(
            mode='composition',
            t=None,
            slab_composition_elements=batch.slab_composition_elements,
            slab_composition_ratios=batch.slab_composition_ratios,
            slab_composition_mask=batch.slab_composition_mask,
            adsorbate_types_padded=batch.adsorbate_types_padded,
            adsorbate_types_mask=batch.adsorbate_types_mask,
            adsorbate_local_distances=getattr(batch, 'adsorbate_local_distances', None),
            fake_atom_types=fake_slab_atom_types,
            fake_batch_mapping=fake_batch_mapping
        )
        # Debug: masking logits/probs stats, threshold/rule, per-graph expected keep counts
        try:
            probs = torch.sigmoid(pred_masking_logits)
            rule = getattr(self, 'composition_keep_rule', 'le')
            thr = float(getattr(self, 'composition_keep_threshold', 0.5))
            q10, q50, q90 = torch.quantile(probs, torch.tensor([0.1, 0.5, 0.9], device=probs.device)).tolist()
            print(
                f"DEBUG[sample]: probs mean={float(probs.mean().item()):.4f} min={float(probs.min().item()):.4f} "
                f"q10/50/90={q10:.4f}/{q50:.4f}/{q90:.4f} max={float(probs.max().item()):.4f} | rule={rule} thr={thr:.3f}"
            )
            # Per-graph expected keep (sum of probabilities over fake atoms)
            expected_keep = []
            for i in range(batch_size):
                mask_i = (fake_batch_mapping == i)
                expected_keep.append(float(probs[mask_i].sum().item()))
            true_slab = [int(n.item()) for n in batch.slab_num_atoms]
            print(f"DEBUG[sample]: expected_keep_per_graph={expected_keep} | true_slab_num_atoms={true_slab}")
        except Exception as e:
            print(f"DEBUG[sample]: failed to compute masking stats due to: {e}")
        
        # Step 2: Apply masking to get final slab atom types
        predicted_atom_types, predicted_num_atoms = self._apply_masking_to_generate_system_atom_types(
            batch, fake_slab_atom_types, fake_batch_mapping, fake_num_atoms_per_batch, pred_masking_logits
        )
        
        # Update batch for consistency
        batch_updated = batch.clone()
        batch_updated.atom_types = predicted_atom_types
        batch_updated.num_atoms = predicted_num_atoms
        batch_updated.num_nodes = predicted_atom_types.size(0)
        
        # Create new batch mapping
        batch_updated.batch = torch.repeat_interleave(torch.arange(batch_size, device=batch.batch.device), predicted_num_atoms)

        # Step 3: Initialize diffusion sampling with updated dimensions
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        if self.coord_type == 'cart':
            x_T = torch.randn([predicted_atom_types.size(0), 3]).to(self.device)  # Use new total atom count
        else:
            x_T = torch.rand([predicted_atom_types.size(0), 3]).to(self.device)   # Use new total atom count

        if self.keep_coords:
            if self.coord_type == 'cart':
                # Convert fractional coordinates to cartesian coordinates
                x_T = frac_to_cart_coords(batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms, lattices=lattice_params_to_matrix_torch(batch.lengths, batch.angles))
            else:
                # Use fractional coordinates directly
                x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : predicted_num_atoms,  # Use predicted num_atoms
            'atom_types' : predicted_atom_types,  # Use predicted atom types
            'coords' : x_T if self.coord_type == 'cart' else x_T % 1.,
            'lattices' : l_T,
            'pred_masking_logits' : pred_masking_logits  # Store masking prediction
        }}

        # Step 4: Iterative denoising with predicted atom_types
        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # Center coords per-graph before model input, without mutating x_t
            coords_for_model = x_t
            if self.coord_type == 'cart' and self.center_cart_coords:
                graph_means = scatter(coords_for_model, batch_updated.batch, dim=0, reduce='mean', dim_size=batch_size)
                coords_for_model = coords_for_model - graph_means[batch_updated.batch]

            pred_l, pred_x = self.decoder(
                mode='diffusion',
                t=time_emb,
                atom_types=predicted_atom_types,  # Use predicted atom types
                coords=coords_for_model,
                lattices=l_t,
                num_atoms=predicted_num_atoms,  # Use predicted num_atoms
                slab_composition_elements=batch.slab_composition_elements,
                slab_composition_ratios=batch.slab_composition_ratios,
                slab_composition_mask=batch.slab_composition_mask,
                slab_num_atoms=batch.slab_num_atoms,
                adsorbate_types_padded=batch.adsorbate_types_padded,
                adsorbate_types_mask=batch.adsorbate_types_mask,
                adsorbate_local_distances=getattr(batch, 'adsorbate_local_distances', None),
                node2graph=batch_updated.batch  # Use updated batch mapping
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # Center coords per-graph before model input, without mutating x_t_minus_05
            coords_for_model = x_t_minus_05
            if self.coord_type == 'cart' and self.center_cart_coords:
                graph_means = scatter(coords_for_model, batch_updated.batch, dim=0, reduce='mean', dim_size=batch_size)
                coords_for_model = coords_for_model - graph_means[batch_updated.batch]

            pred_l, pred_x = self.decoder(
                mode='diffusion',
                t=time_emb,
                atom_types=predicted_atom_types,  # Use predicted atom types
                coords=coords_for_model,
                lattices=l_t_minus_05,
                num_atoms=predicted_num_atoms,  # Use predicted num_atoms
                slab_composition_elements=batch.slab_composition_elements,
                slab_composition_ratios=batch.slab_composition_ratios,
                slab_composition_mask=batch.slab_composition_mask,
                slab_num_atoms=batch.slab_num_atoms,
                adsorbate_types_padded=batch.adsorbate_types_padded,
                adsorbate_types_mask=batch.adsorbate_types_mask,
                adsorbate_local_distances=getattr(batch, 'adsorbate_local_distances', None),
                node2graph=batch_updated.batch  # Use updated batch mapping
            )

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                'num_atoms' : predicted_num_atoms,  # Keep predicted num_atoms
                'atom_types' : predicted_atom_types,  # Keep predicted atom types
                'coords' : x_t_minus_1 if self.coord_type == 'cart' else x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'pred_masking_logits' : pred_masking_logits              
            }

        traj_stack = {
            'num_atoms' : predicted_num_atoms,  # Use predicted num_atoms
            'atom_types' : predicted_atom_types,  # Use predicted atom types
            'all_coords' : torch.stack([traj[i]['coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)]),
            'pred_masking_logits' : pred_masking_logits
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_masking = output_dict['loss_masking']  # Changed from loss_slab_num_atoms
        masking_accuracy = output_dict['masking_accuracy']  # Changed from slab_num_atoms_accuracy
        masking_precision = output_dict['masking_precision']
        masking_recall = output_dict['masking_recall']
        masking_f1 = output_dict['masking_f1']
        masking_target_keep_ratio = output_dict['masking_target_keep_ratio']
        masking_target_remove_ratio = output_dict['masking_target_remove_ratio']
        slab_num_atoms_mae = output_dict['slab_num_atoms_mae']
        loss = output_dict['loss']
        # batch_len = batch.num_graphs
        # print(f"Batch info: num_graphs={batch.num_graphs}, total_atoms={batch.num_nodes}, atoms_per_graph={batch.num_atoms.tolist()[:10]}...")


        # Step + Epoch logging (fast metrics)
        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'masking_loss': loss_masking,
            'masking_accuracy': masking_accuracy,
            'masking_precision': masking_precision,
            'masking_recall': masking_recall,
            'masking_f1': masking_f1,
            'masking_target_keep_ratio': masking_target_keep_ratio,
            'masking_target_remove_ratio': masking_target_remove_ratio,
            'slab_num_atoms_mae': slab_num_atoms_mae},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        # Epoch-only logging (disabled metrics)
        # self.log('per_type_count_mae', output_dict['per_type_count_mae'], on_step=False, on_epoch=True, prog_bar=False)
        # self.log('per_type_topk_acc', output_dict['per_type_topk_acc'], on_step=False, on_epoch=True, prog_bar=False)

        # Log composition lr (param group 1) if available
        if hasattr(self, 'trainer') and self.trainer is not None:
            optimizers = getattr(self.trainer, 'optimizers', None)
            if optimizers and len(optimizers) > 0:
                param_groups = getattr(optimizers[0], 'param_groups', None)
                if param_groups and len(param_groups) > 1 and 'lr' in param_groups[1]:
                    self.log('composition_lr', param_groups[1]['lr'], on_step=True, on_epoch=False, prog_bar=False)

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_masking = output_dict['loss_masking']  # Changed from loss_slab_num_atoms
        masking_accuracy = output_dict['masking_accuracy']  # Changed from slab_num_atoms_accuracy
        masking_precision = output_dict['masking_precision']
        masking_recall = output_dict['masking_recall']
        masking_f1 = output_dict['masking_f1']
        masking_target_keep_ratio = output_dict['masking_target_keep_ratio']
        masking_target_remove_ratio = output_dict['masking_target_remove_ratio']
        slab_num_atoms_mae = output_dict['slab_num_atoms_mae']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_masking_loss': loss_masking,  # Changed from slab_num_atoms_loss
            f'{prefix}_masking_accuracy': masking_accuracy,  # Changed from slab_num_atoms_accuracy
            f'{prefix}_masking_precision': masking_precision,
            f'{prefix}_masking_recall': masking_recall,
            f'{prefix}_masking_f1': masking_f1,
            f'{prefix}_masking_target_keep_ratio': masking_target_keep_ratio,
            f'{prefix}_masking_target_remove_ratio': masking_target_remove_ratio,
            f'{prefix}_slab_num_atoms_mae': slab_num_atoms_mae
        }

        return log_dict, loss

    def on_after_backward(self) -> None:
        """
        Log gradient norm of composition-specific modules to detect no-grad issues.
        """
        try:
            decoder = self.decoder
            grad_sq_sum = 0.0
            n_params_with_grad = 0
            comp_modules = []
            for name in ['unified_masking_head', 'atom_to_latent', 'atomic_num_embedding', 'ratio_embedding', 'element_context_projection']:
                if hasattr(decoder, name):
                    comp_modules.append(getattr(decoder, name))
            if hasattr(decoder, 'comp_mlp_layers'):
                comp_modules.extend(list(decoder.comp_mlp_layers))
            for module in comp_modules:
                for p in module.parameters(recurse=True):
                    if p.grad is not None:
                        grad_sq_sum += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
                        n_params_with_grad += 1
            grad_l2 = float(grad_sq_sum) ** 0.5 if grad_sq_sum > 0 else 0.0
            # Also compute total grad norm (all trainable parameters)
            total_grad_sq = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_grad_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
            total_grad_l2 = float(total_grad_sq) ** 0.5 if total_grad_sq > 0 else 0.0

            # Log to both step and epoch, show comp_grad_norm on progress bar
            self.log('comp_grad_norm', grad_l2, on_step=True, on_epoch=True, prog_bar=True)
            self.log('total_grad_norm', total_grad_l2, on_step=True, on_epoch=True, prog_bar=False)
            self.log('comp_params_with_grad', n_params_with_grad, on_step=True, on_epoch=True, prog_bar=False)
        except Exception:
            pass

    