import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, get_pbc_distances, radius_graph_pbc, frac_to_cart_coords, repeat_blocks

MAX_ATOMIC_NUM=100
MAX_COMP_NUM=3
MAX_ADSO_NUM=11

class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1,-2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):

    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 300,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        edge_style = 'fc',
        cutoff = 6.0,
        max_neighbors = 20,
        ln = False,
        ip = True,
        smooth = False,
        pred_type = False,
        pred_scalar = False,
        coord_type = 'frac'
    ):
        super(CSPNet, self).__init__()

        self.latent_dim = latent_dim
        
        self.ip = ip
        self.smooth = smooth
        self.coord_type = coord_type
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(MAX_ATOMIC_NUM, hidden_dim)  # Use MAX_ATOMIC_NUM for atomic numbers
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        # For composition mode: project per-atom embedding (hidden_dim) to latent_dim
        self.atom_to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Composition embedding: separate atomic numbers and ratios
        self.atomic_num_embedding = nn.Embedding(MAX_ATOMIC_NUM, latent_dim // 2)
        self.ratio_embedding = nn.Linear(1, latent_dim // 2)  # Each ratio is a single value
        
        self.ads_element_embedding = nn.Linear(MAX_ADSO_NUM, hidden_dim)
        self.ads_local_dist_embedding = nn.Linear(MAX_ADSO_NUM**2, hidden_dim)

        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )            
        self.num_layers = num_layers
        self.coord_out_frac = nn.Linear(hidden_dim, 3, bias = False)
        self.coord_out_cart = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias = False)
        # Note: masking_out is unused in current design; removed to avoid confusion
        
        # Additional projection layer for adsorbate embedding
        self.adsorbate_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Adsorbate local distances embedding (if available)
        self.adsorbate_dist_embedding = nn.Linear(MAX_ADSO_NUM * MAX_ADSO_NUM, hidden_dim)
        
        # Composition-specific MLP layers (for composition mode)
        self.comp_mlp_layers = nn.ModuleList()
        for i in range(num_layers):
            self.comp_mlp_layers.append(
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    self.act_fn,
                    nn.Dropout(0.1),
                    nn.Linear(latent_dim, latent_dim)
                )
            )
        
        # Enhanced composition features
        self.element_context_projection = nn.Linear(2, latent_dim // 4)  # ratio + position info
        self.unified_masking_head = nn.Sequential(
            nn.Linear(latent_dim + latent_dim + latent_dim // 4, latent_dim),  # atom_emb + comp_context + element_context
            self.act_fn,
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1)  # Single output per atom
        )
        
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        
        # Composition-specific layer norms
        if self.ln:
            self.comp_layer_norm = nn.LayerNorm(latent_dim)
            self.comp_elem_layer_norm = nn.LayerNorm(latent_dim // 4)
            self.comp_combined_layer_norm = nn.LayerNorm(latent_dim + latent_dim + (latent_dim // 4))
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):

        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, distance_vectors)

            return edge_index_new, -edge_vector_new
    
    def _process_adsorbate_info(self, batch_size, adsorbate_types_padded, adsorbate_types_mask, adsorbate_local_distances):
        """
        Process adsorbate information (types and optional local distances)
        Returns combined adsorbate embedding
        """
        if adsorbate_types_padded is None or adsorbate_types_mask is None:
            return None
            
        # Reshape adsorbate data
        adsorbate_types_padded = adsorbate_types_padded.view(batch_size, MAX_ADSO_NUM)
        adsorbate_types_mask = adsorbate_types_mask.view(batch_size, MAX_ADSO_NUM)
        
        # Process adsorbate types
        adsorbate_types_clamped = torch.clamp(adsorbate_types_padded, min=0, max=MAX_ATOMIC_NUM - 1)
        adsorbate_emb = self.node_embedding(adsorbate_types_clamped)  # [batch_size, MAX_ADSO_NUM, hidden_dim]
        adsorbate_mask_expanded = adsorbate_types_mask.unsqueeze(-1).float()  # [batch_size, MAX_ADSO_NUM, 1]
        adsorbate_emb_masked = adsorbate_emb * adsorbate_mask_expanded  # 마스크로 패딩 제거
        adsorbate_emb_pooled = adsorbate_emb_masked.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Process adsorbate local distances if available
        if adsorbate_local_distances is not None:
            # Reshape and process distances
            adsorbate_local_distances = adsorbate_local_distances.view(batch_size, MAX_ADSO_NUM * MAX_ADSO_NUM)
            adsorbate_dist_emb = self.adsorbate_dist_embedding(adsorbate_local_distances)  # [batch_size, hidden_dim]
            
            # Combine type and distance embeddings
            adsorbate_emb_pooled = adsorbate_emb_pooled + adsorbate_dist_emb
        
        return adsorbate_emb_pooled
            

    def forward(self, mode='diffusion', t=None, atom_types=None, coords=None, lattices=None, 
               num_atoms=None, slab_composition_elements=None, slab_composition_ratios=None, 
               slab_composition_mask=None, slab_num_atoms=None, 
               adsorbate_types_padded=None, adsorbate_types_mask=None, adsorbate_local_distances=None, node2graph=None,
               fake_atom_types=None, fake_batch_mapping=None):

        if mode == 'composition':
            # Enhanced composition mode with per-atom prediction
            if fake_atom_types is None:
                raise ValueError("fake_atom_types is required for enhanced composition mode")
            
            # Determine batch size from composition data
            total_elements = slab_composition_elements.shape[0]
            batch_size = total_elements // MAX_COMP_NUM
            num_fake_atoms = len(fake_atom_types)
            
            # Reshape composition data
            slab_composition_elements = slab_composition_elements.view(batch_size, MAX_COMP_NUM)
            slab_composition_ratios = slab_composition_ratios.view(batch_size, MAX_COMP_NUM)
            slab_composition_mask = slab_composition_mask.view(batch_size, MAX_COMP_NUM)
            
            # 1. Per-atom embeddings (what type of atom is this?)
            atom_embs = self.node_embedding(fake_atom_types - 1)  # [num_fake_atoms, hidden_dim]
            # Directly project to latent_dim for composition path (avoids size mismatch across configs)
            atom_embs_projected = self.atom_to_latent(atom_embs)  # [num_fake_atoms, latent_dim]
            
            # 2. Global composition context (what's the overall composition?)
            atomic_emb = self.atomic_num_embedding(slab_composition_elements)  # [batch_size, MAX_COMP_NUM, latent_dim//2]
            ratio_emb = self.ratio_embedding(slab_composition_ratios.unsqueeze(-1))  # [batch_size, MAX_COMP_NUM, latent_dim//2]
            comp_emb_per_element = torch.cat([atomic_emb, ratio_emb], dim=-1)  # [batch_size, MAX_COMP_NUM, latent_dim]
            
            mask_expanded = slab_composition_mask.unsqueeze(-1).float()
            comp_emb_masked = comp_emb_per_element * mask_expanded
            comp_emb = comp_emb_masked.sum(dim=1)  # [batch_size, latent_dim]
            
            # Expand to each fake atom
            # Batch-aware broadcast of composition context to each fake atom
            if fake_batch_mapping is None:
                if batch_size == 1:
                    comp_context = comp_emb.expand(num_fake_atoms, -1)  # [num_fake_atoms, latent_dim]
                else:
                    raise ValueError("fake_batch_mapping is required when batch_size>1 in composition mode")
            else:
                comp_context = comp_emb.index_select(0, fake_batch_mapping)  # [num_fake_atoms, latent_dim]
            
            # 2-1. Inject adsorbate context into composition path
            # Reuse adsorbate processing from diffusion path
            adsorbate_emb_pooled = self._process_adsorbate_info(
                batch_size,
                adsorbate_types_padded,
                adsorbate_types_mask,
                adsorbate_local_distances
            )
            if adsorbate_emb_pooled is not None:
                # Project to latent_dim and add to comp_emb
                ads_latent = self.adsorbate_proj(adsorbate_emb_pooled)
                comp_emb = comp_emb + ads_latent

            # 3. Element-specific context (what's the target ratio for this atom type?)
            # Build per-batch lookup: atomic_num -> (index_in_composition, ratio)
            batch_lookup = []
            for b in range(batch_size):
                lookup = {}
                for i in range(MAX_COMP_NUM):
                    if slab_composition_mask[b, i]:
                        an = slab_composition_elements[b, i].item()
                        rr = float(slab_composition_ratios[b, i].item())
                        lookup[an] = (float(i), rr)
                batch_lookup.append(lookup)

            element_contexts = []
            if fake_batch_mapping is None:
                # Single-batch case
                lookup = batch_lookup[0]
                for atom_type in fake_atom_types.tolist():
                    idx, rr = lookup.get(int(atom_type), (-1.0, 0.0))
                    element_contexts.append([rr, idx])
            else:
                for at, b in zip(fake_atom_types.tolist(), fake_batch_mapping.tolist()):
                    lookup = batch_lookup[int(b)]
                    idx, rr = lookup.get(int(at), (-1.0, 0.0))
                    element_contexts.append([rr, idx])
            element_contexts = torch.tensor(element_contexts, dtype=torch.float, device=fake_atom_types.device)  # [num_fake_atoms, 2]
            element_context_embs = self.element_context_projection(element_contexts)  # [num_fake_atoms, latent_dim//4]
            
            # Optional LayerNorms in composition path for stability/consistency
            if self.ln:
                atom_embs_projected = self.comp_layer_norm(atom_embs_projected)
                comp_context = self.comp_layer_norm(comp_context)
                element_context_embs = self.comp_elem_layer_norm(element_context_embs)

            # 4. Combine all information
            combined_features = torch.cat([
                atom_embs_projected,      # What atom type is this?
                comp_context,             # What's the overall composition? (+ adsorbate)
                element_context_embs      # What's this atom's role in the composition?
            ], dim=-1)  # [num_fake_atoms, latent_dim + latent_dim + latent_dim//4]
            if self.ln:
                combined_features = self.comp_combined_layer_norm(combined_features)
            
            # 5. Unified prediction head
            masking_logits = self.unified_masking_head(combined_features).squeeze(-1)  # [num_fake_atoms]
            
            return masking_logits

        elif mode == 'diffusion':
            # Original diffusion mode processing with structure information
            # Shared coordinate processing
            if self.coord_type == 'cart':
                # Convert cartesian coordinates to fractional coordinates for edge generation
                inv_lattice = torch.linalg.pinv(lattices)
                inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
                frac_coords = torch.einsum('bi,bij->bj', coords, inv_lattice_nodes) % 1.
            else:
                # Use fractional coordinates directly
                frac_coords = coords

            # Shared node embedding
            if self.smooth:
                node_features = self.node_embedding(atom_types)
            else:
                node_features = self.node_embedding(atom_types - 1)

            # Use time embedding
            t_per_atom = t.repeat_interleave(num_atoms, dim=0)
            node_features = torch.cat([node_features, t_per_atom], dim=1)
            node_features = self.atom_latent_emb(node_features)
            
            # Process adsorbate information and add to node features
            batch_size = lattices.size(0)
            adsorbate_emb_pooled = self._process_adsorbate_info(
                batch_size, adsorbate_types_padded, adsorbate_types_mask, adsorbate_local_distances
            )
            
            if adsorbate_emb_pooled is not None:
                # Expand adsorbate embedding to all nodes
                adsorbate_emb_per_node = adsorbate_emb_pooled.repeat_interleave(num_atoms, dim=0)  # [total_atoms, hidden_dim]
                # Add adsorbate information to node features
                node_features = node_features + adsorbate_emb_per_node

            # Shared backbone
            edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
            edge2graph = node2graph[edges[0]]
            
            # Pass through CSP layers
            for i in range(0, self.num_layers):
                node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)

            if self.ln:
                node_features = self.final_layer_norm(node_features)

            # Coordinate output (node-level)
            if self.coord_type == 'cart':
                coord_out = self.coord_out_cart(node_features)
            else:
                coord_out = self.coord_out_frac(node_features)

            # Graph-level outputs
            graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean')

            if self.pred_scalar:
                return self.scalar_out(graph_features)

            lattice_out = self.lattice_out(graph_features)
            lattice_out = lattice_out.view(-1, 3, 3)
            if self.ip:
                lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
            if self.pred_type:
                type_out = self.type_out(node_features)
                return lattice_out, coord_out, type_out

            return lattice_out, coord_out
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

