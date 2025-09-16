"""
Crystal structure centering utilities for better visualization and analysis
"""

import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize import view


def center_atoms_in_cell(atoms, direction='z', method='center_of_mass'):
    """
    Center atoms in the specified direction within the unit cell.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        ASE atoms object
    direction : str or int
        Direction to center ('x', 'y', 'z' or 0, 1, 2)
    method : str
        'center_of_mass' or 'geometric_center'
    
    Returns:
    --------
    ase.Atoms
        Centered atoms object
    """
    # Convert direction to index
    if isinstance(direction, str):
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map[direction.lower()]
    else:
        dir_idx = direction
    
    atoms_copy = atoms.copy()
    positions = atoms_copy.get_positions()
    cell = atoms_copy.get_cell()
    
    if method == 'center_of_mass':
        masses = atoms_copy.get_masses()
        center = np.average(positions[:, dir_idx], weights=masses)
    else:  # geometric_center
        center = np.mean(positions[:, dir_idx])
    
    # Get cell dimension in the specified direction
    cell_length = np.linalg.norm(cell[dir_idx])
    
    # Calculate shift to center the atoms
    cell_center = cell_length / 2
    shift = cell_center - center
    
    # Apply shift
    positions[:, dir_idx] += shift
    
    # Wrap coordinates to stay within cell
    # Convert to fractional coordinates, wrap, then back to cartesian
    frac_coords = np.linalg.solve(cell.T, positions.T).T
    frac_coords = frac_coords % 1.0  # Wrap fractional coordinates
    positions = frac_coords @ cell
    
    atoms_copy.set_positions(positions)
    return atoms_copy


def center_structure_pymatgen(structure, direction='z', method='center_of_mass'):
    """
    Center pymatgen Structure in the specified direction with proper periodic boundary handling.
    
    Parameters:
    -----------
    structure : pymatgen.core.Structure
        Pymatgen structure object
    direction : str or int
        Direction to center ('x', 'y', 'z' or 0, 1, 2)
    method : str
        'center_of_mass' or 'geometric_center'
    
    Returns:
    --------
    pymatgen.core.Structure
        Centered structure
    """
    # Convert direction to index
    if isinstance(direction, str):
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map[direction.lower()]
    else:
        dir_idx = direction
    
    structure_copy = structure.copy()
    frac_coords = structure_copy.frac_coords.copy()
    
    # Handle periodic boundaries properly by using angles
    coords_1d = frac_coords[:, dir_idx]
    
    # Convert fractional coordinates to angles (0 to 2π)
    angles = 2 * np.pi * coords_1d
    
    if method == 'center_of_mass':
        masses = np.array([site.specie.atomic_mass for site in structure_copy.sites])
        # Calculate center of mass in angular space
        sin_avg = np.average(np.sin(angles), weights=masses)
        cos_avg = np.average(np.cos(angles), weights=masses)
    else:  # geometric_center
        # Calculate geometric center in angular space
        sin_avg = np.mean(np.sin(angles))
        cos_avg = np.mean(np.cos(angles))
    
    # Convert back to fractional coordinate center
    center_angle = np.arctan2(sin_avg, cos_avg)
    if center_angle < 0:
        center_angle += 2 * np.pi
    center_frac = center_angle / (2 * np.pi)
    
    # Calculate shift to center at 0.5
    shift = 0.5 - center_frac
    
    # Apply shift
    frac_coords[:, dir_idx] += shift
    
    # Wrap coordinates to [0, 1)
    frac_coords = frac_coords % 1.0
    
    # Create new structure with centered coordinates
    centered_structure = Structure(
        lattice=structure_copy.lattice,
        species=[site.specie for site in structure_copy.sites],
        coords=frac_coords,
        coords_are_cartesian=False
    )
    
    return centered_structure


def visualize_before_after(structure_or_atoms, direction='z', method='center_of_mass'):
    """
    Visualize structure before and after centering for comparison.
    """
    if isinstance(structure_or_atoms, Structure):
        # Pymatgen structure
        original = structure_or_atoms
        centered = center_structure_pymatgen(original, direction, method)
        
        # Convert to ASE for visualization
        original_atoms = AseAtomsAdaptor.get_atoms(original)
        centered_atoms = AseAtomsAdaptor.get_atoms(centered)
        
    else:
        # ASE atoms
        original_atoms = structure_or_atoms
        centered_atoms = center_atoms_in_cell(original_atoms, direction, method)
    
    print("Original structure:")
    view(original_atoms, viewer='ngl')
    
    print("\nCentered structure:")
    view(centered_atoms, viewer='ngl')
    
    return centered_atoms


def analyze_distribution(structure_or_atoms, direction='z'):
    """
    Analyze the distribution of atoms along a specific direction.
    """
    if isinstance(structure_or_atoms, Structure):
        atoms = AseAtomsAdaptor.get_atoms(structure_or_atoms)
    else:
        atoms = structure_or_atoms
    
    # Convert direction to index
    if isinstance(direction, str):
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map[direction.lower()]
    else:
        dir_idx = direction
    
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    cell_length = np.linalg.norm(cell[dir_idx])
    
    coords = positions[:, dir_idx]
    
    print(f"Distribution along {direction}-direction:")
    print(f"Cell length: {cell_length:.3f} Å")
    print(f"Min coordinate: {coords.min():.3f} Å")
    print(f"Max coordinate: {coords.max():.3f} Å")
    print(f"Range: {coords.max() - coords.min():.3f} Å")
    print(f"Center of mass: {np.average(coords, weights=atoms.get_masses()):.3f} Å")
    print(f"Geometric center: {np.mean(coords):.3f} Å")
    print(f"Cell center: {cell_length/2:.3f} Å")
    
    # Check if atoms are clustered at edges
    quarter_length = cell_length / 4
    top_quarter = np.sum(coords > 3 * quarter_length)
    bottom_quarter = np.sum(coords < quarter_length)
    
    print(f"Atoms in top quarter: {top_quarter}")
    print(f"Atoms in bottom quarter: {bottom_quarter}")
    
    if top_quarter + bottom_quarter > len(coords) / 2:
        print("⚠️  Atoms appear to be clustered at cell edges - centering recommended!")
    else:
        print("✅ Atoms are reasonably distributed within the cell")


if __name__ == "__main__":
    # Example usage
    from pymatgen.core import Structure
    
    # Load a structure that might need centering
    structure = Structure.from_file("/home/minkyu/DiffCSP/hydra/singlerun/2025-08-21/CSP_oc20_dense/eval_diff_train_set.dir/0/cif/7.cif")
    
    print("=== Analysis of Original Structure ===")
    analyze_distribution(structure, 'z')
    
    print("\n=== Centering Structure ===")
    centered = center_structure_pymatgen(structure, 'z', 'center_of_mass')
    
    print("=== Analysis of Centered Structure ===")
    analyze_distribution(centered, 'z')
    
    # Save centered structure
    centered.to(filename="centered_structure.cif")
    print("\nCentered structure saved as 'centered_structure.cif'")
