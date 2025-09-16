from fairchem.core import pretrained_mlip, FAIRChemCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from tqdm import tqdm
import pandas as pd
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from io import StringIO
from ase.visualize import view
import ast
import numpy as np

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")

df = pd.read_csv("/home/minkyu/DiffCSP/data/oc20_2M/test.csv") 

diff_energy_list = []

for i in tqdm(range(100)):
  try:
    pred_structure = Structure.from_file(f"/home/minkyu/DiffCSP/hydra/singlerun/2025-09-16/no_shuffle_soft_label/eval_diff.dir/0/cif/{i}.cif")
    pred_slab = AseAtomsAdaptor.get_atoms(pred_structure)

    pred_slab = Atoms(
      symbols=pred_slab.get_chemical_symbols(),
      positions=pred_slab.get_positions(),
      cell=pred_slab.get_cell(),
      pbc=[True, True, True]
    )
    
    # Reconstruct structure from CSV data
    atom_types = ast.literal_eval(df.loc[i, 'atom_types'])
    lengths = ast.literal_eval(df.loc[i, 'lengths'])
    angles = ast.literal_eval(df.loc[i, 'angles'])
    frac_coords = ast.literal_eval(df.loc[i, 'frac_coords'])
    
    # Create lattice from lengths and angles
    lattice = Lattice.from_parameters(lengths[0], lengths[1], lengths[2], 
                                    angles[0], angles[1], angles[2])
    
    # Create structure
    true_slab = Structure(lattice, atom_types, frac_coords)

    # 2. Structure → ASE Atoms 변환
    true_slab = AseAtomsAdaptor.get_atoms(true_slab)
    
    true_slab = Atoms(
      symbols=true_slab.get_chemical_symbols(),
      positions=true_slab.get_positions(),
      cell=true_slab.get_cell(),
      pbc=[True, True, True]
    )
    
    pred_slab.calc = calc
    true_slab.calc = calc
      
    diff_energy = pred_slab.get_potential_energy() - true_slab.get_potential_energy()
    diff_energy_list.append(diff_energy)
    
  except Exception as e:
    print(f"Error at sample_{i}: {e}")
  
if len(diff_energy_list) > 0:
    print(f"Average energy difference: {sum(diff_energy_list)/len(diff_energy_list)}")
    print(f"Successfully calculated {len(diff_energy_list)} out of 100 samples")
else:
    print("No energy differences were successfully calculated")

# Save results to file
if len(diff_energy_list) > 0:
    with open('/home/minkyu/DiffCSP/diff_energy_results.txt', 'w') as f:
        for energy in diff_energy_list:
            f.write(f"{energy}\n")
    print("Results saved to diff_energy_results.txt")
else:
    print("No results to save")