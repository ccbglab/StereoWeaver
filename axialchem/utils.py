######################################################################################
#   Backend Codes (utils.py)
#   Mayuk Joddar.
######################################################################################

# Standard library
import os
import json
import shutil
import cclib
import itertools
import warnings
from pathlib import Path
from collections import defaultdict
# Third-party
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ase.io import read as ase_read
# RDKit
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools, rdmolops, rdMolTransforms, rdchem, rdDetermineBonds, SDWriter
from rdkit.Chem.rdmolops import SanitizeMol
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
# Mordred
from mordred import Calculator, descriptors
# Silence warnings
warnings.filterwarnings("ignore")

#Global
metal_symbols = {'Fe', 'Cu', 'Pd', 'Co'}

######################################################################################
# Extracts Symbols and Coordinate From .xyz File.
######################################################################################
def xyz2coords(filepath):
    """
    Input: str, Path to the .xyz file.
    Read an XYZ file and return atomic symbols and Cartesian coordinates.
    """
    coords = []
    symbols = []
    with open(filepath, 'r') as file:
        lines = file.readlines()[2:]  # Skip the first two lines
        for line in lines:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(symbols),np.array(coords)

######################################################################################
# Finds The Atom Closest To The Centre.
######################################################################################
def find_closest_to_center(coords):
    """
    Input: coords (N*3) → index of closest atom to center.
    """
    center = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    return np.argmin(distances)

######################################################################################
# Find Priority order.
######################################################################################
def get_cip_ranks(mol, atom_indices):
    ranks = []
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPRank'):
            ranks.append((idx, int(atom.GetProp('_CIPRank'))))
        else:
            ranks.append((idx, 0))
    return ranks

######################################################################################
# Convert log to XYZ file to save DFT geometry.
######################################################################################
def log2xyz(log_file, xyz_file, n_atoms=None):
    data = cclib.io.ccread(str(log_file))
    atoms = data.atomnos
    coords = data.atomcoords[-1]

    if n_atoms and len(atoms) > n_atoms:
        atoms = atoms[:n_atoms]
        coords = coords[:n_atoms]

    pt = Chem.GetPeriodicTable()
    symbols = [pt.GetElementSymbol(int(at_num)) for at_num in atoms]

    with open(xyz_file, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Converted from {log_file}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"[INFO] Converted log_file → xyz_file")

######################################################################################
# SMILES + XTB = CXSMILES
######################################################################################
def smiles_xyz_to_cxsmiles(smiles, xyz_file, validate=True):
    """
    Convert SMILES + xTB optimized XYZ → CXSMILES with embedded coordinates.

    """
    # Build molecule from SMILES
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles,params)
    mol = Chem.AddHs(mol)

    if mol is None:
        print("[INFO] Invalid SMILES input for CXSMILES")

    order = sorted(range(mol.GetNumAtoms()),key=lambda i:mol.GetAtomWithIdx(i).GetAtomMapNum())
    mol = Chem.RenumberAtoms(mol,order)  

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # Create conformer container (coordinates will be overwritten)
    mol.RemoveAllConformers()
    conf=Chem.Conformer(mol.GetNumAtoms())
    conf_id = mol.AddConformer(conf, assignId=True)
    conf = mol.GetConformer(conf_id)

    # Read optimized geometry
    elements, coords = xyz2coords(xyz_file)
    
    # Validation
    if validate:
        if mol.GetNumAtoms() != len(coords):
            print("[INFO] Atom count mismatch between SMILES and XYZ for CXSMILES")
            return smiles

        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != elements[i]:
                print(f"[INFO ]Atom mismatch (CXSMILES) at index {i}: {atom.GetSymbol()} vs {elements[i]}")
                return smiles
    
    # Inject XYZ coordinates into RDKit conformer
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))

    cxsmiles = Chem.MolToCXSmiles(mol)
    return cxsmiles

######################################################################################
# Maps xTB geometry onto SMILES and export MOL
######################################################################################
def smiles_xyz_to_molfile(smiles, xyz_file, output_molfile, validate=True):
    """
    Convert SMILES + xTB optimized XYZ → MOL file with embedded coordinates.
    If validation fails, writes MOL from SMILES only (no coordinate injection).
    """
    # Build molecule from SMILES
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles,params)
    mol = Chem.AddHs(mol)
    if mol is None:
        print("[INFO] Invalid SMILES input for CXSMILES")

    order = sorted(range(mol.GetNumAtoms()),key=lambda i:mol.GetAtomWithIdx(i).GetAtomMapNum())
    mol = Chem.RenumberAtoms(mol,order)  

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # Create conformer container (coordinates will be overwritten)
    mol.RemoveAllConformers()
    conf=Chem.Conformer(mol.GetNumAtoms())
    conf_id = mol.AddConformer(conf, assignId=True)
    conf = mol.GetConformer(conf_id)

    # Read optimized geometry
    elements, coords = xyz2coords(xyz_file)

    # Validation 
    if validate:
        if mol.GetNumAtoms() != len(coords):
            print("[INFO] Atom count mismatch → writing MOL without xTB coordinates")
            Chem.MolToMolFile(mol, output_molfile)
            return output_molfile

        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() != elements[i]:
                print(f"[INFO] Atom mismatch at index {i} → writing MOL without xTB coordinates")
                Chem.MolToMolFile(mol, output_molfile)
                return output_molfile

    # Inject XYZ coordinates into RDKit conformer
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))

    Chem.MolToMolFile(mol, output_molfile)
    return output_molfile

######################################################################################
# To Generate List Of Metal And Possible Neighbors Based On Distance<=2.6.
######################################################################################
def get_metal_and_neighbors(mol, xyz_file, dummy_idx=999):

    symbols, coords = xyz2coords(xyz_file)
    used_atom_indices = set()

    # SMARTS database to Identify the donor atoms
    base_dir = Path(__file__).resolve().parent
    smarts_path = base_dir / "SMARTS"/ "SMARTS_DESC.json"
    
    with open(smarts_path, "r") as f:
         smarts_spec = json.load(f)


    if smarts_spec is None:
        raise ValueError("[ERROR] Please provide a SMARTS_DESC.json file.")
    if mol is None:
        raise ValueError("[ERROR] Please check mol file")
    # Step 1: Get metal index
    metal_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_symbols:
            metal_idx = atom.GetIdx()
            break
    if metal_idx is None:
        raise ValueError(f"[Error] No metal atom from {metal_symbols} found.")

    metal_coord = coords[metal_idx]
    neighbor_indices = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx == metal_idx:
            continue
        dist = np.linalg.norm(coords[idx] - metal_coord)
        if mol.GetBondBetweenAtoms(metal_idx, idx) or dist <= 2.6:
            neighbor_indices.append(idx)

    print(f"[INFO] Metal index: {metal_idx}, Total neighbors: {len(neighbor_indices)}")

    matched_fragments = []

    # Step 2: Check each SMARTS pattern and for Diazo group
    for idx in neighbor_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue
        if idx in used_atom_indices:
            continue

        for nbr in atom.GetNeighbors():
            oxy_count = sum(1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == "O")
            if oxy_count == 2:
                matched_fragments.append((33, idx, "Diazo_carbon"))
                used_atom_indices.add(idx)
                break

    for entry in smarts_spec:
        name = entry['name']
        patt = Chem.MolFromSmarts(entry['smarts'])
        donor_idxs = entry['donor_idxs']
        rank = entry['rank']
        condition_atoms = entry.get("condition", None)

        max_matches = 1 

        matches = mol.GetSubstructMatches(patt)
        count = 0

        for match in matches:
            
            if condition_atoms is not None:
                if len(match) != condition_atoms:
                    continue

            if any(idx in used_atom_indices for idx in match):
                continue

            matched_neighbors = set(neighbor_indices) & set(match)
            if not matched_neighbors:
                continue

            for i, rel_idx in enumerate(donor_idxs):
                try:
                    donor_idx = match[rel_idx]
                    if donor_idx in matched_neighbors:
                        matched_fragments.append((rank, donor_idx, f"{name}_{i+1}"))
                    else:
                        matched_fragments.append((rank, dummy_idx, f"{name}_{i+1}"))
                except IndexError:
                    matched_fragments.append((rank, dummy_idx, f"{name}_{i+1}"))

            used_atom_indices.update(match)
            count += 1
            if count >= max_matches:
                break

        # Adding dummy idx for missing match
        while count < max_matches:
            for j in range(len(donor_idxs)):
                matched_fragments.append((rank, dummy_idx, f"{name}_{j+1}"))
            count += 1

    # Step 3: Fill unmatched neighbor indices
    for idx in neighbor_indices:
        already_used = any(idx == frag[1] for frag in matched_fragments)
        if not already_used:
            matched_fragments.append((999, idx, "unassigned"))

    # Step 4: Sort by rank
    matched_fragments.sort(key=lambda x: x[0])

    donor_indices = [entry[1] for entry in matched_fragments]
    donor_names = [entry[2] for entry in matched_fragments]

    return metal_idx, donor_indices, donor_names

######################################################################################
# To Generate List Of Carbene Centre And it's Neighbors.
######################################################################################
def get_carbene_and_neighbors(mol, coords, metal_idx, distance_cutoff=2.5):
    """
    Detect 'carbene carbon' based on metal connectivity and COO / CF3 -like group:
    - A carbon bonded to a metal (or within distance_cutoff)
    - Neighbor of this carbon has 2 oxygens OR 3 Fluorine
    Returns:
    - carbene carbon index
    - sorted list of neighboring atom indices (rank 0: neighbor with 2 O, rank 1: other neighbors)
    """
    carbene_idx = None
    ranked_neighbors = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        c_idx = atom.GetIdx()

        # Check if bonded to metal or within distance cutoff
        bonded_to_metal = mol.GetBondBetweenAtoms(c_idx, metal_idx) is not None
        dist_to_metal = np.linalg.norm(coords[c_idx] - coords[metal_idx])
        if not bonded_to_metal and dist_to_metal > distance_cutoff:
            continue

        # Check neighbors of this carbon
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == metal_idx:
                continue  # skip metal

            oxy_count = sum(1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == "O")
            f_count = sum (1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == "F")
            if oxy_count >= 2 or f_count >= 3:
                # Found carbene carbon
                carbene_idx = c_idx

                # Assign ranks to neighbors
                for n in atom.GetNeighbors():
                    n_idx = n.GetIdx()
                    if n_idx == metal_idx:
                        continue
                    if oxy_count >= 2:
                     rank = 0
                     ranked_neighbors.append((rank, n_idx))
                    elif  n.GetSymbol() == "C":
                      rank = 1  
                      ranked_neighbors.append((rank, n_idx))

                    if len(ranked_neighbors) == 2:
                        break
                break  # stop checking other neighbors once carbene found

        if carbene_idx is not None:
            break

    if carbene_idx is None:
        raise ValueError("No carbene carbon found.")

    # Sort neighbors by rank
    sorted_neighbor_indices = [idx for (rank, idx) in sorted(ranked_neighbors, key=lambda x: x[0])]
    donor_name = ['C_2O', 'C_R']
    return carbene_idx, sorted_neighbor_indices, donor_name
   
######################################################################################
# To Generate 2D Descriptors-RDKIT, Mordred, Morgan FingerPrint.
######################################################################################
def Compute_2d_descriptors(mol):
    
    if mol is None:
        raise ValueError("Mol object is None.")

    rdkit_desc = {}
    mordred_desc = {}
    morgan_desc = {}

    # 1. RDKit Descriptors 
    try:
        rdkit_desc_names = [desc[0] for desc in Descriptors._descList]
        rdkit_calc = MolecularDescriptorCalculator(rdkit_desc_names)
        rdkit_values = rdkit_calc.CalcDescriptors(mol)
        for name_, val in zip(rdkit_desc_names, rdkit_values):
            rdkit_desc[f'RDKit_{name_}'] = val
    except Exception as e:
        print(f"[RDKit Error] {e}")
        for name_ in rdkit_desc_names:
            rdkit_desc[f'RDKit_{name_}'] = np.nan

    # 2. Mordred Descriptors
    try:
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_df = calc.pandas([mol])
        mordred_df = mordred_df.select_dtypes(include=[np.number])
        mordred_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        mordred_df.dropna(axis=1, inplace=True)
        for k, v in mordred_df.iloc[0].items():
            mordred_desc[f'Mordred_{str(k)}'] = v
    except Exception as e:
        print(f"[Mordred Error] {e}")

    # 3. Morgan Fingerprint
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048)
        for i in range(fp.GetNumBits()):
            morgan_desc[f'Morgan_{i}'] = int(fp[i])
    except Exception as e:
        print(f"[Morgan FP Error] {e}")
        for i in range(2048):
            morgan_desc[f'Morgan_{i}'] = 0

    return rdkit_desc, mordred_desc, morgan_desc

######################################################################################
# RMSD Descriptor
######################################################################################
def RMSD(xyz_file_1, xyz_file_2):
    def kabsch(P, Q):
        P_cent = P - P.mean(axis=0)
        Q_cent = Q - Q.mean(axis=0)

        H = P_cent.T @ Q_cent
        U, S, Vt = np.linalg.svd(H)

        # reflection correction
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[-1, :] *= -1

        R = Vt.T @ U.T
        return P_cent @ R + Q.mean(axis=0)

    # Read files
    atoms1, P = xyz2coords(xyz_file_1)
    atoms2, Q = xyz2coords(xyz_file_2)

    if list(atoms1) != list(atoms2):
        raise ValueError("[ERROR]DFT-Atom order mismatch between files")

    # Align 
    P_aligned = kabsch(P, Q)

    # RMSD (all atoms) 
    diff = P_aligned - Q
    rmsd_all = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # Heavy atom RMSD 
    mask = np.array([a != 'H' for a in atoms1])
    diff_heavy = P_aligned[mask] - Q[mask]
    rmsd_heavy = np.sqrt(np.mean(np.sum(diff_heavy**2, axis=1)))

    # Per-atom deviation 
    distances = np.sqrt(np.sum(diff**2, axis=1))
    per_atom = list(zip(atoms1, distances))

    # Max deviation 
    max_idx = np.argmax(distances)
    max_dev = distances[max_idx]
    max_atom = atoms1[max_idx]

    # Metal deviation 
    M_dev = None
    for atom, d in per_atom:
        if atom in metal_symbols:
            M_dev = d
            break

    desc_row = {
        "RMSD_all": rmsd_all,
        "RMSD_heavy": rmsd_heavy,
        "Max_dev": max_dev,
        "mETAL_dev": M_dev
    }
    return pd.Series(desc_row)

######################################################################################
# To Calculate Bond Length Between Given Key Index and Neighboring Index.
######################################################################################
def get_bond_length(xyz_file, key_idx, neighbor_indices, donor_names=None):
    """
    Computes bond length of key_idx to each neighbor, optionally labeled by donor name.

    Parameters:
        xyz_file (str): Path to the .xyz file
        key_idx (int): Index of the central atom (0-based)
        neighbor_indices (list): List of neighbor atom indices (0-based or 999)
        donor_names (list or None): List of donor names corresponding to each neighbor

    Returns:
        pd.Series: {Bond_Len_*: length}
    """
    symbol,coords = xyz2coords(xyz_file)
    key_coord = coords[key_idx]
    
    desc_row = {}

    if donor_names is None or len(donor_names) != len(neighbor_indices):
        # Generic labeling if donor names are not provided
        for i, n_idx in enumerate(neighbor_indices, start=1):
            bond_length = 0.0 if n_idx == 99 else np.linalg.norm(key_coord - coords[n_idx])
            desc_row[f"Bond_Len_{i}"] = bond_length
    else:
        for n_idx, name in zip(neighbor_indices, donor_names):
            bond_length = 0.0 if n_idx == 999 else np.linalg.norm(key_coord - coords[n_idx])
            desc_row[f"{name}"] = bond_length

    return pd.Series(desc_row)

######################################################################################
# To Calculate Bond Angle Between Given Key Index and Neighboring Two Index.
######################################################################################
def get_bond_angles(xyz_file, key_idx, neighbor_indices, donor_names=None, pad_value=0.0):
    """
    Computes bond angles between key_idx and all neighbor pairs.
    Angles are named based on neighbor position in list (1-based): Angle_1_2, etc.
    """
    symbol, coords = xyz2coords(xyz_file)
    key_coord = coords[key_idx]

    if len(neighbor_indices) < 2:
        return pd.Series()

    angles = []

    for idx1, idx2 in itertools.combinations(range(len(neighbor_indices)), 2):
        atom1 = neighbor_indices[idx1]
        atom2 = neighbor_indices[idx2]
        if donor_names is None or len(donor_names) != len(neighbor_indices):
            label = f"Angle_{idx1+1}_{idx2+1}"
        else:
            label = f"Angle_{donor_names[idx1]}-{donor_names[idx2]}"

        if 999 in (atom1, atom2):
            angles.append((label, pad_value))
            continue

        vec1 = coords[atom1] - key_coord
        vec2 = coords[atom2] - key_coord
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            angle_deg = pad_value
        else:
            cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))

        angles.append((label, angle_deg))

    return pd.Series(dict(angles))

######################################################################################
# To Calculate Dihedral Angle Between Given Indices.
######################################################################################
def dihedral(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

def get_axial_dihedrals(xyz_file, mol, pad_value=0.0):

    _, coords = xyz2coords(xyz_file)

    base_dir = Path(__file__).resolve().parent
    smarts_path = base_dir / "SMARTS"/ "SMARTS_AC.json"
    
    with open(smarts_path, "r") as f:
         smarts_patterns = json.load(f)
         
    mol = Chem.AddHs(mol)

    diherdral_series = {
        "D_spiro_1": pad_value,
        "D_spiro_2": pad_value,
        "D_spiro_3": pad_value,
        "D_biimidazole_1": pad_value,
        "D_biimidazole_2": pad_value,
        "D_biphenyl/bipyridine_1": pad_value,
    }

    for name, smart in smarts_patterns.items():
        patt = Chem.MolFromSmarts(smart)
        if not patt:
            continue

        if mol.HasSubstructMatch(patt):
            match = mol.GetSubstructMatch(patt)

            if name in ("spiro_C", "spiro_Si"):
                torsions = [
                    (match[7], match[8], match[16], match[11]),
                    (match[2], match[3], match[8], match[16]),
                    (match[3], match[8], match[16], match[15])
                ]
                for i, (a,b,c,d) in enumerate(torsions):
                    diherdral_series[f"D_spiro_{i+1}"] = dihedral(coords[a], coords[b], coords[c], coords[d])

            elif name == "Biimidazole":
                torsions = [
                    (match[7], match[11], match[12], match[16]),
                    (match[4], match[3], match[23], match[18])
                ]
                for i, (a,b,c,d) in enumerate(torsions):
                    diherdral_series[f"D_biimidazole_{i+1}"] = dihedral(coords[a], coords[b], coords[c], coords[d])

            elif name in ("Bipyridine", "Biphenyl"):
                torsions = [(match[5], match[4], match[3], match[2])
                ]
                for i, (a,b,c,d) in enumerate(torsions):
                    diherdral_series[f"D_biphenyl/bipyridine_{i+1}"] = dihedral(coords[a], coords[b], coords[c], coords[d])

            break

    return pd.Series(diherdral_series)

######################################################################################
# Code to Generate Metallocarbene.
######################################################################################
def link_multiple_ligs_to_metal(lig_mol_list, metal_mol, coord_idx_list_per_lig, bond_types_per_lig=None):
    """
    lig_mol_list: list of RDKit molecules for ligands.
    metal_mol: RDKit molecule with one metal atom (e.g., [Pd+2])
    coord_idx_list_per_lig: list of lists; each inner list has indices in the ligand to bond to metal
    """
    assert metal_mol.GetNumAtoms() == 1

    rw_mol = Chem.RWMol()
    atom_mapping = []

    # copy ligands with full stereo 
    for lig_mol_orig in lig_mol_list:
        lig_mol = Chem.Mol(lig_mol_orig)
        amap = {}

        # copy atoms
        for atom in lig_mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetChiralTag(atom.GetChiralTag())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            new_atom.SetNoImplicit(atom.GetNoImplicit())

            new_idx = rw_mol.AddAtom(new_atom)
            amap[atom.GetIdx()] = new_idx

        # copy bonds INCLUDING stereo
        for bond in lig_mol.GetBonds():
            i = amap[bond.GetBeginAtomIdx()]
            j = amap[bond.GetEndAtomIdx()]
            rw_mol.AddBond(i, j, bond.GetBondType())
            new_bond = rw_mol.GetBondBetweenAtoms(i, j)
            new_bond.SetStereo(bond.GetStereo())
            new_bond.SetBondDir(bond.GetBondDir())

        atom_mapping.append(amap)

    # add metal atom 
    metal_atom = metal_mol.GetAtomWithIdx(0)
    m = Chem.Atom(metal_atom.GetSymbol())
    m.SetFormalCharge(metal_atom.GetFormalCharge())
    metal_idx = rw_mol.AddAtom(m)

    # add coordination bonds 
    for lig_idx, coord_list in enumerate(coord_idx_list_per_lig):
        amap = atom_mapping[lig_idx]
        if bond_types_per_lig is None:
           bond_types = [Chem.BondType.DATIVE] * len(coord_list)
        else:
           bond_types = bond_types_per_lig[lig_idx]
        for lig_atom_idx, bond_type in zip(coord_list, bond_types):
            rw_mol.AddBond(amap[lig_atom_idx], metal_idx, bond_type)

    mol = rw_mol.GetMol()

    # re-assign stereochemistry from preserved flags
    Chem.AssignStereochemistry(mol, force=True, cleanIt=False)

    return mol

def process_diazo_ligand(diazo_mol):
    """
    Process diazo ligand: Remove N2 and return carbene carbon atom
    Input: R-C=[N+]=[N-]
    Output: R-C (carbene) molecule with the carbon atom index
    """
    rw_mol = Chem.RWMol(diazo_mol)
    
    # Find the C=[N+]=[N-] pattern
    diazo_pattern = Chem.MolFromSmarts('[C]=[N+]=[N-]')
    match = rw_mol.GetSubstructMatch(diazo_pattern)
    
    if not match:
        raise ValueError("Diazo pattern not found in molecule")
    
    carbon_idx = match[0]
    n1_idx = match[1]
    n2_idx = match[2]
    
    # Remove nitrogen atoms (in reverse order to maintain indices)
    if n2_idx > n1_idx:
        rw_mol.RemoveAtom(n2_idx)
        rw_mol.RemoveAtom(n1_idx)
    else:
        rw_mol.RemoveAtom(n1_idx)
        rw_mol.RemoveAtom(n2_idx)
    
    # Adjust carbon index based on removed atoms
    new_carbon_idx = carbon_idx
    if n1_idx < carbon_idx:
        new_carbon_idx -= 1
    if n2_idx < carbon_idx:
        new_carbon_idx -= 1
    
    carbene_mol = rw_mol.GetMol()
    Chem.SanitizeMol(carbene_mol)

    c_atom = carbene_mol.GetAtomWithIdx(new_carbon_idx)
    c_atom.SetFormalCharge(0)
    c_atom.SetNumExplicitHs(0)
    c_atom.SetNoImplicit(True)
    c_atom.SetNumRadicalElectrons(2) 

    for o_atom in carbene_mol.GetAtoms():
        if o_atom.GetSymbol() == "O":
            o_atom.SetNoImplicit(True)       
            o_atom.SetNumExplicitHs(0)        
            o_atom.SetFormalCharge(0)         
            

    return carbene_mol, new_carbon_idx

def filter_mecn_from_smiles(smiles):
    """
    To filer out 4MeCN mol to 2MeCN as when the bidentata ligand 
    attach to the molecule there is release of 2MeCN
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")  

    mol_with_H = Chem.AddHs(mol)
    fragments = Chem.GetMolFrags(mol_with_H, asMols=True, sanitizeFrags=False)

    if len(fragments) < 5:
        return smiles

    ligand = fragments[-1]
    non_ligand_frags = fragments[:-1]

    mecn_query = Chem.MolFromSmiles("CC#N")
    mecn_frags = []
    other_frags = []
    

    for frag in non_ligand_frags:
        if frag.HasSubstructMatch(mecn_query):
            mecn_frags.append(frag)
        else:
            other_frags.append(frag)

    if len(mecn_frags) == 4:
        #print("Found 4 MeCN fragments — keeping only 2.")
        mecn_frags = mecn_frags[:2]
    else:
        return smiles

    updated_frags = other_frags + mecn_frags + [ligand]
    combined = updated_frags[0]
    for frag in updated_frags[1:]:
        combined = rdmolops.CombineMols(combined, frag)

    Chem.AssignStereochemistry(combined, force=True, cleanIt=False)
    return Chem.MolToSmiles(combined, canonical=False, isomericSmiles=True)

def build_active_metal_ligand_catalyst(smiles):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    smiles = filter_mecn_from_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles, params)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=False)

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    
    for f in fragments:
        Chem.AssignStereochemistry(f, force=True, cleanIt=False)

    if len(fragments) == 1:
        return smiles

    metal_atom = None
    metal_frag = None

    for frag in fragments:
        for atom in frag.GetAtoms():
            if atom.GetSymbol() in metal_symbols:
                metal_atom = atom
                metal_idx= atom.GetIdx()
                metal_frag = frag
                break
        if metal_atom:
            break

    if not metal_atom:
        print("[INFO]No metal found.")
        return smiles
    
    # Check if Cu+2 and change to Cu+1
    if metal_atom.GetSymbol() == "Cu" and metal_atom.GetFormalCharge() == 2:
        metal_frag_editable = Chem.RWMol(metal_frag)
        for atom in metal_frag_editable.GetAtoms():
            if atom.GetSymbol() == "Cu" and atom.GetFormalCharge() == 2:
                atom.SetFormalCharge(1)
                break
        metal_frag = metal_frag_editable.GetMol()
        for atom in metal_frag.GetAtoms():
            if atom.GetSymbol() == "Cu":
                metal_atom = atom
                break

    # Check if Fe+3 and change to Fe+2
    if metal_atom.GetSymbol() == "Fe" and metal_atom.GetFormalCharge() == 3:
        metal_frag_editable = Chem.RWMol(metal_frag)
        for atom in metal_frag_editable.GetAtoms():
            if atom.GetSymbol() == "Fe" and atom.GetFormalCharge() == 3:
                atom.SetFormalCharge(2)
                break
        metal_frag = metal_frag_editable.GetMol()
        for atom in metal_frag.GetAtoms():
            if atom.GetSymbol() == "Fe":
                metal_atom = atom
                break        

    def is_anionic(frag):
        return sum([a.GetFormalCharge() for a in frag.GetAtoms()]) < 0

    def is_neutral(frag):
        return sum([a.GetFormalCharge() for a in frag.GetAtoms()]) == 0

    anionic_frags = [Chem.Mol(f) for f in fragments if is_anionic(f)]

    ligand_frags = []
    for f in fragments:
     if is_neutral(f):
        l = Chem.Mol(f)
        Chem.AssignStereochemistry(l, force=True, cleanIt=False)
        ligand_frags.append(l)

    anion_handlers = {
        'Cl': {'query': Chem.MolFromSmiles('[Cl-]'), 'donor_idx': [0]},
        'Br': {'query': Chem.MolFromSmiles('[Br-]'), 'donor_idx': [0]},
        'OTf': {'query': Chem.MolFromSmiles('O=S(=O)([O-])C(F)(F)F'), 'donor_idx': [3]},
        'AcO': {'query': Chem.MolFromSmiles('CC(=O)[O-]'), 'donor_idx': [3]},
        'ClO4': {'query': Chem.MolFromSmiles('[O-][Cl+3]([O-])([O-])[O-]'), 'donor_idx': [0]},
    }

    base_dir = Path(__file__).resolve().parent
    smarts_path = base_dir / "SMARTS"/ "Ligands_SMARTS.json"
    
    with open(smarts_path, "r") as f:
         Ligands_SMARTS = json.load(f)
  

    ligand_handlers = {}

    for name, entry in Ligands_SMARTS.items():
        if "Smarts" in entry:
            query = Chem.MolFromSmarts(entry["Smarts"])
        elif "Smiles" in entry:
            query = Chem.MolFromSmiles(entry["Smiles"])
        else:
            continue

        ligand_handlers[name] = {
            "query": query,
            "donor_idx": entry["donor_idx"],
            "condition": entry.get("condition", None)
         }

    lig_mol_list = []
    coord_idx_list_per_lig = []
    bond_types_per_lig = []
    supporting_ligs = []
    unused_supporting_ligs = []
    for ligs in ligand_frags:
        matched = False
        for name, handler in ligand_handlers.items():
            cond = handler.get("condition", None)
            if cond is not None and ligs.GetNumAtoms() != cond:
               continue
            query = handler['query']
            donor_idx = handler['donor_idx']
            match = ligs.GetSubstructMatch(query)
            if match:
                lig_donor_idx = [match[i] for i in donor_idx]
                if name in ['MeCN', 'PhCN','Allyl','Dba','COD']:
                    bond_types = [Chem.BondType.DATIVE] * len(lig_donor_idx)
                    supporting_ligs.append((Chem.Mol(ligs), lig_donor_idx, bond_types))
                    matched = True
                    break
                if name == 'L04_07_17': #Here we are only fixing the donor idx
                   if metal_atom.GetSymbol() == 'Cu':
                      lig_donor_idx = [lig_donor_idx[0], lig_donor_idx[2]]
                   else:
                      lig_donor_idx = lig_donor_idx
                if name == 'Diazo':
                    try:
                        # Process diazo to carbene and remove N2
                        carbene_mol, carbene_idx = process_diazo_ligand(ligs)
                        lig_mol_list.append(carbene_mol)
                        coord_idx_list_per_lig.append([carbene_idx])
                        bond_types_per_lig.append([Chem.BondType.DOUBLE])
                        matched = True
                        break
                    except Exception as e:
                        print(f"Error processing Diazo ligand: {e}") 
                if name not in ['MeCN', 'PhCN','Allyl','Dba', 'Diazo']:
                    lig_mol_list.append(Chem.Mol(ligs))
                    coord_idx_list_per_lig.append(lig_donor_idx)
                    bond_types_per_lig.append([Chem.BondType.DATIVE] * len(lig_donor_idx))
                matched = True
                break
        if not matched:
            print("No match found for ligand fragment.")

    supporting_anions = []

    for anion in anionic_frags:
        for name, handler in anion_handlers.items():
            if anion.HasSubstructMatch(handler['query']):
                donor_idx = handler['donor_idx'][0]
                supporting_anions.append((anion, donor_idx))
                break

    n_support_ligs = len(supporting_ligs)
    n_support_anions = len(supporting_anions)
    n_attach = min(n_support_ligs, n_support_anions)

     
    # This section depends on the chemical environment and metal
    if n_support_ligs==2 and n_support_anions==1 :
        lig_mol_list.append(supporting_anions[0][0])
        coord_idx_list_per_lig.append([supporting_anions[0][1]])
        bond_types_per_lig.append([Chem.BondType.DATIVE])
        for i in range(n_support_ligs):
            unused_supporting_ligs.append(supporting_ligs[i][0])

    elif n_support_ligs==2 and n_support_anions==2 :
        lig_mol_list.append(supporting_anions[0][0])
        coord_idx_list_per_lig.append([supporting_anions[0][1]])
        bond_types_per_lig.append([Chem.BondType.DATIVE])
        for i in range(n_support_ligs):
            unused_supporting_ligs.append(supporting_ligs[i][0])

    elif n_support_ligs==1 and n_support_anions==1 :
       for i in range(n_attach): 
          lig_mol_list.append(supporting_anions[i][0])
          coord_idx_list_per_lig.append([supporting_anions[i][1]])
          bond_types_per_lig.append([Chem.BondType.DATIVE])

    elif n_support_ligs==1 and n_support_anions==2 :
         lig_mol_list.append(supporting_anions[0][0])
         coord_idx_list_per_lig.append([supporting_anions[0][1]])
         bond_types_per_lig.append([Chem.BondType.DATIVE])
         for i in range(n_support_ligs):
             unused_supporting_ligs.append(supporting_ligs[i][0]) 

    elif n_support_ligs == 0 :
       if metal_atom.GetSymbol() == "Cu":
          pass 
       else:
          if n_support_anions != 0:    
            lig_mol_list.append(supporting_anions[0][0])
            coord_idx_list_per_lig.append([supporting_anions[0][1]]) 
            bond_types_per_lig.append([Chem.BondType.DATIVE])  

    elif n_support_anions == 0 :
       if metal_atom.GetSymbol() == "Cu" :
           for i in range(n_support_ligs):
               unused_supporting_ligs.append(supporting_ligs[i][0]) 
       else:
          if n_support_ligs != 0 :
             lig_mol_list.append(supporting_ligs[0][0])
             coord_idx_list_per_lig.append(supporting_ligs[0][1])
             bond_types_per_lig.append(supporting_ligs[0][2])
    
    final_mol = link_multiple_ligs_to_metal(lig_mol_list, metal_frag, coord_idx_list_per_lig,bond_types_per_lig)
     
    return Chem.MolToSmiles(final_mol, canonical=False)




