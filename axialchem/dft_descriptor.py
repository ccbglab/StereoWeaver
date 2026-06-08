##################################################################################################################
#   Main code block to process geometries, and compute descriptors. (dft_descriptor.py)
#   Mayuk Joddar.
##################################################################################################################

# Standard library
import os, shutil, tempfile, warnings, time
import json
import subprocess
from subprocess import run, PIPE
from collections import OrderedDict
from pathlib import Path
# Third-party
import pandas as pd
import numpy as np
from numpy import pad
# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms, Draw
# Descriptors (DScribe)
from dscribe.descriptors import MBTR, ACSF, SOAP
# Morfeus (steric descriptors)
from morfeus import BiteAngle, Sterimol, read_xyz, ConeAngle, SASA, BuriedVolume, SolidAngle, VisibleVolume
# ASE
from ase.io import read as ase_read
#utils.py
from .utils import *
# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RDLogger.DisableLog('rdApp.*')


def dft_globaldesc(log_file, n_atoms=None):
    """
    Extract global DFT descriptors from a Gaussian log file.
    Optionally pass n_atoms to truncate Mulliken charges to that length.
    """
    # do not require cclib for parsing lines (we still use if available)
    desc = {
        'global': {},
        'local': {}
    }

    with open(log_file, "r") as f:
        lines = f.readlines()

    mulliken_charges = []
    H = L = None

    for i, line in enumerate(lines):
        # SCF
        if "SCF Done" in line:
            parts = line.split()
            try:
                desc['global']["SCF_energy"] = float(parts[4])
            except:
                pass

        # Zero point / thermals
        if "Zero-point correction=" in line:
            try:
                desc['global']["ZPE_correction"] = float(line.split()[2])
            except:
                pass
        if "Thermal correction to Energy=" in line:
            try:
                desc['global']["Thermal_Energy"] = float(line.split()[-1])
            except:
                pass
        if "Thermal correction to Enthalpy" in line:
            try:
                desc['global']["Enthalpy"] = float(line.split()[-1])
            except:
                pass
        if "Thermal correction to Free Energy" in line:
            try:
                desc['global']["Free_energy"] = float(line.split()[-1])
            except:
                pass

        # Dipole
        if "Dipole moment" in line:
            for j in range(i+1, i+5):
                if j < len(lines) and "Tot=" in lines[j]:
                    parts = lines[j].replace("=", " ").split()
                    try:
                        desc['global']["Dipole_X"] = float(parts[1])
                        desc['global']["Dipole_Y"] = float(parts[3])
                        desc['global']["Dipole_Z"] = float(parts[5])
                        desc['global']["Dipole_Tot"] = float(parts[7])
                    except:
                        pass

        # Polarizability
        if "Isotropic polarizability" in line:
            try:
                desc['global']["Polarizability"] = float(line.split()[-2])
            except:
                pass

        # Mulliken charges
        if " Mulliken charges:" in line or "Mulliken charges and spin densities:" in line:
            # read from i+2 until "Sum of Mulliken charges ="
            for j in range(i+2, len(lines)):
                if "Sum of Mulliken charges" in lines[j]:
                    break
                parts = lines[j].split()
                if len(parts) >= 2:
                    try:
                        mulliken_charges.append(float(parts[-1]))
                    except:
                        continue

        # HOMO / LUMO lines
        if "Alpha  occ. eigenvalues" in line:
            vals = line.split()
            try:
                H = float(vals[-1])
            except:
                pass
        if "Alpha virt. eigenvalues" in line:
            vals = line.split()
            if len(vals) > 4:
                try:
                    L = float(vals[4])
                except:
                    pass

    desc['local']["Mulliken_charges"] = mulliken_charges

    # derived
    if H is not None:
        desc['global']["HOMO"] = H
    if L is not None:
        desc['global']["LUMO"] = L
    if ("HOMO" in desc['global']) and ("LUMO" in desc['global']):
        Hv = desc['global']["HOMO"]
        Lv = desc['global']["LUMO"]
        desc['global']["GAP"] = (Lv - Hv)
        desc['global']["mu"] = (Hv + Lv) / 2.0
        desc['global']["eta"] = (Lv - Hv) / 2.0
        if desc['global']["eta"] and desc['global']["eta"] != 0:
            desc['global']["omega"] = (desc['global']["mu"] ** 2) / (2.0 * desc['global']["eta"])
        else:
            desc['global']["omega"] = None
        desc['global']["N_index"] = -Hv

    return desc

def extract_nbo_charge(nbo_file):
    """
    Extract NBO charges from a Gaussian NBO section.
    Returns a list of floats in the order of atoms.
    """
    nbo_charges = []
    with open(nbo_file, "r") as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        # Detect the NBO section header
        if "Summary of Natural Population Analysis" in line:
            continue
        if "Atom" in line and "No" in line and "Charge" in line:
             start_reading = True
             continue
        if start_reading:
            if (line.strip() == ""
                or "====" in line):
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    charge = float(parts[2])  # 'Charge' is the 3rd column
                    nbo_charges.append(charge)
                except ValueError:
                    continue
    return nbo_charges

class StericDescriptor():
    '''
    0-index
    '''
    def __init__(self,xyz_file):
        self.xyz_file = xyz_file
        self.elements, self.coordinates = read_xyz(self.xyz_file)
    def BV(self,metal_index, excluded_atoms=None, radii=None, include_hs=False, radius=3.5, radii_type='bondi',
           radii_scale=1.17, density=0.001, z_axis_atoms=None, xz_plane_atoms=None):
        '''
        Buried Volume
        '''
        if excluded_atoms != None:
            excluded_atoms = [idx+1 for idx in excluded_atoms]
        if z_axis_atoms != None:
            z_axis_atoms = [idx+1 for idx in z_axis_atoms]
        if xz_plane_atoms != None:
            xz_plane_atoms = [idx+1 for idx in xz_plane_atoms]
        bv = BuriedVolume(self.elements,self.coordinates,metal_index+1,
                          excluded_atoms, radii, include_hs, radius, radii_type,
                          radii_scale, density, z_axis_atoms, xz_plane_atoms)
        return [bv.fraction_buried_volume]

    def Sterimol(self,dummy_index,attached_index,radii=None, radii_type='crc', n_rot_vectors=3600, excluded_atoms=None, calculate=True):
        dummy_index = dummy_index + 1
        attached_index = attached_index + 1
        if excluded_atoms != None:
            excluded_atoms = [idx+1 for idx in excluded_atoms]
        sterimol = Sterimol(self.elements, self.coordinates, dummy_index, attached_index, radii=radii, radii_type=radii_type, n_rot_vectors=n_rot_vectors, excluded_atoms=excluded_atoms, calculate=calculate)
        return [sterimol.L_value, sterimol.B_1_value, sterimol.B_5_value]
    

def process_dft(input_excel, dft_folder, output_excel, nbo_folder=None, compute_descriptor= False): 
    # --- Read input Excel/CSV ---
    if input_excel.endswith(".csv"):
        df = pd.read_csv(input_excel)
    elif input_excel.endswith((".xls", ".xlsx")):
        df = pd.read_excel(input_excel)
    else:
        raise ValueError("Unsupported input format")
    
    all_descs = []

    # --- Create top-level folder for DFT processed files ---
    dft = Path(dft_folder) / "dft"
    dft.mkdir(exist_ok=True, parents=True)
   
    # --- Loop over all molecules in input ---
    for idx, row in df.iterrows():
        name = row["Name"]
        smile= row["SMILES"]
        start = time.time()
        print(f"\n▶ Processing: {name}")

        # --- Create subfolder inside dft/ ---
        dft_sub = dft / name
        dft_sub.mkdir(exist_ok=True, parents=True)

        log_file = Path(dft_folder) / f"{name}.log"
        if not log_file.exists():
            print(f"⚠ Log file not found: {log_file}")
            continue
        
        if nbo_folder is not None:
           nbo_file= Path(nbo_folder) / f"{name}.log" 
           if not nbo_file.exists():
              print(f"⚠ NBO Log file not found: {nbo_file}")
              continue
        
        # --- Copy log file to dft folder ---
        log_file_dest = dft_sub / f"{name}.log"
        shutil.copy2(log_file, log_file_dest)

        # --- Copy NBO log file to dft folder ---
        if nbo_folder is not None:
           nbo_file_dest = dft_sub / f"{name}_nbo.log"
           shutil.copy2(nbo_file, nbo_file_dest)

        # --- Construct mol file path inside dft folder ---
        mol_file = dft_sub / f"{name}.mol"
        xyz_file = dft_sub / f"{name}.xyz"

        try:
            # Convert log to xyz to process DFT data
            log2xyz(log_file_dest, xyz_file)

            #Convert SMILES + DFT to Mol file
            smiles_xyz_to_molfile(smile,xyz_file,mol_file)
            mol_object = Chem.MolFromMolFile(mol_file, removeHs=False)
            # for atom in mol_object.GetAtoms():
            #     atom.SetProp("atomLabel", f"{atom.GetSymbol()}{atom.GetIdx()}")
            # img = Draw.MolToImage(mol_object, size=(600,600))
            # img.show()
            
            symbols,coords = xyz2coords(xyz_file)

            if name.startswith('L') or compute_descriptor == False:
                print(f"[INFO] Skipping descriptor generation for ligand: {name}")
                desc_row = OrderedDict()
                desc_row["Name"] = name
                desc_row["SMILES"] = smile
                desc_row["CXSMILES"] = smiles_xyz_to_cxsmiles(smile, xyz_file)
                all_descs.append(desc_row)
                end = time.time()
                print(f"[INFO] Time: {name}: {end - start:.2f} sec")
                continue

            donor_name = None
            if 'M'  in name:
                key_idx, neighbor_idx, donor_name = get_metal_and_neighbors(mol_object, xyz_file)
                c_idx, c_nbrs, c_donor_name = get_carbene_and_neighbors(mol_object, coords, key_idx)
                species = ['Br', 'C', 'Cl', 'Co', 'Cu', 'F', 'Fe', 'I', 'N', 'O', 'P', 'Pd', 'S', 'Si', 'H']
            else:
                key_idx = find_closest_to_center(coords)
                neighbor_idx = []
                species = ['C', 'Cl', 'F', 'N', 'O','H']
                
            # Initialize descriptors
            acsf = ACSF(species=species, r_cut=6.0,
                        g2_params=[[1, 1], [1, 2], [1, 3]],
                        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]])
            soap = SOAP(species=species, r_cut=6.0, n_max=4, l_max=3)
            mbtr = MBTR(
                species=species,
                geometry={"function": "inverse_distance"},
                grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                periodic=False,
                normalization="l2",
            )

            rdkit_desc, mordred_desc, morgan_desc = Compute_2d_descriptors(mol_object)
            desc_bond_len = get_bond_length(xyz_file, key_idx, neighbor_idx, donor_name)
            desc_angle = get_bond_angles(xyz_file, key_idx, neighbor_idx, donor_name)
            desc_dihedral= get_axial_dihedrals(xyz_file,mol_object)
            desc_c_bond_len= get_bond_length(xyz_file, c_idx, c_nbrs, c_donor_name)
            desc_c_angle = get_bond_angles(xyz_file, c_idx, c_nbrs)

            ase_atoms = ase_read(xyz_file, format='xyz')
            acsf_desc = acsf.create(ase_atoms, centers=[key_idx]).reshape(-1)
            soap_desc = soap.create(ase_atoms, centers=[key_idx]).reshape(-1)
            mbtr_desc = mbtr.create(ase_atoms).reshape(-1)

            dft_desc = dft_globaldesc(log_file_dest)
            global_desc = dft_desc["global"]
            charges_desc = np.asarray(dft_desc['local']["Mulliken_charges"])

            # --- Combine all descriptors ---
            desc_row = OrderedDict()

            desc_row["Name"] = name
            desc_row["SMILES"] = smile
            desc_row["CXSMILES"] = smiles_xyz_to_cxsmiles(smile, xyz_file)
 
            desc_row["Key_idx"] = key_idx
            desc_row["Neighbor_idx"] = ','.join(map(str, neighbor_idx))

            desc_row.update(desc_bond_len)
            desc_row.update(desc_c_bond_len)
            desc_row.update({f"Carbene_{k}": v for k, v in desc_c_angle.items()})
            desc_row.update(desc_dihedral)
            desc_row.update(desc_angle)

            desc_row.update(rdkit_desc)
            desc_row.update(mordred_desc)
            desc_row.update(morgan_desc)


            for i, val in enumerate(acsf_desc, start=1):
                desc_row[f'acsf{i}'] = val
            for i, val in enumerate(soap_desc, start=1):
                desc_row[f'soap{i}'] = val
            for i, val in enumerate(mbtr_desc, start=1):
                desc_row[f'mbtr{i}'] = val

            for key, val in global_desc.items():
                desc_row[key] = val

            # --- Mulliken charges Descriptor ---
            desc_row[f'Mulliken_Charge_KeyIdx'] = charges_desc[key_idx]
            charge_counter = 1
            if donor_name is None:
                for nbr in neighbor_idx:
                    val = 0 if nbr == 999 else charges_desc[nbr]
                    desc_row[f'Mulliken_Charge_{charge_counter}'] = val
                    charge_counter += 1
            elif len(donor_name) == len(neighbor_idx):
                for d_name, nbr in zip(donor_name, neighbor_idx):
                    val = 0 if nbr == 999 else charges_desc[nbr]
                    desc_row[f'Mulliken_Charge_{d_name}'] = val
            else:
                print(f"[Warning] donor_name length != neighbor_idx length. Falling back to numbered labels.")
                for nbr in neighbor_idx:
                    val = 0 if nbr == 999 else charges_desc[nbr]
                    desc_row[f'Mulliken_Charge_{charge_counter}'] = val
                    charge_counter += 1

            # --- NBO Charges Descriptor ---
            if nbo_folder is not None:
               nbo_charges= extract_nbo_charge(nbo_file_dest)
               desc_row[f'NBO_Charge_KeyIdx'] = nbo_charges[key_idx]
               charge_counter = 1
               if donor_name is None:
                  for nbr in neighbor_idx:
                    val = 0 if nbr == 999 else nbo_charges[nbr]
                    desc_row[f'NBO_{charge_counter}'] = val
                    charge_counter += 1
               elif len(donor_name) == len(neighbor_idx):
                  for d_name, nbr in zip(donor_name, neighbor_idx):
                    val = 0 if nbr == 999 else nbo_charges[nbr]
                    desc_row[f'NBO_{d_name}'] = val
               else:
                  print(f"[Warning] donor_name length != neighbor_idx length. Falling back to numbered labels.")
                  for nbr in neighbor_idx:
                    val = 0 if nbr == 999 else nbo_charges[nbr]
                    desc_row[f'NBO_{charge_counter}'] = val
                    charge_counter += 1

            ## --- Sterics Descriptor ---
            try:
                sterics = StericDescriptor(xyz_file)
                desc_row["Burried Volume"] = sterics.BV(key_idx)[0]

                if donor_name and len(donor_name) == len(neighbor_idx):
                    labels = donor_name
                else:
                    if donor_name:
                        print(f"[Warning] donor_name length != neighbor_idx length. Falling back to numbered labels.")
                    labels = [f"{i+1}" for i in range(len(neighbor_idx))]

                for label, nbr in zip(labels, neighbor_idx):
                    if nbr == 999:
                        l = b1 = b5 = 0.0
                    else:
                        l, b1, b5 = sterics.Sterimol(nbr, key_idx)
                    desc_row[f"Sterimol_{label}_L"] = l
                    desc_row[f"Sterimol_{label}_B1"] = b1
                    desc_row[f"Sterimol_{label}_B5"] = b5

            except Exception:
                desc_row["Burried Volume"] = 0.0
                labels = donor_name if (donor_name and len(donor_name) == len(neighbor_idx)) else [f"{i+1}" for i in range(len(neighbor_idx))]
                for label in labels:
                    desc_row[f"Sterimol_{label}_L"] = 0.0
                    desc_row[f"Sterimol_{label}_B1"] = 0.0
                    desc_row[f"Sterimol_{label}_B5"] = 0.0

            all_descs.append(desc_row)
            end = time.time()
            print(f"[INFO] Time: {name}: {end - start:.2f} sec")

        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            continue

    # --- Save final descriptor CSV ---
    df_final = pd.DataFrame(all_descs)
    df_cleaned = df_final.dropna(axis=1)
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    df_cleaned.to_csv(output_excel, index=False, na_rep='NaN')
