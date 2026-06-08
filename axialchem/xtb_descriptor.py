##################################################################################################################
#   Main code block to process geometries, run xTB calculations, and compute all descriptors. (xTB_descriptors.py)
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

######################################################################################
# Generates 3D Geometries with fixed Axial Chirality and writes the output to .xyz and
# .sdf Files.
######################################################################################
def generate_3d_xyz(smile, stereo_tag, xyz_filename, folder):

    # For defining axial chirality
    base_dir = Path(__file__).resolve().parent
    smarts_path = base_dir / "SMARTS"/ "SMARTS_AC.json"
    
    with open(smarts_path, "r") as f:
         smarts_patterns = json.load(f)

    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)

    if all(x in xyz_filename for x in ['M', 'D']):
      # Process metal–carbene topology to improve RDKit embedding
      carbene_idx = None
      metal_idx = None
      # Find metal index
      metals = ('Fe', 'Cu', 'Pd', 'Co')
      for atom in mol.GetAtoms():
        if atom.GetSymbol() in metals:
            metal_idx= atom.GetIdx()
      # Find carbene index       
      for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        c_idx = atom.GetIdx()
        # Check if bonded to metal
        if mol.GetBondBetweenAtoms(c_idx, metal_idx) is None:
           continue
        # Check neighbors of this carbon
        for nbr in atom.GetNeighbors():
            nbr_idx= nbr.GetIdx()
            if nbr_idx == metal_idx:
                continue  # skip metal
            oxy_count = sum(1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == "O")
            if oxy_count >= 2:
                # Found carbene carbon
                carbene_idx = c_idx
               
      #Re-building Metal-carbene  
      # Step 1: Removal of metal-carbene bond 
      mol_nb = Chem.RWMol(mol)
      bond = mol_nb.GetBondBetweenAtoms(metal_idx, carbene_idx)
      if bond is not None:
       mol_nb.RemoveBond(metal_idx, carbene_idx)
      else:
        print(f"[ERROR] No bond found between metal: {metal_idx}, and carbene: {carbene_idx}")          
      mol_nb.UpdatePropertyCache(strict=False)
      mol_nb = mol_nb.GetMol()

      #Indentifying Fragments
      fragments = Chem.GetMolFrags(mol_nb, asMols=True, sanitizeFrags=False)

      # Recombine fragments deterministically and re-add metal-carbene bond
      combined = Chem.CombineMols(fragments[0], fragments[1])
      mol_b = Chem.RWMol(combined)
      metal_idx_new = None
      carbene_idx_new = None
      for atom in mol_b.GetAtoms():
        if atom.GetSymbol() in metals:
            metal_idx_new= atom.GetIdx()
            
      for atom in mol_b.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        c_idx = atom.GetIdx()
        # Check neighbors of this carbon
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == metal_idx_new:
                continue  # skip metal
            oxy_count = sum(1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == "O")
            if oxy_count >= 2:
                # Found carbene carbon
                carbene_idx_new = c_idx
      atom = mol_b.GetAtomWithIdx(carbene_idx_new)
      atom.SetFormalCharge(0)
      atom.SetNumExplicitHs(0)
      atom.SetNoImplicit(True)
      atom.SetNumRadicalElectrons(2)       

      bond = mol_b.GetBondBetweenAtoms(metal_idx_new, carbene_idx_new)
      if bond is None:
       mol_b.AddBond(carbene_idx_new,metal_idx_new, Chem.BondType.DOUBLE)
      else:
       print(f"[Error] No Double bond found between metal: {metal_idx_new}, and carbene: {carbene_idx_new}")          
      mol_b.UpdatePropertyCache(strict=False)
      mol_b = mol_b.GetMol()

      if mol_b.GetNumConformers() == 0:
       print("[INFO]New Metal-Carbene generated. Embedding molecule...")
       params = AllChem.ETKDGv3()
       params.randomSeed = 0xf00d
       AllChem.EmbedMolecule(mol_b, params)
      
      #Add Atom mapping 
      for atom in mol_b.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)   
      U_smile = Chem.MolToSmiles(mol_b,canonical=False,isomericSmiles=True)
      #Re-initialise mol  
      mol = mol_b
      mol = Chem.AddHs(mol) 
    else:
      for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)   
      U_smile = Chem.MolToSmiles(mol,canonical=False,isomericSmiles=True)

    # SMARTS matching & Dihedral Angle set up
    maxattempts = 10000
    matched_atoms = []

    for s_name, smarts in smarts_patterns.items():
        substructure = Chem.MolFromSmarts(smarts)
        if substructure and mol.HasSubstructMatch(substructure):
            match = mol.GetSubstructMatch(substructure)
            print(f"[INFO] Matched SMARTS: {s_name}")
            if s_name =="spiro_C":
                matched_atoms.append([match[7], match[8], match[16], match[11]])
                matched_atoms.append([match[2], match[3], match[8], match[16]]) 
                matched_atoms.append([match[3], match[8], match[16], match[15]])
            elif s_name =="spiro_Si":
                matched_atoms.append([match[7], match[8], match[16], match[11]])
                matched_atoms.append([match[2], match[3], match[8], match[16]]) 
                matched_atoms.append([match[3], match[8], match[16], match[15]])
            elif s_name == "Biimidazole":
                matched_atoms.append([match[7], match[11], match[12], match[16]])
                matched_atoms.append([match[4], match[3], match[23], match[18]])
            elif s_name == "Bipyridine":
                matched_atoms.append([match[5], match[4], match[3], match[2]])
            elif s_name == "Biphenyl":
                matched_atoms.append([match[5], match[4], match[3], match[2]])
            elif s_name == "Allene":
                # match: (left_C, central_C, right_C)
                alC3, alC2, alC1 = match

                Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

                # neighbors excluding axis
                alC3_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(alC3).GetNeighbors() if n.GetIdx() != alC2]
                alC1_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(alC1).GetNeighbors() if n.GetIdx() != alC2]

                # require two substituents on each side
                if len(alC3_nbrs) < 2 or len(alC1_nbrs) < 2:
                   print("[INFO] Not axially chiral allene")
                else:
                   # CIP ranks
                   alC3_ranks = get_cip_ranks(mol, alC3_nbrs)
                   alC1_ranks = get_cip_ranks(mol, alC1_nbrs)

                   # highest priority substituents
                   alC3_high = max(alC3_ranks, key=lambda x: x[1])[0]
                   alC1_high = max(alC1_ranks, key=lambda x: x[1])[0]

                   matched_atoms.append([alC3_high, alC3, alC1, alC1_high])
            break
    if not matched_atoms: 
        s_name = None
    geometry_embedded = False
    if matched_atoms and stereo_tag != 0:
        if stereo_tag == 2:
            stereo_pattern = [-1, 1] 
        elif stereo_tag == -2:
            stereo_pattern = [1, -1]
        else:
            stereo_pattern = [np.sign(stereo_tag)] * (len(matched_atoms))
        for i in range(maxattempts):
            # print(f"Attempt {i+1}/{maxattempts}")
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.randomSeed = i + 1
            if AllChem.EmbedMolecule(mol,params) == 0:
                conf = mol.GetConformer()
                dihedral_check = True
                dihedrals = []

                for idx, quartet in enumerate(matched_atoms):
                    dihedral = Chem.rdMolTransforms.GetDihedralDeg(conf, *quartet)
                    # print(dihedral)
                    if not (dihedral * stereo_pattern[idx] >= 0 ):
                       dihedral_check = False
                       break

                if dihedral_check:
                    AllChem.MMFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    if any(
                          Chem.rdMolTransforms.GetDihedralDeg(conf, *q) * stereo_pattern[idx] <= 0
                          for idx, q in enumerate(matched_atoms)):
                        continue

                    dihedrals = [Chem.rdMolTransforms.GetDihedralDeg(conf, *q) for q in matched_atoms]
                    print("[INFO] Dihedral angles initialised:", ", ".join(f"{d:.2f}°" for d in dihedrals))
                    geometry_embedded = True
                    break

    if not geometry_embedded:
        print("[INFO] Using general ETKDGv3 embedding.")
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id < 0 or mol.GetNumConformers() == 0:
            print("[INFO] ETKDGv3 failed; using random-coordinate distance geometry")
            conf_id = AllChem.EmbedMolecule(mol, useRandomCoords=True)

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Export to XYZ
    os.makedirs(folder, exist_ok=True)
    xyz_path = os.path.join(folder, f"{xyz_filename}.xyz")
    with open(xyz_path, 'w') as f:
        f.write(f"{len(atoms)}\n{xyz_filename}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

    return xyz_path, xyz_filename, U_smile, mol, s_name, matched_atoms

######################################################################################
# Creates constraint tmp.inp File for Metal-Ligand Complexes.
######################################################################################
def write_xtb_constraints_from_xyz(xyz_path, matched_atoms, ligand_name):

    metal_symbols = {'Fe', 'Cu', 'Pd', 'Co'}
    atoms, coords = xyz2coords(xyz_path)
    constraints = []
    d_constraints= []
    added_pairs = set()
    dihedrals = []

    for i, (sym_i, coord_i) in enumerate(zip(atoms, coords)):
        if sym_i not in metal_symbols:
            continue

        for j, (sym_j, coord_j) in enumerate(zip(atoms, coords)):
            if i == j:
                continue

            dist = np.linalg.norm(coord_i - coord_j)

            # Constraint for typical dative-type interactions
            if sym_j not in metal_symbols and dist < 2.6:
                pair = tuple(sorted((i, j)))
                if pair not in added_pairs:
                    constraints.append((i + 1, j + 1, dist))
                    added_pairs.add(pair)

    for i, (a,b,c,d) in enumerate(matched_atoms):       
        D = dihedral(coords[a], coords[b], coords[c], coords[d])  
        d_constraints.append((a+1, b+1, c+1, d+1, D)) 
        dihedrals.append(D)     

    if not constraints:
        print("[INFO] No constraints found in XYZ.")
        return None

    # Write constraints in the requested format
    folder = os.path.dirname(xyz_path) or "."
    os.makedirs(folder, exist_ok=True)
    constrain_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".inp", dir=folder)

    constrain_file.write("$constrain\n")
    for i, j, dist in constraints:
        # xTB expects space-separated fields
        if dist < 2.3:
           constrain_file.write(f"  distance: {i}, {j}, {dist:.2f}\n")
        else:
           constrain_file.write(f"  distance: {i}, {j}, 2.3\n")
    for a, b, c, d, D in d_constraints:
        if abs(D) < 15:
           D += 20 * np.sign(D)
           constrain_file.write(f"  dihedral: {a}, {b}, {c}, {d}, {D:.2f}\n") 
        else:  
           constrain_file.write(f"  dihedral: {a}, {b}, {c}, {d}, {D:.2f}\n")   
    constrain_file.write("$end\n")

    print(f"[INFO] Temporary constraint file generated")
    return constrain_file.name

######################################################################################
# Runs xTB geometry optimization followed by single-point calculations.
######################################################################################
class xTBCalculator():
    def __init__(self, xyz_path, xyz_filename, mol, matched_atoms, ligand_name, method=2):
        self.mol = mol
        self.matched_atoms = matched_atoms
        self.ligand_name = ligand_name
        self.charge = Chem.GetFormalCharge(mol)
        self.method = method
        self.cwd = os.getcwd()
        self.input_xyz = xyz_path
        self.xyz_file = f"{xyz_filename}.xyz"
        self.xyz_name = xyz_filename
        self.xyz_dir = os.path.dirname(xyz_path)
        self.spin_flags = self.get_spin_flags()
       
        # Setup xTB working directory
        self.xtb_dir = os.path.join(self.xyz_dir, 'xTB')
        self.xtb_calc_dir = os.path.join(self.xtb_dir, self.xyz_name)
        os.makedirs(self.xtb_calc_dir, exist_ok=True)
        
        #Copy the input.xyz file in the working directory
        self.xyz_path = os.path.join(self.xtb_calc_dir, self.xyz_file)
        shutil.copyfile(self.input_xyz, self.xyz_path)
        os.remove(self.input_xyz)
        
        #Change to the working directory
        os.chdir(self.xtb_calc_dir)
       
    def get_spin_flags(self):
        for atom in self.mol.GetAtoms():
            Z = atom.GetSymbol()
            q = atom.GetFormalCharge()

            if Z == "Fe" and q == 2:
               return " --uhf 4"
            elif Z == "Fe" and q == 3:
               return " --uhf 5"
            elif Z == "Co" and q == 2:
               return " --uhf 3"
            elif Z == "Cu" and q == 1:
               return " --uhf 0"
            elif Z == "Pd":
               return " --uhf 0"

        return ""   # closed-shell default

    def run_xtb(self, input_file=None, output_name="xtbopt.xyz"):
        inp_flag = f"--input {input_file}" if input_file else ""

        cmd = f"xtb {self.xyz_file} --opt --gfn {self.method} --charge {self.charge} {self.spin_flags} {inp_flag}".strip()
        cmd += " > xtboptlog"
        
        run(cmd, shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)

        return os.path.join(self.xtb_calc_dir, output_name)

    def constrained_optimization(self): 
       # Try to create constraint file
       constraint_path = write_xtb_constraints_from_xyz(self.xyz_path, self.matched_atoms,self.ligand_name)

       # If constraints are found and file created
       if constraint_path and os.path.exists(constraint_path):
          inp_path = os.path.join(self.xtb_calc_dir, 'xtb.inp')
          shutil.copyfile(constraint_path, inp_path)
          os.remove(constraint_path)

          print(f"[INFO] Starting constrained optimization for {self.xyz_name}")
          opt_path = self.run_xtb(input_file='xtb.inp')

          # Replace current input with optimized geometry
          if os.path.exists(opt_path):
             shutil.copyfile(opt_path, self.xyz_path)
             os.remove(opt_path)

          # Cleanup constraint file
          os.remove(inp_path)
          print(f"[INFO] Deleted temporary constraint file")

          # Continue to unconstrained optimization
          self.unconstrained_optimization()

       else:
          # No constraints — directly run unconstrained
          self.unconstrained_optimization()

    def unconstrained_optimization(self):
        print(f"[INFO] Starting unconstrained optimization for {self.xyz_name}")
        opt_path = self.run_xtb()
        if os.path.exists(opt_path):
           print("[INFO] Optimization succeeded")

    def single_point(self):
        os.chdir(f'{self.xtb_dir}/{self.xyz_name}')
        xtbopt_file = f'{self.xtb_dir}/{self.xyz_name}/xtbopt.xyz'
        if not os.path.exists(xtbopt_file):
            print(f'[ERROR] xTB optimization task failed ({self.xyz_name})')
            return False
        cmd_lst = [
            f'xtb xtbopt.xyz --gfn {self.method} --charge {self.charge} {self.spin_flags} > splog',
            f'xtb xtbopt.xyz --gfn {self.method} --charge {self.charge} {self.spin_flags} --acc 1.0 --vfukui > vfukuilog',
            f'xtb xtbopt.xyz --gfn {self.method} --charge {self.charge} {self.spin_flags} --acc 1.0 --vipea > vipealog',
            f'xtb xtbopt.xyz --gfn {self.method} --charge {self.charge} {self.spin_flags} --acc 1.0 --vomega > vomegalog' 
        ]
        print(f'[INFO] Single point calculations for {self.xyz_name}')
        for cmd in cmd_lst:
            run(cmd, shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        return True
    
    def full_workflow(self):
        self.constrained_optimization()
        self.single_point()
        
    def get_output_dir(self):
        return self.xtb_calc_dir
 
######################################################################################
# Records all possible local and global descriptors.
######################################################################################         
class xTBDescriptor():
    def __init__(self):
        pass
    def read_desc(self,xtb_calc_dir,force=False):
        self.xtb_calc_dir = xtb_calc_dir
        self.charges_f = f'{xtb_calc_dir}/charges'
        self.splog = f'{xtb_calc_dir}/splog'
        self.vipealog = f'{xtb_calc_dir}/vipealog'
        self.vomegalog = f'{xtb_calc_dir}/vomegalog'
        if not os.path.exists(f'{xtb_calc_dir}/xtbopt.xyz'):
            return 
        with open(f'{xtb_calc_dir}/xtbopt.xyz','r') as fr:
            lines = fr.readlines()
        self.atom_num = int(lines[0].strip())
        self.desc_ens = {}
        self.desc_ens['global'] = {}
        self.desc_ens['local'] = {}
        sp_all_done_flag = True
        if force or not self.load():
            if self.is_exists(self.charges_f):
                self.read_charges()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.splog):
                self.read_splog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.vipealog):
                self.read_vipealog()
            else:
                sp_all_done_flag = False
            if self.is_exists(self.vomegalog):
                self.read_vomegalog()
            else:
                sp_all_done_flag = False
        if sp_all_done_flag:
            self.save()
    def is_exists(self,file):
        return os.path.exists(file)
    def read_charges(self):
        with open(self.charges_f,'r') as fr:
            lines = fr.readlines()
        charges = np.array([float(chrg.strip()) for chrg in lines])
        self.desc_ens['charges'] = charges
        self.desc_ens['local']['charges'] = charges.reshape(-1,1)
    def read_splog(self):
        with open(self.splog,'r',encoding='utf-8', errors='ignore') as fr:
            lines = fr.readlines()
        for i,line in enumerate(lines):
            if '(HOMO)' in line:
                homo = float(line.strip().split()[-2])
                self.desc_ens['HOMO'] = homo
                self.desc_ens['global']['HOMO'] = homo
            elif '(LUMO)' in line:
                lumo = float(line.strip().split()[-2])
                self.desc_ens['LUMO'] = lumo
                self.desc_ens['global']['LUMO'] = lumo
            elif 'TOTAL ENERGY' in line:
                tot_energy = float(line.strip().split()[-3])
                self.desc_ens['E'] = round(tot_energy,8)
                self.desc_ens['global']['E'] = round(tot_energy,8)
            elif 'molecular dipole' in line:
                qx,qy,qz = [float(q) for q in lines[i+2].strip().split()[-3:]]
                full_x,full_y,full_z,full_tot = [float(q) for q in lines[i+3].strip().split()[-4:]]
                self.desc_ens['dipole_q_xyz'] = np.array([qx,qy,qz])
                self.desc_ens['dipole_full_xyz'] = np.array([full_x,full_y,full_z])
                self.desc_ens['dipole_tot'] = full_tot
                self.desc_ens['global']['dipole'] = np.concatenate([np.array([qx,qy,qz]),
                                                                    np.array([full_x,full_y,full_z]),
                                                                    np.array([full_tot])])
        self.desc_ens['GAP'] = round(self.desc_ens['LUMO'] - self.desc_ens['HOMO'],8)
        self.desc_ens['global']['GAP'] = self.desc_ens['GAP']  
    def read_vipealog(self):
        with open(self.vipealog,'r',encoding='utf-8', errors='ignore') as fr:
            lines = fr.readlines()
        for line in lines:
            if 'delta SCC IP (eV):' in line:
                vip = float(line.strip().split()[-1]) # vertical IP
                self.desc_ens['VIP'] = vip
                self.desc_ens['global']['VIP'] = vip
            elif 'delta SCC EA (eV):' in line:
                vea = float(line.strip().split()[-1]) # vertical EA
                self.desc_ens['VEA'] = vea
                self.desc_ens['global']['VEA'] = vea
    def read_vomegalog(self):
        with open(self.vomegalog,'r',encoding='utf-8', errors='ignore') as fr:
            lines = fr.readlines()
        for line in lines:
            if "Global electrophilicity index (eV):" in line:
                GEI = float(line.strip().split()[-1])
                self.desc_ens['GEI'] = GEI
                self.desc_ens['global']['GEI'] = GEI
    def save(self,path=None):
        if path == None:
            path = f'{self.xtb_calc_dir}/desc_ens.npy'
        np.save(path,self.desc_ens)
    def load(self,path=None):
        if path == None:
            path = f'{self.xtb_calc_dir}/desc_ens.npy'
        try:
            self.desc_ens = np.load(path,allow_pickle=True).item()
            return True
        except:
            return False

######################################################################################
# Calcultes the steric descriptos
######################################################################################        
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

######################################################################################
# Processes a .xlsx file with smiles data and generates all descriptors.
######################################################################################  
def process_xtb(smiles_file, xtb_folder, output_excel, dft_folder=None, compute_descriptor= False):

    if smiles_file.endswith(".csv"):
        df = pd.read_csv(smiles_file)
    elif smiles_file.endswith((".xls", ".xlsx")):
        df = pd.read_excel(smiles_file)
    else:
        raise ValueError("[ERROR]Unsupported input format")

    all_descs = []

    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        name = row["Name"]
        stereo_tag = int(row["Stereo_tag"])
        start = time.time()
        print(f"\n▶ Processing: {name}")

        try:
            input_smile = build_active_metal_ligand_catalyst(smiles)
            print(f"Input SMILE: {input_smile}")
            
            xyz_path, xyz_filename, U_smile, mol, cluster_name, matched_atoms = generate_3d_xyz(input_smile, stereo_tag, name, xtb_folder)

            calc = xTBCalculator(xyz_path, xyz_filename, mol, matched_atoms, cluster_name)
            xtb_dir = calc.get_output_dir()

            opt_xyz_file = os.path.join(xtb_dir, 'xtbopt.xyz')
            mol_file = os.path.join(xtb_dir, 'xtbopt.mol')
            if dft_folder is not None:
                dft_xyz_file = os.path.join(dft_folder, name, f"{name}.xyz")

            if not os.path.exists(opt_xyz_file):
               calc.constrained_optimization()
               calc.single_point()

            smiles_xyz_to_molfile(U_smile,opt_xyz_file,mol_file)
            mol_object = Chem.MolFromMolFile(mol_file, removeHs=False)
            # img = Draw.MolToImage(mol_object, size=(600,600))
            # img.show()

            symbols,coords = xyz2coords(opt_xyz_file)
           
            if name.startswith('L') or compute_descriptor == False:
                print(f"[INFO] Skipping descriptor generation for ligand: {name}")
                desc_row = OrderedDict()
                desc_row["Name"] = name
                desc_row["SMILES"] = U_smile
                desc_row["CXSMILES"] = smiles_xyz_to_cxsmiles(U_smile, opt_xyz_file)
                all_descs.append(desc_row)
                end = time.time()
                print(f"[INFO] Time: {name}: {end - start:.2f} sec")
                continue

            donor_name = None
            if 'M'  in name:
                key_idx, neighbor_idx, donor_name = get_metal_and_neighbors(mol_object, opt_xyz_file)
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
            desc_bond_len = get_bond_length(opt_xyz_file, key_idx, neighbor_idx, donor_name)
            desc_angle = get_bond_angles(opt_xyz_file, key_idx, neighbor_idx, donor_name)
            desc_dihedral= get_axial_dihedrals(opt_xyz_file,mol_object)
            desc_c_bond_len= get_bond_length(opt_xyz_file, c_idx, c_nbrs, c_donor_name)
            desc_c_angle = get_bond_angles(opt_xyz_file, c_idx, c_nbrs)


            ase_atoms = ase_read(opt_xyz_file, format='xyz')
            acsf_desc = acsf.create(ase_atoms, centers=[key_idx]).reshape(-1)
            soap_desc = soap.create(ase_atoms, centers=[key_idx]).reshape(-1)
            mbtr_desc = mbtr.create(ase_atoms).reshape(-1)

            xtb_desc = xTBDescriptor()
            xtb_desc.read_desc(xtb_dir)
            global_desc = xtb_desc.desc_ens["global"]
            charges_desc = xtb_desc.desc_ens['local']['charges']

            CLUSTER_MAP = {
               "spiro_C": 1,
               "spiro_Si": 2,
               "Biimidazole": 3,
               "Bipyridine": 4,
               "Biphenyl": 5,
             }
            cluster = CLUSTER_MAP.get(cluster_name, 6) 

            desc_row = OrderedDict()

            desc_row["Name"] = name
            desc_row["SMILES"] = U_smile
            desc_row["CXSMILES"] = smiles_xyz_to_cxsmiles(U_smile,opt_xyz_file)
 
            desc_row["Key_idx"] = key_idx
            desc_row["Neighbor_idx"] = ','.join(map(str, neighbor_idx))
            if any(x in name for x in ['M', 'L']):
               desc_row["Cluster"] = cluster

            if dft_folder is not None:
                desc_rmsd = RMSD(opt_xyz_file, dft_xyz_file)
                desc_row.update(desc_rmsd)

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
               if key == "dipole" and isinstance(val, (list, tuple, np.ndarray)) and len(val) == 7:
                 dipole_labels = ['qx', 'qy', 'qz', 'full_x', 'full_y', 'full_z', 'full_total']
                 for i, v in enumerate(val):
                   desc_row[f'dipole_{dipole_labels[i]}'] = v
               else:
                 desc_row[key] = val

            charges_desc = np.asarray(charges_desc).flatten()
            desc_row[f'Charge_KeyIdx'] = charges_desc[key_idx]
            charge_counter = 1
            if donor_name is None:
               for nbr in neighbor_idx:
                   if nbr == 999:
                      val = 0
                   else:
                      val = charges_desc[nbr]
                   desc_row[f'Charge_{charge_counter}'] = val
                   charge_counter += 1
            elif len(donor_name) == len(neighbor_idx):
                 for d_name, nbr in zip(donor_name, neighbor_idx):
                     val = 0 if nbr == 999 else charges_desc[nbr]
                     desc_row[f'Charge_{d_name}'] = val
            else:
             print(f"[Warning] donor_name length != neighbor_idx length. Falling back to numbered labels.")
             for nbr in neighbor_idx:
                 if nbr == 999:
                    val = 0
                 else:
                    val = charges_desc[nbr]
                 desc_row[f'Charge_{charge_counter}'] = val
                 charge_counter += 1


            try:
                sterics = StericDescriptor(opt_xyz_file)
                desc_row["Burried Volume"] = sterics.BV(key_idx)[0]

                if donor_name is not None and len(donor_name) == len(neighbor_idx):
                   labels = donor_name
                else:
                   if donor_name is not None:
                      print(f"[Warning] donor_name length != neighbor_idx length. Falling back to numbered labels.")
                   labels = [f"{i+1}" for i in range(len(neighbor_idx))]

                # Sterimol calculation
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

    # Final DataFrame
    df_final = pd.DataFrame(all_descs)
    df_cleaned = df_final.dropna(axis=1)
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]
    df_cleaned.to_csv(output_excel, index=False, na_rep='NaN')


