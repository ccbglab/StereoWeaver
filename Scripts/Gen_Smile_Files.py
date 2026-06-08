import glob
import os
import re
from pathlib import Path
import comtypes.client as w32
from rdkit import Chem
import pandas as pd

# ==============================
# Name-Normalization Functions
# ==============================
def smart_normalize(name):
    name = Path(str(name)).stem.strip()
    match = re.match(r"([A-Za-z]+)(\d+)$", name)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(number)}"
    return name

# ==============================
# Main Function
# ==============================

def cdx2smi(cdx_folder, dest_folder):
    """
    This function reads ChemDraw files (*.cdx, *.cdxml) from a folder,
    extracts SMILES strings using ChemDraw's COM API, validates them with RDKit,
    and saves the results into a CSV file.

    Parameters:
    -----------
    cdx_folder : str
        Folder containing .cdx or .cdxml files.
    
    dest_folder : str
        Destination folder where the 'smile.csv' will be saved.
    """
    files = glob.glob(os.path.join(cdx_folder, '*.cdx')) + \
            glob.glob(os.path.join(cdx_folder, '*.cdxml'))

    try:
        # Initialize a COM instance of ChemDraw.
        # Make sure the correct ProgID is used, based on your installed version.
        # You can find the ProgID via Windows Registry under: (Win + R -> regedit)
        # HKEY_CLASSES_ROOT\ (e.g., ChemDraw_x64.Application or ChemDraw.Application.19)
        ChemDraw = w32.CreateObject("ChemDraw_x64.Application")
    except Exception as e:
        print(f"❌ Error initializing ChemDraw: {e}")
        return

    smi_dict = {'Name': [], 'SMILES': []}

    for tmp_file in files:
        try:
            file_name = Path(tmp_file).stem
            file_name = smart_normalize(file_name)
            # Open the ChemDraw file using the COM API
            tmp_compound = ChemDraw.Documents.Open(tmp_file)
            # Extract SMILES from the opened ChemDraw document
            # Data format "chemical/x-smiles" is a standard ChemDraw export type
            smiles = tmp_compound.Objects.Data("chemical/x-smiles")
            
            # Use RDKit to sanitize and standardize the SMILES string
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                canonical=True
            ) if mol else ''

        except Exception as e:
            print(f"❌ Error processing {tmp_file}: {e}")
            smiles = ''

        smi_dict['Name'].append(file_name)
        smi_dict['SMILES'].append(smiles)

    ChemDraw.Quit()

    # ------------------------------
    # Create destination folder
    # ------------------------------
    os.makedirs(dest_folder, exist_ok=True)

    df = pd.DataFrame.from_dict(smi_dict)

    # ------------------------------
    # Save full CSV
    # ------------------------------
    full_path = os.path.join(dest_folder, 'smile.csv')
    df.to_csv(full_path, index=False)
    print(f"Full SMILES saved: {full_path}")

    # ------------------------------
    # Strict grouping (M, N, D, L, S + numbers ONLY)
    # ------------------------------
    prefix_groups = {
        "M": [],
        "N": [],
        "D": [],
        "L": [],
        "S": []
    }
    prefix_file_names = {
    "M": "Active_Metal_Species.csv",
    "N": "Nucleophiles.csv",
    "D": "Diazos.csv",
    "L": "Ligands.csv",
    "S": "Solvents.csv"
}
    others = []

    for _, row in df.iterrows():
        name = row["Name"]

        match = re.match(r"^([MNDLS])\d+$", name)

        if match:
            prefix = match.group(1)
            prefix_groups[prefix].append(row)
        else:
            others.append(row)

    # ------------------------------
    # Save grouped CSVs
    # ------------------------------
    for prefix, rows in prefix_groups.items():
       if rows:
        sub_df = pd.DataFrame(rows)
        sub_df["sort_num"] = (sub_df["Name"].str.extract(r'(\d+)').astype(int))
        sub_df = (sub_df.sort_values("sort_num").drop(columns="sort_num"))
        filename = prefix_file_names.get(prefix, f"{prefix}.csv")
        file_path = os.path.join(dest_folder, filename)
        sub_df.to_csv(file_path, index=False)
        print(f"Saved {prefix}: {file_path}")

    # ------------------------------
    # Save others
    # ------------------------------
    if others:
        others_path = os.path.join(dest_folder, "others.csv")
        pd.DataFrame(others).to_csv(others_path, index=False)
        print(f"Saved others: {others_path}")


# ==============================
# Run Script
# ==============================

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.dirname(script_dir)

    # Define the folder containing ChemDraw files (.cdx/.cdxml)
    cdx_folder = os.path.join(base_dir, "Data", "Chemdraw")

    # Define the folder where the resulting CSV will be saved
    dest_folder =os.path.join(base_dir, "Data", "SMILES")

    cdx2smi(cdx_folder, dest_folder)

    print("SMILES extracted and categorized.")