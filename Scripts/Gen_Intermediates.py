import os
import pandas as pd
from itertools import product

# ---------- Read files ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

metal_df = pd.read_csv(os.path.join(base_dir,"Data","SMILES","Active_Metal_Species.csv"))

diazo_df = pd.read_csv(os.path.join(base_dir,"Data","SMILES","Diazos.csv"))

ligand_df = pd.read_excel( os.path.join( base_dir, "Data", "Ligands.xlsx"))

# ---------- User selections ----------
metal_names = ['M4']
diazo_names = ['D1']
ligand_names = None

# ---------- Filter selected rows ----------

if metal_names is not None:
    selected_metals = metal_df[ metal_df['Name'].isin(metal_names)]
else:
    selected_metals = metal_df

if diazo_names is not None:
    selected_diazos = diazo_df[diazo_df['Name'].isin(diazo_names)]
else:
    selected_diazos = diazo_df

if ligand_names is not None:
    selected_ligands = ligand_df[ligand_df['Name'].isin(ligand_names)]
else:
    selected_ligands = ligand_df  

# ---------- Generate combinations ----------

rows = []

for (_,m),(_,l),(_,d) in product(
        selected_metals.iterrows(),
        selected_ligands.iterrows(),
        selected_diazos.iterrows()
):

    combo_name = f"{m['Name']}_{l['Name']}_{d['Name']}"

    combo_smiles = (
        f"{m['SMILES']}."
        f"{l['SMILES']}."
        f"{d['SMILES']}"
    )

    rows.append({

        'Name':combo_name,
        'SMILES':combo_smiles,
        'Stereo_tag':l['Stereo_tag'],

        'Metal_Name':m['Name'],
        'Metal_SMILES':m['SMILES'],

        'Ligand_Name':l['Name'],
        'Ligand_SMILES':l['SMILES'],
        

        'Diazo_Name':d['Name'],
        'Diazo_SMILES':d['SMILES']

    })


# ---------- Save ----------

comb_df = pd.DataFrame(rows)

output_file = os.path.join(base_dir,"Data","SMILES","Intermediate.csv")

comb_df.to_csv(output_file,index=False)

print(f"Saved {len(comb_df)} combinations")
print(output_file)
