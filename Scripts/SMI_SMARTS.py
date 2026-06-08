from rdkit import Chem

# Input SMILES string
smarts = "O=C(C=Cc1ccccc1)C=Cc1ccccc1"

# Generate the molecule from SMILES
mol = Chem.MolFromSmiles(smarts)

# Convert the molecule to SMARTS
if mol:
    smarts = Chem.MolToSmarts(mol)
    print(f"Generated SMARTS: {smarts}")
else:
    print("Invalid SMILES string!")

# Function to customize SMARTS formatting
def customize_smarts(smarts):
    smarts = smarts.replace('-', '~')
    smarts = smarts.replace('=', '~')
    smarts = smarts.replace(':', '~')
    return smarts

# Apply the custom formatting
customized_smarts = customize_smarts(smarts)
print(f"Customized SMARTS: {customized_smarts}")




