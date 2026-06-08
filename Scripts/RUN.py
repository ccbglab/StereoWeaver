import os

from axialchem.xtb_descriptor import *
from axialchem.dft_descriptor import *

if __name__ == "__main__":
    """
    Keep compute_descriptor=True only for species containing diazo
    compute_descriptor=False (Default)
    """

    # ---------------------------------------------------------
    # Base directories
    # ---------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "Data")
    desc_dir = os.path.join(data_dir, "Descriptor")

    os.makedirs(desc_dir, exist_ok=True)

    # =========================================================
    # Ligand dataset
    # =========================================================

    L_file = os.path.join( data_dir,"Ligands.xlsx")
    L_xtbfolder = os.path.join(data_dir,"Ligands_xtb_files")
    out_L = os.path.join(desc_dir,"Ligands_desc.csv")

    os.makedirs(L_xtbfolder, exist_ok=True)

    process_xtb(L_file, L_xtbfolder, out_L)

    # =========================================================
    # Cu xtb dataset
    # =========================================================

    Cu_file = os.path.join(data_dir,"Cu_Ligand_Carbene_Complex.xlsx")
    Cu_xtbfolder = os.path.join(data_dir,"Cu_xtb_files")
    out_Cu = os.path.join(desc_dir,"Cu_xtb_desc.csv")

    os.makedirs(Cu_xtbfolder, exist_ok=True)

    process_xtb(Cu_file, Cu_xtbfolder, out_Cu, compute_descriptor=True)

    # # =========================================================
    # # Cu dft dataset
    # # =========================================================

    # Cu_dft_file = os.path.join(data_dir,"dft_input.xlsx")
    # Cu_dftfolder = os.path.join(data_dir,"Cu_dft_files")
    # out_dftCu = os.path.join(desc_dir,"Cu_dft_desc.csv")

    # os.makedirs(Cu_dftfolder, exist_ok=True)

    # process_dft(Cu_dft_file, Cu_dftfolder, out_dftCu, compute_descriptor=True)


    # =========================================================
    # POC dataset
    # =========================================================

    POC_file = os.path.join(data_dir,"POC_Complex.xlsx")
    POC_xtbfolder = os.path.join(data_dir,"POC_xtb_files")
    out_POC = os.path.join(desc_dir,"POC_desc.csv")

    os.makedirs(POC_xtbfolder, exist_ok=True)

    process_xtb(POC_file, POC_xtbfolder, out_POC)

