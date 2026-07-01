# 🧬 StereoWeaver: Axial-Chirality-Aware Dataset Generation Pipeline

This repository accompanies the work:
**“Overcoming the Axial Chirality Bottleneck in Automated Intermediate Generation for Machine Learning Applications.”**

Modern machine learning models in asymmetric catalysis often struggle to distinguish stereochemically unique intermediates, as conventional molecular representations (e.g., SMILES) do not fully capture axial chirality. **StereoWeaver** provides a geometry-aware automated workflow designed to preserve stereochemical information during molecular dataset generation.

The pipeline enables:

* Generation of **stereochemically faithful 3D molecular structures** from 2D molecular inputs
* Explicit enforcement of **axial chirality using SMARTS-based recognition and dihedral constraints**
* Automated construction of **chemically meaningful catalytic intermediates**
* Calculation of **quantum, steric, and geometric molecular descriptors using xTB-based workflows**
* Generation of **machine-learning-ready descriptor datasets**

## Contributions

* **Arushi Tyagi**
  Project conception and design, scientific workflow development, DFT calculations, xTB–DFT benchmarking, molecular visualization, and stereochemical validation.

* **Mayuk Joddar**
  Software architecture, code development, automation, and implementation of the computational pipeline.

* **Dr. Garima Jindal**
  Project supervision, scientific guidance, and overall research direction.

## Installation

### 1. Clone the repository

Clone the StereoWeaver repository and navigate to the project directory.

```bash
git clone https://github.com/ccbglab/StereoWeaver.git
cd StereoWeaver
```

---

### 2. Create the Conda environment

Create the Conda environment using the provided environment file.

```bash
conda env create -f environment/environment.yml
```

Activate the environment:

```bash
conda activate xtb-descriptor-env
```

---

### 3. Install StereoWeaver

Install StereoWeaver in editable mode.

```bash
pip install -e .
```

This registers the `axialchem` package within the active Conda environment, allowing it to be imported into any Python script.

---

### 4. Importing StereoWeaver

After installation, the descriptor generation modules can be imported directly into any Python script.

```python
from axialchem.xtb_descriptor import *
from axialchem.dft_descriptor import *
```

Alternatively, specific functions can be imported:

```python
from axialchem.xtb_descriptor import process_xtb
from axialchem.dft_descriptor import process_dft
```

---

### 5. Running StereoWeaver

An example workflow demonstrating the complete descriptor generation pipeline is provided in:

```text
Scripts/RUN.py
```

Execute the workflow using:

```bash
python Scripts/RUN.py
```



