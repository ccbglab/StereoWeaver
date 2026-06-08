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
