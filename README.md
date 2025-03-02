# Molecular Data Analysis: Free-Wilson & Molecule Networks

This repository demonstrates a workflow for analyzing molecular data using two complementary approaches:

1. **Free-Wilson Analysis**  
   A simplified method where molecular scaffolds (extracted using RDKit) are used as features to model the effect on potency/activity. The approach involves:
   - Extracting the Murcko scaffold from SMILES strings.
   - Encoding the scaffolds as categorical variables (via one-hot encoding).
   - Fitting a linear regression model to estimate the contribution (coefficient) of each scaffold.

2. **Molecule Networks Visualization**  
   An interactive network visualization where:
   - Morgan fingerprints are computed for each molecule.
   - Pairwise Tanimoto similarities are calculated.
   - An edge is drawn between molecules if the similarity exceeds a user-defined threshold.
   - The network is visualized interactively using the [PyVis](https://pyvis.readthedocs.io/en/latest/) package.

The entire workflow is integrated into a [Streamlit](https://streamlit.io/) web application.

## Getting Started

### Prerequisites

- **Python 3.7+**  
- **RDKit:** Installation of RDKit is best managed using [Conda](https://docs.conda.io/). For example, you can install RDKit with:
  ```bash
  conda install -c conda-forge rdkit

