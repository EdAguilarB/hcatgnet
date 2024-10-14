# HCat-GNet: Homogeneous Catalyst Graph Neural Network

## Overview
HCat-GNet (Homogeneous Catalyst Graph Neural Network) is a cutting-edge, open-source platform designed to facilitate the virtual evaluation and optimization of homogeneous catalysts. Utilizing Graph Neural Networks (GNNs), HCat-GNet predicts the selectivity of homogeneous catalytic reactions based solely on SMILES representations of participant molecules, significantly speeding up the process of ligand optimization in asymmetric catalysis.

## Features
- **Predictive Accuracy**: Delivers highly accurate predictions of enantioselectivity for metal-ligand catalyzed asymmetric reactions.
- **Interpretability**: Provides insights into how different ligand modifications affect reaction outcomes, enhancing human understanding and guiding experimental design.
- **Flexibility**: Agnostic to reaction type, capable of handling a variety of catalytic processes without the need for domain-specific adjustments.

## Installation

### Prerequisites
- Python 3.8+
- Pip (Python package installer)

### Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/HCat-GNet.git
   cd HCat-GNet

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To run all experiments as described in our paper
   ```bash
   python run_experiments.py
```

To run the experiments using the CircuS descriptors
   ```bash
   python run_experiments.py --descriptors circus
```

### Citation
Aguilar-Bejarano, E., et al. "HCat-GNet: An Interpretable Graph Neural Network for Catalysis Optimization." (Year). Journal/Conference. DOI.

### License
Distributed under the MIT License. See LICENSE for more information.

### Contact
eduardo.aguilar-bejarano@nottingham.ac.uk
g.figueredo@nottingham.ac.uk




   







