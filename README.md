# RAMBOAU: Risk-averse Multiobjective Bayesian Optimization with Aleatoric Uncertainty

This repository contains the implementation of **RAMBOAU** (Risk-averse Multiobjective Bayesian Optimization with Aleatoric Uncertainty), a novel framework for multiobjective optimization under aleatoric uncertainty, with applications to nanomaterial synthesis and other experimental design problems.

## Overview

RAMBOAU addresses the challenge of optimizing multiple objectives when there is heteroscedastic inherent randomness (aleatoric uncertainty) in the evaluation process. The framework introduces risk-averse acquisition functions that considers both epistemic and aleatoric uncertainty in objective values.

## Algorithms

The repository implements several multiobjective Bayesian optimization algorithms:
Risk-averse algorithms:
- **RAqNEHVI**: Risk-averse q-Noisy Expected Hypervolume Improvement
- **RAqLogNEHVI**: Risk-averse q-Log Noisy Expected Hypervolume Improvement  
- **RAqNEIRS**: Multi-Objective Acquisition for Risk-Sensitive optimization
Risk-neutral algorithms:
- **qNEHVI**: Standard q-Noisy Expected Hypervolume Improvement
- **qEHVI**: q-Expected Hypervolume Improvement

## Installation

### Core Dependencies

- Python 3.9
- CUDA 11.8 (for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAMBOAU
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f install_help/env_man.yml
   conda activate ramboau
   ```

3. **Test installation**:
   ```bash
   python main.py --problem bstvert --algo raqneirs
   ```

## Usage

### Basic Usage

```bash
python main.py --problem bstdiag --algo raqneirs --n-iter 20 --n-init-sample 6
```

### All Synthetic Experiments in the Paper

```bash
python run.py --problem bstvert bsthorz bstdiag --algo qnehvi raqneirs raqnehvi --n-seed 20 
```

### Visualization
Generate plots from results:
```bash
python visualization/visualize_batch_all.py --problem bstvert bsthorz bstdiag --algo qnehvi raqneirs raqnehvi --n-seed 20
```

## References
S. Daulton, S. Cakmak, M. Balandat, M. A. Osborne, E. Zhou, and E. Bakshy. “Robust multi-objective bayesian optimization under input noise”. In: International Conference on Machine Learning. PMLR. 2022, pages 4831–4866.

A. Makarova, I. Usmanova, I. Bogunovic, and A. Krause. “Risk-averse Heteroscedastic Bayesian Optimization”. In: Advances in Neural Information Processing Systems.

R.-R. Griffiths, A. A. Aldrick, M. Garcia-Ortegon, V. Lalchand, et al. “Achieving robustness to aleatoric uncertainty with heteroscedastic Bayesian optimisation”. In: Machine Learning: Science and Technology 3.1 (2021), page 015004.

S. Daulton, M. Balandat, and E. Bakshy. “Parallel bayesian optimization of multiple noisy objectives with expected hypervolume improvement”. In: Advances in Neural Information Processing Systems 34 (2021), pages 2187–2200.

N. A. Jose, M. Kovalev, E. Bradford, A. M. Schweidtmann, H. C. Zeng, and A. A. Lapkin. “Pushing nanomaterials up to the kilogram scale–An accelerated approach for synthesizing antimicrobial ZnO with high shear reactors, machine learning and high throughput analysis”. In: Chemical Engineering Journal 426 (2021), page 131345.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{benhicham2025riskaverse,
  author = {Karim K. Ben Hicham and Nicholas A. Jose and Mohammed I. Jeraal and Jan G. Rittig and Alexei A. Lapkin},
  title = {RAMBO: Risk-averse Multiobjective Bayesian Optimization with Aleatoric Uncertainty for Nanomaterial Synthesis},
  year = {2025},
}
```

## Acknowledgments
- The core repository structure is strongly inspired by the DGEMO repository: https://github.com/yunshengtian/DGEMO
