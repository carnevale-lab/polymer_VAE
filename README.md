# Polymer-VAE

Generative Modeling of Entangled Polymers with a Distance-Based Variational Autoencoder

## Model architecture

<p align="center">
  <img width="100%" height="auto" src="https://github.com/carnevale-lab/polymer_VAE/blob/6786ac055f2afe931fad46a3814e0fe064dcffde/model_architecture.png">
</p>

## Data

The datasets associated with the paper can be generated from coarse-grained MD simulations (see the paper for details). For accessibility, we provide a small sample dataset in the **data** directory. These files allow you to test the training, reconstruction, latent space analysis and generation pipelines.

small_matrices.dat     →     30 upper-triangular distance matrices of polyethylene configurations.
Note: Each line corresponds to one matrix and starts with the number of atoms (400 in our case).

small_temps.dat        →     Temperature labels for each sample.
Note: Each line is a scalar value (100, 200, or 300), aligned with the order of matrices in small_matrices.dat

small_energies.dat     →     Energy values for each sample.
Note: Each line contains the potential energy of the corresponding matrix, in the same order as above.

### Setup

This code is implemented to fully utilize a single available GPU.
On systems with multiple GPUs, the scripts will run correctly but will not leverage multi-GPU parallelism — only one GPU is used for training/inference.
If no GPU is available, the code will fall back to CPU execution, but training will be significantly slower.

Install required dependencies:
```
conda env create -f polVAE.yml
```

## General usage in HPC servers:
In order to use this model on a HPC server, you can run the `trainVAE.py` script to train the model. The list of command line arguments used is as follows:

```
usage: trainVAE.py [--input_file INPUT_dat_FILE] [--npy_file INPUT_npy_FILE] [--temps_file TEMP_FILE] [--latent_dim LATENT_DIM]
                   [--atoms ATOMS] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--batch_size_per_gpu BATCH_SIZE]
                   [--verbose VERBOSE]

optional arguments:
  --input_file           Path to the input .dat upper triangular matrices file.
  --npy_file             Output path to the .npy full matrices file.
  --temps_file           Path to temperatures file.
  --latent_dim           Latent dimension size (default = 1200).
  --atoms                Number of atoms/monomers in each system (default = 400).
  Note: All input matrices must have the same number of atoms.
  --epochs               Maximum number of training epochs (default = 1000).
  --learning_rate        Learning rate (default = 4e-7).
  --batch_size_per_gpu   Batch size per GPU (default = 8).
  --verbose              Training verbosity (default = 2).
```

To test the reconstruction of the input samples, you can run the `reconstruct.py` script in the **reconstruction** directory as follows:

```
usage: reconstruct.py [--npy_file INPUT_npy_FILE] [--temps_file TEMP_FILE] [--atoms ATOMS] [--batch_size_per_gpu BATCH_SIZE]
```

To analyze the latent space as described in the paper, you can run the `latent_space_analyze.py` script in the **latent_analysis** directory as follows:

```
usage: latent_space_analyze.py [--npy_file INPUT_npy_FILE] [--temps_file TEMP_FILE] [--energies_file ENERGY_FILE] [--atoms ATOMS] [--batch_size_per_gpu BATCH_SIZE]

optional arguments:
  --energies_file        Path to potential energies file.
```

To generate de novo distance matrices of polyethylene configurations, you can run the `generation.py` script in the **denovo_generation** directory as follows:

```
usage: generation.py [--npy_file INPUT_npy_FILE] [--temps_file TEMP_FILE] [--atoms ATOMS] [--batch_size_per_gpu BATCH_SIZE] [--gen_samples NUMBER_OF_GENERATED_SAMPLES]
```

## Citation
If you use our model in any project or publication, please cite our paper [Generative Modeling of Entangled Polymers with a Distance-Based Variational Autoencoder](XXXlinkXXX)

```
@article{chiarantoni2025genVAEpol,
  title={Generative Modeling of Entangled Polymers with a Distance-Based Variational Autoencoder},
  author={Chiarantoni, Pietro and Serra, Oscar and Mowlaei, Mohammad Erfan and Choutipalli, Venkata Surya Kumar and DelloStritto, Mark and Shi, Xinghua Mindy and Klein, Micheal L. and Carnevale, Vincenzo},
  year={2025},
  journal={XXXXX}
}
