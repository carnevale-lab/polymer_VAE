import sys
import os
import numpy as np
import argparse
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trainVAE as tv
import tensorflow as tf
from tensorflow import keras

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

keras.saving.get_custom_objects().clear()

def pprint(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")
    
def get_indices_by_temperature(temperatures, target_temperatures=[100, 200, 300]):
    indices_by_T = []
    for T in target_temperatures:
        indices = np.where(np.isclose(temperatures.flatten(), T))[0].tolist()
        indices_by_T.append(indices[:3])  # Take first 3 per T
    return indices_by_T

#Function to plot input or reconstructed distance matrices
def visualize_distance_matrices_withT(matrices, indices_by_T, save_path="./input_distance_matrices.pdf"):
    indices = [idx for group in indices_by_T for idx in group]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True, facecolor='white')
    vmin, vmax = 0, 45
    font_size = 17
    plt.subplots_adjust(left=0.08, right=0.85, top=0.93, bottom=0.08, wspace=0.3, hspace=0.15)
    row_titles = [r"$T=100$ K", r"$T=200$ K", r"$T=300$ K"]

    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            idx = indices[row_idx * 3 + col_idx]
            im = ax.imshow(matrices[idx], cmap="RdGy", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Backbone index j", fontsize=font_size-1, labelpad=3)
            ax.set_ylabel("Backbone index i", fontsize=font_size-1, labelpad=3)

    for i, title in enumerate(row_titles):
        axes[i, 1].set_title(title, fontsize=font_size + 2, pad=14)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.6, pad=0.04, aspect=25)
    cbar.set_label("Distance", fontsize=font_size + 2)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.yaxis.set_label_position('left')
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')

def compute_degree_of_violation(matrix):
    n = matrix.shape[0]
    degree_values = []

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dij, djk, dik = matrix[i, j], matrix[j, k], matrix[i, k]
                if dik != 0:
                    degree_values.append((dij + djk) / dik)

    return np.array(degree_values)

def plot_violation_distribution_subplots(degree_dict_by_T, save_path, peak_output_path):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    peak_lines = []

    for ax, (T_val, all_degrees) in zip(axs, degree_dict_by_T.items()):
        flat_degrees = np.concatenate([np.asarray(d, dtype=np.float64) for d in all_degrees])
        flat_degrees = flat_degrees[np.isfinite(flat_degrees)]
        flat_degrees = flat_degrees[flat_degrees < 1e6]  # Adjust threshold if needed
        kde = gaussian_kde(flat_degrees)
        x_min = 0
        x_max = 10
        x_vals = np.linspace(x_min, x_max, 500)
        pdf = kde(x_vals)
        red_mask = x_vals < 1
        red_values = flat_degrees[flat_degrees < 1]
        red_mean = np.mean(red_values) if red_values.size > 0 else 0.0
        peak_val = np.max(pdf)
        red_area = np.trapz(pdf[red_mask], x_vals[red_mask])
        count_red = len(red_values)
        total_count = len(flat_degrees)
        fraction_red = count_red / total_count if total_count > 0 else 0.0

        print(f"***************T = {T_val} K | Area (X < 1) = {red_area:.6f}**************")
        print(f"Count (<1) = {count_red} / {total_count}  |  Fraction = {fraction_red:.6f}")

        ax.plot(x_vals, pdf, color='black', lw=1.8)
        ax.fill_between(x_vals, pdf, 0, where=red_mask, color='red', alpha=0.3)
        ax.fill_between(x_vals, pdf, 0, where=~red_mask, color='green', alpha=0.3)
        ax.axvline(x=1, color='black', linestyle='-', lw=1.8)
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        y_ticks = np.arange(0.0, 0.51, 0.1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['0' if y == 0 else f'{y:.1f}' for y in y_ticks])
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Triangle violation degree, v", fontsize=14)
        ax.set_ylabel("Probability density, P(v)", fontsize=14)
        ax.set_title(f"T = {T_val} K", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        peak_lines.append(f"{T_val} {peak_val:.6f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    with open(peak_output_path, 'w') as f:
        f.write('\n'.join(peak_lines))

def set_diagonal_zero(matrix):
    for i in range(matrix.shape[0]):
        np.fill_diagonal(matrix[i], 0)

custom_objects = {"TransformerBlock": tv.TransformerBlock,
                "CCTTokenizer": tv.CCTTokenizer,
                "Sampling": tv.Sampling,
                "To1D": tv.To1D,
                "ConvBlock": tv.ConvBlock,
                "build_VAE": tv.build_VAE,
                "MMTVAE": tv.MMTVAE}

def main(args):
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Number of GPUs used: {N_REPLICAS}")
   
    number_atoms = args.atoms
    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE_PER_REPLICA = args.batch_size_per_gpu
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = GLOBAL_BATCH_SIZE

    X = np.load(args.npy_file)
    T = np.loadtxt(args.temps_file)
    T = np.array(T).reshape(-1, 1)
    pprint(f"Input shape: {X.shape}")
    pprint(f"Temperature shape: {T.shape}")

    num_samples = X.shape[0]
    X_reshaped = X.reshape(num_samples, number_atoms, number_atoms).astype(np.float32)
    pprint(f"Input reshaped to: {X_reshaped.shape}")

    indices_by_T = get_indices_by_temperature(T)

    pprint("Selected indices for each temperature:")
    for i, idx_list in zip([100, 200, 300], indices_by_T):
        pprint(f"T = {i} K: {idx_list}")
    
    visualize_distance_matrices_withT(X_reshaped, indices_by_T) #Input samples

    vae = tf.keras.models.load_model(
        "./model_outputs/model/full.ckpt",
        custom_objects=custom_objects,
        compile=False
    )

    steps_per_epoch = np.ceil(len(X_reshaped[:100])/BATCH_SIZE).astype(int)
    sample_dataset = tf.data.Dataset.from_tensor_slices((X_reshaped[:100], T[:100]))\
                    .prefetch(AUTO)\
                    .batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    sample_dataset = sample_dataset.with_options(options)
    sample_dataset = strategy.experimental_distribute_dataset(sample_dataset)
    
    results = []
    with strategy.scope():
        for batch in tqdm(sample_dataset):
            ss, ee = batch
            results.append(vae([ss, ee])[0])
        results = np.vstack(results)
        results = 0.5 * (results + np.transpose(results, (0, 2, 1, 3)))
    results = np.squeeze(results, axis=-1)
    set_diagonal_zero(results)

    visualize_distance_matrices_withT(results, indices_by_T, save_path="./Reconstructed_samples.pdf")

    with open("4k_upT_rec_Tcond.dat", "w") as f:
        for sample in results:
            up_T = sample[np.triu_indices(sample.shape[0], k=1)]
            up_T = np.concatenate(([400], up_T))
            np.savetxt(f, [up_T], fmt="%.5f")

    ###### TRIANGLE INEQUALITY VIOLATION ANALYSIS ON RECONSTRUCTED SAMPLES ######
    # degree_dict_by_T = {100: [], 200: [], 300: []}
    # for i, T_val in enumerate(T[:300].flatten()):
    #     degrees = compute_degree_of_violation(results[i])
    #     degree_dict_by_T[int(T_val)].append(degrees)

    # plot_violation_distribution_subplots(degree_dict_by_T, save_path="violation_distribution_RECON.pdf", peak_output_path="max_values_REC.dat")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--atoms", type=int, default=400, help="Number of atoms in each system")
    parser.add_argument("--npy_file", type=str, required=True, help="Path to the .npy full matrices file")
    parser.add_argument("--temps_file", type=str, required=True, help="Path to temperatures file")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, help="Batch size per GPU")
    args = parser.parse_args()
    main(args)