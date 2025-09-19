import os
import numpy as np
import argparse
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
    E = np.loadtxt(args.energies_file)
    E = np.array(E).reshape(-1, 1)
    pprint(f"Input shape: {X.shape}")
    pprint(f"Temperature shape: {T.shape}")
    pprint(f"Energy shape: {E.shape}")

    num_samples = X.shape[0]
    X_reshaped = X.reshape(num_samples, args.atoms, args.atoms).astype(np.float32)
    pprint(f"Input reshaped to: {X_reshaped.shape}")

    #####LATENT_SPACE_EMBEDDING_MEAN_SIGMA#####
    encoder = tf.keras.models.load_model(
       "./model_outputs/model/enc.ckpt",
       custom_objects=custom_objects,
       compile=False
    )

    z_mean, z_log_var, _ = encoder.predict([X_reshaped, T])
    z_mean = np.squeeze(z_mean, axis=-1)
    pprint(z_mean.shape)
    z_log_var = np.squeeze(z_log_var, axis=-1)
    pprint(z_log_var.shape)

    np.savetxt("latent_MEAN.dat", z_mean, fmt="%.5f", delimiter="\t")

    sigma = np.exp(0.5 * z_log_var)

    np.savetxt("latent_SIGMA.dat", sigma, fmt="%.5f", delimiter="\t")


    #####LATENT_VARIABLE_PERCENTILES#####
    percentiles_mean = np.percentile(z_mean, [25, 50, 75], axis=0)
    percentiles_sigma = np.percentile(sigma, [25, 50, 75], axis=0)

    q50_indices_sigma = np.argsort(percentiles_sigma[1])
    percentiles_mean_sorted = percentiles_mean[:, q50_indices_sigma]
    percentiles_sigma_sorted = percentiles_sigma[:, q50_indices_sigma]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    #---Plot for μ---
    axs[0].fill_between(range(latent_dim), percentiles_mean_sorted[0], percentiles_mean_sorted[2],
                        alpha=0.3, color='orange')
    axs[0].plot(range(latent_dim), percentiles_mean_sorted[1], color='orange', linewidth=2.1)
    axs[0].plot(range(latent_dim), percentiles_mean_sorted[0], color='orange', linestyle='--', linewidth=1.9)
    axs[0].plot(range(latent_dim), percentiles_mean_sorted[2], color='orange', linestyle='--', linewidth=1.9)

    axs[0].set_xlim([-50, latent_dim + 50])
    axs[0].set_ylim([-1, 1])
    axs[0].set_xticks(np.arange(0, latent_dim + 1, 200))
    axs[0].set_yticks(np.arange(-1, 1.1, 0.5))
    axs[0].tick_params(axis='x', labelsize=12)
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].set_title(r'$\mu$', fontsize=19)
    axs[0].set_xlabel("Latent Dimension Index", fontsize=17)
    axs[0].set_ylabel("Percentile", fontsize=17)
    axs[0].grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    #---Plot for σ---
    axs[1].fill_between(range(latent_dim), percentiles_sigma_sorted[0], percentiles_sigma_sorted[2],
                        alpha=0.3, color='gray')
    axs[1].plot(range(latent_dim), percentiles_sigma_sorted[1], color='gray', linewidth=2.1)
    axs[1].plot(range(latent_dim), percentiles_sigma_sorted[0], color='gray', linestyle='--', linewidth=1.9)
    axs[1].plot(range(latent_dim), percentiles_sigma_sorted[2], color='gray', linestyle='--', linewidth=1.9)

    axs[1].set_xlim([-50, latent_dim + 50])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(0, latent_dim + 1, 200))
    axs[1].set_yticks(np.arange(0, 1.1, 0.25))
    axs[1].tick_params(axis='x', labelsize=12)
    axs[1].tick_params(axis='y', labelsize=12)
    axs[1].set_title(r'$\sigma$', fontsize=19)
    axs[1].set_xlabel("Latent Dimension Index", fontsize=17)
    axs[1].set_ylabel("Percentile", fontsize=17)
    axs[1].grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig("latent_variable_percentile_plots.pdf", dpi=300, bbox_inches='tight')

    #####PCA_LATENT_SPACE#####COLORED BY TEMPERATURE VALUE#####
    embeddings = []
    steps = int(np.ceil(len(X_reshaped)/BATCH_SIZE))
    with strategy.scope():
        for step in range(steps):
            start_idx = step*BATCH_SIZE
            end_idx = min(start_idx+BATCH_SIZE, len(X_reshaped))
            embeddings.append(encoder.predict((X_reshaped[start_idx:end_idx], T[start_idx:end_idx]), verbose=0)[0])
        embeddings = np.vstack(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=6)
    embeddings_pca = pca.fit_tmransform(embeddings)

    pairs = [
        (0, 1),  # PC1 vs PC2
        (0, 2),  # PC1 vs PC3
        (0, 3),  # PC1 vs PC4
        (1, 2),  # PC2 vs PC3
        (1, 3),  # PC2 vs PC4
        (2, 3),  # PC3 vs PC4
    ]

    all_indices = set([idx for pair in pairs for idx in pair])
    mins = {idx: embeddings_pca[:, idx].min() for idx in all_indices}
    maxs = {idx: embeddings_pca[:, idx].max() for idx in all_indices}

    def get_limits(i, j, margin_ratio=0.05):
        x_min, x_max = mins[i], maxs[i]
        y_min, y_max = mins[j], maxs[j]
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_margin = margin_ratio * x_range
        y_margin = margin_ratio * y_range
        
        return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)

    color_list = [(1, 0, 0), (0.65, 0.65, 0.65), (0, 0, 1)]
    cmap_redgrayblue = LinearSegmentedColormap.from_list("redgrayblue", color_list)

    T_categories = [100, 200, 300]
    colors_for_legend = {
        100: cmap_redgrayblue(1.0),
        200: cmap_redgrayblue(0.5),
        300: cmap_redgrayblue(0.0)
    }
    labels = {100: '100 K', 200: '200 K', 300: '300 K'}

    new_T = T.flatten()

    fig, axs = plt.subplots(3, 3, figsize=(24, 24))
    axs = axs.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axs[idx]
        for T_val in T_categories:
            mask = (new_T == T_val)
            ax.scatter(
                embeddings_pca[mask, i], embeddings_pca[mask, j],
                c=[colors_for_legend[T_val]], marker='o',
                alpha=1.0, s=18, zorder=2)

        ax.set_xlabel(f'PC{i+1}', fontsize=17)
        ax.set_ylabel(f'PC{j+1}', fontsize=17)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(np.arange(-8, 8, 2))
        ax.set_yticks(np.arange(-8, 8, 2))
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.set_box_aspect(1)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axs[len(pairs):]:
        ax.remove()

    plt.subplots_adjust(wspace=0.18, hspace=0.06, top=0.94)

    handles = [plt.scatter([], [], c=[colors_for_legend[T_val]], edgecolors='k', marker='o', s=120, alpha=0.6)
            for T_val in T_categories]

    fig.legend(handles=handles, labels=[labels[T_val] for T_val in T_categories],
            loc='upper center', bbox_to_anchor=(0.5, 0.965),
            ncol=3, fontsize=19, frameon=False, handletextpad=0.5, columnspacing=1.0)

    plt.savefig("pca_temperature_plot.pdf", dpi=300, bbox_inches='tight')

    #####PCA_LATENT_SPACE#####COLORED BY ENERGY VALUE#####
    fig, axs = plt.subplots(3, 3, figsize=(24, 24))
    axs = axs.flatten()

    norm = Normalize(vmin=-800, vmax=-500)

    for idx, (i, j) in enumerate(pairs):
        ax = axs[idx]
        sc = ax.scatter(embeddings_pca[:, i], embeddings_pca[:, j],
                        c=E, cmap=cmap_redgrayblue, norm=norm, marker='o',
                        alpha=1.0, s=18, zorder=2)
        
        ax.set_xlabel(f'PC{i+1}', fontsize=17)
        ax.set_ylabel(f'PC{j+1}', fontsize=17)
        ax.set_box_aspect(1)
        ax.set_xticks(np.arange(-8, 8, 2))
        ax.set_yticks(np.arange(-8, 8, 2))
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
        ax.set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axs[len(pairs):]:
        ax.remove()

    plt.subplots_adjust(wspace=0.18, hspace=0.002, right=0.80, top=0.94, bottom=0.06)

    cbar_ax = fig.add_axes([0.85, 0.372, 0.02, 0.55])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Energy [kJ/mol]', fontsize=19, rotation=90, labelpad=10)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.tick_params(labelsize=19)

    plt.savefig("pca_energy_plot.pdf", dpi=300, bbox_inches='tight')

    ####EIGENVALUES SPECTRUM#####
    pca_full = PCA(n_components=1200)
    pca_full.fit(embeddings)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(pca_full.explained_variance_) + 1), pca_full.explained_variance_, 'o-', linewidth=2)
    plt.title('PCA Eigenvalues')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.savefig("./PCA_eigenvalues_raw.pdf")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--atoms", type=int, default=400, help="Number of atoms in each system")
    parser.add_argument("--npy_file", type=str, required=True, help="Path to the .npy full matrices file")
    parser.add_argument("--temps_file", type=str, required=True, help="Path to temperatures file")
    parser.add_argument("--energies_file", type=str, required=True, help="Path to energies file")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, help="Batch size per GPU")
    args = parser.parse_args()
    main(args)