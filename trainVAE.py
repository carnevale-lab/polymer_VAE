import os
import math
import random
import numpy as np
import argparse
from sklearn import metrics, mixture
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(tf.__version__)

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

def create_directories(save_dir,
                       models_dir="model") -> None:
    for dd in [save_dir,
               f"{save_dir}/{models_dir}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)
    pass

def get_indices_by_temperature(temperatures, target_temperatures=[100, 200, 300]):
    indices_by_T = []
    for T in target_temperatures:
        indices = np.where(np.isclose(temperatures.flatten(), T))[0].tolist()
        indices_by_T.append(indices[:3])
    return indices_by_T

#Function to plot input distance matrices
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

#Class converts a .dat file of upper triangular matrices into a .npy file of full N x N matrices
class DatToNpyConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def full_distance_matrix(self, flat_array):
        k = len(flat_array)
        n = int(np.ceil((1 + np.sqrt(1 + 8 * k)) / 2))
        matrix = np.zeros((n, n))
        upper_indices = np.triu_indices(n, k=1)
        matrix[upper_indices] = flat_array
        matrix = matrix + matrix.T

        return matrix

    def convert(self):
        upT_arrays = []
        with open(self.input_file, "r") as file:
            for line in file:
                columns = line.strip().split()
                line = " ".join(columns[1:])
                arr = np.fromstring(line, sep=" ")
                upT_arrays.append(arr)
        np_upT_arrays = np.asarray(upT_arrays)
        
        full_arrays = []
        for arr in np_upT_arrays:
            full_matrix = self.full_distance_matrix(arr)
            full_arrays.append(full_matrix.flatten())
        
        np_full_arrays = np.asarray(full_arrays)
        np.save(self.output_file, np_full_arrays)
        return np_full_arrays

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, activation=tf.nn.gelu,
                 dropout_rate=0.2, use_ffn=True, use_ln=True,
                 kernel_constraints=None):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_ffn = use_ffn
        self.use_ln = use_ln
        self.att0 = layers.MultiHeadAttention(num_heads=self.num_heads,
                                              key_dim=self.embed_dim,
                                              dropout=dropout_rate,
                                              kernel_constraint=kernel_constraints)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation=self.activation,
                                kernel_constraint=kernel_constraints),
                layers.Dense(self.embed_dim, activation=self.activation,
                                kernel_constraint=kernel_constraints),
            ]
        )
        self.layer_norm0 = layers.LayerNormalization()
        self.layer_norm1 = layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "use_ffn": self.use_ffn,
                "use_ln": self.use_ln,
                "att0": self.att0,
                "ffn": self.ffn,
                "layer_norm0": self.layer_norm0,
                "layer_norm1": self.layer_norm1,
            }
        )

        return config

    def call(self, inputs, training):
        if self.use_ln:
            x = self.layer_norm0(inputs)
        else:
            x = inputs
        attn_output = self.att0(x, x)
        out1 = x + attn_output
        if self.use_ln:
            out1 = self.layer_norm1(out1)
        if self.use_ffn:
            ffn_output = self.ffn(out1)
            x = out1 + ffn_output
        else:
            x = out1
        return x

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class CCTTokenizer(layers.Layer):
    def __init__(
      self,
      up_scale=False,
      kernel_size=3,
      stride=1,
      padding=1,
      pooling_kernel_size=3,
      pooling_stride=2,
      num_conv_layers=2,
      num_output_channels=[64, 128],
      positional_emb=True,
      kernel_constraints=None,
      **kwargs,
    ):
        super().__init__(**kwargs)
        self.up_scale = up_scale
        self.conv_model = tf.keras.Sequential()
        if not up_scale:
            for i in range(num_conv_layers):
                self.conv_model.add(
                    layers.Conv2D(
                        num_output_channels[i],
                        kernel_size,
                        pooling_stride,
                        padding="same",
                        use_bias=False,
                        activation=tf.nn.gelu,
                        kernel_initializer="he_normal",
                        kernel_constraint=kernel_constraints
                    )
                )
        else:
            for i in range(num_conv_layers):
                self.conv_model.add(
                    layers.Conv2DTranspose(
                        num_output_channels[i],
                        kernel_size,
                        pooling_stride,
                        padding="same",
                        use_bias=False,
                        activation=tf.nn.gelu,
                        kernel_initializer="he_normal",
                        kernel_constraint=kernel_constraints
                    )
                )
        self.positional_emb = positional_emb

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "up_scale": self.up_scale,
                "conv_model": self.conv_model,
                "positional_emb": self.positional_emb
            }
        )
        return config

    def call(self, images):
        if not self.up_scale:
            outputs = self.conv_model(images)
            reshaped = tf.reshape(
                outputs,
                (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
            )
            return reshaped
        else:
            h = w = tf.cast(tf.math.sqrt(tf.cast(tf.shape(images)[1], tf.float32)), tf.int32)
            reshaped = tf.reshape(
                images,
                (-1, w, h, tf.shape(images)[-1]),
            )
            outputs = self.conv_model(reshaped)
            return outputs

    def positional_embedding(self, image_size):
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, image_size, 2))
            dummy_outputs = self.call(dummy_inputs)
            number_atoms = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]

            embed_layer = layers.Embedding(
                input_dim=number_atoms, output_dim=projection_dim
            )
            return embed_layer, number_atoms
        else:
            return None

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class Sampling(layers.Layer):
    def __init__(self, name=None):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        length = tf.shape(z_mean)[1]
        dim = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, length, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class To1D(layers.Layer):
    def __init__(self, atom_nu, depth, name=None):
        super(To1D, self).__init__(name=name)
        self.atom_nu = atom_nu
        self.depth = depth

    def get_config(self):
        config = super().get_config()
        config.update({"atom_nu": self.atom_nu,"depth": self.depth})
        return config

    def call(self, inputs, training=None):
        return layers.Reshape((self.atom_nu, self.depth))(inputs)

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class ConvBlock(layers.Layer):
    def __init__(self, embed_dim, kernel_constraints=None):
        super(ConvBlock, self).__init__()
        self.embed_dim = embed_dim
        self.const = kernel_constraints
        self.conv000 = layers.Conv1D(embed_dim, 3, padding="same", activation=tf.nn.gelu,
                                    kernel_constraint=self.const,
                        )
        self.conv010 = layers.Conv1D(embed_dim, 5, padding="same", activation=tf.nn.gelu,
                                    kernel_constraint=self.const,
                        )
        self.conv011 = layers.Conv1D(embed_dim, 7, padding="same", activation=tf.nn.gelu,
                                    kernel_constraint=self.const,
                        )
        self.conv020 = layers.Conv1D(embed_dim, 7, padding="same", activation=tf.nn.gelu,
                                    kernel_constraint=self.const,
                        )
        self.conv021 = layers.Conv1D(embed_dim, 15, padding="same", activation=tf.nn.gelu,
                                    kernel_constraint=self.const,
                        )
        self.add = layers.Add()
        self.conv100 = layers.Conv1D(embed_dim, 3, padding="same",
                                    activation=tf.nn.gelu,
                                    kernel_constraint=self.const,)
        self.bn0 = layers.BatchNormalization()
        self.bn1 = layers.BatchNormalization()
        self.dw_conv = layers.DepthwiseConv1D(embed_dim, 1, padding="same")
        self.activation = layers.Activation(tf.nn.gelu)
  
    def get_config(self):
        config = super().get_config()
        
        config.update(
        {
        "embed_dim": self.embed_dim,
        "const": self.const,
        "conv000": self.conv000,
        "conv010": self.conv010,
        "conv011": self.conv011,
        "conv020": self.conv020,
        "conv021": self.conv021,
        "add": self.add,
        "conv100": self.conv100,
        "bn0": self.bn0,
        "bn1": self.bn1,
        "dw_conv": self.dw_conv,
        "activation": self.activation,
        }
        )

        return config

    def call(self, inputs, training):

        xa = self.conv000(inputs)
        xb = self.conv010(xa)
        xb = self.conv011(xb)
        xc = self.conv020(xa)
        xc = self.conv021(xc)
        xa = self.add([xb, xc])
        xa = self.conv100(xa)
        xa = self.bn0(xa)
        xa = self.dw_conv(xa)
        xa = self.bn1(xa)
        xa = self.activation(xa)

        return xa

encoder_struct_positional_emb = True
encoder_struct_conv_layers = 3
emb_dim = 64
NUM_CA_BLOCKS = 1
gen_transformer_layers = 1
num_heads = 32
encoder_struct_output_channels = [emb_dim*2**i for i in range(encoder_struct_conv_layers)]
encoder_struct_projection_dim = encoder_struct_output_channels[-1]

@tf.keras.saving.register_keras_serializable(package="MyLayers", name="build_VAE")
def build_VAE(latent_dim, num_heads, number_atoms):
    temperature_inputs = tf.keras.Input(shape=(1,))  
    struct_inputs = tf.keras.Input(shape=(number_atoms, number_atoms, 1))
    temperature = layers.Dense(number_atoms**2, activation=tf.nn.relu)(temperature_inputs)   
    temperature = layers.Reshape((number_atoms, number_atoms, 1))(temperature)
    encoder_inputs = layers.Concatenate(axis=-1)([struct_inputs, temperature])

    cct_tokenizer_encoder = CCTTokenizer(up_scale=False,
        kernel_size=7,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=encoder_struct_conv_layers,
        num_output_channels=encoder_struct_output_channels,
        positional_emb=encoder_struct_positional_emb)
    
    encoded_patches = cct_tokenizer_encoder(encoder_inputs)

    if encoder_struct_positional_emb:
        pos_embed, atoms = cct_tokenizer_encoder.positional_embedding(number_atoms)
        positions = tf.range(start=0, limit=atoms, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    encoder_struct_representation = encoded_patches

    for i in range(NUM_CA_BLOCKS):
        i_conv_n_channels = encoder_struct_output_channels[-1]
        encoder_struct_representation = ConvBlock(i_conv_n_channels)(encoder_struct_representation)

    encoder_struct_conv_shape = K.int_shape(encoder_struct_representation)[1:]
    encoder_str_transformer_block = TransformerBlock(encoder_struct_conv_shape[-1],
                                                    num_heads,
                                                    encoder_struct_conv_shape[-1])
    for _ in range(gen_transformer_layers):
        encoder_struct_representation = encoder_str_transformer_block(encoder_struct_representation)
 
    encoder_struct_representation = layers.Flatten()(encoder_struct_representation)
    encoder_struct_z_mean = layers.Dense(latent_dim, name="struc_z_mean")(encoder_struct_representation)
    encoder_struct_z_mean = To1D(latent_dim,1)(encoder_struct_z_mean)
    encoder_struct_z_log_var = layers.Dense(latent_dim, name="struc_z_log_var")(encoder_struct_representation)
    encoder_struct_z_log_var = To1D(latent_dim,1)(encoder_struct_z_log_var)
    
    z_mean = encoder_struct_z_mean
    z_log_var = encoder_struct_z_log_var

    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model([struct_inputs, temperature_inputs],[z_mean, z_log_var, z])

    z_out = layers.Flatten()(z)

    combined_latent = layers.Concatenate()([z_out, temperature_inputs])

    decoder_struct_representation = layers.Dense(encoder_struct_conv_shape[0]*encoder_struct_conv_shape[1],
                                                activation=tf.nn.gelu)(combined_latent)
    decoder_struct_representation = layers.Reshape((encoder_struct_conv_shape[0],
                                                    encoder_struct_conv_shape[1]))(decoder_struct_representation)
    decoder_struct_representation = TransformerBlock(encoder_struct_conv_shape[-1],
                                            num_heads, encoder_struct_conv_shape[-1])(decoder_struct_representation)
                                            
    for j in list(range(NUM_CA_BLOCKS))[::-1]:
        i_conv_n_channels = encoder_struct_output_channels[-1]
        decoder_struct_representation = ConvBlock(i_conv_n_channels)(decoder_struct_representation)

    cct_tokenizer_decoder = CCTTokenizer(up_scale=True,
        kernel_size=7,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=encoder_struct_conv_layers,
        num_output_channels=encoder_struct_output_channels[::-1],
        positional_emb=False)

    decoded_patches = cct_tokenizer_decoder(decoder_struct_representation)
    structure_reconstructed = layers.Conv2DTranspose(1, 3, padding="same",
                                                    name="struct_rec",
                                                    activation=tf.nn.relu)(decoded_patches)

    decoder = tf.keras.Model([z, temperature_inputs], structure_reconstructed)

    return encoder, decoder

@tf.keras.saving.register_keras_serializable(package="MyModels")
class MMTVAE(tf.keras.Model):
    def __init__(self, latent_dim, number_atoms,
                 num_attention_heads=5, global_batch_size=16,
                 **kwargs):
        super(MMTVAE, self).__init__(**kwargs)
        self.global_batch_size = global_batch_size
        self.latent_dim = latent_dim
        self.encoder, self.decoder = build_VAE(latent_dim,
                                               num_attention_heads,
                                               number_atoms)
        self.mse_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.reconstruction_loss_tracker_str = tf.keras.metrics.Mean(name="rec_loss_str")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.loss_tracker_total = tf.keras.metrics.Mean(name="loss_total")
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "global_batch_size": self.global_batch_size,
                "latent_dim": self.latent_dim,
                "encoder": self.encoder,
                "decoder": self.decoder,
                "mse_fn": self.mse_fn,
                "reconstruction_loss_tracker_str": self.reconstruction_loss_tracker_str,
                "kl_loss_tracker": self.kl_loss_tracker,
                "loss_tracker_total": self.loss_tracker_total,
                "d_optimizer": self.d_optimizer,
                "g_optimizer": self.g_optimizer
            }
        )
        return config

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker_str,
            self.kl_loss_tracker,
            self.loss_tracker_total
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction_str = self.decoder([z, inputs[1]])
        reconstruction_str = 0.5 * (reconstruction_str + tf.transpose(reconstruction_str, perm=[0, 2, 1, 3]))
        return reconstruction_str, z_mean, z_log_var, z

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def compute_loss(self, loss_object, labels, predictions, model_losses):
        per_example_loss = loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    
    def compute_self_loss(self, loss_object, predictions, model_losses):
        per_example_loss = loss_object(predictions)
        loss = tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def compute_kl_loss(self, z_log_var, z_mean, model_losses):
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        per_example_loss = tf.reduce_sum(kl_loss, axis=[1, 2])
        loss = tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    
    def train_step(self, data):
        input_str, energies, target_str = data
        mean_target = tf.math.reduce_mean(target_str)
        variance_target = tf.math.reduce_variance(target_str)
        identity_matrix = tf.eye(400) # (400,400)
        identity_tensor = tf.tile(tf.expand_dims(identity_matrix, axis=0), [8, 1, 1]) # (8,400,400)
        target_str = target_str + identity_tensor
    
        with tf.GradientTape() as tape_rec:
            reconstruction_str, z_mean, z_log_var, z = self([input_str, energies],  training=True)
            reconstruction_str = tf.squeeze(reconstruction_str, axis=-1)
            reconstruction_str = tf.linalg.set_diag(reconstruction_str, tf.ones((8, 400)))
            one_tensor = tf.ones((8, 400, 400))
            reconstruction_loss_str = self.compute_loss(self.mse_fn, one_tensor, reconstruction_str / target_str, None)
            kl_loss = self.compute_kl_loss(z_log_var, z_mean, None) * 0.01
            total_loss = reconstruction_loss_str + kl_loss
            
        rec_grads = tape_rec.gradient(total_loss, self.trainable_weights)
        self.g_optimizer.apply_gradients(zip(rec_grads, self.trainable_weights))
        self.reconstruction_loss_tracker_str.update_state(reconstruction_loss_str)
        self.kl_loss_tracker.update_state(kl_loss)
        self.loss_tracker_total.update_state(total_loss)
    
        return {
            "rec_loss_str": self.reconstruction_loss_tracker_str.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss_total": self.loss_tracker_total.result()
        }

    def test_step(self, data):
        input_str, energies, target_str = data
        mean_target = tf.math.reduce_mean(target_str)
        variance_target = tf.math.reduce_variance(target_str)
        identity_matrix = tf.eye(400) # (400,400)
        identity_tensor = tf.tile(tf.expand_dims(identity_matrix, axis=0), [8, 1, 1]) # (8,400,400)
        target_str = target_str + identity_tensor
        one_tensor = tf.ones((8, 400, 400))
    
        reconstruction_str, z_mean, z_log_var, z = self([input_str, energies],  training=False)
        reconstruction_str = tf.squeeze(reconstruction_str, axis=-1)
        reconstruction_str = tf.linalg.set_diag(reconstruction_str, tf.ones((8, 400)))
        reconstruction_loss_str = self.compute_loss(self.mse_fn, one_tensor, reconstruction_str / target_str, None)
        kl_loss = self.compute_kl_loss(z_log_var, z_mean, None) * 0.01
        total_loss = reconstruction_loss_str + kl_loss

        self.reconstruction_loss_tracker_str.update_state(reconstruction_loss_str)
        self.kl_loss_tracker.update_state(kl_loss)
        self.loss_tracker_total.update_state(total_loss)
    
        return {
            "rec_loss_str": self.reconstruction_loss_tracker_str.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss_total": self.loss_tracker_total.result()
        }

def create_callbacks(early_stopping=True, metric = "val_rec_loss_str"):

    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor= metric,
        mode="auto",
        factor=0.5,
        patience=3,
        min_lr=1e-12,
        verbose=1
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor= metric,
        mode="auto",
        patience= 10,
        verbose=1,
        restore_best_weights=True
    )

    callbacks = [reducelr,
                 earlystop] if early_stopping else [reducelr]

    return callbacks

custom_objects = {"TransformerBlock": TransformerBlock,
                "CCTTokenizer": CCTTokenizer,
                "Sampling": Sampling,
                "To1D": To1D,
                "ConvBlock": ConvBlock,
                "build_VAE": build_VAE,
                "MMTVAE": MMTVAE}

def main(args):
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Number of GPUs used: {N_REPLICAS}")
   
    number_atoms = args.atoms
    AUTO = tf.data.AUTOTUNE
    latent_dim = args.latent_dim
    BATCH_SIZE_PER_REPLICA = args.batch_size_per_gpu
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    BATCH_SIZE = GLOBAL_BATCH_SIZE
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    
    dat_to_npy = DatToNpyConverter(args.input_file, args.npy_file)
    X = dat_to_npy.convert()
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
    
    visualize_distance_matrices_withT(X_reshaped, indices_by_T)

    create_directories("./model_outputs")

    enc, dec = build_VAE(latent_dim, num_heads, number_atoms)

    pprint("\nEncoder:")
    enc.summary()
    pprint("\nDecoder:")
    dec.summary()

    train_indices, test_indices = train_test_split(range(num_samples),
                                   test_size=0.2,
                                   random_state=2023,
                                   shuffle=True,
                                   )
    train_indices, valid_indices = train_test_split(train_indices,
                                   test_size=0.2,
                                   random_state=2023,
                                   shuffle=True,
                                   )

    steps_per_epoch = len(train_indices)//BATCH_SIZE
    validation_steps = len(valid_indices)//BATCH_SIZE
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_reshaped[train_indices], T[train_indices], X_reshaped[train_indices]))\
                    .shuffle(len(train_indices), reshuffle_each_iteration=True)\
                    .repeat()\
                    .prefetch(AUTO)\
                    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    train_dataset = train_dataset.with_options(options)
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_reshaped[valid_indices], T[valid_indices], X_reshaped[valid_indices]))\
                    .prefetch(AUTO)\
                    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    valid_dataset = valid_dataset.with_options(options)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    K.clear_session()
    callbacks = create_callbacks()
    with strategy.scope():
        g_opt = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        d_opt = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
        vae = MMTVAE(latent_dim, number_atoms)
        vae.compile(g_opt, d_opt)
        history = vae.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        validation_data=valid_dataset, validation_steps=validation_steps,
                        callbacks=callbacks, verbose=args.verbose)
        vae.encoder.save_weights("./model_outputs/model/enc")
        vae.decoder.save_weights("./model_outputs/model/dec")
        vae.save_weights("./model_outputs/model/full")
        vae.encoder.save("./model_outputs/model/enc.ckpt")
        vae.decoder.save("./model_outputs/model/dec.ckpt")
        vae.save("./model_outputs/model/full.ckpt")
    
        fig = plt.figure(figsize=(12, 14))
        num_plots = 3
        effective_epochs = len(history.history["rec_loss_str"])
        plt.subplot(num_plots, 1, 1)
        plt.title("Distribution Loss Structure")
        plt.plot(range(effective_epochs), history.history["rec_loss_str"])
        plt.plot(range(effective_epochs), history.history["val_rec_loss_str"])
        plt.subplot(num_plots, 1, 2)
        plt.title("KL Loss")
        plt.plot(range(effective_epochs), history.history["kl_loss"])
        plt.plot(range(effective_epochs), history.history["val_kl_loss"])
        plt.subplot(num_plots, 1, 3)
        plt.title("Cumulative Loss")
        plt.plot(range(effective_epochs), history.history["loss_total"])
        plt.plot(range(effective_epochs), history.history["val_loss_total"])
        plt.savefig("./losses_progress.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--latent_dim", type=int, default=1200, help="Latent dimension size")
    parser.add_argument("--atoms", type=int, default=400, help="Number of atoms in each system")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=4e-7, help="Learning rate")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .dat upper triangular matrices file")
    parser.add_argument("--npy_file", type=str, required=True, help="Path to the .npy full matrices file")
    parser.add_argument("--temps_file", type=str, required=True, help="Path to temperatures file")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level")
    args = parser.parse_args()
    main(args)
