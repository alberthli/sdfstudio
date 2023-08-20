import yaml
from pathlib import Path, PosixPath
from typing import Sequence

import flax.linen as linen
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax import lax

# needed for int64
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

def load_trained_pipeline(
    config_path: PosixPath,
    data_path: PosixPath,
    ckpt_path: PosixPath,
    device: torch.device,
):
    """Loads a trained pipeline.

    Parameters
    ----------
    config_path : PosixPath
        The path to the config YAML file associated with a trained model.
    data_path : PosixPath
        The path to the data that trained the model.
    ckpt_path : PosixPath
        The path to the ckpt file you want to load.
    device : torch.device
        The device.
    """
    # setting up pipeline config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager.dataparser.data = data_path
    config.trainer.load_dir = config.get_checkpoint_dir()
    config.pipeline.datamanager.eval_image_indices = None

    # loading pipeline with saved checkpoint
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    loaded_state = torch.load(ckpt_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    pipeline.eval()
    return pipeline

# ##### #
# TORCH #
# ##### #

class SDFWrapper(nn.Module):
    """A wrapper for the SDF and the S-density from the NeuS paper."""

    def __init__(self, pipeline):
        """Initializes the wrapper."""
        super().__init__()
        self.sdf_model = pipeline.model.field
        self.device = pipeline.device

    def forward(self, x: torch.tensor) -> torch.tensor:
        """SDF value."""
        h = self.sdf_model.forward_geonetwork(x[None, ...])
        s = torch.split(h, [1, self.sdf_model.config.geo_feat_dim], dim=-1)[0]
        return s[0, ...]

    def density(self, x: torch.tensor) -> torch.tensor:
        """Queries the volume density at x using the method of volSDF."""
        s = self.forward(x)  # sdf value
        return self.sdf_model.laplace_density(s)

# ### #
# JAX #
# ### #

class PeriodicVolumeEncoding:
    """Jax implementation of the periodic volume encoding.

    Note for self: the hashing operation is super non-differentiable,
    which is fine - we only differentiate through the position encoder
    part, don't need gradients through the hash table.
    """

    def __init__(
        self,
        _scalings: torch.tensor,
        _hash_table: torch.tensor,
        num_levels: int = 16,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        smoothstep: bool = False,
    ) -> None:
        """Initializes the encoding.

        Parameters
        ----------
        _hash_table : torch.tensor
            Trained hash table from torch.
        """
        # ##################################### #
        # COPIED FROM SDFSTUDIO W/MINOR CHANGES #
        # ##################################### #

        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.n_output_dims = num_levels * features_per_level
        self.smoothstep = smoothstep

        levels = jnp.arange(num_levels)
        self.periodic_volume_resolution = 2 ** (log2_hashmap_size // 3)
        self.per_level_weights = 1.0

        # ####### #
        # MY PORT #
        # ####### #

        self.scalings = jnp.array(_scalings.detach().cpu().numpy())
        self.hash_offset = levels * self.hash_table_size
        self.hash_table = jnp.array(_hash_table.detach().cpu().numpy())

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(
        self,
        in_tensor: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns hash tensor using method described in Instant-NGP.

        WARNING: does NOT assume the inputs are batched.

        Parameters
        ----------
        in_tensor : jnp.ndarray, shape=(num_levels, 3)
            The tensor to hash.
        """
        # round to make it periodic
        x = in_tensor
        x = jnp.mod(x, self.periodic_volume_resolution)

        # xyz to index
        pvr = self.periodic_volume_resolution
        out = x[:, 0] * pvr ** 2 + x[:, 1] * pvr + x[:, 2] + self.hash_offset
        return out.astype(jnp.int64)

    def forward(self, in_tensor: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of encoder.

        Parameters
        ----------
        in_tensor : jnp.ndarray, shape=(3,)
            Query point in space.

        Returns
        -------
        hash_encoding : jnp.ndarray, shape=(num_levels * features_per_level,)
            The hash encoding.
        """
        scaled = jnp.outer(self.scalings, in_tensor)  # (num_levels, 3)
        scaled_c = jnp.ceil(scaled).astype(jnp.int32)
        scaled_f = jnp.floor(scaled).astype(jnp.int32)

        # this if statement is OK because it isn't input-conditioned
        offset = scaled - scaled_f
        if self.smoothstep:
            offset = offset ** 2 * (3.0 - 2.0 * offset)

        # computing hash encoding
        # inputs into hash_fn are (num_levels, 3)
        hashed_0 = self.hash_fn(scaled_c)  # (num_levels,)
        hashed_1 = self.hash_fn(jnp.stack([scaled_c[:, 0], scaled_f[:, 1], scaled_c[:, 2]], axis=-1))
        hashed_2 = self.hash_fn(jnp.stack([scaled_f[:, 0], scaled_f[:, 1], scaled_c[:, 2]], axis=-1))
        hashed_3 = self.hash_fn(jnp.stack([scaled_f[:, 0], scaled_c[:, 1], scaled_c[:, 2]], axis=-1))
        hashed_4 = self.hash_fn(jnp.stack([scaled_c[:, 0], scaled_c[:, 1], scaled_f[:, 2]], axis=-1))
        hashed_5 = self.hash_fn(jnp.stack([scaled_c[:, 0], scaled_f[:, 1], scaled_f[:, 2]], axis=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(jnp.stack([scaled_f[:, 0], scaled_c[:, 1], scaled_f[:, 2]], axis=-1))

        f_0 = self.hash_table[hashed_0]  # (num_levels, features_per_level)
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[:, 0:1] + f_3 * (1 - offset[:, 0:1])
        f_12 = f_1 * offset[:, 0:1] + f_2 * (1 - offset[:, 0:1])
        f_56 = f_5 * offset[:, 0:1] + f_6 * (1 - offset[:, 0:1])
        f_47 = f_4 * offset[:, 0:1] + f_7 * (1 - offset[:, 0:1])

        f0312 = f_03 * offset[:, 1:2] + f_12 * (1 - offset[:, 1:2])
        f4756 = f_47 * offset[:, 1:2] + f_56 * (1 - offset[:, 1:2])

        # (num_levels, features_per_level)
        encoded_value = f0312 * offset[:, 2:3] + f4756 * (1 - offset[:, 2:3])  
        return encoded_value.flatten()

class SDFEncoding:
    """Jax implementation of the SDF positional encoding."""

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        off_axis: bool = False,
    ) -> None:
        """Initialize the encoding."""
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.off_axis = off_axis
        self.P = jnp.array(
            [
                [0.8506508, 0, 0.5257311],
                [0.809017, 0.5, 0.309017],
                [0.5257311, 0.8506508, 0],
                [1, 0, 0],
                [0.809017, 0.5, -0.309017],
                [0.8506508, 0, -0.5257311],
                [0.309017, 0.809017, -0.5],
                [0, 0.5257311, -0.8506508],
                [0.5, 0.309017, -0.809017],
                [0, 1, 0],
                [-0.5257311, 0.8506508, 0],
                [-0.309017, 0.809017, -0.5],
                [0, 0.5257311, 0.8506508],
                [-0.309017, 0.809017, 0.5],
                [0.309017, 0.809017, 0.5],
                [0.5, 0.309017, 0.809017],
                [0.5, -0.309017, 0.809017],
                [0, 0, 1],
                [-0.5, 0.309017, 0.809017],
                [-0.809017, 0.5, 0.309017],
                [-0.809017, 0.5, -0.309017],
            ]
        ).T

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        if self.off_axis:
            out_dim = self.P.shape[1] * self.num_frequencies * 2
        else:
            out_dim = self.in_dim * self.num_frequencies * 2
        return out_dim

    def forward(self, in_tensor: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        WARNING: not batched! Also assumes "covs" is None, unlike
        the original implementation.

        Parameters
        ----------
        in_tensor : jnp.ndarray, shape=(input_dim,)
            The input tensor.
        """
        freqs = 2 ** jnp.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        if self.off_axis:
            scaled_inputs = jnp.outer(self.P.T @ in_tensor, freqs)  # (21, num_scales)
        else:
            scaled_inputs = jnp.outer(in_tensor, freqs)  # (input_dim, num_scales)
        scaled_inputs = scaled_inputs.flatten()
        encoded_inputs = jnp.sin(
            jnp.concatenate((scaled_inputs, scaled_inputs + jnp.pi / 2.0), axis=-1)
        )
        return encoded_inputs

class MLPFromTorch(linen.Module):
    """Loads an MLP from torch. Specifically assumes softplus activations."""
    W_list: Sequence[jnp.ndarray]

    def setup(self):
        self.linear_layers = [
            linen.Dense(features=W.shape[0], name=f"glin{i}")
            for i, W in enumerate(self.W_list)
        ]
        self.num_layers = len(self.linear_layers)

    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear_layers[i](x)
            x = linen.activation.softplus(100 * x) / 100
        x = self.linear_layers[-1](x)
        return x

class JaxSDF:
    """Jax version of the SDF wrapper."""

    def __init__(self, pipeline):
        """Initializes the flax model from a trained torch pipeline."""
        # pulling values from the torch SDF
        _sdf = pipeline.model.field  # torch wrapper
        self.n_output_dims = _sdf.encoding.n_output_dims
        self.use_position_encoding = _sdf.config.use_position_encoding
        self.use_grid_feature = _sdf.use_grid_feature
        self.num_layers = _sdf.num_layers
        if _sdf.config.off_axis:
            self.pe_out_dim = 2 * 21 * _sdf.config.position_encoding_max_degree
        else:
            self.pe_out_dim = 2 * 3 * _sdf.config.position_encoding_max_degree
        self.beta = jnp.array(_sdf.laplace_density.beta.detach().cpu().numpy())

        # hash encoding mask
        self.hash_encoding_mask = jnp.ones(
            _sdf.num_levels * _sdf.features_per_level
        )

        # periodic volume encoding
        self.pve = PeriodicVolumeEncoding(
            _sdf.encoding.scalings,
            _sdf.encoding.hash_table,
            num_levels = _sdf.num_levels,
            log2_hashmap_size=18,
            features_per_level=_sdf.features_per_level,
            smoothstep=_sdf.config.hash_smoothstep,
        )

        # sdf encoding
        self.position_encoding = SDFEncoding(
            in_dim=3,
            num_frequencies=_sdf.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=_sdf.config.position_encoding_max_degree - 1,
            off_axis=_sdf.config.off_axis,
        )

        # initializing MLP using parameters from torch model
        name_list = []
        W_list = []
        b_list = []
        for l in range(0, self.num_layers - 1):
            name = "glin" + str(l)
            torch_layer = getattr(_sdf, name)
            W = jnp.array(torch_layer.weight.detach().cpu().numpy())
            b = jnp.array(torch_layer.bias.detach().cpu().numpy())
            name_list.append(name)
            W_list.append(W)
            b_list.append(b)

        self.mlp = MLPFromTorch(W_list)
        self.params = {'params': {}}
        for name, W, b in zip(name_list, W_list, b_list):
            self.params['params'][name] = {
                'kernel': W.T,
                'bias': b,
            }

    def sdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the SDF value. Implements `forward_geonetwork` from sdfstudio."""
        # hash feature
        if self.use_grid_feature:
            position = (x + 2.0) / 4.0
            feature = self.pve.forward(position)
            feature = feature * self.hash_encoding_mask
        else:
            feature = torch.zeros(self.n_output_dims)

        # position encoding
        if self.use_position_encoding:
            pe = self.position_encoding.forward(x)
        else:
            pe = jnp.zeros(self.pe_out_dim)

        # MLP
        inputs = jnp.concatenate((x, pe, feature), axis=-1)
        h = self.mlp.apply(self.params, inputs)
        return h[0]

    def density(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the sdf-equivalent density."""
        s = self.sdf(x)  # sdf value

        # computing laplace density value
        beta = jnp.abs(self.beta) + 0.0001  # numerical stability
        alpha = 1.0 / beta
        density = 0.5 * alpha * (
            1.0 + jnp.sign(s) * (jnp.exp(-jnp.abs(s) / beta) - 1)
        )
        return density

if __name__ == "__main__":
    # hardcoded paths using pwd

    # ################################# #
    # USING TCNN HASHMAP IMPLEMENTATION #
    # ################################# #

    # trains very fast, but the hashmap implementation is uninterpretable under
    # the hood

    # config_path = Path(
    #     "/home/albert/research/"
    #     "sdfstudio/outputs/neus-facto-dtu65/"
    #     "neus-facto/2023-08-03_164714/config.yml"
    # )
    # data_path = Path(
    #     "/home/albert/research/"
    #     "sdfstudio/data/sdfstudio-demo-data/dtu-scan65"
    # )
    # ckpt_path = Path(
    #     "/home/albert/research/"
    #     "sdfstudio/outputs/neus-facto-dtu65/"
    #     "neus-facto/2023-08-03_164714/sdfstudio_models/step-000020000.ckpt"
    # )

    # ####################### #
    # USING PERIODIC ENCODING #
    # ####################### #

    # trains a bit slower, but we can transfer the implementation

    config_path = Path(
        "/home/albert/research/"
        "sdfstudio/outputs/neus-facto-dtu65/"
        "neus-facto/2023-08-04_172312/config.yml"
    )
    data_path = Path(
        "/home/albert/research/"
        "sdfstudio/data/sdfstudio-demo-data/dtu-scan65"
    )
    ckpt_path = Path(
        "/home/albert/research/"
        "sdfstudio/outputs/neus-facto-dtu65/"
        "neus-facto/2023-08-04_172312/sdfstudio_models/step-000020000.ckpt"
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    pipeline = load_trained_pipeline(config_path, data_path, ckpt_path, device)

    # testing in torch
    s = SDFWrapper(pipeline)
    x = torch.tensor([-0.1, -0.1, -0.1], device=device)  # (3,)
    print(s(x))
    print(s.density(x))

    # ############################### #
    # TESTING SDF AND ITS DERIVATIVES #
    # ############################### #
    from jax import hessian, jit, jacobian

    jsdf = JaxSDF(pipeline)
    sdf_func = jit(jsdf.sdf)
    density_func = jit(jsdf.density)

    Dsdf_func = jit(jacobian(jsdf.sdf))
    D2sdf_func = jit(hessian(jsdf.sdf))
    D3sdf_func = jit(jacobian(jacobian(jacobian(jsdf.sdf))))
    x_jax = jnp.array(x.cpu().numpy())
    print(density_func(x_jax))
    print(sdf_func(x_jax))
    print(Dsdf_func(x_jax))
    print(D2sdf_func(x_jax))
    print(D3sdf_func(x_jax))

    """
    When forcing the jax device to CPU, the first 3 of these are faster than on GPU,
    while the third of these is slower.
    """

    import time

    start = time.time()
    density_func(x_jax).block_until_ready()
    end = time.time()
    print(end - start)

    start = time.time()
    sdf_func(x_jax).block_until_ready()
    end = time.time()
    print(end - start)

    start = time.time()
    Dsdf_func(x_jax).block_until_ready()
    end = time.time()
    print(end - start)

    start = time.time()
    D2sdf_func(x_jax).block_until_ready()
    end = time.time()
    print(end - start)

    start = time.time()
    D3sdf_func(x_jax).block_until_ready()
    end = time.time()
    print(end - start)
    breakpoint()

    # beta = s.sdf_model.laplace_density.beta  # torch tensor still