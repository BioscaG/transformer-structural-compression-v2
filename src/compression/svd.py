import copy

import torch
import torch.nn as nn


class SVDLinear(nn.Module):
    """Drop-in replacement for nn.Linear using a low-rank SVD factorization.

    Replaces W (m x n) with two matrices: W1 (m x k) and W2 (k x n),
    where k is the truncated rank.  Bias lives only on the second linear
    layer so the math stays correct: y = W2 @ (W1 @ x) + b.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.second.bias is not None}"
        )


def decompose_linear(layer: nn.Linear, rank: int) -> SVDLinear:
    """Decompose a nn.Linear layer into a low-rank SVDLinear via truncated SVD.

    The SVD is computed in float32 for numerical stability, then the result
    is cast back to the original dtype.
    """
    W = layer.weight.data.float()  # (out_features, in_features)
    has_bias = layer.bias is not None
    orig_dtype = layer.weight.dtype

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Truncate to rank k
    U_k = U[:, :rank]          # (out, k)
    S_k = S[:rank]             # (k,)
    Vh_k = Vh[:rank, :]        # (k, in)

    # Absorb singular values into U: U_k * diag(S_k)
    U_k = U_k * S_k.unsqueeze(0)  # broadcast multiply columns

    # W ≈ U_k @ Vh_k  =>  y = (U_k @ Vh_k) @ x + b = U_k @ (Vh_k @ x) + b
    # first.weight = Vh_k  (k x in_features)
    # second.weight = U_k  (out_features x k)
    svd_layer = SVDLinear(
        layer.in_features, layer.out_features, rank, bias=has_bias,
    )
    svd_layer.first.weight = nn.Parameter(Vh_k.to(orig_dtype))
    svd_layer.second.weight = nn.Parameter(U_k.to(orig_dtype))
    if has_bias:
        svd_layer.second.bias = nn.Parameter(layer.bias.data.clone())

    return svd_layer


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Retrieve a nested sub-module by its dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a nested sub-module by its dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def get_target_layer_names(model: nn.Module) -> list[str]:
    """Return names of all nn.Linear layers inside bert.encoder.

    These are the layers we want to compress (query, key, value, dense in
    attention + intermediate/output in FFN).  Excludes the classifier head
    and the pooler.
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "bert.encoder" in name:
            names.append(name)
    return names


def apply_svd_compression(
    model: nn.Module,
    rank: int,
    layer_names: list[str] | None = None,
    inplace: bool = False,
) -> nn.Module:
    """Apply truncated SVD compression to linear layers in the model.

    Parameters
    ----------
    model : nn.Module
        The original (fine-tuned) model.
    rank : int
        Truncated rank for the SVD decomposition.
    layer_names : list[str] | None
        Which layers to compress.  Defaults to all encoder Linear layers.
    inplace : bool
        If False (default), works on a deep copy so the original is untouched.

    Returns
    -------
    nn.Module
        The compressed model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    if layer_names is None:
        layer_names = get_target_layer_names(model)

    for name in layer_names:
        linear = _get_module_by_name(model, name)
        if not isinstance(linear, nn.Linear):
            continue
        svd_linear = decompose_linear(linear, rank)
        _set_module_by_name(model, name, svd_linear)

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_compression_ratio(original: nn.Module, compressed: nn.Module) -> float:
    """Compute the ratio of original parameters to compressed parameters."""
    orig_params = count_parameters(original)["total"]
    comp_params = count_parameters(compressed)["total"]
    return orig_params / comp_params


def compute_singular_value_energy(model: nn.Module) -> dict:
    """Compute singular values and cumulative energy for each encoder Linear layer.

    Returns a dict mapping layer name to a dict with:
      - 'singular_values': 1-D tensor of singular values
      - 'cumulative_energy': 1-D tensor of cumulative energy ratios (0 to 1)
      - 'rank_for_90': int, minimum rank to capture 90% of energy
      - 'rank_for_95': int, minimum rank to capture 95% of energy
      - 'rank_for_99': int, minimum rank to capture 99% of energy
    """
    target_names = get_target_layer_names(model)
    result = {}

    for name in target_names:
        layer = _get_module_by_name(model, name)
        W = layer.weight.data.float()
        S = torch.linalg.svdvals(W)

        energy = S ** 2
        total_energy = energy.sum()
        cumulative = torch.cumsum(energy, dim=0) / total_energy

        result[name] = {
            "singular_values": S.cpu(),
            "cumulative_energy": cumulative.cpu(),
            "rank_for_90": int((cumulative >= 0.90).nonzero(as_tuple=True)[0][0].item()) + 1,
            "rank_for_95": int((cumulative >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1,
            "rank_for_99": int((cumulative >= 0.99).nonzero(as_tuple=True)[0][0].item()) + 1,
        }

    return result
