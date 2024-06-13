import torch


def fp32_to_bfp(tensor, group_size, mantissa_bits):
    shape = tensor.shape
    tensor = tensor.view(-1, group_size)

    max_exponents = torch.max(tensor.abs().log2().ceil(), dim=1)[0]

    aligned_mantissas = tensor * 2 ** (
        max_exponents.view(-1, 1) - tensor.abs().log2().ceil()
    )

    scale = 2**mantissa_bits
    truncated_mantissas = torch.floor(aligned_mantissas * scale) / scale

    bfp_values = truncated_mantissas * 2 ** (
        -max_exponents.view(-1, 1) + tensor.abs().log2().ceil()
    )
    bfp_values = bfp_values.view(shape)

    return bfp_values


tensor = torch.randn(32)
print("Tensor before quant: ", tensor)
group_size = 16
mantissa_bits = 4
bfp_tensor = fp32_to_bfp(tensor, group_size, mantissa_bits)

torch.set_printoptions(threshold=10_000)

print("Tensor after quant: ", bfp_tensor)
