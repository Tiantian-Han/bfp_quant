import torch
from mx.specs import finalize_mx_specs
from mx.mx_ops import quantize_mx_op
import mx.mx_mapping as mx_mapping

# Define the MX specs dictionary
mx_specs = {
    'w_elem_format': 'fp4_e2m1',
    'a_elem_format': 'fp4_e2m1',
    'scale_bits': 8,
    'block_size': 32,
    'custom_cuda': False,
}

# Finalize the MX specs
mx_specs = finalize_mx_specs(mx_specs)

# Inject PyTorch operations
mx_mapping.inject_pyt_ops(mx_specs)

# Create a sample tensor
input_tensor = torch.randn(4, 4, dtype=torch.float32)

# Convert the tensor to MX format
quantized_tensor = quantize_mx_op(
    input_tensor,
    mx_specs,
    elem_format=mx_specs['a_elem_format'],
    block_size=mx_specs['block_size'],
    axes=[0, 1],
    round="nearest",
    expand_and_reshape=False,
)

# Inspect the tensor
print("Original Tensor:\n", input_tensor)
print("Quantized Tensor:\n", quantized_tensor)
