# Test whether the apex FusedRMSNorm is the same as burn's RMSNorm
from apex.normalization import FusedRMSNorm
import torch

module = FusedRMSNorm(3, 1e-5)

input = torch.arange(9, dtype=torch.float32).view(3, 3)
output = module(input)
expected = torch.tensor(
    [
        [0.0000, 0.7746, 1.5492],
        [0.7348, 0.9798, 1.2247],
        [0.8514, 0.9933, 1.1352],
    ]
)

assert torch.allclose(output, expected, atol=1e-4)
