
# Nyström Spectral Attention

**Linear-time, cosine-free attention for small language models.**

A practical drop-in replacement for scaled dot-product attention, based on the Nyström spectral method from the 2026 paper by Venky Rao.

**15.8× faster** at seq len 128 • **>30× faster** at seq len 2048 • Perfect for tiny LLMs on CPUs/laptops.

## Features

- O(L m d + m³) complexity (linear in sequence length)
- No cosine similarity or trig functions
- Fully differentiable & causal-ready
- Multi-head support
- Tiny memory footprint

## Installation

Copy [`nystrom_attention.py`](nystrom_attention.py) into your project.

## Usage

```python
import torch
from nystrom_attention import NystromAttention

attn = NystromAttention(d_model=512, num_heads=8, m=16)

x = torch.randn(2, 128, 512)           # (B, L, d)
out = attn(x, causal=True)             # (B, L, d)
```

## Quick Performance

| Length | Speedup vs Standard Attention |
|--------|-------------------------------|
| 128    | **15.8×**                     |
| 2048   | **>30×**                      |

## Citation

```bibtex
@article{rao2026nystrom,
  title={Beyond Scaled Dot-Product: Nyström Spectral Attention for Efficient Small-Scale Language Models},
  author={Venky Rao},
  year={2026}
}
```

## License

MIT

