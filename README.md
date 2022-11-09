# text-sed (wip)
Implementation of Strudel et al.'s ["Self-conditioned Embedding Diffusion for Text Generation"](https://arxiv.org/abs/2211.04236) in PyTorch.


```bash
# Optional CUDA wheels
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install -e ".[dev, train]"
```

# TODOs

- [ ] Add span masking.
- [ ] Add guidance for conditional text generation.