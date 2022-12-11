## text-sed (wip) 📚🔥💭

Implementation of Strudel et al.'s ["Self-conditioned Embedding Diffusion for Text Generation"](https://arxiv.org/abs/2211.04236) in PyTorch and some other goodies 🍭.

## Install 💻

```bash
pip install -e ".[dev, train]"
```

## TODOs

* [ ] Add span masking and cfg for conditional generation.
* [ ] Add Karras samplers.

## Appreciation

* Katherine Crowson's [`k-diffusion` repo](https://github.com/crowsonkb/k-diffusion)


## Citations

```bibtex
@article{strudel2022self,
  title={Self-conditioned Embedding Diffusion for Text Generation},
  author={Strudel, Robin and Tallec, Corentin and Altch{\'e}, Florent and Du, Yilun and Ganin, Yaroslav and Mensch, Arthur and Grathwohl, Will and Savinov, Nikolay and Dieleman, Sander and Sifre, Laurent and others},
  journal={arXiv preprint arXiv:2211.04236},
  year={2022}
}
```

```bibtex
@article{dieleman2022continuous,
  title={Continuous diffusion for categorical data},
  author={Dieleman, Sander and Sartran, Laurent and Roshannai, Arman and Savinov, Nikolay and Ganin, Yaroslav and Richemond, Pierre H and Doucet, Arnaud and Strudel, Robin and Dyer, Chris and Durkan, Conor and others},
  journal={arXiv preprint arXiv:2211.15089},
  year={2022}
}
```

```bibtex
@article{Chen2022AnalogBG,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey E. Hinton},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.04202}
}
```

```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```
