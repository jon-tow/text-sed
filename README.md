## text-sed (wip) 📚🔥💭

Implementation of Strudel et al.'s ["Self-conditioned Embedding Diffusion for Text Generation"](https://arxiv.org/abs/2211.04236) in PyTorch and some other goodies 🍭.

## Install 💻

```bash
pip install -e ".[dev, train]"
```

## Experimental Samples 

Non-cherry picked samples from various experiments to track progress with the eventual goal of reproducing the results from the paper using large scale pre-training. 

* __Unconditionallly__ generated samples obtained from training text-sed on the simple [E2E](https://huggingface.co/datasets/e2e_nlg) dataset for 17k steps with the config [here](configs/e2e.yaml).
  ```markdown
  ➜ The Golden Palace is a mid priced restaurant that has a rating of 1 out of 5.
  ➜ There is a children friendly priced restaurant that offers English food called The Twenty Two.
  ➜ Taste of Cambridge is a nice family friendly fast food pub in the riverside area, near The Sorrento.
  ➜ In the city centre lies The Golden Palace, a above average coffee shop serving French cuisine. Previous its customers,, unfortunately because it 3 out of 5.
  ➜ Strada specializes in Chinese food. They are a pub near Yippee Noodle Bar and has a rating of 1 out of 5.
  ➜ The Vaults is a high priced restaurant serving Indian food. It is kid friendly and is moderately priced.
  ➜ The Waterman is a kid friendly restaurant that serves Japanese food near the city center. They are moderately priced.
  ➜ The Punter is a Chinese restaurant, with an average rating. The food is cheap.
  ```

## TODOs

* [ ] Add span masking and cfg for conditional generation.
* [ ] Add EMA warmup.
* [ ] Add Karras samplers.
* [ ] Add conditional generation examples/samples.

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
