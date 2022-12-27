import setuptools

setuptools.setup(
    name="text-sed",
    version="0.0.1",
    description="A PyTorch implementation of Self-conditioned Embedding Diffusion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jon Tow",
    author_email="jonathantow1@gmail.com",
    url="http://github.com/jon-tow/text-sed",
    license="MIT",
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        "einops",
        "numpy",
        "torch>=1.12.1",
        "transformers",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort"],
        "train": ["datasets", "omegaconf", "wandb", "tqdm"],
    },
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp machinelearning text-generation",
)
