import setuptools


setuptools.setup(
    name="text-sed",
    version="0.0.1",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    author="Jon Tow",
    author_email="jonathantow1@gmail.com",
    url="http://github.com/jon-tow/text-sed",
    license="Apache 2.0",
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        "einops>=0.4",
        "ml_collections",
        "numpy",
        "torch>=1.12",
        "torchtyping",
        "transformers",
        "wandb",
    ],
    extras_require={
        "test": ["pytest"],
    },
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp machinelearning",
)
