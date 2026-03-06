import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wimusim",
    version="0.2.0",
    description="WIMUSim — Wearable IMU Simulation Framework (SMPL branch)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/si-runnan/WIMUSim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "pybullet",
        "scipy",
        "tqdm",
        "wandb",
        # torch and pytorch3d better to be installed via conda
        # conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
        # conda install -c fvcore -c iopath -c conda-forge fvcore iopath
        # conda install pytorch3d -c pytorch3d
        "torch",
        "pytorch3d",
    ],
    extras_require={
        "dev": [
            "jupyterlab",
            "pytest>=6.0.2"
        ]
    },
    python_requires='>=3.8',
)
