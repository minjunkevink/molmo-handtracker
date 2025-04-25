from setuptools import setup, find_packages

setup(
    name="molmo_cotrack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "imageio",
        "numpy",
        "matplotlib",
        "h5py",
        "tqdm",
        # Note: cotracker needs to be installed separately:
        # pip install git+https://github.com/facebookresearch/co-tracker.git
    ],
    description="Hand tracking tools using optical flow with CoTracker and SAM for MOLMO project",
    author="Kevin Kim",
    author_email="kimkj@usc.edu",
    url="https://github.com/minjunkevink/molmo_cotrack",
) 