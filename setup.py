from setuptools import setup, find_packages
import os

# Get the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Collect data files
data_files = []

# Add test_data directory
test_data_dir = os.path.join(base_dir, 'test_data')
if os.path.exists(test_data_dir):
    test_data_files = []
    for root, dirs, files in os.walk(test_data_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                test_data_files.append(rel_path)
    if test_data_files:
        data_files.append(('nexus/test_data', test_data_files))

# Add tutorial directory
tutorial_dir = os.path.join(base_dir, 'tutorial')
if os.path.exists(tutorial_dir):
    tutorial_files = []
    for root, dirs, files in os.walk(tutorial_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                tutorial_files.append(rel_path)
    if tutorial_files:
        data_files.append(('nexus/tutorial', tutorial_files))

setup(
    name="nexus",
    version="0.1.0",
    description="NEXUS: A contrastive learning model for single-cell RNA sequencing analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch==2.0.1",
        "torch-geometric==2.6.1",
        "scanpy==1.10.2",
        "igraph==0.11.8",
        "leidenalg==0.10.2",
        "h5py==3.13.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "harmonypy==0.0.10",
        "anndata==0.10.8",
        "matplotlib==3.10.1",
        "seaborn==0.13.2",
        "shap==0.47.1",
        "scikit-learn==1.6.1",
        "statannotations==0.7.2",
    ],
    data_files=data_files,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

