# NEXUS
NEXUS is a computational model designed to biologically meaningful encoding of single-cell transcriptomic data at the sample level. It enables the dimensional encoding of single-cell transcriptomes at the sample scale and leverages these encodings for downstream analyses such as sample population clustering.


# Installation
Method 1: Install from Source (Recommended)
Clone the repository:
git clone https://github.com/JingtaoLab/NEXUS.gitcd NEXUS
Install dependencies and the package:
pip install -r requirements.txtpip install -e .
Alternatively, install using setup.py:
python setup.py install

Method 2: Install via pip
If the package is published to PyPI:
pip install nexus
Download Test Dataset


# The test dataset
The test dataset test.h5ad is available in GitHub Releases. You can download it using the following methods:
Method 1: Download via Browser
Visit the Releases page:
https://github.com/JingtaoLab/NEXUS/releases/tag/test-dataset
Find the test.h5ad file in the Assets section and click to download

Method 2: Download via Command Line
Using wget (Linux/Mac):
wget https://github.com/JingtaoLab/NEXUS/releases/download/test-dataset/test.h5ad
Or using curl:
curl -L -O https://github.com/JingtaoLab/NEXUS/releases/download/test-dataset/test.h5ad
In Windows PowerShell:
Invoke-WebRequest -Uri https://github.com/JingtaoLab/NEXUS/releases/download/test-dataset/test.h5ad -OutFile test.h5ad

Method 3: Download using Python
import urllib.requesturl = "https://github.com/JingtaoLab/NEXUS/releases/download/test-dataset/test.h5ad"urllib.request.urlretrieve(url, "test.h5ad")


# Usage
For examples and tutorials, please refer to the Jupyter Notebooks in the tutorial/ directory.


# System Requirements
Python 3.10.16
Dependencies are listed in requirements.txt





