## Optical Pooled Screens of Essential Genes

This repository contains code and computational tools related to the publication, [*The phenotypic landscape of essential human genes*](https://pubmed.ncbi.nlm.nih.gov/36347254/).

For new projects using optical pooled screens, it is highly recommended to use the Github repository accompanying our Nature Protocols paper, [*Pooled genetic perturbation screens with image-based phenotypes*](https://pubmed.ncbi.nlm.nih.gov/35022620/): https://github.com/feldman4/OpticalPooledScreens.

## More about this repository

This repository contains additional application-specific resources for our study of essential gene function using optical pooled screens.

This includes:
- Many additional image features implemented as functions operating on scikit-image RegionProps objects (features come from CellProfiler and additional sources)
- Functions for analyzing live-cell optical pooled screens (calling TrackMate for cell tracking)

## Installation (OSX)

**WARNING: many versions of dependencies will have trouble installing on Python 3.8. It is currently recommended to use Python 3.6. Setting up a Python 3.6 conda environment may be a convenient solution, set-up guide [here](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python).**

Download the OpticalPooledScreens directory (e.g., on Github use the green "Clone or download" button, then "Download ZIP").

In Terminal, go to the OpticalPooledScreens project directory and create a Python 3 virtual environment using a command like:

```bash
python3 -m venv venv
```

If the python3 command isn't available, you might need to specify the full path. E.g., if [Miniconda](https://conda.io/miniconda.html) is installed in the home directory:

```bash
~/miniconda3/bin/python -m venv venv
```

This creates a virtual environment called `venv` for project-specific resources. The commands in `install.sh` add required packages to the virtual environment:

```bash
sh install.sh
```

The `ops` package is installed with `pip install -e`, so the source code in the `ops/` directory can be modified in place.

Once installed, activate the virtual environment from the project directory:

```bash
source venv/bin/activate
```

## Data Access

Raw image data from the fixed-cell screen presented in the [publication](https://pubmed.ncbi.nlm.nih.gov/36347254/) can be accessed from the public [BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) data repository (study [`S-BIAD394`](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD394)). Note that the web interface for this study is currently inactive. However, all data can be accessed using Aspera and the corresponding command line tool, `ascp`.

First, download the IBM Aspera Command Line Interface (includes `ascp`) from [here](http://www.ibm.com/support/fixcentral/swg/quickorder?parent=ibm%7EOther%20software&product=ibm/Other+software/IBM+Aspera+CLI&release=All&platform=All&function=all&source=fc), and complete the install by following the platform-specific user guides available [here](https://www.ibm.com/docs/en/aci/latest).

Then, the image data can be downloaded using the following helper function, executed from the command line while the virtual environment created during installation of this repository is activated:

```bash
python -m ops.download download_from_bioimage_archive \
<local filesystem destination> \
--plate <plate> \
--dataset <dataset>
--well <well> \
--site <tile> \
--ascp <path/to/ascp/executable>
```

Where valid `plate`s are:

| |
|-------------|
|20200202_6W-LaC024A|
|20200202_6W-LaC024B|
|20200202_6W-LaC024C|
|20200202_6W-LaC024D|
|20200202_6W-LaC024E|
|20200202_6W-LaC024F|
|20200206_6W-LaC025A|
|20200206_6W-LaC025B|

`dataset` must be either `sequencing` or `phenotype`. Any or all of `plate`, `well`, and `tile` can be set to `all` to download all of the corresponding images. Note that if `ascp` is in your path, you can use `--ascp=ascp`. The full list of available files is present in this repository [here](https://github.com/lukebfunk/OpticalPooledScreens/blob/master/paper/BioImage_Archive.csv.gz).
