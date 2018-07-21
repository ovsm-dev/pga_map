# pga_map.py
Parses event XML files (ShakeMap) and plots PGAs in mg on a map.

(c) 2017-2018 - Claudio Satriano (IPGP), Félix Léger (IPGP/OVSM), Jean-Marie Saurel (IPGP)

## Installation

### Installation of Python environment through Miniconda

  - Download Miniconda from
    [conda.io/miniconda.html](https://conda.io/miniconda.html).
    Python version 3.x is preferred over 2.x.

  - Run the installer. You might want to answer "no" to the
    request of updating your `.bashrc`, to avoid ovverriding
    the system `python` executable with the Miniconda one
    (Other WebObs PROCs might use different versions of Python.)

  - Install `matplotlib`, `cartopy` and `adjusttext` through the
    `conda` package manager:

        ~/miniconda3/bin/conda install matplotlib
        ~/miniconda3/bin/conda install -c conda-forge cartopy
        ~/miniconda3/bin/conda install -c phlya adjusttext

  - Install `pdfkit` using `pip`:

        ~/minicoda3/bin/pip install pdfkit


### Installation of the WebObs PROC

  - If not included in your WebObs release, place the PGA_MAP.conf
    file in `/etc/webobs.d/../CODE/tplates`:

        cp PROC.PGA_MAP /etc/webobs.d/../CODE/tplates/

  - Place the Python script under `/etc/webobs.d/../CODE/python/`:

        cp pga_map.py /etc/webobs.d/../CODE/python/

  - Place the bash wrap-up script under
    `/etc/webobs.d/../CODE/shells/`:

        cp pga_map.sh /etc/webobs.d/../CODE/shells/

  - Under `WebObs`, check the variable `PYTHON_PRGM` from
    `/etc/webobs.d/WEBOBS.rc` according to your Python
    installation:

        PYTHON_PRGM|/opt/webobs/miniconda3/python

  - Create a new PROC and select
    `PROC : Strong Motion mapping` from the list.

  - Edit the `RAWDATA` variable to point to the root directory
    of your shakemap files.

  - Edit the `MAP_XYLIM` variable with your boundaries list:

        lonmin,lonmax,latmin,latmax

  - Create a job in the scheduler with the following values:

        xeq1 = $WEBOBS{ROOT_CODE}/shells/pga_map.sh

  - Set the first argument to the PROC NAME:

        xeq2 = PYRAP

  - Set the delay the PROC looks for modifications according to
    the job interval. For example, if the scheduler launches the
    PROC every 5 minutes (300 seconds), set the delay to 15
    minutes:

        xeq3 = 15

