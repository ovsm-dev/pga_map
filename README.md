# pga_map.py

Parses event XML files (ShakeMap) and plots PGAs in mg on a map.

(c) 2017-2022 - Claudio Satriano (IPGP), Félix Léger (IPGP/OVSM), Jean-Marie Saurel (IPGP)

## Installation

### Installation of Python environment through Miniconda

- Download Miniconda3 from
  [conda.io/miniconda.html](https://conda.io/miniconda.html).

- Run the installer. You might want to answer "no" to the
  request of updating your `.bashrc`, to avoid ovverriding
  the system `python` executable with the Miniconda one
  (Other WebObs PROCs might use different versions of Python.)

- Create the `pga_map` environment through:

      ~/miniconda3/bin/conda env create -f environment.yml

- Install `pdfkit` using `pip`:

      ~/minicoda3/envs/pga_map/bin/pip install pdfkit

## Testing the Python code

Run the test:

    ~/miniconda3/envs/pga_map/bin/python pga_map.py test/event.xml test/event_dat.xml test

You should get the following files in the `test` directory:

    ├── 2016
    │   └── 12
    │       └── 02
    │           └── ovsm2016xroz
    │               ├── 20161202T221159_ovsm2016xroz_pga_dist_fig.png
    │               ├── 20161202T221159_ovsm2016xroz_pga_map.html
    │               ├── 20161202T221159_ovsm2016xroz_pga_map.pdf
    │               ├── 20161202T221159_ovsm2016xroz_pga_map.txt
    │               ├── 20161202T221159_ovsm2016xroz_pga_map_fig.png
    │               ├── OVS_logo.png
    │               ├── RAP_logo.png
    │               ├── pga_map.pdf
    │               ├── pga_map.txt
    │               └── styles.css

Note: if you get an error like this one (generally on macOS):

    OSError: Could not find lib c or load any of its variants [].

reinstall `shapely` via `pip`:

    pip uninstall shapely && pip install --no-binary :all: shapely

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
