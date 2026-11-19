## How to Run This
1. Click the green **Code** button above and select **Download ZIP**.
2. Unzip the folder onto your Desktop.
3. Open **Jupyter Notebook** (or **JupyterLab**) and open that unzipped folder.
4. Click **New** → **Python 3** to open a blank notebook.
5. Type `%run downloadWAM.py` into the first box and press **Shift + Enter** to run it.

NOTE:
1. This script prompts for the start/end date and downloads the netcdf model output
2. The script automatically pulls the most updated WAM-IPE model available (v1.2, v1.1, v1.0)
3. Each day of data is about 1GB
4. ML-TDM data for the SWIFT re-entry event (0630 UTC 11/4/26) has been archived and attached here as "mltdm_archive.tgz"
