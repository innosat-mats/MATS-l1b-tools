# MATS-l1b-tools
Repository to hold tools for manipulating, plotting and analyzing MATS L1b dataset. 
For assistance and discussion please consider joining our MATS discord community
https://discord.com/channels/1197482909997727774/1197482911763533836

Start by downloading this repository (click on Code->Download Zip), optionally clone this repository to your working directory using:
```bash
git clone git@github.com:innosat-mats/MATS-l1b-tools.git
```

# Optional: Consider special environment 
Next, you may want to consider creating a special environment for your MATS project. Python 3.11 is the recommended version for compatibility. You do this by running the following command:
conda create -n MATSlevel1b python=3.11 

# Optional: Installing this repository 
You can install this repository by executing the following commands. 
```bash
cd MATS-l1b-tools
pip install -e .
```
This allows you to execute the files from other working directories.

#### MATS data download with get_zarr.py script
If you only wish to use the data download script, it should be sufficient to install its dependencies by (inside the MATS-l1b-tools directory) running

    pip install -r requirements.txt

The script can be executed by running 

    python get_zarr.py [options]

 Available options:

| Option    | Description |
| --------- | ----------- |                       
|-h or --help                           | Show short help info and exit |
|-c or --channel                        | Data channel to download.<br>Can be IR1, IR2, IR3, IR4, UV1, UV2.<br>This is mandatory unless advanced options are used. |
|-b or --start_time YYYY MM DD hh mm ss | Start time for data set |
|-e or --stop_time YYYY MM DD hh mm ss  | Stop time for data set |
|-f or --filter [variable] [lower bound] [upper bound] | Filter data set by values of given variable |
|-n or --ncdf_out [filename]            | Generate output as netCDF4 file with given name |
|-z or --zarr_out [filename]            | Generate output as zarr file with given name |
|-t or --tp_coords                      | Interpolate tangent point coordinates for every pixel |

Note that filtering can only be used with variables that have one value per image. Trying to use it with more complex variables would return an error. 
If neither zarr nor netCDF output is specified, netCDF file with a default name will be created.

By default, the latest data version on the Bolin centre server (https://bolin.su.se/data/s3/) will be used. The following options can be added to use a different zarr archive and/or a different server:

| Advanced option   | Description |
| ---------         | ----------- |  
|-u or --url        | Use non-default server with the specified URL. |
|-p or --path_base  | Specify non-default file path on the server (without channel name) |
|-P or --path_full  | Specify non-default full file path on the server (overrides path_base) |
|-a or --address    | Explicitly specify full address of zarr archive (overrides all of the above) |

An example command for running get_zarr.py:

    python get_zarr.py -c IR1 -b 2023 2 20 0 0 0 -e 2023 3 1 0 0 0  -f TPlon 10 30
