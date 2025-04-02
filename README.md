# MATS-l1b-tools
Repository to hold tools for manipulating, plotting and analyzing MATS L1b dataset. 


#### MATS data download with get_zarr.py script
If you only wish to use the data download script, it should be sufficient to install its dependencies by running

    pip install -r requirements.txt

The script can be executed by running 

    python get_zarr.py <address> [options]

The address should be the web address of the zarr archive to be read. Must be preceded with "zip::" if the archive is zipped. Available options:

| Option    | Description |
| --------- | ----------- |                       
|-h or --help                           | Show short help info and exit |
|-b or --start_time YYYY MM DD hh mm ss | Start time for data set |
|-e or --stop_time YYYY MM DD hh mm ss  | Stop time for data set |
|-f or --filter [variable] [lower bound] [upper bound] | Filter data set by values of given variable |
|-n or --ncdf_out [filename]            | Generate output as netCDF4 file with given name |
|-z or --zarr_out [filename]            | Generate output as zarr file with given name |
|-t or --tp_coords                      | Interpolate tangent point coordinates for every pixel |

Note that filtering can only be used with variables that have one value per image. Trying to use it with more complex variables would return an error.

An example command for running get_zarr.py:

    python get_zarr.py zip::https://bolin.su.se/data/s3/data/mats-level-1b-limb-cropd-1.0-root/mats-level-1b-limb-cropd-UV2.zarr.zip -b 2023 2 20 0 0 0 -e 2023 3 1 0 0 0  -f TPlon 10 30
