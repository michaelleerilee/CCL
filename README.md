# CCL

<!--- -*- coding: utf-8 -*- --->

Copyright © 2018 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

For license information see the file LICENSE that should have accompanied this source.

Connected Component Labeling (CCL) using OpenCV for identifying structures in a stack of 2D data slices.

Simplest usage:

  labels = ccl_marker_stack().make_labels_from(data_slices,data_threshold_mnmx)

where data_slices is a list of 2D numpy arrays and data_threshold_mnmx is a (mn,mx) tuple.




# Depencencies

```bash
pip3 install dask['distributed']
pip3 install opencv-python
```

to run the examples:

jupyter
hdf5
matplotlib


## Zero-To examples

```bash
conda create --name ccl 
conda activate ccl 

conda install -c conda-forge jupyterlab
conda install -c conda-forge matplotlib
conda install -c conda-forge dask['distributed']
conda install -c conda-forge ipympl
conda install -c conda-forge nodejs
conda install -c conda-forge proj4 
conda install -c conda-forge basemap
conda install -c conda-forge netCDF4
 
conda install -c conda-forge proj4 basemap matplotlib ipympl nodejs ipykernel dask['distributed'] pip

pip install opencv-python￼
pip install pyhdf
pip install connected-components-3d
conda 



python -m ipykernel install --user --name=ccl
