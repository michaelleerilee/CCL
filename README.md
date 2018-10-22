# CCL

<!--- -*- coding: utf-8 -*- --->

Copyright © 2018 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

For license information see the file LICENSE that should have accompanied this source.

Connected Component Labeling (CCL) using OpenCV for identifying structures in a stack of 2D data slices.

Simplest usage:

  labels = ccl_marker_stack().make_labels_from(data_slices,data_threshold_mnmx)

where data_slices is a list of 2D numpy arrays and data_threshold_mnmx is a (mn,mx) tuple.



