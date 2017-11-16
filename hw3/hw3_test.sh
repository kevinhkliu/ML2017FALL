#!/bin/bash
wget 'https://www.dropbox.com/s/umu8nojm2zks9x9/CNN_model.hdf5?dl=1' -O 'CNN_model.hdf5' 
python3 hw3_test.py $1 $2 
