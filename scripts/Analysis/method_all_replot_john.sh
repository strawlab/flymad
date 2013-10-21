#!/bin/sh

python flymad_speed_multiplot.py ~/Dropbox/Straw/FLYMAD_data_prep_john/DATA_speed --only-plot --show &
python flymad_velocity_multiplot.py ~/Dropbox/Straw/FLYMAD_data_prep_john/DATA_velocity/ --only-plot --show &
python flymad_courtship_plots_10minexp.py ~/Dropbox/Straw/scored_data/courtship_10min/ --only-plot --show &

wait
