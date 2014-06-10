# Plotting Basics

## Scripts for plotting bag files

* bag/plot_one.py
  
  plots the trajectory, laser state, and velocity of all
  tracked objects in a bag file

* bag/plot_ttm_performance.py

  used to quantifying latency and performance of TTM tracking. Given a 
  bag file that contains eith TTM-head or TTM-body tracking, it plots the
  image processing time, accuracy and latency of TTM.

* bag/plot_speed.py

  plots aligned timeseries, for all genotypes, showing speed when laser was on.
  see the help file for information.
  
  ```python bag/plot_speed.py path/*.bag --show --genotypes foo-ok371,bar-ok371```

## Scripts for plotting csv files

* csv/plot_scored_courtship.py

  plots all courtship plots as seen in figure4 of the paper. see the
  help file for information.

  in the paper this was run the following ways

  ```python csv/plot_scored_courtship.py /data/ --show --exp-genotype wGP --other-genotypes wtrpmyc,40347trpmyc,G323,40347```

  and

  ```python csv/plot_scored_courtship.py /data/exemplary/ --show --calibration-file foo.filtered.yaml --exp-genotype wGP --other-genotypes wtrpmyc,40347trpmyc,G323,40347 --only-trajectories 100L```

  in the directory of csv files there should be one mp4 file which allows you to
  define the target locations.

