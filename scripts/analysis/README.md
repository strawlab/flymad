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
