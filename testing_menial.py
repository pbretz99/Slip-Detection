
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def stick_intervals(slips):
     def key_func(interval):
          return interval[0]
     slips = sorted(slips, key=key_func)
     sticks = []
     if len(slips) > 0:
          prev_stop = slips[0][1]
     for start, stop in slips:
          if start > prev_stop:
               sticks.append([prev_stop, start])
          prev_stop = stop
     return sticks

def interior_intervals(intervals, R=0):
     new_intervals = []
     for x, y in intervals:
          if x + R <= y - R:
               new_intervals.append([x + R, y - R])
     return new_intervals

def right_intervals(intervals, size=50):
     new_intervals = []
     for x, y in intervals:
          if x <= y - size:
               new_intervals.append([y - size, y])
     return new_intervals

slip_intervals = [[410, 600], [0, 100], [200, 350], [300, 400]]
print(stick_intervals(slip_intervals))
print(interior_intervals(stick_intervals(slip_intervals), R=10))
print(right_intervals(stick_intervals(slip_intervals), size=50))