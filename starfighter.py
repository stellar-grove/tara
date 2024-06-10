# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 07:02:54 2023

@author: DanielKorpon
"""

# --------------------------------------------------------------

dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
import pint
import numpy as np

units = pint.UnitRegistry()

a = 9.8 * units.meter / units.second**2
a.magnitude
t = 3.4 * units.second
t
a * t**2/2
h = 381 * units.meter
t = np.sqrt(2 * h / a)
t
v = a * t
v
mile = units.mile
hour = units.hour
v = v.to(mile/hour)
v + (1 * (mile/hour))


import modsim as sim

bikeshare = sim.State(olin=10, wellesly=2)
bikeshare.olin += 1

def bike_to_wellesly():
    bikeshare.olin -= 1
    bikeshare.wellesly += 1
def bike_to_olin():
    bikeshare.olin += 1
    bikeshare.wellesly -= 1    

