"""
temp_setup.py - this script conducts the absolute path mapping for an ipython session needed
to temporarily map the location of the SG_Seven_Decades module to memory.
"""

import os
import sys

path = os.getcwd()
sys.path.append(os.path.split(path)[0])