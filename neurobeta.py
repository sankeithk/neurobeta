from nb_functions import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nilearn
import click
import sys
import os
import click
import nipype
from nipype.interfaces import fsl

if __name__ == "__main__":
    i,o = args_checker(*sys.argv)
    print(f"Input file: {i}")
    print(f"Output directory: {o}")
    nb_basename = nb_getbasename(i)
    bet_file = brain_extractor(i, o, nb_basename)
    segmented_file = segmenter(bet_file, nb_basename)
    dicv_file = coregister(segmented_file, bet_file, nb_basename, dof = 12)
    




