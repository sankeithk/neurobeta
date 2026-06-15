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
    run_dartel(i, o)




