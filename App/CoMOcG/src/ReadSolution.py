######### Libraries #########
import numpy as np
import pandas as pd

######### Libraries #########

def ReadDataFrame(filename):
    df = pd.read_csv(filename)
    return df