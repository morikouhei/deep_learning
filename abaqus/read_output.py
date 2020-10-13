import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_fieldoutput(filename, value):
    df = pd.read_csv(filename)
    data = df[[value]]
    return data.describe()
def read_historyoutput(filename):
    df = pd.read_csv(filename)
    l = []
    for _,ind in df.itertuples():
        x,y = ind.split()
        l.append(-eval(y))

    l2 = pd.DataFrame(l)
    return l2.describe()