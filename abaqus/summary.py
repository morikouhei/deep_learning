import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_output import read_fieldoutput, read_historyoutput
from collections import namedtuple

def write_to_summary(field_name,field_value,history_name,summary_name):
    df_field = read_fieldoutput(field_name,field_value)
    df_history = read_historyoutput(history_name)
    df = pd.DataFrame([{'name': 1, 'mises-mean': df_field.loc["mean"][0], 'mises-max': df_field.loc["max"][0],
                        'rf-mean': df_history.loc["mean"][0], 'rf-max': df_history.loc["max"][0]},
                       ])
    df.to_csv(summary_name,mode="a",header=False)
pastresult = namedtuple("pastresult", ["mises_mean", "mises_max", "rf_mean", "rf_max"])

def read_summary(summary_name):
    pastresult = namedtuple("pastresult", ["mises_mean", "mises_max", "rf_mean", "rf_max"])
    df = pd.read_csv("summary.csv")
    dic = {}
    for i in df.itertuples():
        dic[i[2]] = (pastresult(i[3], i[4], i[5], i[6]))

    return dic