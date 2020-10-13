import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_output import read_fieldoutput, read_historyoutput
from summary import write_to_summary, read_summary, pastresult
from collections import namedtuple


write_to_summary("field.csv","       S-Mises", "rf1.csv","summary.csv")
dic = read_summary("summary.csv")
if 1 in dic:
    print(dic[1])

x = pastresult(1,2,3,4)
print(x)