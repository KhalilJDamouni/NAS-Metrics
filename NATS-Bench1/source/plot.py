import csv
import pandas as pd
import matplotlib.pyplot as plt

filename = "layers.csv"
x = "KG-1/MC-1"
y = "test_acc"

data = pd.read_csv("outputs/"+filename)
plt.plot(data[x],data[y], '+', color='blue')
plt.xlabel(x)
plt.ylabel(y)
#plt.ylim([0,1.1])
plt.show()