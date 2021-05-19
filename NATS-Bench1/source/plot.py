import csv
import pandas as pd
import matplotlib.pyplot as plt
import correlate


filename = "correlation-05-19-2021_10-42-47.csv"
x = "mqBE_L5"
y = "g-gap"

data = pd.read_csv("outputs/"+filename)

print(correlate.pearson_corr(data[x],data[y]))
print(correlate.rank_order_corr(data[x],data[y]))

plt.plot(data[x],data[y], '+', color='blue')
plt.xlabel(x)
plt.ylabel(y)
#plt.ylim([0,1.1])
plt.show()
