from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def display(x, y, degree = None): #Add Type
    #Add Docstring
    plt.scatter(x, y)
    plt.show()
    #if(degree != None):
    #    break
    #    ##Doesn't work yet. Will fix.
    #    poly = np.polyfit(x, y, degree)
    #    plt.plot(sort(x), np.polyval(poly, sort(x)))
    #    plt.show()

def rank_order_corr(x, y): #Add Type
    #Add Docstring
    rho, pval = stats.spearmanr(x, y)
    return rho, pval

def pearson_corr(x, y): ## Add type
    ## Add Docstring
    return stats.pearsonr(x,y)

if __name__ == "__main__":
    print("hello")
    x = np.random.rand(100)
    y = np.random.rand(100)
    print(rank_order_corr(x,y))
    print(pearson_corr(x,y))
    display(x, y)
    display(x, y, 2)