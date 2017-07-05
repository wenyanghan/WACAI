import numpy as np
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
    
total =np.array( [5954,5396,5354,5611,5575,5020,5239,5635,5588,7233])
bad = np.array([215,215,259,269,308,308,329,402,418,778])

def get_ks_max(total,bad):
    sum_total = np.sum(total)
    sum_bad= np.sum(bad)
    ks = []
    for i in range(9):
        ks_temp = -(-(sum(total[:i+1])-sum(bad[:i+1]))/(sum_total-sum_bad)+sum(bad[:i+1])/sum_bad)
        ks.append(ks_temp)   
    max_ks = np.amax(ks)
    return max_ks

def show_hist(array):
    plt.hist(array,bins =10)
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
    
show_hist(total)