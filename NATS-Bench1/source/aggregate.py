import numpy.linalg as LA
import numpy as np

def norm(x, L):
    #L: L1, or L2.
    if(L not in  [1, 2]):
        print("Error: L must be 1 or 2")
        exit()
    if(L == 1):
        return sum(abs(x)) 
    if(L == 2):
        return LA.norm(x)


if __name__ == "__main__":
    x = np.array([3,4,5.5])
    print(norm(x, 1))
    print(norm(x, 2))
    print(norm(x, 4))

