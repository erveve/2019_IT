import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos


def get_pareto_indices(X):
    N, M = X.shape
    inds = np.array([])
    for i, x1 in enumerate(X):        
        for j, x2 in enumerate(X):
            if (i!=j and (np.prod(x2>=x1))):
                break
            
            if j==(N-1):
                inds = np.append(inds,[i])
    
    return inds.astype(int)


def get_polar_coords(X):
    N, M = X.shape 
    # y[i,0,j] - массив x
    # y[i,1,j] - массив y
    y = np.zeros((N,2,M))  
    y[:,0] = np.copy(X)
    
    for i in range(M):
        matr = rotation_matrix(2*pi*i/M)
        y[:,:,i] = (np.dot(matr, y[:,:,i].T)).T

    return y


def rotation_matrix(fi):
    return np.array([[cos(fi), sin(fi)],\
                     [-sin(fi), cos(fi)]])


def draw_axis(M, length):
    x0=np.array([length,0])
    
    for i in range(M):
        x1 = np.dot(rotation_matrix(2*pi*i/M),x0)

        plt.plot([0,x1[0]], 
                 [0,x1[1]],
                 linestyle = '--',
                 linewidth = 1,
                 color = 'grey')

        plt.annotate( f"x{i}",(x1[0],x1[1]))
        

def draw_points(X):
    N, M = X.shape 
    y = get_polar_coords(X)    
    
    colors = "bgcmy"
    for j in range(N):
        plt.plot(y[j,0,:], y[j,1,:], c=colors[j], alpha=0.6)
        plt.plot([y[j,0,-1], y[j,0,0]], 
                 [y[j,1,-1], y[j,1,0]],
                 c=colors[j], alpha=0.6)


def draw_pareto(X):
    N, M = X.shape 
    y = get_polar_coords(X)   
    y = [np.reshape(y[:,0,:], N*M),
         np.reshape(y[:,1,:], N*M)]
    plt.scatter(y[0], y[1], c='r', marker='*', alpha=1)


def main():
    X=np.array([[1, 2, 3, 4, 7],
                [2, 3, 4, 5, 6],
                [1, 1, 1, 1, 1]])
    N, M = X.shape 

    draw_axis(M, np.max(X)+1)
    draw_points(X)
    draw_pareto(X[get_pareto_indices(X)])
    
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()    
