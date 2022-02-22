'''
Helpful Function involving Matrices
'''

# Libraries
import numpy as np
from numpy import sin, cos
from scipy.linalg import block_diag

def add_above_diag(G, val=1):
     n = G.shape[0]
     for j in range(1, n):
          G[j-1,j] = val

def poly_mats(W_list, V):
     deg = len(W_list)
     W = np.diag(np.array(W_list))
     F = np.zeros((1, deg))
     F[0,0] = 1
     G = np.identity(deg)
     add_above_diag(G)
     return G, F, W, V

def trig_evol_mat(omega):
     H = np.array([[cos(omega), sin(omega)], [-sin(omega), cos(omega)]])
     return H

def trig_mats(omega, q, trig_var, V):
     G = trig_evol_mat(omega)
     F = np.zeros((1, q * 2))
     F[0,0] = 1
     #W = np.ones((q*2, q*2)) * trig_var
     W = np.identity(q*2) * trig_var
     for j in range(1, q):
          H_current = trig_evol_mat(omega * (j+1))
          G = block_diag(G, H_current)
          F[0,j*2] = 1
     return G, F, W, V

def trig_inits(q, init_var):
     m = np.zeros((q*2, 1))
     C = np.identity(q*2) * init_var
     return m, C


