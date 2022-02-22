'''
DLM Models and Relevant Functionality
'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
from scipy.linalg import block_diag

# Local Code
from Utilities import load_data, check_shape, check_square
from Matrix_Utilities import poly_mats, trig_mats, trig_inits

# Filter a sample
def filter_sample(Model, Data, init, final, set_init=True):
     if set_init: Model.m[0,0] = Data[init]
     point_est = []
     innovation = []
     obs_var = []
     for t in range(init, final):
          ret = Model.filter(Data[t], return_results=True, return_innovation=True, return_obs_var=True, return_forecast=True)
          point_est.append(Model.m[0,0])
          #point_est.append(ret['forecast'][0,0])
          innovation.append(ret['innovation'][0,0])
          obs_var.append(ret['obs_var'][0,0])
     return np.array(point_est), np.array(innovation), np.array(obs_var)

# DLM parent class
class DLM:
     def __init__(self, m, C, G, F, W, V):

          # State
          self.m = check_shape(m)
          self.C = check_square(C)

          # Forecast matrix
          self.G = check_square(G)
          self.G_T = np.transpose(check_shape(G))

          # Observation matrix
          self.F = check_shape(F, column=False)
          self.F_T = np.transpose(check_shape(F, column=False))

          # Forecast covariance
          self.W = check_square(W)

          # Observation covariance
          self.V = check_square(V)
     
     def copy(self):
          return DLM(self.m, self.C, self.G, self.F, self.W, self.V)

     def to_discount(self, df, alpha, beta):
          return DLMDiscount(self.m, self.C, self.G, self.F, df, alpha, beta)

     def add_model(self, M):

          # State
          self.m = np.concatenate((self.m, M.m))
          self.C = block_diag(self.C, M.C)

          # Forecast matrix
          self.G = block_diag(self.G, M.G)
          self.G_T = block_diag(self.G_T, M.G_T)

          # Observation matrix
          self.F = np.concatenate((self.F, M.F), axis=1)
          self.F_T = np.concatenate((self.F_T, M.F_T))

          # Forecast covariance
          self.W = block_diag(self.W, M.W)

          # Observation covariance
          self.V = self.V + M.V

     def filter(self, z, return_results=False, **kwargs):

          # Forecast step
          self.m, self.C = self.forecast()

          # Data assimilation step
          ret = self.data_assimilation(z, **kwargs)
          self.m, self.C = ret['m'], ret['C']

          if return_results: return ret     

     def forecast(self):

          # Forecast distribution parameters
          m_forecast = np.dot(self.G, self.m)
          C_forecast = np.dot(self.G, np.dot(self.C, self.G_T)) + self.W

          return m_forecast, C_forecast
     
     def data_assimilation(self, obs, return_innovation=False, return_obs_var=False, return_forecast=False):

          # Predictive distribution parameters
          f = np.dot(self.F, self.m)
          Q = np.dot(self.F, np.dot(self.C, self.F_T)) + self.V

          # Forecast error
          innovation = obs - f

          # Kalman gain
          K = self.K_gain(Q)

          # Assimilate data
          m_analysis = self.m + np.dot(K, innovation)
          C_analysis = np.dot((np.identity(self.C.shape[0]) - np.dot(K, self.F)), self.C)
          ret = {'m': m_analysis, 'C': C_analysis}

          # Optional returns
          if return_innovation: ret['innovation'] = innovation
          if return_obs_var: ret['obs_var'] = Q
          if return_forecast: ret['forecast'] = f

          return ret

     # Get Kalman Gain, given Q
     def K_gain(self, Q):

          Q_inv = np.linalg.inv(Q)
          K = np.dot(self.C, np.dot(self.F_T, Q_inv))

          return K
     
     # Print attributes
     def print_model(self):
          text_G = '\nForecast Matrix G = \n'
          text_F = '\nObservation Matrix F = \n'
          text_W = '\nForecast Covariance W = \n'
          text_V = '\nObservation Covariance V = \n'
          print(text_G, self.G, text_F, self.F, text_W, self.W, text_V, self.V)

# Polynomial model
class DLMPoly(DLM):
     def __init__(self, m, C, W_list, V):
          G, F, W, V = poly_mats(W_list, V)
          super().__init__(m, C, G, F, W, V)

# Periodic model
class DLMTrig(DLM):
     def __init__(self, init_var, omega, q, trig_var, V):
          G, F, W, V = trig_mats(omega, q, trig_var, V)
          m, C = trig_inits(q, init_var)
          super().__init__(m, C, G, F, W, V)

# Discount model
class DLMDiscount(DLM):
     def __init__(self, m, C, G, F, df, alpha, beta):
          W = np.identity(C.shape[0])
          V = np.array([[1]])
          super().__init__(m, C, G, F, W, V)
          self.df = df
          self.alpha = alpha
          self.beta = beta
     
     def copy(self):
          return DLMDiscount(self.m, self.C, self.G, self.F, self.df, self.alpha, self.beta)

     def filter(self, z, return_results=False, **kwargs):

          # Forecast step
          self.m, self.C, self.alpha, self.beta = self.forecast()

          # Data assimilation step
          ret = self.data_assimilation(z, **kwargs)
          self.m, self.C, self.alpha, self.beta = ret['m'], ret['C'], ret['alpha'], ret['beta']

          if return_results: return ret     

     def forecast(self):

          # Forecast distribution parameters
          m_forecast = np.dot(self.G, self.m)
          C_forecast = (1 / self.df) * np.dot(self.G, np.dot(self.C, self.G_T))

          return m_forecast, C_forecast, self.alpha, self.beta
     
     def data_assimilation(self, obs, return_innovation=False, return_obs_var=False, return_forecast=False):

          # Predictive distribution parameters
          f = np.dot(self.F, self.m)
          Q = np.dot(self.F, np.dot(self.C, self.F_T)) + self.V
          Q_inv = np.linalg.inv(Q)

          # Forecast error
          innovation = obs - f

          # Kalman gain
          K = self.K_gain(Q)

          # Assimilate data
          m_analysis = self.m + np.dot(K, innovation)
          C_analysis = np.dot((np.identity(self.C.shape[0]) - np.dot(K, self.F)), self.C)
          alpha_analysis = self.alpha + 0.5
          beta_analysis = self.beta + 0.5 * np.dot(np.transpose(innovation), np.dot(Q_inv, innovation))
          ret = {'m': m_analysis, 'C': C_analysis, 'alpha': alpha_analysis, 'beta': beta_analysis}

          # Optional returns
          if return_innovation: ret['innovation'] = innovation
          if return_obs_var: ret['obs_var'] = Q * self.beta / (self.alpha - 1)
          if return_forecast: ret['forecast'] = f

          return ret