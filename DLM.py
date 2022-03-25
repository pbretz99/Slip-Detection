'''
DLM Models and Relevant Functionality
'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
from scipy.linalg import block_diag

# Local Code
from Utilities import load_data, check_shape, check_square, print_tracker
from Matrix_Utilities import poly_mats, trig_mats, trig_inits

class Results:
     def __init__(self):
          self.m = None
          self.C = None
          self.forecast = []
          self.filter = []
          self.innovation = []
          self.obs_var = []
     
     def append(self, ret):
          self.m = ret['m']
          self.C = ret['C']
          self.forecast.append(ret['forecast'][0,0])
          self.filter.append(ret['filter'][0,0])
          self.innovation.append(ret['innovation'][0,0])
          self.obs_var.append(ret['obs_var'][0,0])
     
     def point_estimate(self):
          return np.array(self.filter)
     
     def standardized_error(self):
          innovation = np.array(self.innovation)
          obs_var = np.array(self.obs_var)
          return innovation / np.sqrt(obs_var)
     
class ResultsDiscount(Results):
     def __init__(self):
          super().__init__()
          self.alpha = []
          self.beta = []
     
     def append(self, ret):
          self.forecast.append(ret['forecast'][0,0])
          self.filter.append(ret['filter'][0,0])
          self.innovation.append(ret['innovation'][0,0])
          self.obs_var.append(ret['obs_var'][0,0])
          self.alpha.append(ret['alpha'])
          self.beta.append(ret['beta'])
     
     def var_point_estimate(self):
          alpha = np.array(self.alpha)
          beta = np.array(self.beta)
          return beta / (alpha - 1)

# Filter a sample
def filter_sample(Model, Data, init, final, set_init=True, discount_model=True, reset_to_zero=False, forgetful=False, memory=100, verbose=False):
     Temp_Model = Model.copy()
     if set_init: Temp_Model.m[0,0] = Data[init]
     if reset_to_zero: Temp_Model.m[0,0] = 0
     if discount_model: results = ResultsDiscount()
     else: results = Results()
     for t in range(init, final):
          if verbose: print_tracker(t-init, final-init)
          if t > init + memory: ret = Temp_Model.filter(Data[t], return_results=True, forgetful=forgetful, memory=memory)
          else: ret = Temp_Model.filter(Data[t], return_results=True)
          results.append(ret)
     if verbose: print('Complete!')
     return results

# Initialize discount filter with local level and drift term + periodic
def set_up_drift_discount_filter(init_val, omega, df, alpha, beta, J=2, my_EKF=False):

     m = np.array([[init_val], [0]])
     C = np.array([[0.05, 0.01], [0.01, 0.05]])

     Model = DLMPoly(m, C, [0, 0], np.array([[0]]))
     if J > 0:
          ModelTrig = DLMTrig(1, omega, J, 0, np.array([[0]]))
          Model.add_model(ModelTrig)
     
     discount_Model = Model.to_discount(df, alpha, beta)
     discount_Model.my_EKF = my_EKF
     
     return discount_Model

# Initialize discount filter with local level + periodic
def set_up_local_discount_filter(init_val, omega, df, alpha, beta, J=2, my_EKF=False):

     m = np.array([[init_val]])
     C = np.array([[0.05]])

     Model = DLMPoly(m, C, [0], np.array([[0]]))
     ModelTrig = DLMTrig(1, omega, J, 0, np.array([[0]]))
     Model.add_model(ModelTrig)
     
     discount_Model = Model.to_discount(df, alpha, beta)
     discount_Model.my_EKF = my_EKF
     
     return discount_Model

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

     def set_inits(self, results):
          self.m = results.m
          self.C = results.C

     def filter(self, z, return_results=False):

          # Forecast step
          self.m, self.C = self.forecast()

          # Data assimilation step
          ret = self.data_assimilation(z)
          self.m, self.C = ret['m'], ret['C']

          if return_results: return ret     

     def forecast(self):

          # Forecast distribution parameters
          m_forecast = np.dot(self.G, self.m)
          C_forecast = np.dot(self.G, np.dot(self.C, self.G_T)) + self.W

          return m_forecast, C_forecast
     
     def data_assimilation(self, obs):

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
          ret['forecast'] = f
          ret['filter'] = m_analysis
          ret['innovation'] = innovation
          ret['obs_var'] = Q
          
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
     def __init__(self, m, C, G, F, df, alpha, beta, my_EKF=False):
          W = np.identity(C.shape[0])
          V = np.array([[1]])
          super().__init__(m, C, G, F, W, V)
          self.df = df
          self.alpha = alpha
          self.beta = beta
          self.my_EKF = my_EKF
     
     def copy(self):
          return DLMDiscount(self.m, self.C, self.G, self.F, self.df, self.alpha, self.beta)

     def filter(self, z, return_results=False, forgetful=False, memory=100):

          # Forecast step
          self.m, self.C, self.alpha, self.beta = self.forecast()

          # Data assimilation step
          ret = self.data_assimilation(z, forgetful, memory)
          self.m, self.C, self.alpha, self.beta = ret['m'], ret['C'], ret['alpha'], ret['beta']

          if return_results: return ret     

     def forecast(self):

          # Test out EKF filter with update of the form mu <- mu * e^b + nu
          if self.my_EKF:
               self.G[0,0] = np.exp(self.m[1,0])
               self.G[0,1] = self.m[0,0] * np.exp(self.m[1,0])

          # Forecast distribution parameters
          m_forecast = np.dot(self.G, self.m)
          C_forecast = (1 / self.df) * np.dot(self.G, np.dot(self.C, self.G_T))

          # Set forecast m directly
          if self.my_EKF: m_forecast[0,0] = self.m[0,0] * np.exp(self.m[1,0])

          return m_forecast, C_forecast, self.alpha, self.beta
     
     def data_assimilation(self, obs, forgetful=False, memory=100):

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
          if forgetful:
               alpha_analysis = self.alpha
               beta_analysis = (1 - 1 / memory) * self.beta + 0.5 * np.dot(np.transpose(innovation), np.dot(Q_inv, innovation))
          else:
               alpha_analysis = self.alpha + 0.5
               beta_analysis = self.beta + 0.5 * np.dot(np.transpose(innovation), np.dot(Q_inv, innovation))
          ret = {'m': m_analysis, 'C': C_analysis, 'alpha': alpha_analysis, 'beta': beta_analysis[0,0]}

          # Optional returns
          ret['forecast'] = f
          ret['filter'] = m_analysis
          ret['innovation'] = innovation
          ret['obs_var'] = Q * self.beta / (self.alpha - 1)
          
          return ret
