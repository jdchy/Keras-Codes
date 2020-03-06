import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson,expon,nbinom

poisson_lambda = 4.3
p_arr = []

distribution = poisson(poisson_lambda)
for transactions in range(0,10):
     p_arr.append(distribution.pmf(transactions))

plt.ylabel('Probability')
plt.xlabel('Number of Transactions')
plt.xticks(range(0, 10))
plt.title('Poisson Probability Mass Function')
plt.plot(p_arr, color='black', linewidth=0.7, zorder=1)
plt.scatter(range(0, 10), p_arr, color='purple', edgecolor='black', linewidth=0.7, zorder=2)
# plt.show()

gamma_shape = 9
gamma_scale = 0.5

for customer in range(0, 100):
    distribution = poisson(np.random.gamma(shape=gamma_shape, scale=gamma_scale))
    p_arr = []
    for transactions in range(0,9):
        p_arr.append(distribution.pmf(transactions))
    plt.plot(p_arr, color='black', linewidth=0.7, zorder=1)
    
plt.ylabel('Probability')
plt.xlabel('Number of Transactions')
plt.xticks(range(0,9))
plt.title('Poisson Probability Distribution Curves 100 Customers')
# plt.show()

p = 0.52
p_arr = []

for i in range(0,10): 
  proba_inactive = p*(1-p)**(i-1)
  p_arr.append(proba_inactive)
p_arr = np.array(p_arr)
p_arr /= p_arr.sum()

plt.plot(range(0, 10), p_arr, color='black', linewidth=0.7, zorder=1)
plt.ylabel('Probability inactive')
plt.xlabel('Number of Transactions')
plt.xticks(range(0, 10))
plt.title('(Shifted) Geometric Probability Mass Function')
# plt.show()

beta_a = 2
beta_b = 3

for customer in range(0, 10):
  p_arr = []
  beta = np.random.beta(a=beta_a, b=beta_b)
  for transaction in range(1,10): 
    proba_inactive = beta*(1-beta)**(transaction-1)
    p_arr.append(proba_inactive)
  p_arr = np.array(p_arr)
  plt.plot(p_arr, color='black', linewidth=0.7, zorder=1)

plt.ylabel('Probability Inactive')
plt.xlabel('Number of Transactions')
plt.xticks(range(1, 10))
plt.title('Geometric Probability Mass Function 10 customers')
plt.show()