import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def VaR(horizon, confidenceLevel, P, inMu, inSigma):
    alpha = norm.ppf(1 - confidenceLevel)
    return P - np.exp(inMu * horizon + alpha * inSigma * np.sqrt(horizon) + np.log(P))

hArray = np.linspace(0, 60, 10)
cArray = np.linspace(0.99, 0.9, 20)
mu = 0.0
sigma = 0.25
alpha = 0.1
[hMesh, cMesh] = np.meshgrid(hArray, cArray)
varSurface = VaR(hMesh, cMesh, 1, mu, sigma)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(hMesh, cMesh, varSurface, cmap=cm.coolwarm,
                       linewidth=2, antialiased=False)
ax.set_xlabel('horizon [Y]')
ax.set_ylabel('confidence level')
ax.set_zlabel('VaR')
print(varSurface)
plt.show()