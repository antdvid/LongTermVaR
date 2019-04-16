from scipy.special import *
import matplotlib.pyplot as plt
import scipy.optimize


def match_error(nv, T):
    ''' integration of st distribution from x* to +inf '''
    res = (nv+1)*gamma(0.5*(nv+1))/(np.sqrt(np.pi)*gamma(nv/2) * T**(0.5*(nv-2)) * (np.log(T))**(nv/2))
    return res

def target_fun(nv, conf_level):
    T = 250
    return match_error(nv, T) - conf_level/2

conf_levels = np.linspace(0.001, 0.01, 20)
sols = []
sol = 4.0
for conf_level in conf_levels:
    sol = scipy.optimize.fsolve(target_fun, x0=np.array([sol]), args=np.array([conf_level]))
    print("conf_level = ", conf_level, ", sol = ", sol, ", res = ", target_fun(sol, conf_level))
    sols.append(sol)
plt.plot(conf_levels, sols)
plt.show()