import scipy.special
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def st_pdf(x, mu, sigma, nu):
    return scipy.special.gamma((nu + 1) / 2) / (
                sigma * np.sqrt(nu * np.pi) * scipy.special.gamma(nu / 2)) * \
           np.power(1 + ((x - mu) / sigma) ** 2 / nu, -(nu + 1) / 2)

def uniform_pdf(x, a, b):
    return scipy.stats.uniform(loc=a,scale=b-a).pdf(x)

def vg_pdf(x, mu, sigma, kappa, theta):
    horizon = 1
    coeff_1 = np.sqrt(2) * np.exp(theta * (x - mu * horizon) / (2 * sigma ** 2)) \
              / (sigma * np.sqrt(np.pi) * kappa ** (horizon / kappa) * scipy.special.gamma(
        horizon / kappa))
    coeff_2 = np.power(np.abs(x - mu * horizon) / np.sqrt(2 * sigma ** 2 / kappa + theta ** 2),
                       horizon / kappa - 0.5)
    bessel_order = horizon / kappa - 0.5
    bessel_param = np.abs(x - mu * horizon) * np.sqrt(
        2 * sigma ** 2 / kappa + theta ** 2) / sigma ** 2
    bessel = scipy.special.kv(bessel_order, bessel_param)
    res = coeff_1 * coeff_2 * bessel
    return res

# mu = 0
# nu = 3
# sigma_norm = np.sqrt(5)
# sigma = sigma_norm/np.sqrt(nu)

mu = 0.0002621652698799067
sigma = 0.009228227387922006
nu = 2.809213663439115
sigma_norm = np.sqrt(nu) * sigma

x = np.linspace(-10, 10, 100000)
pdf = st_pdf(x, mu, sigma, nu)
#pdf = vg_pdf(x, 0, np.sqrt(5), 0.5, 0)
#pdf = uniform_pdf(x, -1, 1)
delta = x[1] - x[0]
last_pmf = pdf * delta
one_pmf = pdf * delta
plt.plot(x, last_pmf / delta, label="PDF, n = 0")
for i in range(1, 255):
    last_pmf = scipy.signal.fftconvolve(one_pmf, last_pmf, 'same')
    last_pdf = last_pmf / delta
    if i == 1 or i == 50 or i == 100 or i == 254:
        plt.plot(x/np.sqrt(i+1), last_pdf * np.sqrt(i+1), label="PDF, n = %s" % (i))
plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma_norm), 'k-', label='N')
plt.xlim([-0.2, 0.2])
plt.ylim([1e-3, 100])
plt.yscale('log')
plt.legend()
plt.show()