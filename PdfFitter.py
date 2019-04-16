# this class is an implementation a method to fit the pdf
# the fit was obtained by minimizing the mean squared error between theoretical and observed CDF in the case
# of ST and VG distribution, and by moment matching in the Normal case
import abc
import numpy as np
import scipy.special
import scipy.optimize
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt


class DistributionFitter(abc.ABC):
    def __init__(self, is_pdf):
        self.residual = 10000000
        self.is_pdf = is_pdf

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def get_pdf(self, x):
        """this function should support vector input"""
        pass

    @abc.abstractmethod
    def get_label(self):
        pass

    @abc.abstractmethod
    def fit_to_pdf(self, grid, distribution):
        pass

    def get_mse(self):
        """return the mean square error of the fit"""
        return self.residual

    def get_cdf(self, grid):
        cdf_analy = []
        for upper_bound in grid:
            integ = scipy.integrate.quad(self.get_pdf, -np.inf, upper_bound)
            cdf_analy.append(integ[0])
            #print("upper_bound = ", upper_bound, "cdf = ", cdf_analy)
        return np.array(cdf_analy)

    @staticmethod
    def compute_error(d1, d2):
        return np.mean(np.square(d1 - d2))

    @staticmethod
    def hist_cdf(data, size):
        hist, bin_edges = np.histogram(data, density=True, bins=size)
        area = hist * (bin_edges[1:]-bin_edges[:-1])
        return area.cumsum(), bin_edges[1:]

    @staticmethod
    def hist_pdf(data, size):
        hist, bin_edges = np.histogram(data, density=True, bins=size)
        return hist, (bin_edges[:-1] + bin_edges[1:])/2

class NormalFitter(DistributionFitter):
    def __init__(self, horizon, grid_size, is_pdf):
        self.mean = 0
        self.var = 0
        self.horizon = horizon
        self.grid_size = grid_size
        DistributionFitter.__init__(self, is_pdf)

    def get_label(self):
        return "N"

    def fit(self, data):
        self.mean = np.mean(data)/self.horizon
        self.var = np.var(data)/self.horizon
        if self.is_pdf:
            pdf_data, grid = self.hist_pdf(data, self.grid_size)
            pdf_analy = self.get_pdf(grid)
            self.residual = DistributionFitter.compute_error(pdf_analy, pdf_data)
        else:
            cdf_data, grid = self.hist_cdf(data, self.grid_size)
            cdf_analy = self.get_cdf(grid)
            self.residual = DistributionFitter.compute_error(cdf_analy, cdf_data)

    def fit_to_pdf(self, pdf_grid, pdf):
        init_guess = np.array([0, 1])
        self.target_pdf = pdf
        self.target_pdf_grid = pdf_grid
        fit_res = scipy.optimize.minimize(self.target_fun, init_guess, method="Nelder-Mead", tol=1e-10)

    def target_fun(self, x):
        self.mean = x[0]
        self.var = x[1]
        pdf = self.get_pdf(self.target_pdf_grid)
        self.residual = self.compute_error(pdf, self.target_pdf)
        return self.residual

    def get_params(self):
        return {"mean": self.mean,
                "var": self.var}

    def get_pdf(self, x):
        return 1/np.sqrt(2 * np.pi * self.var * self.horizon) * np.exp(-(x - self.mean * self.horizon)**2/(2 * self.var * self.horizon))

    def get_cdf(self, x):
        limit = (x - self.mean * self.horizon)/np.sqrt(self.var * self.horizon)
        return 0.5 * (1 + scipy.special.erf(limit/np.sqrt(2)))


class StFitter(DistributionFitter):
    def __init__(self, grid_size, is_pdf):
        self.mu = 0
        self.sigma = 0
        self.nu = 0
        self.grid_size = grid_size
        self.is_pdf = is_pdf
        DistributionFitter.__init__(self, is_pdf)

    def get_label(self):
        return "ST"

    def fit(self, data):
        if self.is_pdf:
            self.distr_data, self.grid = self.hist_pdf(data, self.grid_size)
        else:
            self.distr_data, self.grid = self.hist_cdf(data, self.grid_size)

        init_guess = np.array([0.0, np.sqrt(5), 3])
        lower_bounds = np.array([0.0, np.sqrt(5), 3]) * 0.01
        upper_bounds = np.array([0.0, np.sqrt(5), 3]) * 1.99
        bounds = scipy.optimize.Bounds(lower_bounds, upper_bounds)
        fit_res = scipy.optimize.minimize(self.target_fun, init_guess, method="Nelder-Mead", tol=1e-10)

    def set_params_from_array(self, params):
        self.mu = params[0]
        self.sigma = np.abs(params[1])
        self.nu = params[2]

    def target_fun(self, params):
        self.set_params_from_array(params)
        if self.is_pdf:
            pdf_analy = self.get_pdf(self.grid)
            residual = DistributionFitter.compute_error(pdf_analy, self.distr_data)
        else:
            cdf_analy = self.get_cdf(self.grid)
            residual = DistributionFitter.compute_error(cdf_analy, self.distr_data)
        #print("params = ", params, ", residual = ", residual)
        self.residual = residual
        return residual

    def fit_to_pdf(self, pdf_grid, pdf):
        self.is_pdf = True
        init_guess = np.array([0.0, np.sqrt(5), 3])
        self.grid = pdf_grid
        self.distr_data = pdf
        fit_res = scipy.optimize.minimize(self.target_fun, init_guess, method="Nelder-Mead", tol=1e-10)
        return self.residual

    def get_params(self):
        return {"mu": self.mu,
                "sigma": self.sigma,
                "nu": self.nu}

    def get_pdf(self, x):
        return scipy.special.gamma((self.nu + 1)/2)/(np.abs(self.sigma) * np.sqrt(self.nu * np.pi) * scipy.special.gamma(self.nu/2)) * \
               np.power(1 + ((x - self.mu)/self.sigma)**2/self.nu, -(self.nu + 1)/2)

    def get_cdf(self, x):
        x_pos = [xi for xi in x if xi >= self.mu]
        x_neg = [xi for xi in x if xi < self.mu]
        if len(x_neg) > 0:
            cdf_pos = self.get_cdf(np.array(x_pos))
            cdf_neg = (1 - self.get_cdf(2*self.mu - np.array(x_neg)))
            cdf = np.concatenate([cdf_neg, cdf_pos])
            return cdf
        else:
            t = (x - self.mu)/self.sigma
            x_t = self.nu / (t**2 + self.nu)
            ibeta = scipy.special.betainc(self.nu/2, 0.5, x_t)
            cdf = 1 - 0.5 * ibeta
            return cdf


class VgFitter(DistributionFitter):
    def __init__(self, grid_size, horizon, is_pdf):
        self.grid_size = grid_size
        self.horizon = horizon
        self.mu = 0
        self.sigma = 0
        self.kappa = 1
        self.theta = 0
        DistributionFitter.__init__(self, is_pdf)

    def get_label(self):
        return "VG"

    def fit(self, data):
        if self.is_pdf:
            self.distr_data, self.grid = self.hist_pdf(data, self.grid_size)
        else:
            self.distr_data, self.grid = self.hist_cdf(data, self.grid_size)
        init_guess = np.array([0.0, 0.1, 5.0, 0.0])
        fit_res = scipy.optimize.minimize(self.target_fun, init_guess, method="Nelder-Mead", tol=1e-10)

    def fit_to_pdf(self, pdf_grid, pdf):
        self.is_pdf = True
        self.grid = pdf_grid
        self.distr_data = pdf
        init_guess = np.array([0.0, np.sqrt(5), 0.5, 0.0])
        fit_res = scipy.optimize.minimize(self.target_fun, init_guess, method="Nelder-Mead", tol=1e-10)
        return self.residual

    def set_params_from_array(self, params):
        self.mu = params[0]
        self.sigma = params[1]
        self.kappa = params[2]
        self.theta = params[3]

    def target_fun(self, params):
        self.set_params_from_array(params)
        if self.is_pdf:
            distr_analy = self.get_pdf(self.grid)
        else:
            distr_analy = self.get_cdf(self.grid)
        residual = DistributionFitter.compute_error(distr_analy, self.distr_data)
        #print("params = ", params, ", residual = ", residual)
        self.residual = residual
        return residual

    def get_params(self):
        return {"mu": self.mu,
                "sigma": self.sigma,
                "kappa": self.kappa,
                "theta": self.theta}

    def get_pdf(self, x):
        coeff_1 = np.sqrt(2) * np.exp(self.theta * (x - self.mu * self.horizon)/(2 * self.sigma ** 2))\
                /(self.sigma * np.sqrt(np.pi) * self.kappa ** (self.horizon/self.kappa) * scipy.special.gamma(self.horizon/self.kappa))
        coeff_2 = np.power(np.abs(x - self.mu * self.horizon)/np.sqrt(2*self.sigma ** 2/self.kappa + self.theta**2), self.horizon/self.kappa - 0.5)
        bessel_order = self.horizon/self.kappa - 0.5
        bessel_param = np.abs(x - self.mu * self.horizon) * np.sqrt(2 * self.sigma ** 2/self.kappa + self.theta ** 2)/self.sigma**2
        bessel = scipy.special.kv(bessel_order, bessel_param)
        res = coeff_1 * coeff_2 * bessel
        return res

    def get_cdf(self, x):
        #compute cdf using composite simpson
        x_pos = [xi for xi in x if xi >= self.mu]
        x_neg = [xi for xi in x if xi < self.mu]

        cdf_neg = []
        cdf_pos = []
        if len(x_pos) > 0:
            x_pos = 2 * self.mu * self.horizon - np.array(x_pos[::-1])
            lower = np.array(x_pos[:-1])
            upper = np.array(x_pos[1:])
            mid = (upper + lower) * 0.5
            h = (upper - lower) * 0.5
            int_simpson = h * (self.get_pdf(lower) + 4 * self.get_pdf(mid) + self.get_pdf(upper))/3.0
            int_cum = np.cumsum(int_simpson)
            cdf_pos = 1 - np.concatenate([[0], int_cum])[::-1]

        if len(x_neg) > 0:
            lower = np.array(x_neg[:-1])
            upper = np.array(x_neg[1:])
            mid = (upper + lower) * 0.5
            h = (upper - lower) * 0.5
            int_simpson = h * (self.get_pdf(lower) + 4 * self.get_pdf(mid) + self.get_pdf(upper))/3.0
            int_cum = np.cumsum(int_simpson)
            cdf_neg = np.concatenate([[0], int_cum])

        return np.concatenate([cdf_neg, cdf_pos])

def plot_vg_cdf():
    vg_fitter = VgFitter(50, 1, True)
    vg_fitter.mu = 0
    vg_fitter.sigma = 0.65355449
    vg_fitter.theta = 0.000001
    vg_fitter.kappa = 0.05
    x = np.linspace(-5, 5, 50)
    pdf = vg_fitter.get_pdf(x)
    cdf = vg_fitter.get_cdf(x)
    plt.subplot(1,2,1)
    plt.plot(x, pdf)
    plt.subplot(1,2,2)
    plt.plot(x, cdf)
    plt.show()


def fit_to_pdf_comparison():
    mu = 0
    std = 2.5
    grid_size = 100
    grid = np.linspace(-10, 10, grid_size)
    target_pdf = scipy.stats.norm.pdf(grid, mu, std)
    horizon = 1
    is_pdf = True

    plt.plot(grid, target_pdf, 'ob', label='target')
    fitters = [NormalFitter(horizon, grid_size, is_pdf), StFitter(grid_size, is_pdf), VgFitter(grid_size, horizon, is_pdf)]
    for fitter in fitters:
        fitter.fit_to_pdf(grid, target_pdf)
        print(fitter.get_label(), " residual = ", fitter.residual)
        print(fitter.get_params())
        plt.plot(grid, fitter.get_pdf(grid), label="%s"%fitter.get_label())
    plt.legend()
    plt.show()

def plot_pdf_fit_comparison():
    data = scipy.stats.norm.rvs(loc=0, scale=np.sqrt(5), size=100000)
    grid_size = 50
    fit_to_pdf = False
    pdf, pdf_grid = DistributionFitter.hist_pdf(data, grid_size)
    cdf, cdf_grid = DistributionFitter.hist_cdf(data, grid_size)
    T = 1

    # Norm fitter
    norm_fitter = NormalFitter(T, grid_size, fit_to_pdf)
    norm_fitter.fit(data)
    print("Fit residual for N = ", norm_fitter.residual)

    # VG fitter
    vg_fitter = VgFitter(grid_size, T, fit_to_pdf)
    vg_fitter.fit(data)
    print("Fit residual for VG = ", vg_fitter.residual)

    #ST fitter
    st_fitter = StFitter(grid_size, fit_to_pdf)
    st_fitter.fit(data)
    print("Fit residual for ST = ", st_fitter.residual)

    #plot_pdf
    plt.subplot(1,2,1)
    plt.plot(pdf_grid, pdf, 'o')
    plt.plot(pdf_grid, norm_fitter.get_pdf(pdf_grid), 'b-', label='N')
    plt.plot(pdf_grid, vg_fitter.get_pdf(pdf_grid), '--g', label='VG')
    plt.plot(pdf_grid, st_fitter.get_pdf(pdf_grid), '--r', label='ST')
    plt.ylim([1e-3, 1])
    plt.yscale('log')
    plt.legend()

    #plot cdf
    plt.subplot(1,2,2)
    plt.plot(cdf_grid, cdf, 'o')
    plt.plot(cdf_grid, norm_fitter.get_cdf(cdf_grid), 'b-', label='N')
    plt.plot(cdf_grid, vg_fitter.get_cdf(cdf_grid), '--g', label='VG')
    plt.plot(cdf_grid, st_fitter.get_cdf(cdf_grid), '--r', label='ST')
    plt.ylim([1e-3, 1])
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_pdf_cdf_data():
    data = scipy.stats.norm.rvs(loc=0, scale=np.sqrt(5), size=100000)
    grid_size = 50
    pdf, pdf_grid = DistributionFitter.hist_pdf(data, grid_size)
    cdf, cdf_grid = DistributionFitter.hist_cdf(data, grid_size)
    plt.subplot(1,2,1)
    plt.plot(pdf_grid, pdf, 'o-')
    #plt.yscale('log')
    plt.subplot(1,2,2)
    plt.plot(cdf_grid, cdf, 'o-')
    #plt.yscale('log')
    plt.show()

# test functions
if __name__ == "__main__":
    #plot_pdf_cdf_data()
    #plot_vg_cdf()
    #plot_student_t_cdf()
    #plot_pdf_fit_comparison()
    fit_to_pdf_comparison()