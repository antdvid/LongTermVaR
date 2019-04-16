# this file implement the long-term VaR calculation based on the research paper
# Luca Spadafora, Marco Dubrovich and Marcello Terraneo, Value-at-Risk time scaling for long-term risk estimation, 2013
import numpy as np
import matplotlib.pyplot as plt
from GetObservationFromDb import *
from PdfFitter import *
import scipy.optimize
from scipy import signal


class LucaVaRCalculator:
    def __init__(self, horizon, confidence_level):
        self.horizonInDays = horizon
        self.confidence_level = confidence_level

    def compute_var(self):
        axid = '1692' #axiomadataid for ibm stock

        self.print_splitter()
        print("Getting data from database")
        dates, daily_data = get_data_from_dataid(axid)

        self.print_splitter()
        print("Computing return from daily data")
        # window is how many days in the period to compute a return
        num_days_to_compute_return = 1
        return_data = self.compute_return_data_for_horizon(num_days_to_compute_return, daily_data)

        self.print_splitter()
        print("Using return data to fit the distribution")
        grid_size = 50
        best_fitter = self.fit_to_distribution(return_data, grid_size, num_days_to_compute_return)

        self.print_splitter()
        print("Best fitter is: ", best_fitter.get_label())
        print("Computing convolution using FFT, Iteration %d times"%self.horizonInDays)
        pdf_grid = np.linspace(-10, 10, 100000)
        pdf_in_horizon = self.compute_scaling(self.horizonInDays, best_fitter, pdf_grid)

        self.print_splitter()
        print("Computing VaR using computed PDF in horizon")
        var_1d = self.find_var_from_pdf(best_fitter.grid, best_fitter.get_pdf(best_fitter.grid), best_fitter)
        var_nd = self.find_var_from_pdf(pdf_grid, pdf_in_horizon, best_fitter)
        print("VaR = %f, for confidence level = %f, horizon = %dd" % (var_1d, self.confidence_level, 1))
        print("VaR = %f, for confidence level = %f, horizon = %dd" % (var_nd, self.confidence_level, self.horizonInDays))
        return var_nd

    def find_var_from_pdf(self, grid, pdf_in_horizon, best_fitter):
        residual = best_fitter.fit_to_pdf(grid, pdf_in_horizon)
        print("Fit pdf, residual = %e" % residual)
        target = lambda x: best_fitter.get_cdf(x) - (1 - self.confidence_level)
        target_ret = scipy.optimize.fsolve(target, x0=np.array([0]))[0]
        return np.abs(target_ret)

    def compute_scaling(self, horizon, fitter, grid):
        """compute scaling by computing the convolution many times"""
        total_num = horizon
        delta = (grid[-1] - grid[0])/(len(grid)-1)
        pmf_nd = fitter.get_pdf(grid) * delta
        pmf_1d = fitter.get_pdf(grid) * delta
        ax = plt.gca()
        ax.plot(grid, pmf_nd/delta, label='1d')
        for i in range(total_num):
            pmf_nd = scipy.signal.fftconvolve(pmf_1d, pmf_nd, 'same')

        pdf_nd = pmf_nd/delta
        ax.plot(grid/np.sqrt(total_num), pdf_nd*np.sqrt(total_num), label='%dd' %total_num)
        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('pdf')
        ax.set_xlabel('x/sqrt(n)')
        ax.set_xlim([grid[0]/np.sqrt(total_num), grid[-1]/np.sqrt(total_num)])
        plt.show()
        plt.savefig('figure/scaling.png')
        return pdf_nd

    def fit_to_distribution(self, return_data, grid_size, horizon_in_days):
        data = return_data
        T = horizon_in_days
        fit_to_pdf = False
        # Norm fitter
        norm_fitter = NormalFitter(T, grid_size, fit_to_pdf)
        norm_fitter.fit(data)
        print("Fit residual for N = ", norm_fitter.residual)
        print(norm_fitter.get_params())

        # VG fitter
        vg_fitter = VgFitter(grid_size, T, fit_to_pdf)
        vg_fitter.fit(data)
        print("Fit residual for VG = ", vg_fitter.residual)
        print(vg_fitter.get_params())

        # ST fitter
        st_fitter = StFitter(grid_size, fit_to_pdf)
        st_fitter.fit(data)
        print("Fit residual for ST = ", st_fitter.residual)
        print(st_fitter.get_params())

        self.plot_comparison(return_data, grid_size, [norm_fitter, vg_fitter, st_fitter])

        best_fitter = min([norm_fitter, vg_fitter, st_fitter], key=lambda item: item.residual)
        return best_fitter

    def print_splitter(self):
        print("*****************************")

    def plot_comparison(self, data, grid_size, fitters):
        pdf, pdf_grid = DistributionFitter.hist_pdf(data, grid_size)
        cdf, cdf_grid = DistributionFitter.hist_cdf(data, grid_size)
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(pdf_grid, pdf, 'o')
        ax2.plot(cdf_grid, cdf, 'o')

        for fitter in fitters:
            # plot_pdf
            ax1.plot(pdf_grid, fitter.get_pdf(pdf_grid), label=fitter.get_label())
            ax1.set_yscale('log')
            ax1.legend()

            # plot cdf
            ax2.plot(cdf_grid, fitter.get_cdf(cdf_grid), label=fitter.get_label())
            ax2.set_yscale('log')
            ax2.legend()
        ax1.set_title('pdf')
        ax2.set_title('cdf')
        ax1.set_ylim([1e-2, 100])
        ax2.set_ylim([1e-3, 1])
        #plt.show()
        plt.savefig('figure/fit_distr.png')
        plt.close()

    @staticmethod
    def compute_return_data_for_horizon(horizon_in_days, daily_data):
        data_length = len(daily_data)
        window = horizon_in_days + 1
        rets = []
        for i in range(0, data_length - window + 1):
            one_ret = LucaVaRCalculator.calculate_return(daily_data[i], daily_data[i + window - 1])
            rets.append(one_ret)
        return rets

    @staticmethod
    def calculate_return(p_start, p_end):
        return (p_end - p_start)/p_start

def test_convolution():
    grid_size = 200
    fit_to_pdf = True
    # student-T distribution
    fitter = StFitter(grid_size, fit_to_pdf)
    fitter.set_params_from_array([0, np.sqrt(5/3), 3])
    # VG distribution
    #fitter = VgFitter(grid_size, 1, True)
    #fitter.set_params_from_array([0, np.sqrt(5), 0.5, 0])

    x = np.linspace(-1000, 1000, 100000)
    pdf = fitter.get_pdf(x)
    delta = x[1] - x[0]
    last_pmf = pdf * delta
    one_pmf = pdf * delta
    plt.plot(x, last_pmf/delta, label="ST, n = 0")
    for i in range(1, 251):
        last_pmf = scipy.signal.fftconvolve(one_pmf, last_pmf, 'same')
        last_pdf = last_pmf/delta
        if i == 1 or i == 20 or i==250:
            plt.plot(x/np.sqrt(i), last_pdf, label="ST, n = %s"%(i))
    plt.xlim([-10, 10])
    plt.ylim([1e-3, 1])
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    var_cal = LucaVaRCalculator(255, 0.999)
    var = var_cal.compute_var()
    print("VaR = ", var)
    plt.ion()
    plt.show()
    #test_convolution()