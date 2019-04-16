import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from LucaSpadafora2013 import *
from PdfFitter import *

dates, daily_data = LucaVaRCalculator.get_data_from_dataid('1692')
print("Computing return from daily data")
# window is how many days in the period to compute a return
return_data = LucaVaRCalculator.compute_return_data_for_horizon(2, daily_data)

grid_size = 50
pdf, pdf_grid = DistributionFitter.hist_pdf(return_data, grid_size)
grid = pdf_grid
plt.plot(pdf_grid, pdf, 'ob', label = 'Data')

normfitter = NormalFitter(1, grid_size, True)
normfitter.mean = 0.00022257813788181352
normfitter.var = 0.00023115706052162909
plt.plot(grid, normfitter.get_pdf(grid), 'b-', label='N')
print('normfitter_residual = ', normfitter.residual, ", real residual = ", DistributionFitter.compute_error(normfitter.get_pdf(grid), pdf))

vgfitter = VgFitter(grid_size, 1, True)
vgfitter.set_params_from_array(
    [-0.0006488627692761936, 0.012302232518493886, 0.5140112237145882, 0.0028449547490870595])
plt.plot(grid, vgfitter.get_pdf(grid), '--g', label='VG')
print(vgfitter.get_params())
print('vg_residual = ', normfitter.residual, ", real residual = ", DistributionFitter.compute_error(vgfitter.get_pdf(grid), pdf))

stfitter = StFitter(grid_size, True)
stfitter.set_params_from_array([0.00022956084052690114, 0.009316396474593376, 2.831613380693119])
plt.plot(grid, stfitter.get_pdf(grid), 'r--', label='ST')
print('st_residual = ', normfitter.residual, ", real residual = ", DistributionFitter.compute_error(stfitter.get_pdf(grid), pdf))

#plt.plot(grid, scipy.stats.norm.pdf(grid, normfitter.mean, np.sqrt(normfitter.var)), 'b', label='N-sci')

# plt.ylim([1e-3, 1])
# plt.yscale('log')
plt.legend()
plt.show()