# This script illustrates the central limit theorem:
# sqrt(N)(X1 + X2 + ... + XN)/N -> Norm(mu, sigma^2)
# Xi are i.i.d with mean mu, variance sigma^2
import numpy as np
import scipy as scp
import scipy.stats
import matplotlib.pyplot as plt
import scipy.signal


def GetPdfFromDiscrete(numVariables, lower, upper, numSamples):
    sums = []
    gridSize = 40
    for i in range(numSamples):
        vars = scp.random.uniform(lower, upper, numVariables)
        sums.append(sum(vars))
    sums = np.array(sums)
    if DivideSqrtN:
        sums = sums/np.sqrt(numVariables)
    hist, binEdges = np.histogram(sums, gridSize, density=True)
    return (binEdges[1:] + binEdges[0:-1]) * 0.5, hist


def GetPdfFromConvolution(numVariables, lower, upper):
    lowerBd = 5.0 * lower
    upperBd = 5.0 * upper
    gridSize = 10000
    grid = np.linspace(lowerBd, upperBd, gridSize)
    delta = grid[1] - grid[0]
    pdf = scp.stats.uniform.pdf(grid, lower, upper - lower)
    pmf = pdf * delta
    lastPdf = pdf
    lastPmf = lastPdf * delta

    for i in range(numVariables-1):
        lastPmf = scipy.signal.fftconvolve(lastPmf, pmf, mode='same')
    if DivideSqrtN:
        lastPdf = lastPmf / delta * np.sqrt(numVariables)
        return grid/np.sqrt(numVariables), lastPdf
    else:
        lastPdf = lastPmf / delta
        return grid, lastPdf


lower = -1
upper = 1
f, axes = plt.subplots(1,2)
ax1 = axes[0]
ax2 = axes[1]
ax1.set_title('Samples')
ax2.set_title('Convolution')
DivideSqrtN = True
for i in range(1, 5, 1):
    gridDisc, pdfDisc = GetPdfFromDiscrete(i, lower, upper, 10000)
    gridCon, pdfCon = GetPdfFromConvolution(i, lower, upper)
    ax1.plot(gridDisc, pdfDisc, 'o', label="n = %s"%i)
    ax2.plot(gridCon, pdfCon, '-', label="n = %s"%i)
plt.legend()
plt.show()