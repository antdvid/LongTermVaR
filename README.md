#Introduction

This python code is to replicate the results of the paper "L. Spadafora, M. Dubrovich and M. Terraneo, Value-at-Risk time scaling for long-term risk estimation, 2013" The main idea of this algorithm is to estimate the long-term (more than one year) value at risk by computing the convolution of the probability density function (p.d.f.) of one-day return, which is estimated and chosen by performing fittings to three distributions: normal, student-T and variance-gamma. This method uses the principle that the p.d.f. of two days return equals to the convolution of the p.d.fs of the one-day return on each day. This method also assumes that the distributions of the one-day return in the risk horizon are identical, which may be problematic. For people interested on a more descent long-term risk estimation method, please refer to risk metrics 2006 methodolgy.

#Code 
PdfFitter.py: the class try to compute the fitting parameters of the target distribution and also produces the fitting error
LucaSpadafora2013.py: the main class for the algorithm, implementing the methodology from the paper.
