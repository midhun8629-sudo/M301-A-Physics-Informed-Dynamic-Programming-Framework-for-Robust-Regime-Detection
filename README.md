# Abstract
This study evaluates two regression paradigms—Global Ordinary Least Squares (OLS) and FixedInterval Segmented Least Squares (SLS) for characterizing heterogeneous time-series regimes. While
Global OLS provides a baseline linear approximation, it fails to capture local variances and structural breaks inherent in non-stationary physical systems. Similarly, SLS, though capable of piecewise
approximation, is limited by arbitrary interval selection, which frequently results in boundary discontinuities and suboptimal error minimization. To resolve these limitations, We propose an Auto
Fixed-Interval Segmented Least Squares (SLS) framework governed by a penalty term based on the
Bayesian Information Criterion (BIC) [1]. This hybrid approach automates the detection of the optimal number of segments (k), ensuring that the model complexity is mathematically justified by
the data. Furthermore, to identify the ideal segment break-points, we integrate Bellman’s Principle of Optimality, which recursively minimizes the cumulative error cost function to identify optimal
regime transitions [2]. The results demonstrate that the Bellman-optimized Dynamic-Segmented Least
Squares Method, constrained by BIC, offers a superior balance of parsimony and accuracy, providing
a robust computational tool for detecting subtle phase transitions and functional shifts in complex
material datasets [3].
