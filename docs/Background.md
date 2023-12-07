---
layout: page
title: "Background"
permalink: /background
---

# Sensitivity Analysis using the Morris Method

Morris method is a reliable and efficient sensitivity analysis method that defines the sensitivity of a model input as the ratio of the change in an output variable to the change in an input feature. Given a model $f(.)$, and a set of $k$ input features $X (x_1, \dots, x_{k})$, the Morris sensitivity \cite{morris1991factorial} of a model input feature $x_{i}$ can be defined as follows:

\begin{equation}
\label{eqn:morris}
    Sensitivity(\boldsymbol{X}, i)=\frac{f(x_{1},\dots,x_{i}+\Delta,\dots,x_{k}) - f(\mathbf{X})}{\Delta}
\end{equation}

where $\Delta$ is the small change to the input feature $x_i$. Since the original Morris method was for static variables, we expand it for our predictions for the high dimensional spatial and temporal datasets. 

\textbf{Algorithm \ref{alg:norm_morris}} shows the implementation of the modified Morris method for our study, where we normalized the output value change by the number of input days, counties, and delta $\Delta$. Which we call the normalized Morris index $\hat{\mu*}$. We further scaled this index using the standard deviation ($\sigma$) of the input feature $x_{i}$, which we call the scaled Morris index ($\hat{\mu*} \times \sigma$). This scaling takes the underlying distribution of the feature $x_{i}$ when ranking the features by sensitivity. In the rest of the study, by Morris score, we refer to the scaled Morris index.

[ Related Works ]({{ site.baseurl }}{% link Related_Works.md %})