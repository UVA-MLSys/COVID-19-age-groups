---
layout: page
title: "Problem Statement"
permalink: /problem_statement
---

We consider a multivariate multi-horizon time series setting with length $T$, the number of input features $J$, and total $N$ instances. $X_{j,t} \in \mathbb{R}^{J \times T}$ is the input feature $j$ at time $t \in \{0, \cdots, T-1\}$. We use past information within a fixed look-back window $L$, to forecast for the next $\tau_{max}$ time steps. The target output at time $t$ is $y_t$. Hence our black-box model $f$ can be defined as $\hat{y}_{t} = f(X_t)$ where,

\begin{equation}
\begin{aligned}
    % \hat{y}_{t} & = f(X_t)  \\ \text{where, } 
 X_t &= x_{t-(L-1):t} \\
 &= [x_{t-(L-1)}, x_{t-(L-2)}, \cdots, x_t] \\
&= \{ x_{j, l, t}\}, ~ j \in \{1, \cdots, J\}, ~ l \in \{1, \cdots, L\}
\end{aligned}
\end{equation}

$\hat{y}_{t}$ is the forecast at $\tau \in \{1, \cdots, \tau_{max}\}$ time steps in the future.  $ X_t$ is the input slice at time $t$ of length $L$. 
An individual covariate at position $(n, l)$ in the full covariate matrix at time step $t$ is denoted as $x_{j, l, t}$. 

For interpretation, our target is to construct the importance matrix $\phi_t = \{ \phi_{j, l, t} \}$ for each output $o \in O$ and prediction horizon $\tau \in \{1, \cdots, \tau_{max}\}$. So this is a matrix of size $O \times \tau_{max} \times J \times L$. We find the relevance of the feature $x_{j, l, t}$ by masking it in the input matrix $X_t$ and output change from the model,
\begin{equation}
    \phi_{j, l, t} = | (f(X_t) - f(X_t ~\text{\textbackslash}~ x_{j, l, t})|
\end{equation}
where $X_t ~\text{\textbackslash}~ x_{j, l, t}$ is the feature matrix achieved after masking entry $x_{j, l, t}$.

