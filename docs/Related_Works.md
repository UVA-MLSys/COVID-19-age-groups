---
layout: page
title: "Related Works"
permalink: /related_works
---

This section describes the related works on time series interpretation and also the interpretation works done on COVID-19 infection prediction. 

\subsection{Time Series Interpretation}
A wide range of interpretation methods has been proposed in the literature \citep{rojat2021explainable, turbe2023evaluation}. Including \textit{gradient based methods} such as Integrated Gradients \citep{Sundararajan2017AxiomaticAF}, GradientSHAP \citep{Erion2019LearningEM} which uses the gradient of the model predictions to input features to generate importance scores. \textit{Feature removal based} methods such as Feature Occlusion \citep{Zeiler2013VisualizingAU}, Feature Ablation \citep{Suresh2017ClinicalIP}, and Sensitivity Analysis \citep{morris1991factorial} replace a feature or a set of features from the input using some fixed baselines or generated samples and measure the importance based on the model output change. These methods have been popularly used in time series datasets \cite{ozyegen2022evaluation, zhao2023interpretation, turbe2023evaluation}. 

% \textit{Model based saliency} methods such as \citep{Choi2016RETAINAI, Song2017AttendAD, Xu2018RAIMRA, Kaji2019AnAB, lim2021temporal} use the model architecture e.g. attention layers, to generate importance scores.

\subsection{Interpreting COVID-19 Infection}
DeepCOVID \cite{rodriguez2021deepcovid} utilized RNN with auto-regressive inputs to predict COVID-19 cases. Then recursively eliminating input signals to quantify the model output deviation without those signals and use that to rank the signal importance. Unlike local interpretation methods, this can't show the change of feature importance within the input window and needs to retrain the model. DeepCOVIDNet \cite{ramchandani2020deepcovidnet} used a classification approach to predict regions with high, medium, and low COVID-19 case growth. The interpreted feature importance using Feature Occlusion on part of the training data. 

COVID-EENet \cite{kim2022covid} interpreted the economic impact of COVID-19 on local businesses. Self-Adaptive Forecasting \cite{arik2022self} used the model's attention weights to interpret state-level COVID-19 death forecasts. However, this is model-dependent and can't be applied to models without the attention mechanism. 

[ Problem Statement ]({{ site.baseurl }}{% link Problem_Statement.md %})