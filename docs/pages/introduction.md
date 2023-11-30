# Introduction
With the increased application of deep learning models in
different health, finance, social, and other vital domains
(Zhao et al. 2023) the interpretability of these methods is
becoming more important for better transparency and relia-
bility of the model’s decision (Amann et al. 2020).
These interpretability methods benefit us by showing dif-
ferent factors contributing to the model decisions (Rojat
et al. 2021), reveal incompleteness in the problem formaliza-
tion, and improve our scientific understanding (Doshi-Velez
and Kim 2017). Existing interpretability methods are either:
1) Global: Explains the entire behavior of the model 2) Lo-
cal: Explains the reasons behind a specific model decision
on an input instance.
Explaining time series models is challenging due to the
dynamic nature of the dataset. Most time series interpreta-
tion works have focused on interpreting classification tasks
or using baseline models. It is important to understand how
we can generalize these interpretation methods in state-of-
the-art time series models while still getting the best perfor-
mance.
We use the highly dynamic COVID-19 data in a multivari-
ate, multi-horizon, and multi-time series setting with state-
of-the-art time series transformer models. COVID-19 is a
recent pandemic taking millions of lives and causing many
research efforts to forecast the infection spread using statisti-
cal learning, epidemiological, and machine learning models
(Clement et al. 2021).
We focus on local interpretation methods to show the con-
tribution of each input feature to the prediction and give an
input sample. This allows us to give a more granular analy-
sis. We collect around three years of COVID-19 data daily
for 3,142 US counties. Each county contributes to one time
series in the dataset (hence multi-time series). We use the
last 14 days of data to predict the COVID-19 cases for the
next 14 days. The best-performing model on the test set is
later used for interpretation. However, our approach is black-
box and hence could be used for the other models too.
We benchmark our interpretation using eight recent meth-
ods. Then evaluate the interpreted attribution scores follow-
ing the latest practices (Ozyegen, Ilic, and Cevik 2022). We
also propose an innovative way to evaluate sensitivities of
age group features using real COVID-19 cases by age group.
The rest of the sections are organized as follows: Section
2 describes the related works. Section 4 defines the prob-
lem statement of both forecasting and interpretation tasks.
Section 5 describes the data collection and pre-processing
steps. Section 6 summarizes the experimental setup, training
steps, and test results. Section 7 lists the interpretation meth-
ods used, how we evaluated their performance and visualiza-
tions. Section 8 discusses our findings from the experiments.
Finally in Section 9 we have the concluding remarks.