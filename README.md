# Decision-Focused Learning for Transport Network Design

_MSc thesis in computer science at [UdeM](https://www.umontreal.ca) and [EPFL](https://www.epfl.ch)_

Supervised by:
* [Prof. Emma Frejinger](https://www.emmafrejinger.org) - University of Montreal
* [Prof. Michel Bierlaire](https://people.epfl.ch/michel.bierlaire) - EPFL

## Description

Solving operations research problems in transportation presents many challenges. Linear programming models are often large, with many integer variables. Model parameters such as freight demand are subject to uncertainty and must be estimated from noisy history data. This is usually performed separately from the optimization by machine learning or time-series models. These prediction models are trained using a predictive accuracy loss function.

A recent approach to handling uncertainty is Decision-Focused Learning. It incorporates the optimization problem into the predictive model loss function, ensuring the predictions are aligned with the objective of making good decisions.

The goal of this Master's project is to explore the use of DFL to the Periodic Freight Demand Estimation problem for Service Network Design. This problem involves predicting a periodic demand from external time-series forecasts. This periodic demand is used to plan a freight transportation network. Using DFL to predict periodic demand could lead to more effective and less costly network designs. This approach could be applied to other transportation problems, improving solution quality for a large class of operations research problems.

