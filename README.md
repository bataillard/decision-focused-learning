# Decision-Focused Learning for Transport Network Design

_MSc thesis in computer science at [UdeM](https://www.umontreal.ca) and [EPFL](https://www.epfl.ch)_

Supervised by:
* [Prof. Emma Frejinger](https://www.emmafrejinger.org) - University of Montreal
* [Prof. Michel Bierlaire](https://people.epfl.ch/michel.bierlaire) - EPFL

## Description

Solving operations research problems in transportation presents many challenges. Network Design (ND) models are complex combinatorial optimization problems used to plan transportation networks. The parameters of ND problems, such as commodity demand, are subject to uncertainty and must be estimated from noisy historical data. This is typically done separately from the optimization by machine learning or time-series models. These prediction models are trained using a prediction accuracy loss function. A recent approach to handling uncertainty in optimization problems is Decision-Focused Learning (DFL). It incorporates the optimization problem into the learning algorithm, ensuring that the predictions are aligned with the goal of making good decisions.

This master's thesis is an exploratory study of DFL for ND problems where the commodity demands are uncertain. Most research focuses on DFL for problems with continuous variables and uncertain objective costs. However, ND problems are combinatorial and the uncertain demand parameters are in the constraints. Our aim is to integrate the prediction of demands and the optimization of an ND problem so that the prediction results in high quality downstream decisions. Our objective is to conduct a review of the literature on ND, DFL, and Inverse Optimization (IO), identify existing approaches, and test their suitability for this problem.

In this study, we conduct a literature review of the fields of ND, DFL, and IO, and find that existing methods cannot be directly applied to the problem of DFL for ND. We formulate our problem as a stochastic optimization problem, and show how to evaluate the performance of a prediction model on the downstream cost of the ND problem. We show how the regret-based loss, the standard way of evaluating the downstream optimization cost, is mathematically ill-defined when the uncertainty is in the constraints. We formulate \textit{IO-constraint}, an IO model that that trains a linear prediction to predict demands but corresponds to Ordinary Linear Regression. Finally, we reframe DFL as a problem of appropriately weighting the training examples in the loss function, and sketch ideas for finding effective weights using an iterative weight update algorithm.

## Report
Read the full report of the MSc thesis [here](/report/LB-MScThesis-DFLforND.pdf)
