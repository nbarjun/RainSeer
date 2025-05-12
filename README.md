# RainSeer

This Python package implements Bayesian Gaussian Process Regression (GPR) models for predicting long-term changes in mean precipitation and their associated uncertainties at a global scale. It integrates multiple Global Climate Models (GCMs) with observational data, aiming for more accurate future rainfall anomaly predictions.

## Overview

Bayesian methods offer a significant advantage by allowing continuous updates to probabilistic predictions as new observational data become available. This package leverages this advantage by combining a Bayesian framework with Gaussian Process Regression (GPR) to generate accurate precipitation projections while quantifying uncertainty. This quantified uncertainty can be used to predict the Time of Confidence (ToC), which will be the time at which we can be confident of a significant change in climate.

The Bayesian approach aligns with modern climate adaptation frameworks that emphasize flexibility and robustness, advocating for probabilistic and adaptive decision-making strategies. The ability to continuously update projections ensures that adaptation strategies remain responsive to scientific advancements and emerging climate signals. This framework can potentially reduce the cost of adaptation while effectively mitigating climate change impacts.

This package provides implementations for:

* Single Input Single Output Gaussian Process Regression, adapted for time series analysis.
* Multiple Input Single Output Gaussian Process Regression, to account for spatial and temporal dependencies.
* Multitask Gaussian Process Regression, to potentially model relationships between different spatial locations or climate variables.

## Key Features

* **Bayesian Framework**: Enables continuous updating of probabilistic predictions with new data.
* **Integration of Multiple GCMs**: Combines information from various climate models to improve prediction accuracy.
* **Quantified Uncertainty**: Provides robust estimates of prediction uncertainty, crucial for risk assessment and adaptive planning.
* **Time of Confidence (ToC) Prediction**: Allows for the estimation of when a predicted climate signal will emerge with a certain level of confidence (as per \citet{Lickley_2024}).
* **Flexible Kernel Functions**: Supports various kernel functions and their combinations to capture complex spatio-temporal relationships.
* **Hyperparameter Optimization**: Optimizes kernel hyperparameters by maximizing the log marginal likelihood of the training data.
* **Prior Knowledge Incorporation**: The kernel definition allows for the integration of prior knowledge about the expected behavior of precipitation patterns.
* **Iterative Validation**: Implements an iterative testing approach, excluding one GCM at a time to validate the framework against independent benchmarks.

## Implementation Details

### Single Input Single Output GPR

* Predicts precipitation anomaly at a specific location based on year.
* Each grid point can have an independent GPR model initially.
* Uses a leave-one-out strategy across GCMs for validation.

### Multiple Input Single Output GPR

* Predicts precipitation anomaly at a location based on multiple inputs such as year, longitude, and latitude, accounting for spatio-temporal covariance.

### Multitask GPR

* Can be extended to predict precipitation anomalies at multiple locations or related climate variables simultaneously, considering potential covariance between them.

### Core Equations

The package implements the fundamental equations of Gaussian Process Regression within a Bayesian framework, including:

* Calculation of the prior mean and covariance based on the ensemble of GCMs (excluding the validation GCM).
* Definition of various kernel functions (e.g., Squared Exponential, Linear, Matern) suitable for spatio-temporal data.
* Computation of the posterior mean and predictive uncertainty for future precipitation anomalies.
* Maximization of the log marginal likelihood to optimize the hyperparameters of the chosen kernel.

