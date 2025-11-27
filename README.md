---
editor_options: 
  markdown: 
    wrap: 72
---

# Course Projects: Simulation, PDEs, Causal Inference, and LLMs

This repository collects several projects from my coursework in
mathematics and statistics at the University of Michigan, Ann Arbor.

------------------------------------------------------------------------

## 1. Monte Carlo Simulation of the St. Petersburg Game

This is a coding project for STATS 506, where I implement simulations of
St. Petersburg game and document the functions into a custom R package.
The St. Petersburg game is a classical example where the payoff $$
X = 2^K,\quad \mathbb{P}(X = 2^k) = 2^{-k},\ k \ge 1
$$ has infinite expectation, yet empirical averages behave very
irregularly. Monte Carlo simulation of repeated plays of the game enable
us to visualize and study the convergence behavior of sample mean and
scaled averages $$
A_n = \frac{1}{n \log_2 n} \sum_{i=1}^n X_i.
$$ We also discussed the link of this problem to robust estimation.

![Mean v.s. Robust Estimators for the St. Petersburg
game](images/stp.png)

------------------------------------------------------------------------

## 2. PDE Simulation with PINNs and Classical Numerical Methods

This project is related to my coursework in MATH 440, MATH 656, and MATH
651. It focuses on time-dependent PDEs (e.g. the 1D Saint-Venant /
shallow-water system) and compares:

-   **Physics-Informed Neural Networks (PINNs)** that approximate the
    solution by minimizing PDE residuals via automatic differentiation.

| ![](images/SV_h1.gif) | ![](images/SV_h2.gif) |
|:------------------------:|:-------------------------:|

| ![](images/pinn_sol1.png) | ![](images/pinn_sol2.png) |
|:------------------------:|:-------------------------:|

*Figure 1. Solutions of 1D Saint_Venant equations with PINNs.*

-   **Classical numerical methods** such as finite-volume schemes that
    provide a high-resolution reference solution. 

| ![](images/pde_fv.gif) | ![](images/pde_fd.gif) |
|:------------------------:|:-------------------------:|

*Figure 2. Solutions from numerical simulations.*


The implementation of PINNs is based on the source code of
<https://github.com/maziarraissi/PINNs>.

------------------------------------------------------------------------

## 3. Synthetic Difference-in-Differences (SDID) for Causal Inference

This is the final project for STATS 700, which is a literature review of
the method **Synthetic Difference-in-Differences (SDID)**. Here, I
implement SDID for panel data with a treatment introduced at a known
time. ![SDID: trends in consumption over time for California and
weighted average of control states](images/SDID.png)

The code templates and examples are from
<https://synth-inference.github.io/synthdid/index.html>.

------------------------------------------------------------------------

## 4. The Cognitive Tests for LLMs (Work in Progress)

This ongoing project plans to design and build a small demo to compare
multiple benchmark LLMs on semi-independent cognitive tasks.

------------------------------------------------------------------------

## Requirements

-   **R / Quarto**
-   **Python**
