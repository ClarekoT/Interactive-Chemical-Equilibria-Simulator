# Interactive-Chemical-Equilibria-Simulator
*A Python-based toolkit for simulating and visualizing complex chemical equilibria, featuring interactive perturbations and real-time kinetic analysis.*

## Overview
This project is a Python-based dashboard built in a Jupyter Notebook environment to simulate and analyze the kinetics of reversible chemical reactions. It moves beyond static textbook examples by providing a fully interactive interface where users can set initial conditions, apply real-time perturbations, and visualize the system's dynamic response as it approaches a new equilibrium.

The core of the simulation is a numerical ODE solver (`scipy.integrate.solve_ivp`) coupled with a modular, class-based architecture that models the chemical system. 

## Key Architectural Features
The simulation is built around a professional Object-Oriented Programming (OOP) model to ensure robustness, scalability, and clarity.

-   **Encapsulation (`ChemicalSystem` Class):** the fundamental state of a reaction (moles, volume, rate constants) and the physical laws that govern it (the ODE system) are bundled into a single, self-contained `ChemicalSystem` object. This eliminates fragile global variables and ensures data integrity.
-   **Composition (`PerturbationSimulation` Class):** complex, multi-stage workflows are managed by an orchestrator class. A `PerturbationSimulation` object takes a baseline `ChemicalSystem` and manages the three-stage process of applying a stress and simulating the relaxation to a new equilibrium. This is a powerful design pattern that separates high-level logic from the core scientific model.

## Scientific Visualisation Features
- **Interactive Controls (ipywidgets):** set initial concentrations and system volume with sliders.
- **Dynamic Perturbation Engine:** apply stresses to a system at equilibrium, including:
    -   Gradual volume changes (pressure stress).
    -   Continuous injection/removal of chemical species (concentration stress).
-   **Analytical Plots (matplotlib):**
    -   Real-time plots of species concentrations.
    -   A secondary Y-axis to compare the **Reaction Quotient (Qc)** against the **Equilibrium Constant (Kc)**.
    -   Plots of forward, reverse, and net reaction rates to illustrate the principles of **dynamic equilibrium**.

## Case Study 1: $N_2O_4 ⇌ 2NO_2$
The initial implementation of this simulator focuses on the classic gas-phase equilibrium between dinitrogen tetroxide ($N_2O_4$) and nitrogen dioxide ($NO_2$). This system serves as a good case study of Le Chatelier's principle.

*(Link to the model notebook: [models/n2o4_model/N2O4_Equilibrium_Model.ipynb](models/n2o4_model/N2O4_Equilibrium_Model.ipynb))*

| Interactive Controls & Perturbations | Analytical Plot | Comparative Data Table |
| :---: | :---: | :---: |
| ![Controls](models/n2o4_model/N2O4_controls.png) | ![Plot](models/n2o4_model/N2O4_plot.png) | ![Table](models/n2o4_model/N2O4_table.png) |
| Users can set initial conditions and apply gradual volume or injection perturbations over a defined time window. | The main plot visualizes concentrations, forward/reverse rates, and the Qc/Kc relationship on a dual-axis. | A table provides key quantitative metrics, comparing the perturbed system against the original baseline run. |

## Project Roadmap
The $N_2O_4$ model serves as the foundation for a more comprehensive toolkit. Future planned enhancements include:
- **Temperature Dependence:** implementing the Arrhenius equation to model the effect of temperature on rate constants and the equilibrium position, using the Haber Process ($N_2(g) + 3H_2(g) ⇌ 2NH_3$(g)) as the case study.
- **Multi-Step Reaction Mechanisms:** expanding the ODE solver to handle sequential reactions, such as the oxidation of nitric oxide.
- **Coupled Aqueous Equilibria:** applying the toolkit to model the complex, multi-reaction system that governs ocean acidification.
