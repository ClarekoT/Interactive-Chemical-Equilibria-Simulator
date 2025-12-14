# Interactive-Chemical-Equilibria-Simulator
*A Python-based toolkit for simulating, visualising, and probing complex, multi-step chemical equilibria, featuring interactive perturbations and real-time kinetic analysis. The computational engine is generalised and object-oriented.*

## Overview
This project is a Python-based dashboard built in a Jupyter Notebook environment to simulate and analyze the kinetics of reversible chemical reactions. It moves beyond static textbook examples by providing a fully interactive interface where users can set initial conditions, apply real-time perturbations, and visualize the system's dynamic response as it approaches a new equilibrium.

The project is a simulation tool designed to model the dynamic evolution of chemical systems over time. Unlike standard equilibrium calculators which only provide the final state ($\Delta G=0$), this engine simulates the *journey*, visualising how reaction intermediates form, how steady states are established, and how systems respond to real-time perturbations in non-ideal conditions.

The engine utilizes an iterative wrapper around `scipy.integrate.solve_ivp`. This allows the simulation to intelligently "search" for equilibrium, automatically extending its simulation window for slow reactions (low T) or refining it for fast reactions (high T). This was first implemented in the second phase of the project (modelling the Haber process).

## Scientific Visualisation Features
- **Interactive Controls (`ipywidgets`):** set initial concentrations and system volume with sliders.
- **Dynamic Perturbation Engine:** apply stresses to a system at equilibrium, including:
    -   Gradual volume changes (pressure stress).
    -   Continuous injection/removal of chemical species (concentration stress).
- **Analytical Plots (`matplotlib`):**
    -   Real-time plots of species concentrations, rates of changes, and other variables such as pressure.
    -   A secondary Y-axis to compare the **Reaction Quotient (Qc)** against the **Equilibrium Constant (Kc)**, along with **Qp** and **Kp**.
    -   Plots of forward, reverse, and net reaction rates to illustrate the principles of **dynamic equilibrium**.
- **Professional UI:** maximum plot clarity, including a clean `ipywidgets` interface.

## Case Study 1: $N_2O_4 \rightleftharpoons 2NO_2$
*Status: Completed*

The initial implementation of this simulator focuses on the classic gas-phase equilibrium between dinitrogen tetroxide ($N_2O_4$) and nitrogen dioxide ($NO_2$). This reaction is a cornerstone of chemical education, famously used to demonstrate Le Chatelier's Principle, partly due to the distinct visual change as the colourless $N_2O_4$ gas dissociates into the brown $NO_2$ gas. This simulation provides a quantitative, dynamic exploration of the system and its behaviour under various stresses.

This model operates under *ideal gas assumptions* and *isothermal conditions*. It is designed to demonstrate the fundamental mechanical aspects of equilibrium shifting.

*(Link to the model notebook: [models/n2o4_model/N2O4_Equilibrium_Model.ipynb](models/n2o4_model/N2O4_Equilibrium_Model.ipynb))*

### Dashboard Features
| Interactive Controls & Perturbations | Analytical Plot | Comparative Data Table |
| :---: | :---: | :---: |
| ![Controls](models/n2o4_model/N2O4_controls.png) | ![Plot](models/n2o4_model/N2O4_plot.png) | ![Table](models/n2o4_model/N2O4_table.png) |
| Users can set initial conditions and apply gradual volume or injection perturbations over a defined time window. | The main plot visualizes concentrations, forward/reverse rates, and the Qc/Kc relationship on a dual-axis. | A table provides key quantitative metrics, comparing the perturbed system against the original baseline run. |

### Key Assumptions & Limitations for This Model
1.  **Isothermal Conditions:** the model assumes the system is held at a constant temperature (323 K). This is a significant simplification, as the forward reaction (dissociation) is endothermic. In a real, isolated system, a shift in equilibrium would cause a temperature change, which would in turn alter the rate constants and the value of Kc.
2.  **Ideal Gas Behavior:** the calculation of total pressure uses the Ideal Gas Law (`PV = nRT`). This approximation becomes less accurate at the higher pressures that can be generated in the simulation, where intermolecular forces and molecular volume (as accounted for in the van der Waals equation) become non-negligible.
    Another important point with this assumption is that at high pressures, such as >1000 atm, the moleucles can become packed so tightly that there is a phase change. For example, the vapour pressure of $N_2O_4$ is relatively low. Compressing it to 1350 atm would inevitably force a phase change from gas to liquid.
4.  **Perfect & Instantaneous Mixing:** the model assumes that concentrations are uniform throughout the entire volume at all times and that any injected species are mixed instantaneously. This ignores the real-world kinetics of gas diffusion and convection.

## Case Study 2: $N_2(g) + 3H_2(g) \rightleftharpoons 2NH_3$(g), Haber Process
*Status: Completed*

This stage represents a significant leap in physical complexity. Modeling the industrial synthesis of ammonia, this simulation discards some of the approximations in the previous stage to model the messy reality of industrial chemistry.

*(Link to model: [models/haber_process_model/Haber_Process_Model.ipynb](models/haber_process/Haber_Process_Model.ipynb))*

### Key Upgrades
- **Van der Waals Equation of State:** the Ideal Gas Law is replaced by the van der Waals equation ($[P + a(n/V)^2][V - nb] = nRT$) to account for molecular volume and intermolecular forces.
- **Dynamic Mixing Rules:** the simulation calculates the effective van der Waals constants ($a_{mix}$ and $b_{mix}$) for the mixture at every time-step based on the changing mole fractions of the species.
- **Arrhenius Temperature Dependence:** rate constants are no longer fixed; they are dynamically calculated based on Activation Energy ($E_a$) and Temperature ($T$), according to the Arrhenius equation.
- **Solver Algorithm**: reaction rates at 400K vs 700K differ by orders of magnitude. A fixed-time solver would fail or waste resources. I implemented an *adaptive iterative solver* that intelligently detects equilibrium stability to auto-scale the simulation duration.
- **Kp**: calculated $K_p$ from partial pressures, along with plotting Qp and Kp over time, allowing for comparison with Qc and Kc values.
- **Visualisation**: implemented *logarithmic axes* for thermodynamics to show $K_c$ and $K_p$ simultaneously. Created a *dynamic detail plot* that geometrically zooms into perturbation events (e.g., showing a 10s event within a $10^5$s simulation).

### Dashboard Features
The interface is segmented to provide control, broad analysis, and fine-grained detail simultaneously.

| Component | Visual Interface | Scientific Function |
| :--- | :--- | :--- |
| **Control Panel** | ![Controls](models/haber_process_model/haber_controls.png) | **Experimental Design:** allows users to define the reactor conditions. By manipulating density ($N/V$) and Temperature, users can force the system into distinct thermodynamic regimes (such as a case where $P_{real} \gg P_{ideal}$). |
| **Macro-Analysis** | ![Main Plot](models/haber_process_model/haber_main_plot.png) | **System Overview:** <br>• **Top:** Concentrations vs Time.<br>• **Bottom (Left):** Log-scale Reaction Rates. Visualises the definition of dynamic equilibrium (Net Rate $\to$ 0).<br>• **Bottom (Right):** Log-scale Thermodynamics. Visualises the equilibrium constants. |
| **Micro-Analysis** | ![Detail Plot](models/haber_process_model/haber_perturb_detail.png) | **Transient Response:** automatically zooms into the perturbation window (e.g., a 10s injection). This reveals the immediate kinetic "pulse" and the system's restoration of $Q_c \to K_c$, which is often invisible on the macro-scale plot. |
| **Data Table** | ![Data Table](models/haber_process_model/haber_data_table.png) | **Quantitative Verification:** provides a side-by-side comparison of the Baseline vs. Perturbed run. Crucially, it quantifies the error of the Ideal Gas assumption by explicitly listing both **ideal pressure** and **real pressure** (Van der Waals). |

### Key Assumptions & Limitations for This Model
While this is a robust kinetic model, it makes certain assumptions and has some limitations:
1.  **Reaction Mechanism:** the simulation models the global stoichiometry ($N_2 + 3H_2 \rightarrow 2NH_3$) as a pseudo-elementary step. In reality, the Haber process is a complex *heterogeneous catalytic reaction* involving adsorption isotherms (Langmuir-Hinshelwood kinetics). Stage 3 will address multi-step mechanisms.
2.  **Thermodynamic Isolation:** the model treats the reactor as *isothermal* (constant T) unless manually perturbed. It does not model the *adiabatic* temperature rise caused by the reaction's exothermicity ($\Delta H = -92$ kJ/mol), which is a critical engineering constraint in real reactors.
3.  **Temperature independence of $\Delta H$:** the model treats the enthalpy of reaction as a constant -92 kJ/mol across all temperatures. Strictily speaking, $\Delta H$ varies with temperature.

Among other assumptions listed in the Jupyter notebook. Since this project is an investigation into physical chemistry rather than engineering, industrial mechanisms such as continuous flow recycling or heat exchangers are intentionally omitted. This isolation allows for a pure visualisation of the fundamental interplay between kinetics and thermodynamics in a controlled, closed-system environment, without the confounding variables of reactor design.

### Numerical Architecture and Verification
Simulating real-time kinetics across a 300K temperature range presents significant computational challenges. This project implements specific algorithms to solve the **"stiffness problem"** inherent in Arrhenius kinetics.

**The Stiffness Challenge**

Reaction rates scale exponentially with temperature ($k = Ae^{-E_a/RT}$).
*   At high temperatures, the reaction can reach equilibrium in seconds (incredibly fast).
*   At low temperatures, the reaction could take years.
A standard fixed-duration simulation would fail: it would either miss the equilibrium at 400K (stopping too early) or waste computational resources simulating static noise at 700K.

**Solution: The Adaptive Iterative Solver**

Instead of a fixed timeline, the `ChemicalSystem` class implements an **autonomous supervisor algorithm**.
1.  The simulation runs in dynamic time "chunks."
2.  After each chunk, the system state is analyzed.
3.  If equilibrium is not reached, the solver preserves the state vector (`y`) and extends the time horizon geometrically.
4.  This allows the dashboard to seamlessly handle widely different timescales without user intervention.

**Equilibrium Detection**

To prevent false positives during transient states (where the rate might momentarily cross zero), equilibrium is only declared when:
1.  **Kinetic stability:** net rate magnitude $\approx 0$.
2.  **Derivative stability:** the slope of the rate curve $\approx 0$ (ensuring the system isn't just passing through a turning point).
3.  **Thermodynamic consistency:** the observed reaction quotient ($Q_c$) matches the theoretical $K_c$ within a strict tolerance.

The physics engine has been validated against multiple test cases.

**Stress Testing**

By compressing a massive quantity of gas ($>17$ mol) into a microscopic volume ($<0.05$ dm$^3$), the simulation encounters a singularity where the container volume $V$ becomes smaller than the excluded molecular volume $nb$.
*   **The breakdown:** the Van der Waals equation $P = nRT/(V-nb) - \dots$ creates a mathematical singularity as $V \to nb$.
*   **Result:** pressure approaches infinity ($P \to \infty$). Consequently, the Reaction Quotient ($Q_p \propto P^{-2}$) approaches zero.
*   **Significance:** this successfully identifies the physical boundary of the gas-phase model. Beyond this limit, the assumptions of the Equation of State fail, representing a regime where phase transition (solidification) would occur in reality.

## Case Study 3: Oxidation of Nitric Oxide and More
*Status: In Progress*

The core of the project is now a flexible "summation machine" that decouples the physical definitions of molecules from the mathematical logic of the solver.

### System Architecture
The software is built upon a few fundamental classes and functions.
1. `class ChemicalSpecies`
* **Role:** acts as the digital identity for a molecule (e.g., $NO$, $O_2$).
* Encapsulates intrinsic physical properties, specifically the Van der Waals constants ($a$ and $b$), which allows the simulation to automatically calculate real gas deviations for any arbitrary mixture without hard-coded lookup tables.
2. `class Reaction`
* **Role:** represents a single elementary step (e.g., $2NO\rightarrow N_2O_2$).
* **Logic:** it stores the stoichiometry and the Arrhenius parameters ($A$, $E_a$). Crucially, this class is vectorised; it can calculate rate constants and reaction velocities for a single time-point (during integration) or an entire history array (during visualization) efficiently.
* **Key method:** `get_rate_constant(T)`, uses the Arrhenius equation.
* **Key method:** `calculate_rate(concentrations, T)`, implements the *Law of Mass Action*, which states that the rate is proportional to the product of reactant concentrations. It iterates through the `reactants` dictionary, raising each concentration to its stoichiometric power.
3. `class GeneralChemicalSystem`
* **Role:** the container and solver. It holds the state ($n$, $V$, $T$) and the collection of ChemicalSpecies and Reactions.
* **Summation machine:** the engine does not rely on pre-written rate laws. Instead, at every time step, it iterates through every reaction object. It calculates the net rate of change for a species $i$ by summing the stoichiometric contributions of every reaction $j$ it participates in:
$$\frac{dn_i}{dt} = V \sum_{j} \nu_{ij} r_j$$

Where:
-   $V$ is the system volume.
-   $r_j$ is the rate of reaction $j$ (concentration basis).
-   $\nu_{ij}$ is the stoichiometric coefficient of species $i$ in reaction $j$ (negative for reactants, positive for products).

By iterating through every reaction and accumulating the changes in a dictionary (`derivatives[name] += ...`), the system naturally handles complex, coupled mechanisms (where a species is produced by one reaction and consumed by another) without needing explicit instructions on how the steps interact.

4. `class GeneralPerturbationSimulation`
* **Role:** simulates when a stress is applied to the system, manipulating the *conditions* ($n$, $V$, $T$) and asking the System to solve the chemistry.
5. `santize_results`
* **Role:** a data cleaning utility. Because numerical solvers and stitching processes are messy, they often produce duplicate time points. This function calculates `dt = diff(time)`, throws away any point where `dt` is effectively zero, and synchronises this cleaning across all data arrays (Volume, Species, Thermodynamics) so the array lengths remain identical.
6. `generate_plot_and_table`
* **Role:** this is the visaulisation engine, with features such as log-time scaling, smart trimming (cutting off the graph), and dual-twin axes (it layers multiple plots such as concentration, pressure, rate on top of each other using distinct axes (`twinx`) to handle different units.
7. `create_interface`
* **Role:** this builds the user interface. It inspects the `GeneralChemicalSystem` object, creating a slider for every species it finds, links the buttons to the logic functions, and manages the application state (`st.prev`, `st.curr`) to allow comparison between runs.

### Mathematical Framework
Real-world chemical mechanisms often involve processes occurring on vastly different timescales. For example, the oxidation of nitric oxide:
* **Step 1:** $2NO(g) \rightleftharpoons N_2O_2(g)$ (fast equilibrium, $\approx10^{-6}s$)
* **Step 2:** $N_2O_2(g) + O_2(g) \rightarrow 2NO_2(g)$ (slow oxidation, $\approx10^2s$)
Standard explicit integrators (like the one used in the previous model) fail here, as they must take microsecond steps to track the fast equilibrium, making the simulation impossibly slow. This engine utilizes the Radau method (an implicit Runge-Kutta scheme), which solves a system of algebraic equations at each step. This allows the solver to remain stable over large time steps, effectively "stepping over" the fast vibrations to model the bulk reaction progression.

### Auxiliary Functions
Beyond the core physics engine, the project includes a suite of auxiliary functions designed to ensure data integrity, numerical stability, and scientific accuracy.

1. **Data sanitization (`santize_results`).**
2. **Automated diagnostic suite (`run_tests`).**
To prevent regression bugs and ensure physical realism, the engine includes a built-in integration testing suite. It runs a cloned instance of the chemical system through several critical checks:
* *Conservation of matter*: verifies that concentrations remain non-negative (withint floating-point tolerance).
* *Convergence*: confirms that the adaptive time-stepping algorithm successfully drives net reaction rates to zero.
* *Thermodynamic stability*: checks that the reaction quotient ($Q_c$) stabilises over time. In coupled mechanisms, a stable $Q_c$ confirms the establishment of a steady state.
* *Real-gas physics*: validates the Van der Waals implementation by confirming that $P_{real}$ diverges from $P_{ideal}$ based on the specific molecular constants ($a$ and $b$) of the species involved.
* *Stiching integrity*: simulates a perturbation event to ensure the "stitcher" maintains a continuous, monotonic time axis without data loss.

### Why this engine is powerful...
The transition to a generalised architecture represents a leap in the ability of the simulation.
* **Zero-math configuration:** the user does not need to derive or write complex differential equations. You simply define what reacts (e.g., "2 molecules of A react with 1 of B"), and the "Summation Machine" automatically constructs the correct mathematical model ($\frac{dn}{dt}$) based on the Law of Mass Action
* **Handling coupled mechanisms:** the engine excels at multistep, coupled mechanisms. It naturally handles scenarios where a species is being produced by one fast reaction while simultaneously being consumed by a slow "drain" reaction (as seen in the NO oxidation mechanism). The solver resolves the competition between these steps without manual intervention.

### How to use the engine...
Setting up a new simulation follows a standardized 4-step workflow. This design allows researchers to swap out the entire chemical system by changing only the configuration block, without touching the core solver logic.

1. **Define the Species:** instantiate `ChemicalSpecies` objects. You must provide the Van der Waals constants ($a$ and $b$) to enable real-gas physics.
```
# name, vdw_a, vdw_b
NO = ChemicalSpecies('NO', vdw_a=1.34, vdw_b=0.0279)
O2 = ChemicalSpecies('O2', vdw_a=1.36, vdw_b=0.0318)
# intermediates with estimated properties
N2O2 = ChemicalSpecies('N2O2', vdw_a=4.0, vdw_b=0.056)
```

2. **Define the Reactions:** instantiate `Reaction` objects. Stoichiometry is defined using dictionaries, allowing for any number of reactants or products. Kinetics are defined via Arrhenius parameters ($A$ and $E_a$).
```
# forward: 2 NO -> N2O2
r1 = Reaction(
    reactants={'NO': 2}, 
    products={'N2O2': 1}, 
    A=4.0e9, 
    Ea=0.0
)
```

3. **Initialise:** create the `GeneralChemicalSystem` object.
```
system = GeneralChemicalSystem(
    species_list=[NO, O2, N2O2],
    reaction_list=[r1, ...],
    initial_moles={'NO': 0.1, 'O2': 0.1, 'N2O2': 0.0},
    initial_V=1.0, 
    initial_T=300
)
```

4. **Launch the interface:** pass the system object to the UI builder. The software will inspect your objects and automatically generate the appropriate sliders, graphs, and perturbation controls.
```
create_interface(system)
```

## Project Roadmap
This is a project in progress. The $N_2O_4$ and Haber Process models serve as the foundation for a more comprehensive toolkit. Future planned enhancements include:
- **Coupled Aqueous Equilibria:** applying the toolkit to model the complex, multi-reaction system that governs ocean acidification.
