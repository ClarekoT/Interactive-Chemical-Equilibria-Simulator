## Interactive Simulation of N₂O₄ ⇌ 2NO₂ Equilibrium

*This notebook contains an interactive dashboard to simulate the chemical kinetics of the reversible gas-phase reaction $N_2O_4(g) ⇌ 2NO_2(g)$. The tool allows for the setting of initial conditions (concentrations, volume) and the application of real-time perturbations (volume and injections) to quantitatively model and visualise Le Chatelier's principle.*

The simulation employs an Object-Oriented Programming (OOP) architecture to model the chemical system. The core principles of this are:
- __Encapsulation:__ the system's state (moles, volume) and the physical laws that govern it (the ODEs) are bundled into a single, self-contained ```ChemicalSystem``` object.
- __State management:__ the state is managed by passing complete ```ChemicalSystem``` objects to ensure data integrity is maintained.

### 1. Imports


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import copy
```

### 2. Model Configuration
This defines the physical and chemical constants of the simulation. The rate constants ('k_f', 'k_r') are derived for a temperature of 323K. The rate constants and Kc would change with temperature. This temperature is just above the boiling point of $N_2O_4$ ($21.7^oC$).

Because Kc < 1 at equilibrium, the concentration of the reactant ($N_2O_4$) will be significantly greater than the concentration of the product ($NO_2$). At 323K, the equilibrium position lies to the left, favouring the formation of dinitrogen tetroxide.


```python
T = 323 # Kelvin
k_f = 0.5 # s^-1, forward rate constant at 323K
k_r = 0.5/0.052 # dm^3 mol^-1 s^-1, calculated from k_f/Kc
R = 0.08206 # universal gas constant in L atm K^-1 mol^-1
MAX_T = 10 # maximum simulation time in seconds
```

### 3. Simulation Engine
The core of the simulation is a system of ODEs. It takes in the current state (in moles) and system volume, calculates the corresponding concentrations, and returns the rate of change of moles ('dn/dt') for each species. An early version of this model defined the system's state purely by the concentrations of the chemical species. However, to model Le Chatelier's principle for pressure and volume changes, it was essential to re-engineer the simulation to run on more fundamental quantities. This model's state vector therefore tracks the number of moles (n) of each species, while the system volume (V) is treated as a separate, independent variable.
- Forward Reaction ($N_2O_4$ -> $2NO_2$): this is a unimolecular decomposition. The rate law is first-order with respect to $N_2O_4$:
      Forward Rate = k_f * $[N_2O_4]$
- Reverse Reaction ($2NO_2$ -> $N_2O_4$): this is a bimolecular reaction. The rate law is second-order with respect to $NO_2$:
      Reverse Rate = k_r * $[NO_2]^2$

The simulation is built upon __2 primary classes:__ ```ChemicalSystem``` and ```PerturbationSimulation```.

*The ```ChemicalSystem``` Class*

This is the fundamental data structure of the simulation. 
- __Attributes:__ an instance of ```ChemicalSystem``` encapsulates the intrinsic properties and current state of the reaction: number of moles of each species, system volume, and forward/reverse rate constants.
- __Methods:__ it contains the methods that define its behaviour. The ```_ode_system``` method provides the mathematical model of the reaction kinetics for the numerical solver. The ```run_simulation``` method orchestrates the integration of this system over time, and the ```_process_results``` method transforms the raw numerical output into chemically meaningful data, such as concentrations, reaction rates, and equilibrium constants.


```python
class ChemicalSystem:
    """Represents the N₂O₄ ⇌ 2NO₂ chemical system.
    This class encapsulates the state of the system (moles, volume) and its kinetic parameters (rate constants),
    providing the methods to model its dynamic behaviour."""
    
    def __init__(self, initial_moles_N2O4, initial_moles_NO2, V, k_f, k_r):
        """Initialises the chemical system with its starting conditions."""

        self.n_N2O4 = initial_moles_N2O4
        self.n_NO2 = initial_moles_NO2
        self.volume = V
        self.k_f = k_f
        self.k_r = k_r
        self.y0 = [self.n_N2O4, self.n_NO2] # initial state vector for the ODE solver
        self.solution = None # placeholder for the simulation results
        self.results = None # placeholder for the processed results dictionary

    def _ode_system(self, t, y):
        """This defines the system of Ordinary Differential Equations (ODEs) for the reaction.

        This method calculates the rate of change of moles for each species
        based on the current state of the system. It is intended for use
        with an ODE solver like scipy.integrate.solve_ivp.

        It returns a list containing the rate of change for each species [dn_N2O4/dt, dn_NO2/dt]."""
    
        moles_N2O4, moles_NO2 = y[0], y[1]  # unpacking the state vector...

        # calculate concentrations from moles and the system's volume...
        conc_N2O4 = moles_N2O4 / self.volume
        conc_NO2 = moles_NO2 / self.volume

        # calculate forward and reverse reaction rates...
        forward_rate = self.k_f * conc_N2O4
        reverse_rate = self.k_r * (conc_NO2)**2

        # calculate the net rate of change of moles for each species...
        dn_N2O4_dt = self.volume * (-forward_rate + reverse_rate)
        dn_NO2_dt = self.volume * (2 * forward_rate - 2 * reverse_rate)

        return [dn_N2O4_dt, dn_NO2_dt]

    def run_simulation(self, max_t):
        """This runs the simulation by integrating the ODE system. This method calls the ODE solver to compute
        the evolution of the system over time and stores the result within the object. 
        max_t (float) represents the total time duration for the simulation in seconds."""

        t_span = (0, max_t)
        # calling the solver using the object's own ODE system...
        self.solution = solve_ivp(fun = self._ode_system, t_span = t_span, y0 = self.y0, dense_output = True)
        # automatically call the processor...
        self._process_results()

    def _process_results(self):
        """This processes the raw output from the ODE solver into a structured dictionary containing chemically meaningful data
        such as concentrations, rates, Kc, etc."""

        if self.solution is None:
            print("Error: simulation has not been run yet. Cannot process results.")
            return

        # 1. Data extraction.
        max_t = self.solution.t[-1]
        time = np.linspace(0, max_t, 1000)
        moles_vs_time = self.solution.sol(time)
        moles_N2O4_vs_time = moles_vs_time[0]
        moles_NO2_vs_time = moles_vs_time[1]

        # 2. Calculations.
        conc_N2O4_vs_time = moles_N2O4_vs_time / self.volume
        conc_NO2_vs_time = moles_NO2_vs_time / self.volume
        net_rates_N2O4 = -self.k_f * conc_N2O4_vs_time + self.k_r * conc_NO2_vs_time**2
        forward_rates_vs_time = self.k_f * conc_N2O4_vs_time
        reverse_rates_vs_time = self.k_r * conc_NO2_vs_time**2
        
        Qc_vs_time = np.divide(
            conc_NO2_vs_time**2, conc_N2O4_vs_time,
            out=np.full_like(conc_N2O4_vs_time, np.nan),
            where=(conc_N2O4_vs_time != 0))

        # 3. Compute equilibrium values.
        conc_N2O4_eq = conc_N2O4_vs_time[-1]
        conc_NO2_eq = conc_NO2_vs_time[-1]
        moles_N2O4_eq = moles_N2O4_vs_time[-1]
        moles_NO2_eq = moles_NO2_vs_time[-1]
        n_total_eq = moles_N2O4_eq + moles_NO2_eq
        P_total_eq = (n_total_eq * R * T) / self.volume
        Kc = (conc_NO2_eq**2) / conc_N2O4_eq if conc_N2O4_eq != 0 else None

        # finding equilibrium time...
        tolerance = 1e-4
        eq_indices = np.where(np.abs(net_rates_N2O4) < tolerance)[0]
        equilibrium_time = time[eq_indices[0]] if eq_indices.size > 0 else time[-1]

        # 4. Store the results in a dictionary.
        self.results = {
            "time": time,
            "initial_conditions": {
                "V": self.volume,
                "n_N2O4": self.n_N2O4,
                "n_NO2": self.n_NO2},
            "results": {
                "moles_N2O4": moles_N2O4_vs_time,
                "moles_NO2": moles_NO2_vs_time,
                "conc_N2O4": conc_N2O4_vs_time,
                "conc_NO2": conc_NO2_vs_time,
                "rates_N2O4": net_rates_N2O4,
                "forward_rates": forward_rates_vs_time,
                "reverse_rates": reverse_rates_vs_time,
                "Qc_vs_time": Qc_vs_time},
            "equilibrium": {
                "time": equilibrium_time,
                "conc_N2O4": conc_N2O4_eq,
                "conc_NO2": conc_NO2_eq,
                "pressure": P_total_eq,
                "Kc": Kc}}
```

*The ```PerturbationSimulation``` Class*

This class is designed to handle the complex, 3-stage workflow of applying a stress to a system at equilibrium.
- __Composition:__ this class uses composition; it holds a fully-formed ```ChemicalSystem object``` as an attribute (```self.baseline_system```). The ```PerturbationSimulation``` acts *upon* the chemical system.
- __Orchestration:__ its primary method, ```run_perturbation```, executes the full sequence: it extracts the pre-perturbation history from the baseline system (Stage 1), selects the appropriate mathematical model for the perturbation window and simulates the stress (Stage 2), and finally, simulates the system's relaxation to a new equilibrium (Stage 3). It is responsible for concatenating the data from these distinct phases into a single, coherent time-series.
  
__The Ramped Volume Model...__
    This takes the interpolation parameters as arguments. This function is used only for the solve_ivp call in stage 2, the stage where the volume is a function of time.

__The Injection Model...__
    This models a gradual injection of a reactant/product. The total rate of change in the number of moles for a given chemical species (dn/dt) is the sum of the change from the internal reaction and any change from an external source. 
dn/dt_total = dn/dt_reaction + dn/dt_injection. This is used for stage 2 of the simulation. Stage 1 (pre-injection) and stage 3 (post-injection) will revert to the original reversible_model.


```python
class PerturbationSimulation:
    """This manages the three-stage process of applying a perturbation to a baseline ChemicalSystem and simulating the result.

    This class uses composition, holding a ChemicalSystem instance to represent the initial state. Its primary role is to orchestrate the simulation
    before, during, and after a defined stress is applied."""
    
    def __init__(self, baseline_system, perturbation_type, t_start, t_end,
                 V_end=None, injection_rate_N2O4=0.0, injection_rate_NO2=0.0):
        """This initialises the perturbation process manager.
        Args:
            baseline_system (ChemicalSystem): the fully simulated system object to be perturbed.
            perturbation_type (str): the type of stress, either 'volume' or 'injection'.
            t_start, t_end (floats): the start and end times of the perturbation window.
            V_end (float, optional): the target volume for a 'volume' perturbation. Defaults to None.
            injection_rate_N2O4 (float, optional): moles/s for an 'injection' perturbation. Defaults to 0.0.
            injection_rate_NO2 (float, optional): moles/s for an 'injection' perturbation. Defaults to 0.0."""
        
        # store all configuration parameters...
        self.baseline_system = baseline_system
        self.perturbation_type = perturbation_type
        self.t_start = t_start
        self.t_end = t_end

        # store perturbation-specific parameters...
        self.V_start = baseline_system.volume
        self.V_end = V_end
        self.injection_rate_N2O4 = injection_rate_N2O4
        self.injection_rate_NO2 = injection_rate_NO2

        self.results = None # placeholders for results

    def _ramped_volume_model(self, t, y):
        """This is an internal ODE system for the volume perturbation stage (Stage 2). It returns a list which is the rate of change of both species.
        Args:
            t (float): current time from the solver.
            y (list): current state vector [moles_N2O4, moles_NO2]."""
        
        # interpolate to find the volume at the exact time t...
        V_at_t = np.interp(t, [self.t_start, self.t_end], [self.V_start, self.V_end])

        # calculate concentrations using the interpolated volume...
        conc_N2O4 = y[0] / V_at_t
        conc_NO2 = y[1] / V_at_t

        # get rate constants from the baseline system object...
        k_f = self.baseline_system.k_f
        k_r = self.baseline_system.k_r

        # calculate rates...
        forward_rate = k_f * conc_N2O4
        reverse_rate = k_r * (conc_NO2)**2
        dn_N2O4_dt = V_at_t * (-forward_rate + reverse_rate)
        dn_NO2_dt = V_at_t * (2 * forward_rate - 2 * reverse_rate)
        return [dn_N2O4_dt, dn_NO2_dt]

    def _injection_model(self, t, y):
        """This is another internal ODE system for the species injection stage (Stage 2). It returns a list of the rate of change of both species.
        Takes in the same arguments as the _ramped_volume_model."""
        
        moles_N2O4, moles_NO2 = y[0], y[1]
        V = self.baseline_system.volume # volume is constant during injection

        conc_N2O4 = moles_N2O4 / V
        conc_NO2 = moles_NO2 / V

        # get rate constants from the baseline system object...
        k_f = self.baseline_system.k_f
        k_r = self.baseline_system.k_r

        # calculate rates from internal reaction...
        forward_rate = k_f * conc_N2O4
        reverse_rate = k_r * (conc_NO2)**2
        dn_N2O4_dt_from_reaction = V * (-forward_rate + reverse_rate)
        dn_NO2_dt_from_reaction = V * (2 * forward_rate - 2 * reverse_rate)

        # add the external injection rate...
        total_dn_N2O4_dt = dn_N2O4_dt_from_reaction + self.injection_rate_N2O4
        total_dn_NO2_dt = dn_NO2_dt_from_reaction + self.injection_rate_NO2
        
        # prevent moles from becoming negative...
        if moles_N2O4 <= 0 and total_dn_N2O4_dt < 0:
            total_dn_N2O4_dt = 0
        if moles_NO2 <= 0 and total_dn_NO2_dt < 0:
            total_dn_NO2_dt = 0

        return [total_dn_N2O4_dt, total_dn_NO2_dt]

    def run_perturbation(self):
        """This orchestrates the full 3-stage perturbation simulation. This executes the pre-perturbation, during perturbation, and post-perturbation
        stages, then stitches the results together into a single, coherent dataset."""

        # Stage 1: Pre-Perturbation (Data from Baseline)
        baseline_results = self.baseline_system.results
        baseline_time = baseline_results['time']

        # find the index in the baseline data corresponding to the perturbation start time...
        perturb_index = np.argmin(np.abs(baseline_time - self.t_start))

        # slice all baseline arrays to get Stage 1 data...
        stage1_time = baseline_time[:perturb_index + 1]
        stage1_moles_N2O4 = baseline_results['results']['moles_N2O4'][:perturb_index + 1]
        stage1_moles_NO2 = baseline_results['results']['moles_NO2'][:perturb_index + 1]
        stage1_conc_N2O4 = baseline_results['results']['conc_N2O4'][:perturb_index + 1]
        stage1_conc_NO2 = baseline_results['results']['conc_NO2'][:perturb_index + 1]

        # Stage 2: During Perturbation
        stage2_initial_moles = (stage1_moles_N2O4[-1], stage1_moles_NO2[-1])
        t_span_2 = (self.t_start, self.t_end)
        
        # select the correct ODE model based on perturbation type...
        if self.perturbation_type == 'volume':
            ode_func_2 = self._ramped_volume_model
            stage3_V = self.V_end # final volume for Stage 3
        elif self.perturbation_type == 'injection':
            ode_func_2 = self._injection_model
            stage3_V = self.V_start # volume is unchanged for Stage 3
        else:
            raise ValueError("Invalid perturbation type specified.")

        # run the solver for Stage 2...
        stage2_solution = solve_ivp(ode_func_2, t_span_2, stage2_initial_moles, dense_output=True)
        stage2_time = np.linspace(self.t_start, self.t_end, 500)
        stage2_moles_vs_time = stage2_solution.sol(stage2_time)
        stage2_moles_N2O4 = stage2_moles_vs_time[0]
        stage2_moles_NO2 = stage2_moles_vs_time[1]

        # calculate concentrations for Stage 2...
        if self.perturbation_type == 'volume':
            stage2_V_vs_time = np.interp(stage2_time, [self.t_start, self.t_end], [self.V_start, self.V_end])
            stage2_conc_N2O4 = stage2_moles_N2O4 / stage2_V_vs_time
            stage2_conc_NO2 = stage2_moles_NO2 / stage2_V_vs_time
        else: # perturbation_type == 'injection'
            stage2_conc_N2O4 = stage2_moles_N2O4 / self.V_start
            stage2_conc_NO2 = stage2_moles_NO2 / self.V_start

        # Stage 3: Post-Perturbation (Relaxation to New Equilibrium)
        stage3_initial_moles = (stage2_moles_N2O4[-1], stage2_moles_NO2[-1])

        # To simulate stage 3, we create a new, temporary ChemicalSystem object that represents the state of the system at the start of stage 3.

        stage3_system = ChemicalSystem(
            initial_moles_N2O4 = stage3_initial_moles[0], initial_moles_NO2 = stage3_initial_moles[1],
            V = stage3_V, k_f = self.baseline_system.k_f, k_r = self.baseline_system.k_r)
        stage3_system.run_simulation(max_t=(MAX_T - self.t_end))

        # extract results, but shift the time array to start at t_end...
        stage3_results = stage3_system.results
        stage3_time = stage3_results['time'] + self.t_end
        stage3_moles_N2O4 = stage3_results['results']['moles_N2O4']
        stage3_moles_NO2 = stage3_results['results']['moles_NO2']
        stage3_conc_N2O4 = stage3_results['results']['conc_N2O4']
        stage3_conc_NO2 = stage3_results['results']['conc_NO2']

        # combine the data from all three stages, slicing to avoid duplicate time points...
        combined_time = np.concatenate((stage1_time[:-1], stage2_time[:-1], stage3_time))
        combined_moles_N2O4 = np.concatenate((stage1_moles_N2O4[:-1], stage2_moles_N2O4[:-1], stage3_moles_N2O4))
        combined_moles_NO2 = np.concatenate((stage1_moles_NO2[:-1], stage2_moles_NO2[:-1], stage3_moles_NO2))
        combined_conc_N2O4 = np.concatenate((stage1_conc_N2O4[:-1], stage2_conc_N2O4[:-1], stage3_conc_N2O4))
        combined_conc_NO2 = np.concatenate((stage1_conc_NO2[:-1], stage2_conc_NO2[:-1], stage3_conc_NO2))
        
        # calculate final derived quantities on the combined data...
        k_f = self.baseline_system.k_f
        k_r = self.baseline_system.k_r
        combined_rates_N2O4 = -k_f * combined_conc_N2O4 + k_r * combined_conc_NO2**2
        forward_rates = k_f * combined_conc_N2O4
        reverse_rates = k_r * combined_conc_NO2**2
        Qc_vs_time = np.divide(combined_conc_NO2**2, combined_conc_N2O4, out=np.full_like(combined_conc_N2O4, np.nan), where=(combined_conc_N2O4 != 0))
        
        # final equilibrium values are the last point of the combined data...
        conc_N2O4_eq = combined_conc_N2O4[-1]
        conc_NO2_eq = combined_conc_NO2[-1]
        n_total_eq = combined_moles_N2O4[-1] + combined_moles_NO2[-1]
        P_total_eq = (n_total_eq * R * T) / stage3_V
        Kc = (conc_NO2_eq**2) / conc_N2O4_eq if conc_N2O4_eq != 0 else None

        tolerance = 1e-4
        search_start_index = np.argmin(np.abs(combined_time - self.t_end))
        # Search for equilibrium only in the rates after the perturbation.
        post_perturb_rates = combined_rates_N2O4[search_start_index:]
        eq_indices_in_slice = np.where(np.abs(post_perturb_rates) < tolerance)[0]

        if eq_indices_in_slice.size > 0:
            true_eq_index = eq_indices_in_slice[0] + search_start_index
            t_new_eq = combined_time[true_eq_index]
        else:
            t_new_eq = combined_time[-1]  # no new equilibrium found, use last time point

        t_re_equilibration = t_new_eq - self.t_end
        print(f"Time to re-establish equilibrium after perturbation is {t_re_equilibration:.4f} s.")

        # store the final, structured dictionary in self.results...
        self.results = {
            "time": combined_time,
            "initial_conditions": {"V": stage3_V, "n_N2O4": combined_moles_N2O4[0], "n_NO2": combined_moles_NO2[0]},
            "results": {"moles_N2O4": combined_moles_N2O4, "moles_NO2": combined_moles_NO2, "conc_N2O4": combined_conc_N2O4, "conc_NO2": combined_conc_NO2, "rates_N2O4": combined_rates_N2O4, "forward_rates": forward_rates, "reverse_rates": reverse_rates, "Qc_vs_time": Qc_vs_time},
            "equilibrium": {"time": t_new_eq, "conc_N2O4": conc_N2O4_eq, "conc_NO2": conc_NO2_eq, "pressure": P_total_eq, "Kc": Kc}}
```

The '```previous_system_run```' variable is used to store the results of the last baseline simulation, enabling comparison.


```python
previous_system_run = None
```

### 4. Core Logic
This section contains the main event-handler functions that are triggered by user interactions. 

*4.1 Display generation ('```generate_plot_and_table```')...*

This is responsible for the visual output. It takes one or two complete simulation datasets and generate a 'matplotlib' plot and a 'pandas' comparison table. It will draw a vertical marker if a perturbation time is provided. This approach ensures a consistent look for all outputs and prevents code duplication.


```python
def generate_plot_and_table(current_data, previous_data, perturbation_window = None):
    """This generates and displays all visual output for a simulation run. This function takes one or two simulation results dictionaries and produces
    a matplotlib plot as well as a pandas DataFrame comparing key metrics."""
    
    #--- Data Extraction ---
    time = current_data["time"]
    V = current_data["initial_conditions"]["V"]
    conc_N2O4 = current_data["results"]["conc_N2O4"]
    conc_NO2 = current_data["results"]["conc_NO2"]
    rates_N2O4 = current_data["results"]["rates_N2O4"]
    
    fig, ax1 = plt.subplots(figsize = (10, 6))
        
    # find equilibrium time for plotting limits
    tolerance = 1e-4
    if perturbation_window is not None:
        t_start, t_end = perturbation_window
        search_start_index = np.argmin(np.abs(time - t_end))
        post_perturb_rates = rates_N2O4[search_start_index:]
        eq_indices_in_slice = np.where(np.abs(post_perturb_rates) < tolerance)[0]
        if eq_indices_in_slice.size > 0:
            true_eq_index = eq_indices_in_slice[0] + search_start_index
            equilibrium_time = time[true_eq_index]
        else:  # no equilibrium is found
            equilibrium_time = time[-1]
    else:  # this is the baseline run logic, finding the first time the system reaches equilibrium
        eq_indices = np.where(np.abs(rates_N2O4) < tolerance)[0]
        equilibrium_time = time[eq_indices[0]] if eq_indices.size > 0 else time[-1]

    plot_max_t = equilibrium_time * 1.2

    # make the sliders dynamic
    t_perturb_slider.max = plot_max_t * 1.5
    injection_time_slider.max = plot_max_t * 1.5

    conc_handles = []
    rate_handles = []
    thermo_handles = []
    marker_handles = []
    
    # plot current run
    line, = ax1.plot(time, conc_N2O4, label=r"$[N_2O_4]$", color="blue")
    conc_handles.append(line)
    line, = ax1.plot(time, conc_NO2, label=r"$[NO_2]$", color="red")
    conc_handles.append(line)
    line, = ax1.plot(time, rates_N2O4, label=r"Rate of Change of $[N_2O_4]$", color="#808080", linestyle=":", lw=1)
    rate_handles.append(line)

    forward_rates = current_data["results"]["forward_rates"]
    reverse_rates = current_data["results"]["reverse_rates"]

    line, = ax1.plot(time, forward_rates, label='Forward Rate', color="green", linestyle="--", lw=1.5, alpha=0.8)
    rate_handles.append(line)
    line, = ax1.plot(time, reverse_rates, label='Reverse Rate', color="orange", linestyle="--", lw=1.5, alpha=0.8)
    rate_handles.append(line)

    if perturbation_window is not None:
        line = ax1.axvline(x = t_start, color = 'black', linestyle = '--', label = 'Perturbation Window', alpha = 1, lw = 0.5)
        marker_handles.append(line)
        ax1.axvline(x = t_end, color = 'black', linestyle = '--', alpha = 1, lw = 0.5)

    line = ax1.axvline(x = equilibrium_time, color = 'black', linestyle = '-', label = 'Equilibrium Reached', alpha = 0.6, lw = 1)
    marker_handles.append(line)

    if previous_data is not None:
        pr = previous_data
        # overlay previous
        ax1.plot(pr["time"], pr["results"]["conc_N2O4"], color="#ADD8E6", alpha=0.5)
        ax1.plot(pr["time"], pr["results"]["conc_NO2"], color="#F08080", alpha=0.5)
        ax1.plot(pr["time"], pr["results"]["rates_N2O4"], color="#DDA0DD", alpha=0.5, linestyle=":", lw=1)
        ax1.plot(pr["time"], pr["results"]["forward_rates"], color="green", alpha=0.1, linestyle="--", lw=1.5)
        ax1.plot(pr["time"], pr["results"]["reverse_rates"], color="#ff8800", alpha=0.1, linestyle="--", lw=1.5)

        # build comparison table
        table_data = {
            'Info': [
                r'Volume (dm³)', r'Initial N₂O₄ Moles', r'Initial NO₂ Moles',
                'Eq. Time (s)', r'[N₂O₄]eq', r'[NO₂]eq', 'Pressure (atm)', 'Kc'],
            'Previous Run': [
                f"{pr['initial_conditions']['V']:.2f}",
                f"{pr['initial_conditions']['n_N2O4']:.2f}",
                f"{pr['initial_conditions']['n_NO2']:.2f}",
                f"{pr['equilibrium']['time']:.4f}" if pr['equilibrium']['time'] else 'N/A',
                f"{pr['equilibrium']['conc_N2O4']:.2f}",
                f"{pr['equilibrium']['conc_NO2']:.2f}",
                f"{pr['equilibrium']['pressure']:.2f}",
                f"{pr['equilibrium']['Kc']:.8f}" if pr['equilibrium']['Kc'] is not None else "N/A"],
            'Current Run': [
                f"{V:.2f}",
                f"{current_data['initial_conditions']['n_N2O4']:.2f}",
                f"{current_data['initial_conditions']['n_NO2']:.2f}",
                f"{equilibrium_time:.4f}",
                f"{current_data['equilibrium']['conc_N2O4']:.2f}",
                f"{current_data['equilibrium']['conc_NO2']:.2f}",
                f"{current_data['equilibrium']['pressure']:.2f}",
                f"{current_data['equilibrium']['Kc']:.8f}" if current_data['equilibrium']['Kc'] is not None else "N/A"]}
        df = pd.DataFrame(table_data)
        display(df.style.set_table_styles([{'selector': 'th, td', 'props': [('text-align', 'center')]}]).hide(axis="index"))
    else:
        # create a single-column table to display data for the current run...
        table_data = {'Info': [r'Volume (dm^3)', r'Initial N_2O_4 Moles', r'Initial NO_2 Moles', 'Eq. Time (s)', r'[N_2O_4]eq', r'[NO_2]eq', 'Pressure (atm)', 'Kc'],
                      'Current Run': [f"{V:.2f}", f"{current_data['initial_conditions']['n_N2O4']:.2f}", f"{current_data['initial_conditions']['n_NO2']:.2f}", f"{current_data['equilibrium']['time']:.4f}" if current_data['equilibrium']['time'] else 'N/A', f"{current_data['equilibrium']['conc_N2O4']:.2f}", f"{current_data['equilibrium']['conc_NO2']:.2f}", f"{current_data['equilibrium']['pressure']:.2f}", f"{current_data['equilibrium']['Kc']:.8f}" if current_data['equilibrium']['Kc'] is not None else "N/A"]}
        df = pd.DataFrame(table_data)
        print("Baseline Simulation Results:")
        display(df.style.set_table_styles([{'selector': 'th, td', 'props': [('text-align', 'center')]}]).hide(axis="index"))

    ax2 = ax1.twinx() # creates a second Y-axis that shares the same X-axis
    Qc = current_data["results"]["Qc_vs_time"]
    Kc = current_data["equilibrium"]["Kc"] # gets the final Kc value
    line, = ax2.plot(time, Qc, label = 'Qc', color = '#c159de', linestyle = '-.', lw = 1)
    thermo_handles.append(line)
    if Kc is not None:
        line = ax2.axhline(y = Kc, color = '#813cba', linestyle = '-.', label = 'Kc', alpha = 0.7, lw = 1)
        thermo_handles.append(line)

    ax2.set_ylabel(r"Qc / mol $dm^{-3}$", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # finalise plot
    ax1.set_xlabel("Time /s")
    ax1.set_ylabel(r"Concentration / mol $dm^{-3}$") 
    ax1.grid(True, linestyle = '--', alpha = 0.7, which = 'major')
    ax1.grid(True, linestyle = ':', alpha = 0.4, which = 'minor')
    ax1.minorticks_on()
    ax1.set_title("N₂O₄ ⇌ 2NO₂")
    ax1.set_xlim(0, plot_max_t) 
    
    # set ylim based on concentration AND rates
    y_max_main = max(conc_N2O4.max(), conc_NO2.max(), forward_rates.max()) 
    ax1.set_ylim(0, y_max_main * 1.1)
    # making the ylim for Qc axis more sensible...
    if np.nanmax(Qc) > 0:
        ax2.set_ylim(0, np.nanmax(Qc) * 1.5 if np.nanmax(Qc) < 3*Kc else np.nanmax(Qc)*1.2)

    # --- Creating Multiple Legends ---
    leg1 = fig.legend(
        handles = conc_handles + marker_handles,
        loc = 'upper left',
        bbox_to_anchor = (1.02, 1.0),
        title = 'Concentrations and More')

    leg2 = fig.legend(
        handles = rate_handles,
        loc = 'upper left',
        bbox_to_anchor = (1.02, 0.75),  # position below the first legend
        title = 'Rates')

    leg3 = fig.legend(
        handles = thermo_handles,
        loc = 'upper left',
        bbox_to_anchor = (1.02, 0.5),   # position below the second legend
        title = 'Equilibrium')
    
    fig.tight_layout()
    plt.show()
```

*4.2 UI Controllers...*

These functions act as controllers, connecting the user interface (UI) widgets to the backend simulation engine. They are not responsible for performing any chemical calculations themselves. Instead, their role is to:
1. Listen for user events (button clicks).
2. Read the relevant parameters (initial concentrations, perturbation windows) from the UI controls.
3. Instantiate the appropriate backend object (```ChemicalSystem``` for a baseline run or ```PerturbationSimulation``` for a perturbation).
4. Issue a command to that object (e.g., ```.run_simulation()``` or ```.runperturbation()```).
5. Retrieve the final, processed results dictionary from the object.
6. Pass this data onto ```generate_plot_and_table``` function for visualisation.
This keeps the UI logic clean and delegates all complex work to the specialised objects. 


```python
def run_simulation_and_plot(button):
    """This acts as the main controller when the 'Run Simulation' button is clicked.
    
    This function reads parameters from the UI, creates a ChemicalSystem object,
    tells it to run its simulation, and then passes the processed results to the
    display function. It also saves the state for the next run."""
    
    global previous_system_run 
    with output:
        output.clear_output(wait=True)

        # 1. Read initial conditions from the UI sliders...
        V = V_slider.value
        initial_moles_N2O4 = N2O4_slider.value * V
        initial_moles_NO2 = NO2_slider.value * V

        # some input validation
        if initial_moles_N2O4 == 0 and initial_moles_NO2 == 0:
            print("Error: Initial moles for both species cannot be zero. Please set an initial concentration for at least one species.")
            return

        # 2. Instantiate the ChemicalSystem object...
        current_system = ChemicalSystem(
            initial_moles_N2O4 = initial_moles_N2O4,
            initial_moles_NO2 = initial_moles_NO2,
            V = V, k_f = k_f, k_r = k_r)

        # 3. Run the simulation and automatically process the results...
        current_system.run_simulation(MAX_T)

        # 4. Prepare the data dictionaries for the plotting function...
        current_data = current_system.results    # extract the .results dictionary from our objects
        previous_data = previous_system_run.results if previous_system_run else None

        # 5. Call the shared plot and table generation function...
        generate_plot_and_table(current_data, previous_data)

        # 6. Save the state for the next run by storing the entire object...
        previous_system_run = copy.deepcopy(current_system)

        # 7. Enable perturbation controls for the next action...
        t_perturb_slider.disabled = False
        perturb_volume_button.disabled = False
        inject_species_button.disabled = False
        N2O4_injection_rate_widget.disabled = False
        NO2_injection_rate_widget.disabled = False
        injection_time_slider.disabled = False
```

The calculation of pressure uses the Ideal Gas Law (PV = nRT) which makes assumptions about the gas molecules. This is a reasonable approximation for a system under moderate conditions but would become less accurate at very high pressures.


```python
def apply_perturbation(button, perturbation_type):
    """This acts as the controller when a perturbation button is clicked.

    This function reads perturbation parameters from the UI, creates a PerturbationSimulation object to manage the process, tells it to run,
    and then passes the final results to the display function."""
    
    global previous_system_run
    with output:
        output.clear_output(wait=True)

        if previous_system_run is None:
            print("Please run a baseline simulation first.")
            return

        # 1. Read perturbation parameters from the UI...
        if perturbation_type == 'volume':
            print("Applying volume perturbation...")
            t_start, t_end = t_perturb_slider.value
            V_end = V_slider.value
            # for this perturbation type, injection rates are zero
            inject_N2O4 = 0.0
            inject_NO2 = 0.0
        elif perturbation_type == 'injection':
            print("Applying injection perturbation...")
            t_start, t_end = injection_time_slider.value
            inject_N2O4 = N2O4_injection_rate_widget.value
            inject_NO2 = NO2_injection_rate_widget.value
            # for this perturbation type, the end volume is the same as the start
            V_end = previous_system_run.volume
        else:
            return # should not happen

        # 2. Instantiate the process manager class...
        perturbation_sim = PerturbationSimulation(
            baseline_system = previous_system_run,
            perturbation_type = perturbation_type,
            t_start = t_start,
            t_end = t_end,
            V_end = V_end,
            injection_rate_N2O4 = inject_N2O4,
            injection_rate_NO2 = inject_NO2)

        # 3. Run the entire three-stage simulation...
        perturbation_sim.run_perturbation()

        # 4. Get the final results from the simulation object...
        combined_data = perturbation_sim.results
        previous_data = previous_system_run.results # get baseline data for comparison

        # 5. Call the display function with the new and old data...
        generate_plot_and_table(
            current_data = combined_data,
            previous_data = previous_data,
            perturbation_window = (t_start, t_end))
```

*4.4 Utility functions...*

This cell contains smaller helper functions, such as the event handler for the 'Clear' button.


```python
def clear_previous_run_data(button):
    global previous_run_data
    previous_run_data = None
    t_perturb_slider.disabled = True
    perturb_volume_button.disabled = True
    inject_species_button.disabled = True
    N2O4_injection_rate_widget.disabled = True
    NO2_injection_rate_widget.disabled = True
    injection_time_slider.disabled = True
    t_perturb_slider.max = 2.5
    with output:
        output.clear_output(wait = True)
        print("Previous run data has been cleared.")
```

### 5. User Interface (UI)
This cell is where the widgets are defined and configured. This includes the sliders for setting initial conditions and the buttons for controlling the simulation.


```python
# --- UI Widget Creation and Layout ---
slider_layout = widgets.Layout(width='600px')
slider_style = {'description_width': '180px'}

N2O4_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.05, description=r'Initial $[N_{2}O_{4}]$ (mol dm$^{-3}$)', continuous_update=False, layout=slider_layout, style=slider_style)
NO2_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.05, description=r'Initial $[NO_{2}]$ (mol dm$^{-3}$)', continuous_update=False, layout=slider_layout, style=slider_style)
V_slider = widgets.FloatSlider(value=1.0, min=0.2, max=10.0, step=0.05, description=r'Volume (dm$^3$)', continuous_update=False, layout=slider_layout, style=slider_style)
t_perturb_slider = widgets.FloatRangeSlider(value=(0.5, 0.7), min=0.05, max=10.0, step=0.01, description='Volume Perturbation Time Range', continuous_update=False, layout=slider_layout, style=slider_style, disabled = True)
injection_time_slider = widgets.FloatRangeSlider(value=(0.5, 0.7), min=0.05, max=10.0, step=0.05, description='Injection Time', continuous_update=False, layout=slider_layout, style=slider_style, disabled = True)

N2O4_injection_rate_widget = widgets.FloatText(value=0.0, description=r'$N_2O_4$ Injection Rate (mol/s)', disabled = True, layout=slider_layout, style=slider_style)
NO2_injection_rate_widget = widgets.FloatText(value=0.0, description=r'$NO_2$ Injection Rate(mol/s)', disabled = True, layout=slider_layout, style=slider_style)

run_button = widgets.Button(description="Run Simulation", button_style='success')
clear_button = widgets.Button(description="Clear Previous Run", button_style='warning')
inject_species_button = widgets.Button(description="Inject Species", disabled=True)
perturb_volume_button = widgets.Button(description="Perturb System (Volume)", disabled = True)

button_layout = widgets.Layout(width='200px', margin='5px 10px 5px 0')
run_button.layout = button_layout
clear_button.layout = button_layout
perturb_volume_button.layout = button_layout
inject_species_button.layout = button_layout
```

### 6. Application Initialisation
This cell connects the UI widgets to the core logic by setting up the 'on_click' event listeners. Since the function apply_perturbation must know which button (Perturb Volume or Inject Species) is triggered, a labmda function is used. This lambda function calls the main apply_perturbation and also passes a custom string that tells the main function which block of logic to execute.


```python
# --- Event Handling ---
run_button.on_click(run_simulation_and_plot)
clear_button.on_click(clear_previous_run_data)
perturb_volume_button.on_click(lambda b: apply_perturbation(b, 'volume'))
inject_species_button.on_click(lambda b: apply_perturbation(b, 'injection'))
```

### 7. Dashboard
This cell displays and runs the simulation.


```python
# --- Display UI ---
slider_box = widgets.VBox([N2O4_slider, NO2_slider, V_slider, t_perturb_slider], layout=widgets.Layout(align_items='flex-start', padding='10px'))
button_box = widgets.HBox([run_button, clear_button, perturb_volume_button, inject_species_button])
injection_controls = widgets.VBox([injection_time_slider, N2O4_injection_rate_widget, NO2_injection_rate_widget])
output = widgets.Output()
app_layout = widgets.VBox([output, slider_box, injection_controls, button_box])
display(app_layout)
```


    VBox(children=(Output(), VBox(children=(FloatSlider(value=1.0, continuous_update=False, description='Initial $…

