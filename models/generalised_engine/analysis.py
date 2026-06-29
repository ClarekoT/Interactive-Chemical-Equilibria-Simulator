import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import logging
import copy
from chemical_engine import ChemicalSystem

logger = logging.getLogger(__name__)

class ParameterSweeper:
    """
    A class which performs sensitivity analysis, taking in an independent variable and the range the variable is being changed,
    along with a dependent variable. It systematically varies one independent parameter and observes a dependent parameter.
    This allows a graph showing how varying one variable affects another to be plotted.
    """
    def __init__(self, baseline_system: ChemicalSystem, x_variable: str, y_variable: str, variable_range: Tuple[float, float],
                 target_species: Optional[str] = None, target_reaction_idx: Optional[int] = None,
                 target_thermo_pair: Optional[Tuple[int, int]] = None,
                 yield_product: Optional[str] = None,
                 yield_reactant: Optional[str] = None,
                 yield_ratio: Optional[float] = None):
        """
        This function takes the inputs.
        
        The independent variables include (x_variable):
        - Temperature (K): "T"
        - Volume (dm^3): "V"
        - Initial Conc of Species X: "initial_conc" (a separate target_species variable is taken to identify the species)

        The dependent variables include (y_variable):
        - Final Conc of Species X: "final_conc"
        - Time to Equilibrium: "eq_time"
        - Initial Rate of Reaction I: "init_rate" (there could be more than one Reaction object defined, thus we take in target_reaction_idx)
        - Initial Real/Ideal Pressure (atm, only valid if gas species are present): "init_P_real" or "init_P_ideal" (included to see effect of temperature, volume, etc. on pressure)
        - Final Real/Ideal Pressure (atm, only valid if gas species are present): "final_P_real" or "final_P_ideal"
        - Final pH (only valid for acid-base systems): "final_pH"
        - Kc (only valid if there is a reversible reaction): "Kc"
        - Kp (only valid if gas species are present): "Kp"
        - Reaction Yield (%, of chosen product): "rxn_yield"
        """

        self.baseline_system = baseline_system
        self.x_variable = x_variable
        self.y_variable = y_variable
        
        # a tuple (start, end) for the range the independent variable is being varied over
        self.var_start, self.var_end = variable_range 
        self.num_points = self.adaptive_grid_refinement()
        
        # target parameters
        self.target_species = target_species
        self.target_reaction_idx = target_reaction_idx
        self.target_thermo_pair = target_thermo_pair

        # yield parameters passed in from the UI
        self.yield_product = yield_product
        self.yield_reactant = yield_reactant
        self.yield_ratio = yield_ratio

        self.results = None 
    
    def run_sweep(self) -> Dict[str, np.ndarray]:
        """
        This is the main function in the class. It runs repeated simulations (number of simulation depends on num_points) and
        creates a dictionary which contains x_array and y_array. Iterates through the x_variable range, instantiates isolated
        ChemicalSystem objects, runs them, and extracts the y_variable.

        If the graph is non-linear, the sensitivity is not a single number, it is a function. For a curve, the partial derivatie
        changes. This derivative is calculated so that it can be passed onto the plotting function and plotted on the graph as well.
        This allows the user to see where the system is most sensitive.
        """

        # initialise arrays
        x_array = np.linspace(self.var_start, self.var_end, self.num_points)
        y_array = np.zeros(self.num_points)

        logger.info(f"Starting parameter sweep: {self.y_variable} vs. {self.x_variable}({self.num_points} points)...")

        for i, x_val in enumerate(x_array):

            # extract and copy baseline conditions
            init_moles = copy.deepcopy(self.baseline_system.initial_moles)
            V = self.baseline_system.V
            T = self.baseline_system.T

            # modify conditions based on x_variable
            if self.x_variable == 'T':
                T = x_val
            elif self.x_variable == 'V':
                V = x_val
            elif self.x_variable == 'initial_conc':
                if not self.target_species:
                    raise ValueError("target_species must be provided when x_variable is 'initial_conc'")
                init_moles[self.target_species] = x_val * V
            else:
                raise ValueError(f"Unknown x_variable '{self.x_variable}'")

            # create new ChemicalSystem instance
            new_sys = ChemicalSystem(
                self.baseline_system.species_list,
                self.baseline_system.reactions,
                init_moles, V, T,
                method=self.baseline_system.method,
                rtol=self.baseline_system.rtol,
                atol=self.baseline_system.atol,
                overall_reaction=self.baseline_system.overall_reaction,
                system_type=self.baseline_system.system_type
            )

            try:
                new_sys.run_simulation()
            except Exception as e:
                logger.warning(f"Simulation failed at {self.x_variable} = {x_val:.4e}. Error: {e}")
                y_array[i] = np.nan # mark failed runs as NaN
                continue

            if new_sys.results is None:
                y_array[i] = np.nan
                continue


            if self.y_variable == 'final_conc':
                if not self.target_species:
                    raise ValueError("target_species must be provided to extract final_conc")
                final_mole = new_sys.results['final_moles'].get(self.target_species, 0.0)
                y_array[i] = final_mole / new_sys.results['info']['V']

            elif self.y_variable == 'eq_time':
                y_array[i] = new_sys.results['time'][-1]

            elif self.y_variable == 'init_rate':
                if self.target_reaction_idx is None:
                    raise ValueError("target_reaction_idx must be provided for init_rate")
                y_array[i] = new_sys.get_initial_rates()[self.target_reaction_idx]

            elif self.y_variable == 'init_P_real':
                y_array[i] = new_sys.results['P_real'][0]

            elif self.y_variable == 'final_P_real':
                y_array[i] = new_sys.results['P_real'][-1]

            elif self.y_variable == 'init_P_ideal':
                y_array[i] = new_sys.results['P_ideal'][0]
            
            elif self.y_variable == 'final_P_ideal':
                y_array[i] = new_sys.results['P_ideal'][-1]

            elif self.y_variable == 'final_pH':
                # to calculate pH, we take the final concentration of H+ or H3O+ ions and calculate the -log([H+])
                proton_moles = new_sys.results['final_moles'].get('H3O+', new_sys.results['final_moles'].get('H+', 0.0))
                proton_conc = proton_moles / new_sys.results['info']['V']
                y_array[i] = -np.log10(max(proton_conc, 1e-16)) # safety clamp to prevent log(0)

            elif self.y_variable in ['Kc', 'Kp']:
                if not self.target_thermo_pair:
                    raise ValueError("target_thermo_pair (fwd_idx, rev_idx) must be provided for Kc or Kp")
                thermo_data = new_sys.calculate_thermodynamics(self.target_thermo_pair[0], self.target_thermo_pair[1])
                if thermo_data and thermo_data[self.y_variable] is not None:
                    y_array[i] = thermo_data[self.y_variable][-1] # get final equilibrium value
                else:
                    y_array[i] = np.nan

            elif self.y_variable == 'rxn_yield':
                if not all([self.yield_product, self.yield_reactant, self.yield_ratio]):
                    raise ValueError("yield_product, yield_reactant, and yield_ratio must be provided for rxn_yield")
                
                n_initial = new_sys.initial_moles.get(self.yield_reactant, 0.0)
                n_final = new_sys.results['final_moles'].get(self.yield_product, 0.0)
                n_initial_prod = new_sys.initial_moles.get(self.yield_product, 0.0)
                
                n_theoretical_produced = n_initial * self.yield_ratio
                n_actual_produced = max(0.0, n_final - n_initial_prod)
                
                if n_theoretical_produced > 1e-12:
                    y_array[i] = (n_actual_produced / n_theoretical_produced) * 100.0
                else:
                    y_array[i] = 0.0

            else: # invalid y_variable type
                raise ValueError(f"Unknown y_variable '{self.y_variable}'.")
        
        # calculate sensitivity only for valid data that is not NaN
        valid_mask = ~np.isnan(y_array)
        x_valid = x_array[valid_mask]
        y_valid = y_array[valid_mask]

        sensitivity_array = np.zeros_like(y_array)
        sensitivity_array.fill(np.nan)
        if len(x_valid) > 1:
            sensitivity_array[valid_mask] = np.gradient(y_valid, x_valid)

        self.results = {
            'x_label': self.x_variable, 'y_label': self.y_variable,
            'x': x_array, 'y': y_array, 'sensitivity': sensitivity_array
        }
        
        return self.results
    
    def adaptive_grid_refinement(self):
        """
        This is a future development. For now, it will just return a set number. A proper algorithm will be developed.
        Returns the number of points for the analysis.
        """

        # take 3 adjacent points (x1, y1), (x2, y2), (x3, y3), calculate where y2 should be on the straight line passing through y1 and y3
        # (linear interpolation)
        # if actual y2 is vastly different from the interpolated y2, it means the graph has high curvature and is highly non-linear
        # use this to find where more simulation points should be taken

        return 100
    
    def transform_to_vant_hoff_arrhenius_plot(self):
        """
        This function transforms the graph to a Van 't Hoff or Arrhenius plot (if applicable).
        """