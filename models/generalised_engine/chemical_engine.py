import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Dict, Set

# physical constants...
R_J_MOL_K = 8.314 # universal gas constant in J K^-1 mol^-1
R_ATM_L = 0.08206 # universal gas constant in L atm K^-1 mol^-1, for pressure calculations

class ChemicalSpecies:
    """A data class representing a specific chemical entity.
    
    Attributes:
        name (str): the unique identifier for the species (e.g., 'NO', 'H2O').
        vdw_a (float): Van der Waals 'a' constant (dm^6 atm mol^-2).
        vdw_b (float): Van der Waals 'b' constant (dm^3 mol^-1).
        species_type (str): the type of the species. Either 'reactant' (default) or 'pool' (concentration is fixed)."""
    
    def __init__(self, name, vdw_a=0.0, vdw_b=0.0, species_type: str='reactant', phase: str='gas', density: float=1000.0, molar_mass: float=18.015):
        self.name = name
        self.vdw_a = vdw_a
        self.vdw_b = vdw_b

        if species_type not in ['reactant', 'pool']:
            raise ValueError(f"species_type must either be 'reactant' or 'pool' for species {self.name}.")
        self.species_type = species_type

        self.phase = phase.lower()
        if self.phase not in ['gas', 'liquid', 'solid', 'aqueous', 'solvent']:
             raise ValueError(f"Phase must be 'gas', 'liquid', 'solid', 'aqueous', or 'solvent' for species {self.name}.")

        self.density = density
        self.molar_mass = molar_mass

    def __repr__(self):
        return f"Species('{self.name}', phase='{self.phase}')"

class Reaction:
    """Represents a single, elementary reaction step (uni-directional).
    To model a reversible equilibrium, instantiate two Reaction objects: one for forward, one for reverse."""
    
    def __init__(self, reactants: Dict[str, int], products: Dict[str, int], A: float, Ea: float) -> None:
        self.reactants = reactants
        self.products = products
        self.A = A
        self.Ea = Ea

    def get_rate_constant(self, T: float) -> float:
        """Calculates k = A * exp(-Ea / RT). Handles both scalar floats and numpy arrays safely."""
        # first, convert to array (even if scalar) to use numpy features...
        T_vals = np.asarray(T)
        
        # vectorised calculation with safety checks...
        # np.errstate ignores division by zero warnings if T=0 (we fix it in step 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            k = self.A * np.exp(-self.Ea / (R_J_MOL_K * T_vals))
            
        # clean up invalid values (T <= 0)...
        k = np.where(T_vals <= 0, 0.0, k)
        
        if T_vals.ndim == 0:
            return float(k)
        return k

    def calculate_rate(self, concentrations: Dict[str, float], T: float) -> float:
        """Calculates rate = k(T) * product([conc]^order)"""
        k = self.get_rate_constant(T)
        rate = k
        for species_name, order in self.reactants.items():
            conc = concentrations.get(species_name, 0.0)
            rate *= (conc ** order)
        return rate

    def __repr__(self):
        # return a readable string representation of the reaction by formatting reactants and products dictionaries as...
        # ..."coefficients + species" joined with "+" and separated by "->"
        def fmt(coeffs):
            parts = []
            for species, num in coeffs.items():
                prefix = f"{num}" if num != 1 else ""  # "1A" would simply be "A" (coefficient is omitted)
                parts.append(f"{prefix}{species}")
            return " + ".join(parts)

        return f"{fmt(self.reactants)} -> {fmt(self.products)}"

def find_all_equilibrium_pairs(reactions: List[Reaction]) -> List[Tuple[int, int]]:
    """
    Scans a list of Reaction objects and identifies all pairs that are direct reverses of each other. Returns an empty list if no pairs are found.
    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple contains the indices (forward, reverse) of a detected equilibrium pair.
    """
    num_reactions = len(reactions)
    paired_indices = set()
    equilibrium_pairs = []

    # iterate through each reaction as a potential 'forward' step
    for i in range(num_reactions):
        # if this reaction has already been paired, skip it
        if i in paired_indices:
            continue
        r1 = reactions[i]

        # look for its reverse in the rest of the list
        for j in range(i + 1, num_reactions):
            # if this potential partner is already paired, skip it
            if j in paired_indices:
                continue
            r2 = reactions[j]

            # the core logic: check if reactants/products are swapped
            if r1.reactants == r2.products and r1.products == r2.reactants:
                # pair found!
                equilibrium_pairs.append((i, j))
                
                # mark both as paired so they aren't used again
                paired_indices.add(i)
                paired_indices.add(j)
                
                # since we found the partner for reaction 'i', we can stop searching for it
                break 
    return equilibrium_pairs

class ChemicalSystem:
    """The central engine that models the state of a chemical system.
    It holds the species, the reactions, and the current physical conditions (moles, V, T).
    Its primary job is to compute the derivative (dn/dt) for every species by summing the contributions of all reactions in the network."""
    
    def __init__(self, species_list: List[ChemicalSpecies], reaction_list: List[Reaction], initial_moles: Dict[str, float], initial_V: float, initial_T: float, method: str='Radau', rtol:float=1e-6, atol:float=1e-9) -> None:
        self.species_list = species_list
        self.reactions = reaction_list
        self.initial_moles = initial_moles
        self.V = initial_V
        self.T = initial_T

        # if a species is a solvent and has 0 initial moles, calculate them: n = V * (rho / MM)
        for s in self.species_list:
            if s.phase == 'solvent':
                s.species_type = 'pool'
                if self.initial_moles.get(s.name, 0.0) == 0.0:
                    solvent_moles = self.V * (s.density / s.molar_mass)
                    self.initial_moles[s.name] = solvent_moles

        # store solver settings
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.species_names = {s.name for s in species_list}
        self.species_order = sorted(list(self.species_names))
        self.equilibrium_pairs = find_all_equilibrium_pairs(self.reactions)

        self.pool_species_names = {s.name for s in species_list if s.species_type == 'pool'}
        self.reactant_species_names = self.species_names - self.pool_species_names

        # the ODE solver should only solve for the non-pool, reacting species
        self.reactant_species_order = sorted(list(self.reactant_species_names))

        # validation checks:
        self._validate_reactions()
        for name in initial_moles:
            if name not in self.species_names:
                raise ValueError(f"Unknown species '{name}' found in initial_moles.")
                
        self._build_stoichiometry_matrix()

        # placeholders for results
        self.solution = None
        self.results = None
                
    def _validate_reactions(self) -> None:
        """Internal method to ensure reaction definitions match the species list."""
        for i, reaction in enumerate(self.reactions): # loop through each reaction
            for r_name in reaction.reactants: # check reactants
                if r_name not in self.species_names:
                    raise ValueError(f"Reaction #{i+1} contains unknown reactant: '{r_name}'")
            for p_name in reaction.products: # check products
                if p_name not in self.species_names:
                    raise ValueError(f"Reaction #{i+1} contains unknown product: '{p_name}'")
    
    def _build_stoichiometry_matrix(self) -> None:
        """Constructs a (num_species x num_reactions) matrix 'S'. S[i, j] is the coefficient of Species i in Reaction j."""
        
        # counting how big our grid is going to be..
        num_species = len(self.reactant_species_order) # a row for every reactant species...
        num_reactions = len(self.reactions) #... and a column for every reaction

        # create a zero-filled matrix (initialise empty matrix)
        self.S_matrix = np.zeros((num_species, num_reactions))
        # create a lookup map (links name of species to the row number index)
        species_index_map = {name: i for i, name in enumerate(self.reactant_species_order)}

        for j, reaction in enumerate(self.reactions): # loop through every reaction, 'j' is the column index (0 for reaction 1, 1 for reaction 2....)
            # reactants (being consumed, negative coefficient)
            for reactant, coeff in reaction.reactants.items():
                if reactant in species_index_map: # check if it's a reacting species
                    row_idx = species_index_map[reactant] # find corect row for this species
                    self.S_matrix[row_idx, j] -= coeff
            
            # products (being formed, positive coefficient)
            for product, coeff in reaction.products.items():
                if product in species_index_map: 
                    row_idx = species_index_map[product]
                    self.S_matrix[row_idx, j] += coeff

        # self.S_matrix is now a completed mathematical map of the entire chemical mechanism

    def _state_to_array (self, moles_dict: Dict[str, float]) -> np.ndarray:
        """Converts a {Species: Moles} dictionary to a [Moles] numpy array."""
        return np.array([moles_dict.get(s, 0.0) for s in self.reactant_species_order])

    def _array_to_state_dict (self, moles_array: np.ndarray) -> Dict[str, float]:
        """Converts a [Moles] numpy array to a {Species: Moles} dictionary."""
        return {s: moles_array[i] for i, s in enumerate(self.reactant_species_order)}

    def _get_characteristic_timescales(self) -> list[float]:
        """Analyses the reaction list to estimate the characteristic timescales."""
        T0 = self.T
        # calculate initial concentrations
        initial_concentrations = {name: n / self.V for name, n in self.initial_moles.items()}
        
        timescales = []
        for reaction in self.reactions:
            # calculate the initial rate of this specific reaction
            initial_rate = reaction.calculate_rate(initial_concentrations, T0)
            if initial_rate < 1e-20:
                timescales.append(1e12) # if the reaction is inactive at t=0, it has a very long timescale.
                continue
            # calculate the timescale for each reactant involved in this reaction
            for reactant_name in reaction.reactants:
                reactant_conc = initial_concentrations.get(reactant_name, 0.0)
                if reactant_conc > 1e-20:
                    timescale = reactant_conc / initial_rate # = [reactant]/rate
                    timescales.append(timescale)

        # teturn a sorted list of unique timescales (with sensible bounds)
        return sorted(list(set(timescales + [1e-12, 1e9])))

    def _package_results(self, t_arr: np.ndarray, y_arr: np.ndarray, V_arr: np.ndarray, T_arr: np.ndarray) -> dict:
        """Takes raw simulation output and packages it into the standard results dictionary."""
        y_arr = np.maximum(y_arr, 0.0)

        species_data = {}
        for idx, name in enumerate(self.reactant_species_order):
            species_data[name] = y_arr[idx]

        num_time_points = len(t_arr)
        for name in self.pool_species_names:
            initial_moles = self.initial_moles.get(name, 0.0)
            species_data[name] = np.full(num_time_points, initial_moles)
            
        P_real = self.calculate_pressure_array(species_data, V_arr, T_arr)
        
        n_total = sum(species_data.values())
        P_ideal = (n_total * R_ATM_L * T_arr) / V_arr
        
        final_moles_dict = {name: arr[-1] for name, arr in species_data.items()}
        
        return {
            'time': t_arr, 'species_data': species_data,
            'volume': V_arr, 'temperature': T_arr, 'final_moles': final_moles_dict,
            'info': {'V': V_arr[-1], 'T': T_arr[-1]},
            'P_real': P_real, 'P_ideal': P_ideal}
    
    def calculate_reaction_rates_over_time(self, external_results: Optional[dict]=None) -> Dict[str, np.ndarray]:
        """Calculates the instantaneous rate of each individual reaction step over the  entire course of the simulation.
        Can accept external_results (from a perturbation) to calculate rates for stitched timelines."""

        if external_results is not None:
            results = external_results
            print(f"DEBUG: Calculating rates using external results (time points: {len(results['time'])})")
        else:
            results = self.results
            print(f"DEBUG: Calculating rates using internal results (time points: {len(results['time'])})")
        if not results:
            return {}
        
        # get NumPy arrays for volume and temperature over time from the results
        vol_arr = results.get('volume', np.full_like(results['time'], self.V))
        temp_arr = results.get('temperature', np.full_like(results['time'], self.T))
        
        # calculate concentration arrays (n/V) for all species
        # this creates a dictionary of {'SpeciesName': [conc_t1, conc_t2, ...], ...}
        conc_data = {name: arr / vol_arr for name, arr in results['species_data'].items()}
        
        reaction_rates_dict = {}
        for i, reaction in enumerate(self.reactions):
            rate_array = reaction.calculate_rate(conc_data, temp_arr)
            reaction_rates_dict[f"R{i+1}"] = rate_array
            
        return reaction_rates_dict
    
    def get_net_rates(self, current_moles: Dict[str, float], V: float, T: float) -> Dict[str, float]:
        """Calculates the net rate of change (dn/dt) for the species in the system using matrix multiplication."""
        
        concentrations = {
            name: n / V 
            for name, n in current_moles.items()} # loop through each species
        
        # build the rate vector (r) for every reaction
        rate_vector = np.zeros(len(self.reactions))
        
        for j, reaction in enumerate(self.reactions):
            # use the standard rate calculation: k * product([C]^order)
            rate_vector[j] = reaction.calculate_rate(concentrations, T)

        # matrix multiplication
        # S (rows=reactants_only, cols=rxns) @ r (rxns) -> net_change (reactants_only)
        net_change_vector = self.S_matrix @ rate_vector
        
        dndt_vector = net_change_vector * V
        
        # map back to dictionary
        rates_dict = {
            self.reactant_species_order[i]: dndt_vector[i] 
            for i in range(len(self.reactant_species_order))
        }

        for name in self.pool_species_names:
            rates_dict[name] = 0.0

        return rates_dict

    def _ode_system_adapter(self, t: float, y: np.ndarray) -> np.ndarray:
        # safety clamp: prevent negative moles creating math errors...
        y = np.maximum(y, 0)
        full_moles_dict = {
            # first, get the moles of the reacting species from the solver's state vector 'y'
            **self._array_to_state_dict(y), 
            # then add the fixed moles of the pool chemicals from the initial conditions.
            **{name: self.initial_moles[name] for name in self.pool_species_names}}
        
        # calculate rates...
        # note: we use the system's current V and T (these are constant unless perturbed externally)
        rates_dict = self.get_net_rates(full_moles_dict, self.V, self.T)
        
        return self._state_to_array(rates_dict) # convert dict -> array...

    def run_simulation(self, t_end: Optional[float] = None, rate_tolerance: float=1e-7, max_iterations: int=50) -> None:
        """Runs the simulation using an iterative, adaptive timescale. Stops when the maximum net rate falls below rate_tolerance."""
        # set up the initial state vector and parameters for the iterative loop.
        y0 = self._state_to_array(self.initial_moles)

        if t_end is not None: # simulation runs for a fixed duration if t_end is provided
            print(f"Starting fixed-time simulation at T={self.T}K for {t_end}s...")
            # run single shot simulation without convergence checks
            solution = solve_ivp(
                fun=self._ode_system_adapter, t_span=(0, t_end), y0=y0, dense_output=True,
                method=self.method, rtol=self.rtol, atol=self.atol)
            
            self._process_final_results([solution])
            return
        
        # if t_end is not provided, it runs until rate convergence
        solutions_list = []
        time_offset = 0.0
        current_chunk_duration = 0.1  # start with a small time window for fast initial kinetics
        equilibrium_found = False

        # build one master list of time points that ensures high resolution during all critical phases of the reaction...
        # ...to prevent the solver from skipping over fast events
        timescales = self._get_characteristic_timescales()

        # solve the system in segments, checking for equilibrium after each one (the iterative "chunking" loop)
        for i in range(max_iterations):
            t_span = (time_offset, time_offset + current_chunk_duration)
            
            chunk_solution = solve_ivp( # call solver
                fun=self._ode_system_adapter, t_span=t_span, y0=y0, dense_output=True,
                method=self.method, rtol=self.rtol, atol=self.atol)
            
            solutions_list.append(chunk_solution)

            # update the state for the start of the next chunk
            y_last = chunk_solution.y[:, -1]
            y0 = y_last
            time_offset = chunk_solution.t[-1]

            # after each chunk, check if the reaction rates have slowed to the tolerance.
            final_moles = self._array_to_state_dict(y_last)
            final_rates = self.get_net_rates(final_moles, self.V, self.T)
            max_rate = max(abs(v) for v in final_rates.values())

            if max_rate < rate_tolerance:
                # force the solver to run at least 2 chunks before deciding it's done.
                if i >= 2:
                    equilibrium_found = True
                    self.equilibrium_time = time_offset
                    break  
                else:
                    #if rates are low but we just started, expand gently
                    current_chunk_duration *= 2.0
            else:
                # if not at equilibrium, expand the next time chunk
                current_chunk_duration *= 10.0

        if not equilibrium_found:
            print(f"Warning: Equilibrium not reached after {time_offset:.2f}s (Max Iterations).")

        # stitch all the solution chunks together into a single, unified results object
        self._process_final_results(solutions_list)

    def calculate_pressure_array(self, moles_data: Dict[str, np.ndarray], V_array: np.ndarray, T_array: np.ndarray) -> np.ndarray:
        gas_species = [s for s in self.species_list if s.phase == 'gas']
        
        # calculate n_total ONLY for gases
        n_gas_total = np.zeros_like(V_array)
        for s in gas_species:
            n_gas_total += moles_data.get(s.name, 0.0)
            
        # mixing rules
        sqrt_a_sum = np.zeros_like(V_array)
        b_sum = np.zeros_like(V_array)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            for s_obj in gas_species:
                n_i = moles_data[s_obj.name]
                x_i = np.divide(n_i, n_gas_total, where=n_gas_total!=0) # mole fraction
                
                sqrt_a_sum += x_i * np.sqrt(s_obj.vdw_a)
                b_sum += x_i * s_obj.vdw_b
                
        a_mix = sqrt_a_sum ** 2
        b_mix = b_sum
        
        nb = n_gas_total * b_mix
        V_free = V_array - nb
        
        # safety clamp
        V_free = np.maximum(V_free, 1e-6) 
        
        # calculate P based on n_gas_total
        P_ideal = (n_gas_total * R_ATM_L * T_array) / V_free
        P_real = P_ideal - ((a_mix * n_gas_total**2) / (V_array**2))
        
        # if no gas is present, pressure is 0 (ignoring vapour pressure of solids/liquids for now)
        no_gas_mask = (n_gas_total == 0)
        P_ideal[no_gas_mask] = 0.0
        P_real[no_gas_mask] = 0.0
        
        return P_real

    def _process_final_results(self, solutions_list: list) -> None:
        combined_t = np.concatenate([s.t for s in solutions_list])
        combined_y = np.concatenate([s.y for s in solutions_list], axis=1)
        
        V_array = np.full_like(combined_t, self.V)
        T_array = np.full_like(combined_t, self.T)
        
        self.results = self._package_results(combined_t, combined_y, V_array, T_array)
    
    def get_initial_rates(self) -> List[float]:
        """Calculates the rate of every reaction based on the system's initial state."""
        # ensure V is not zero to prevent division errors
        if self.V <= 1e-12:return [0.0] * len(self.reactions)

        initial_concentrations = {name: n / self.V for name, n in self.initial_moles.items() if name in self.species_names}
        initial_rates = [r.calculate_rate(initial_concentrations, self.T) for r in self.reactions]

        return initial_rates
    
    def _calculate_mass_action_ratio(self, reaction: Reaction, activity_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Helper used for both Qc (using concentrations) and Qp (using partial pressures)."""
        first_val = next(iter(activity_data.values()))
        numerator = np.ones_like(first_val)
        denominator = np.ones_like(first_val)

        for species, coeff in reaction.products.items(): # calculate numerator (products)
            a_i = activity_data.get(species, 0.0)
            numerator *= (a_i ** coeff)
        for species, coeff in reaction.reactants.items(): # calculate denominator (products)
            a_i = activity_data.get(species, 0.0)
            denominator *= (a_i ** coeff)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20)
    
    def find_intermediates(self) -> Set[str]:
        """This identifies species that act as steady-state intermediates, exluding pool species.
        The logic is: a species is an intermediate if it is produced from an equilibrium and is the reactant of a non-equilibrium 'drain' reaction."""
        intermediates = set()
        
        # identify all reactions involved in equilibria
        eq_indices = {idx for pair in self.equilibrium_pairs for idx in pair}
        
        for fwd_idx, rev_idx in self.equilibrium_pairs:
            fwd_reaction = self.reactions[fwd_idx]
            
            # candidates are products of the forward equilibrium step...
            potential_candidates = set(fwd_reaction.products.keys())
            # exclude pool species (concentrations are fixed)
            potential_candidates -= self.pool_species_names
            
            for i, reaction in enumerate(self.reactions):
                if i in eq_indices: continue # skip other equilibrium steps
                
                # check if candidate is consumed in a non-equilibrium step
                confirmed = potential_candidates.intersection(reaction.reactants.keys())
                intermediates.update(confirmed)
                
        return intermediates

    def calculate_thermodynamics(self, fwd_idx: int, rev_idx: int, external_results: Optional[dict]=None) -> Optional[Dict[str, np.ndarray]]:
        """Calculates Qc/Kc (concentration) and Qp/Kp (pressure)."""
        # get reaction objects...
        r_fwd = self.reactions[fwd_idx]
        r_rev = self.reactions[rev_idx]
        
        data = external_results if external_results else self.results # select data source
        if data is None: return None
        
        T_arr = data.get('temperature', np.full_like(data['time'], self.T))
        V_arr = data.get('volume', np.full_like(data['time'], self.V))
        P_real = data.get('P_real', np.zeros_like(T_arr))
        species_data = data['species_data']
        
        # calculate Kc (as ratio of the rate constants)
        k_f = r_fwd.get_rate_constant(T_arr)
        k_r = r_rev.get_rate_constant(T_arr)
        Kc = np.divide(k_f, k_r, out=np.zeros_like(k_f), where=k_r!=0)
        
        # create an activity dictionary based on concentration (n/V)...
        conc_activities = {}
        solvent_names = {s.name for s in self.species_list if s.phase == 'solvent'}
        pure_names = {s.name for s in self.species_list if s.phase in ['solid', 'liquid']}
        
        for name in self.species_names:
            if name in solvent_names or name in pure_names:
                # unity activity for solvents and pure phases
                conc_activities[name] = np.ones_like(T_arr)
            else:
                # molar conc for solutes/gases
                conc_activities[name] = species_data[name] / V_arr

        Qc = self._calculate_mass_action_ratio(r_fwd, conc_activities)

        rxn_species_names = list(r_fwd.reactants.keys()) + list(r_fwd.products.keys())
        gas_involved = False
        for name in rxn_species_names:
            s_obj = next((s for s in self.species_list if s.name == name), None)
            if s_obj and s_obj.phase == 'gas':
                gas_involved = True
                break
        
        if not gas_involved:
            # if no gases, Kp and Qp are physically meaningless for this reaction...
            return {'Qc': Qc, 'Kc': Kc, 'Qp': None, 'Kp': None}

        # gas calculations
        moles_prod_gas = sum(coeff for name, coeff in r_fwd.products.items() 
                             if getattr(next((s for s in self.species_list if s.name == name), None), 'phase', 'gas') == 'gas')
        moles_reac_gas = sum(coeff for name, coeff in r_fwd.reactants.items() 
                             if getattr(next((s for s in self.species_list if s.name == name), None), 'phase', 'gas') == 'gas')
        delta_n_gas = moles_prod_gas - moles_reac_gas
        
        Kp = Kc * (R_ATM_L * T_arr)**delta_n_gas
        
        # Qp Calculation (partial pressures)
        n_gas_total = np.zeros_like(T_arr)
        gas_names = {s.name for s in self.species_list if s.phase == 'gas'}
        for name in gas_names:
            n_gas_total += species_data.get(name, 0)
            
        pressure_activities = {}
        with np.errstate(divide='ignore', invalid='ignore'):
            for name in self.species_names:
                if name in gas_names:
                    x_i = np.divide(species_data[name], n_gas_total, out=np.zeros_like(n_gas_total), where=n_gas_total!=0)
                    pressure_activities[name] = x_i * P_real
                else:
                    # non-gas species do not contribute to Qp
                    pressure_activities[name] = np.ones_like(P_real)
                
        Qp = self._calculate_mass_action_ratio(r_fwd, pressure_activities)
        Qp = np.maximum(Qp, 1e-20)
        
        return {'Qc': Qc, 'Kc': Kc, 'Qp': Qp, 'Kp': Kp}

    def calculate_yield(self, product_name: str, limiting_reactant: str, reaction_stoich_ratio: float) -> None:
        """
        Calculates and prints the reaction yield.
        reaction_stoich_ratio = (coeff_product / coeff_reactant) from the balanced equation.
        """
        if not self.results:
            print("Error: No results found. Run simulation first.")
            return
            
        n_initial = self.initial_moles.get(limiting_reactant, 0.0)
        n_final = self.results['final_moles'].get(product_name, 0.0)
        n_initial_prod = self.initial_moles.get(product_name, 0.0)
        
        # theoretical max produced = initial reactant * ratio
        n_theoretical_produced = n_initial * reaction_stoich_ratio
        
        # actual produced = final - initial (in case product started non-zero)
        n_actual_produced = n_final - n_initial_prod
        
        if n_theoretical_produced <= 1e-12:
            print("Theoretical yield is zero.")
            return

        percent_yield = (n_actual_produced / n_theoretical_produced) * 100.0
        
        print(f"Yield             :({product_name})")
        print(f"Limiting Reactant : {limiting_reactant} ({n_initial:.4f} mol)")
        print(f"Theoretical Max   : {n_theoretical_produced:.4f} mol")
        print(f"Actual Produced   : {n_actual_produced:.4f} mol")
        print(f"Percent Yield     : {percent_yield:.2f}%")
    
class PerturbationSimulation:
    """Orchestrates a 3-stage perturbation simulation for a ChemicalSystem.
    Handles volume ramps, temperature ramps, and species injections generically."""
    
    def __init__(self, baseline_system: ChemicalSystem, perturbation_type, t_start: float, t_end: float, 
                 new_V: Optional[float]=None, new_T: Optional[float]=None, injection_rates: Optional[Dict[str, float]]=None, t_global_end: Optional[float]=None) -> None:
        self.baseline_system = baseline_system
        self.ptype = perturbation_type
        self.t_start = t_start # start of the perturbation
        self.t_end = t_end  # end of the perturbation
        self.t_global_end = t_global_end
        
        # perturbation targets
        self.target_V = new_V if new_V is not None else self.baseline_system.V
        self.target_T = new_T if new_T is not None else self.baseline_system.T
        self.injection_rates = injection_rates if injection_rates is not None else {}
        
        self.results = None
    
    def _ode_adapter_perturbation(self, t: float, y: np.ndarray) -> np.ndarray:
        y = np.maximum(y, 0)

        current_moles = self.baseline_system._array_to_state_dict(y) # reconstruct reacting species state

        pool_moles = {} # pool species
        for name in self.baseline_system.pool_species_names:
            n_0 = self.baseline_system.initial_moles[name]
            
            # logic: if this pool species is being injected, increase its moles linearly over time
            if self.ptype == 'injection' and name in self.injection_rates:
                # n(t) = n0 + rate * duration_elapsed
                # clamp duration to 0 to avoid issues if solver tries t < t_start
                dt = max(0.0, t - self.t_start) 
                n_0 += self.injection_rates[name] * dt
                
            pool_moles[name] = n_0

        full_moles_dict = {**current_moles, **pool_moles}

        # interpolate environmental conditions
        current_V = self.baseline_system.V
        current_T = self.baseline_system.T

        if self.ptype == 'volume':
            current_V = np.interp(t, [self.t_start, self.t_end], [self.baseline_system.V, self.target_V])
        elif self.ptype == 'temperature':
            current_T = np.interp(t, [self.t_start, self.t_end], [self.baseline_system.T, self.target_T])
        
        # get rates
        dndt_dict = self.baseline_system.get_net_rates(full_moles_dict, current_V, current_T)
        
        # add injections for reacting species
        if self.ptype == 'injection':
            for species, rate in self.injection_rates.items():
                if species in dndt_dict and species not in self.baseline_system.pool_species_names:
                    dndt_dict[species] += rate
                
        return self.baseline_system._state_to_array(dndt_dict)

    def run_perturbation(self) -> None:
        """Orchestrates the 3-stage perturbation."""
        # stage 1: before the perturbation
        base_res = self.baseline_system.results
        # find the index in the baseline time array closest to t_start
        idx = np.argmin(np.abs(base_res['time'] - self.t_start))
        
        # slice the history data up to that index
        s1_time = base_res['time'][:idx]
        s1_data = {s: arr[:idx] for s, arr in base_res['species_data'].items()}
        
        # the last state of stage 1 becomes the initial state of stage 2
        y0_stage2 = np.array([s1_data[s][-1] for s in self.baseline_system.reactant_species_order])
        
        # stage 2: the perturbation/stress
        t_span = (self.t_start, self.t_end)
        
        fun = self._ode_adapter_perturbation
        final_V = self.target_V if self.ptype == 'volume' else self.baseline_system.V
        final_T = self.target_T if self.ptype == 'temperature' else self.baseline_system.T

        # run the solver for the stress period
        sol = solve_ivp(fun, t_span, y0_stage2, dense_output=True, method=self.baseline_system.method, rtol=self.baseline_system.rtol, atol=self.baseline_system.atol)

        if not sol.success and sol.status == -1: # crash detection
            print(f"ERROR: Solver crashed during perturbation (stage 2).")
            print(f"Reason: {sol.message}")
            print("Injection rate might be too high, or the reaction too stiff. Try reducing the rate.")
            return # Stop here, do not overwrite results with garbage
        
        # generate time points for stage 2
        if sol.t.size > 1:
            s2_time = np.linspace(self.t_start, self.t_end, min(200, len(sol.t) * 2))
        else:
            s2_time = np.array([self.t_start, self.t_end])

        try:
            s2_y = sol.sol(s2_time)
        except Exception as e:
            print(f"ERROR: Stage 2 interpolation failed: {e}")
            return

        s2_data_reactants = {s: s2_y[i] for i, s in enumerate(self.baseline_system.reactant_species_order)}
        s2_data = {**s2_data_reactants}
        for name in self.baseline_system.pool_species_names:
            initial_n = self.baseline_system.initial_moles[name]
            
            if self.ptype == 'injection' and name in self.injection_rates:
                # Create the linear ramp array for the graph
                # n_t = n_0 + rate * (t_array - t_start)
                ramp = initial_n + self.injection_rates[name] * (s2_time - self.t_start)
                s2_data[name] = ramp
            else:
                s2_data[name] = np.full_like(s2_time, initial_n)
        
        # stage 3: after the perturbation (relaxation)
        y0_stage3_dict = {s: arr[-1] for s, arr in s2_data.items()} # initial state of Stage 3 is the end of Stage 2

        new_sys = ChemicalSystem( # create a new system object for the new conditions
            self.baseline_system.species_list, self.baseline_system.reactions,
            y0_stage3_dict, final_V, final_T, method=self.baseline_system.method, rtol=self.baseline_system.rtol, atol=self.baseline_system.atol)
        
        # calculation of relaxation phase duration
        if self.t_global_end is not None and self.t_global_end > self.t_end:
            s3_duration = self.t_global_end - self.t_end
            new_sys.run_simulation(s3_duration) 
        else: # default
            new_sys.run_simulation() # run until reaction is complete
        
        s3_res = new_sys.results
        s3_time = s3_res['time'] + self.t_end
        s3_data = s3_res['species_data']
        
        # stitching and data sanitisation
        raw_time = np.concatenate([s1_time[:-1], s2_time[:-1], s3_time])
        
        raw_species = {}
        for s in self.baseline_system.species_order:
            raw_species[s] = np.concatenate([
                s1_data[s][:-1], s2_data[s][:-1], s3_data[s]])
            
        # stitch volume array
        v1 = np.full_like(s1_time[:-1], self.baseline_system.V)
        if self.ptype == 'volume':
            v2 = np.interp(s2_time[:-1], [self.t_start, self.t_end], [self.baseline_system.V, self.target_V])
        else:
            v2 = np.full_like(s2_time[:-1], self.baseline_system.V)
        v3 = np.full_like(s3_time, final_V)
        raw_vol = np.concatenate([v1, v2, v3])

        # stitch temperature array
        T1 = np.full_like(s1_time[:-1], self.baseline_system.T)
        if self.ptype == 'temperature':
            T2 = np.interp(s2_time[:-1], [self.t_start, self.t_end], [self.baseline_system.T, self.target_T])
        else:
            T2 = np.full_like(s2_time[:-1], self.baseline_system.T)
        T3 = np.full_like(s3_time, final_T)
        raw_temp = np.concatenate([T1, T2, T3])

        # data sanitisation (removing any duplicates)
        if len(raw_time) == 0:
            print("ERROR: No data points in stitched timeline")
            return
            
        sort_idx = np.argsort(raw_time)
        t_sorted = raw_time[sort_idx]

        # create a mask to remove duplicate time points (dt > 1e-12, more permissive)
        if len(t_sorted) > 1:
            time_diffs = np.diff(t_sorted)
            keep_mask = np.concatenate(([True], time_diffs > 1e-12))
        else:
            keep_mask = np.array([True])
        # apply mask to all arrays (to ensure the shapes match)
        final_time = t_sorted[keep_mask]
        final_volume = raw_vol[sort_idx][keep_mask]
        final_temp = raw_temp[sort_idx][keep_mask]
        final_species = {s: arr[sort_idx][keep_mask] for s, arr in raw_species.items()}

        # additional safety check
        if len(final_time) == 0:
            print("ERROR: All data points were filtered out during sanitization")
            print(f"Time range: {raw_time[0]:.2e} to {raw_time[-1]:.2e}")
            print(f"Time diffs min: {np.min(time_diffs):.2e}, max: {np.max(time_diffs):.2e}")
            return

        # calculate pressure using the baseline system's engine
        final_P_real = self.baseline_system.calculate_pressure_array(final_species, final_volume, final_temp)
        
        # ideal pressure for comparison
        n_tot_final = sum(final_species.values())
        for arr in final_species.values(): n_tot_final += arr
        final_P_ideal = (n_tot_final * R_ATM_L * final_temp) / final_volume

        if self.t_global_end is not None and self.t_global_end < final_time[-1]:
            mask = final_time <= self.t_global_end
            final_time = final_time[mask]
            final_volume = final_volume[mask]
            final_temp = final_temp[mask]
            final_species = {s: arr[mask] for s, arr in final_species.items()}
            final_P_real = final_P_real[mask]
            final_P_ideal = final_P_ideal[mask]

        self.results = {'time': final_time, 'species_data': final_species, 'volume': final_volume, 'temperature': final_temp,
            'P_real': final_P_real, 'P_ideal': final_P_ideal,
            'final_moles': s3_res['final_moles'], 'info': {'V': final_V, 'T': final_T}}

        # calculate reaction rates for the stitched timeline
        self.results['reaction_rates'] = self.baseline_system.calculate_reaction_rates_over_time(external_results=self.results)