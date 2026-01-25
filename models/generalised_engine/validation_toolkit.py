from chemical_engine import ChemicalSystem, ChemicalSpecies, PerturbationSimulation, Reaction
import copy
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from scipy.integrate import solve_ivp
import warnings

# filter out overflow warnings from the solver (in case we're running oscillating reactions)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.integrate")

def diagnose_and_verify_steady_state(system: ChemicalSystem):
    """
    Automatically diagnoses a chemical system for steady-state mechanisms and verifies their mathematical accuracy.

    The function operates in three stages:
    1.  Detects all reversible equilibrium pairs and all irreversible "drain" reactions.
    2.  Identifies valid steady-state scenarios where an intermediate from an equilibrium is consumed by a drain reaction.
    3.  For each detected scenario, it calculates the theoretical Qc/Kc ratio based on the steady-state assumption and compares it to the final state of the simulation.
    """
    print("Verifying the steady state...")
    
    # 1) identify all the reactions and equilibrium pairs (categorise every reaction)
    reactions = system.reactions
    all_indices = set(range(len(reactions)))
    
    equilibrium_pairs = system.equilibrium_pairs
    # get the set of all indices involved in an equilibrium
    paired_indices = {idx for pair in equilibrium_pairs for idx in pair}
    # any reaction not in an equilibrium is considered a potential "drain"
    unpaired_indices = all_indices - paired_indices

    # 2) detect steady-state scenarios
    steady_state_scenarios = []
    
    # now we look for a species that is produced by an equilibrium and consumed by a drain
    for fwd_idx, rev_idx in equilibrium_pairs:
        forward_reaction = reactions[fwd_idx]
        potential_intermediates = set(forward_reaction.products.keys())

        for drain_idx in unpaired_indices:
            drain_reaction = reactions[drain_idx]
            drain_reactants = set(drain_reaction.reactants.keys())
            
            # find the intersection: the species that is both a product of equilibrium and a reactant of the drain
            consumed_intermediates = potential_intermediates.intersection(drain_reactants)
            
            for intermediate in consumed_intermediates:
                scenario = {
                    'equilibrium_pair': (fwd_idx, rev_idx),
                    'intermediate': intermediate,
                    'drain_reaction_idx': drain_idx
                }
                steady_state_scenarios.append(scenario)

    # 3) perform verification and report findings
    if not steady_state_scenarios:
        print("\nDIAGNOSIS: No coupled equilibrium/drain mechanisms were found.")
        print("The system is either a simple equilibrium or a series of irreversible steps.")
        return

    print(f"\nDIAGNOSIS: Found {len(steady_state_scenarios)} potential steady-state scenario(s). Verifying each...")

    for i, scenario in enumerate(steady_state_scenarios):
        # extract the details for this specific scenario
        fwd_idx, rev_idx = scenario['equilibrium_pair']
        drain_idx = scenario['drain_reaction_idx']
        intermediate = scenario['intermediate']
        
        # part 1 of report
        print(f"\n--- Verifying Scenario #{i+1} ---")
        print(f"  Equilibrium : {reactions[fwd_idx]}")
        print(f"  Intermediate: {intermediate}")
        print(f"  Drain Step  : {reactions[drain_idx]}")
        print("  " + "-"*25)

        # verification calculations (apply the SSA formula)
        results = system.results
        T = results['info']['T']
        V = results['info']['V']
        final_moles = results['final_moles']

        reverse_reaction = reactions[rev_idx]
        drain_reaction = reactions[drain_idx]

        k_reverse = reverse_reaction.get_rate_constant(T)
        k_drain = drain_reaction.get_rate_constant(T)

        # calculate the full drain term: k_drain * [A]^a * [B]^b ...
        drain_rate_term = k_drain
        for reactant, order in drain_reaction.reactants.items():
            if reactant != intermediate:
                conc = final_moles.get(reactant, 0.0) / V
                drain_rate_term *= (conc ** order)
        
        # the core steady-state assumption formula
        denominator = k_reverse + drain_rate_term
        theoretical_ratio = k_reverse / denominator if denominator > 0 else 1.0

        thermo_data = system.calculate_thermodynamics(fwd_idx, rev_idx)
        if thermo_data and thermo_data['Kc'][-1] > 0:
            simulated_ratio = thermo_data['Qc'][-1] / thermo_data['Kc'][-1]
            discrepancy = abs(theoretical_ratio - simulated_ratio) / theoretical_ratio * 100 if theoretical_ratio > 0 else 0
        else:
            simulated_ratio = float('nan') # mark as Not a Number if data is invalid
            discrepancy = float('inf')

        # part 2 of report (present results of the verification)
        print(f"  Theoretical Qc/Kc Ratio: {theoretical_ratio:.6f}")
        print(f"  Simulated Qc/Kc Ratio  : {simulated_ratio:.6f}")
        print(f"  Discrepancy            : {discrepancy:.4f}%")

        if discrepancy < 1.0: # use a 1% tolerance
            print("  CONCLUSION: SUCCESS!")
        else:
            print("  CONCLUSION: FAILURE! A significant discrepancy exists between simulation and theory.")


def diagnose_real_gas_divergence(system: ChemicalSystem):
    """Compares the theoretical/ideal Kp to the observed/real Kp. The divergence indicates the magnitude of non-ideal gas behavior."""
    print("Real gas divergence...")
    
    # check if we have equilibrium pairs to test
    if not system.equilibrium_pairs:
        print("No equilibrium pairs found to diagnose.")
        return

    # use the first pair found for this test
    fwd_idx, rev_idx = system.equilibrium_pairs[0]
    
    # calculate thermodynamics using the engine
    thermo = system.calculate_thermodynamics(fwd_idx, rev_idx)
    if thermo is None:
        print("No results available to analyze.")
        return

    # extract final values
    Kp_ideal = thermo['Kp'][-1] # based on Kc * (RT)^dn
    Kp_observed = thermo['Qp'][-1] # based on actual partial pressures (P_real * x_i) at the end state
    
    P_real = system.results['P_real'][-1]
    P_ideal = system.results['P_ideal'][-1]
    Z = P_real / P_ideal if P_ideal > 0 else 1.0

    print(f"Reaction Pair: {system.reactions[fwd_idx]}")
    print(f"  System Pressure (Real)  : {P_real:.4f} atm")
    print(f"  System Pressure (Ideal) : {P_ideal:.4f} atm")
    print(f"  Compressibility Factor Z: {Z:.4f}")
    print("-" * 30)
    print(f"  Theoretical Kp (Ideal)  : {Kp_ideal:.4e}")
    print(f"  Observed Kp (Real Gas)  : {Kp_observed:.4e}")
    
    # calculate percent divergence
    divergence = abs(Kp_observed - Kp_ideal) / Kp_ideal * 100
    print(f"  Divergence              : {divergence:.4f}%")
    
    if divergence > 1.0:
        print("  CONCLUSION: Significant non-ideal behavior detected.")
    else:
        print("  CONCLUSION: System behaves ideally (or pressure is too low to notice effects).")
    print("-" * 30)


class SystemTester:
    """A professional-grade diagnostic framework for validating ChemicalSystem logic."""

    def __init__(self, system_template: ChemicalSystem, molar_mass_map: Dict[str, float]):
        self.template = system_template
        self.molar_mass_map = molar_mass_map

        # detect whether the system includes autocatalysis
        # if a species is a reactant AND appears in products with a higher coefficient (e.g., X -> 2X)...
        # ...the system has potential for explosion or oscillation.

        self.is_complex = False
        for r in self.template.reactions:
            for spec, coeff in r.reactants.items():
                if r.products.get(spec, 0) > coeff:
                    self.is_complex = True
                    break
        
        # complex systems will run for a fixed time (to prevent an infinite loop)
        self.sim_args = {'t_end': 500.0} if self.is_complex else {}

    def _get_fresh_instance(self, override_V: float = None, override_moles: Dict[str, float] = None) -> ChemicalSystem:
        """Helper to create a clean instance of the template system."""
        V = override_V if override_V is not None else self.template.V
        moles = override_moles if override_moles is not None else copy.deepcopy(self.template.initial_moles)
        return ChemicalSystem(
            self.template.species_list, self.template.reactions,
            moles, V, self.template.T)

    def test_mass_conservation(self) -> Tuple[bool, str]:
        """Verifies that the total mass (moles * molar_mass) remains constant to machine precision."""
        sys = self._get_fresh_instance()
        try:
            sys.run_simulation(**self.sim_args)
        except Exception as e:
            return False, f"CRITICAL: Solver crashed ({str(e)})."
        
        initial_mass = sum(sys.initial_moles[s] * self.molar_mass_map.get(s, 0.0) for s in sys.species_order)
        final_moles = sys.results['final_moles']
        final_mass = sum(final_moles[s] * self.molar_mass_map.get(s, 0.0) for s in sys.species_order)
        
        mass_diff = abs(final_mass - initial_mass)
        if mass_diff < 1e-7: # suitable tolerance
            return True, f"PASS: Mass conserved. Delta: {mass_diff:.2e} g."
        return False, f"FAIL: Mass conservation violation. Delta: {mass_diff:.2e} g."

    def test_kinetic_convergence(self) -> Tuple[bool, str]:
        """Ensures that the net rates (dn/dt) effectively reach zero unless it is an oscillating reaction."""
        sys = self._get_fresh_instance()
        try:
            sys.run_simulation(**self.sim_args)
        except Exception as e:
            return False, f"CRITICAL: Solver crashed ({str(e)})."
        
        # check rates
        final_rates = sys.get_net_rates(sys.results['final_moles'], sys.V, sys.T)
        max_rate = max(abs(r) for r in final_rates.values())
        
        if self.is_complex:
            # an oscillator must have peaks and troughs, we check if the derivative (slope) flips sign
            is_oscillating = False
            for data in sys.results['species_data'].values():
                # calculate gradient (slope) over time
                gradients = np.gradient(data)
                # count how many times the slope changes from positive to negative (or vice versa)
                # We need > 2 flips to confirm it's not just a single bump but a cycle
                sign_flips = np.count_nonzero(np.diff(np.sign(gradients)))
                if sign_flips > 2:
                    is_oscillating = True
                    break
            
            if is_oscillating:
                return True, f"PASS: System exhibits stable oscillation ({sign_flips} phase changes detected)."
            elif max_rate < 1e-6:
                return True, f"PASS: Autocatalytic system settled to steady state."
            else:
                return False, f"FAIL: System exploded or failed to form stable cycles (Max rate: {max_rate:.2e})."
            
        else:
            # standard convergence check for simple systems
            if max_rate < 1e-6:
                return True, f"PASS: System reached kinetic standstill (max rate: {max_rate:.2e})."
            return False, f"FAIL: System still active at simulation end (max rate: {max_rate:.2e})."

    def test_thermodynamic_equilibrium(self) -> Tuple[bool, str]:
        sys = self._get_fresh_instance()
        if not sys.equilibrium_pairs:
            return True, "SKIPPED: No reversible reactions defined in system."
        
        sys.run_simulation(**self.sim_args)
        
        # analyse the first equilibrium pair found..
        fwd_idx, rev_idx = sys.equilibrium_pairs[0]
        fwd_reaction = sys.reactions[fwd_idx]
        rev_reaction = sys.reactions[rev_idx]
        
        thermo = sys.calculate_thermodynamics(fwd_idx, rev_idx)
        final_Qc = thermo['Qc'][-1]
        final_Kc = thermo['Kc'][-1]
        
        # detect intermediate
        potential_intermediates = set(fwd_reaction.products.keys())
        drain_reaction = None
        target_intermediate = None

        eq_indices = {idx for pair in sys.equilibrium_pairs for idx in pair}
        
        for i, reaction in enumerate(sys.reactions):
            if i in eq_indices: continue
            # check if this reaction consumes any of the equilibrium products
            intersection = potential_intermediates.intersection(reaction.reactants.keys())
            if intersection:
                drain_reaction = reaction
                target_intermediate = list(intersection)[0] # analyse the first active intermediate found
                break
        
        # Case A: Steady State
        if drain_reaction:
            T = sys.T
            V = sys.results['info']['V']
            final_moles = sys.results['final_moles']
            conc_map = {name: n/V for name, n in final_moles.items()}

            # we must balance the rate of production vs consumption of the specific intermediate
            # production: forward reaction, consumption: reverse reaction + drain reaction
            nu_prod = fwd_reaction.products.get(target_intermediate, 0)      
            nu_cons_rev = rev_reaction.reactants.get(target_intermediate, 0)
            nu_cons_drain = drain_reaction.reactants.get(target_intermediate, 0)

            k_r = rev_reaction.get_rate_constant(T)
            r_rev = k_r
            for spec, order in rev_reaction.reactants.items():
                r_rev *= (conc_map.get(spec, 0.0) ** order)

            k_drain = drain_reaction.get_rate_constant(T)
            r_drain = k_drain
            for spec, order in drain_reaction.reactants.items():
                # loop includes the intermediate itself with correct power (order)
                r_drain *= (conc_map.get(spec, 0.0) ** order)

            # at steady state: rate of production of intermediate = rate of consumption of intermediate
            numerator = (nu_cons_rev * r_rev) + (nu_cons_drain * r_drain)
            r_fwd_theoretical = numerator / nu_prod if nu_prod != 0 else 0.0
            
            if r_fwd_theoretical == 0:
                 return False, "FAIL: Invalid steady state physics (theoretical forward rate is 0)."

            theoretical_ratio = r_rev / r_fwd_theoretical
            simulated_ratio = final_Qc / final_Kc

            # quantitative check: within 1% of theoretical physics prediction
            error = abs(theoretical_ratio - simulated_ratio)
            # use relative error if ratios are large, absolute if small
            is_pass = False
            if theoretical_ratio > 1e-9:
                rel_error = error / theoretical_ratio
                if rel_error < 0.01: is_pass = True
            else:
                if error < 1e-5: is_pass = True

            if is_pass:
                return True, f"PASS: Steady state confirmed. Observed ratio {simulated_ratio:.4f} matches theory: {theoretical_ratio:.4f}."
            else:
                return False, f"FAIL: Steady state deviation. Observed ratio: {simulated_ratio:.4f}, theory: {theoretical_ratio:.4f}."

        # Case B: True Equilibrium
        else:
            ratio = final_Qc / final_Kc
            deviation = abs(1.0 - ratio)
            if deviation < 0.01:
                return True, f"PASS: True Equilibrium reached. Qc/Kc = {ratio:.4f}."
            return False, f"FAIL: Equilibrium mismatch. Qc/Kc = {ratio:.4f} (target 1.0)."

    def test_real_gas_physics(self, compression_factor: float = 0.05) -> Tuple[bool, str]:
        """Validates Van der Waals pressure deviations under high compression."""
        has_vdw = any(s.vdw_a > 0 or s.vdw_b > 0 for s in self.template.species_list)
        if not has_vdw:
            return True, "SKIPPED: System uses ideal gas assumptions (all VdW constants are 0)."
        
        sys = self._get_fresh_instance(override_V=compression_factor)
        sys.run_simulation(**self.sim_args)
        
        P_real = sys.results['P_real'][-1]
        P_ideal = sys.results['P_ideal'][-1]
        
        # avoid division by zero
        if P_ideal == 0: return False, "FAIL: Zero pressure detected."

        deviation_pct = ((P_real - P_ideal) / P_ideal) * 100
        has_vdw = any(s.vdw_a > 0 or s.vdw_b > 0 for s in sys.species_list)
        
        if not has_vdw:
            if abs(deviation_pct) < 0.1:
                return True, "PASS: Ideal gas behavior confirmed (VdW constants are 0)."
            return False, f"FAIL: Phantom real-gas deviation ({deviation_pct:.2f}%) without VdW constants."
            
        if abs(deviation_pct) > 1.0:
            return True, f"PASS: Real gas non-ideality detected ({deviation_pct:+.2f}% divergence)."
        
        return False, "FAIL: Real gas effects not observed despite VdW constants and compression."

    @staticmethod
    def test_le_chatelier_principle(expansion_factor: float = 2.0) -> Tuple[bool, str]:
        """
        Tests a simple, true equilibrium system (A <=> 2B), simulating a volume expansion.
        A simple, true equilibrium system was chosen to avoid ambiguity from kinetically-controlled steady states where the principle may not apply in its simple form.
        """
        S_A = ChemicalSpecies("A", 0, 0)
        S_B = ChemicalSpecies("B", 0, 0)
        
        # A -> 2B
        r_fwd = Reaction({'A': 1}, {'B': 2}, A=10.0, Ea=0.0)
        # 2B -> A
        r_rev = Reaction({'B': 2}, {'A': 1}, A=10.0, Ea=0.0)
        
        # start with pure A
        sys = ChemicalSystem(
            [S_A, S_B], [r_fwd, r_rev], 
            {'A': 1.0, 'B': 0.0}, 
            initial_V=1.0, initial_T=300.0)
        
        # establish initial equilibrium
        sys.run_simulation()
        n_init = sum(sys.results['final_moles'].values())
        
        # apply perturbation: volume expansion -> pressure decrease -> system must increase n_total to compensate
        target_V = sys.V * expansion_factor
        perturb = PerturbationSimulation(sys, 'volume', t_start=2.0, t_end=4.0, new_V=target_V)
        perturb.run_perturbation()
        
        n_final = sum(perturb.results['final_moles'].values())
        delta_n = n_final - n_init
        
        # since A -> 2B creates moles, equilibrium must shift right
        if delta_n > 1e-5:
            return True, f"PASS: Synthetic A<->2B system shifted right (dn={delta_n:+.2e}) on expansion."
        elif delta_n < -1e-5:
            return False, f"FAIL: Synthetic system shifted LEFT (dn={delta_n:+.2e}) on expansion, violating Le Chatelier."
        else:
            return False, "FAIL: Synthetic system showed no response to volume expansion."

    @staticmethod
    def run_stiffness_test() -> Tuple[bool, str]:
        """Verifies solver stability on a stiff system. Uses a fixed time horizon to prevent infinite simulation of slow steps."""
        S1 = ChemicalSpecies("Fast", 0, 0)
        S2 = ChemicalSpecies("Inter", 0, 0)
        S3 = ChemicalSpecies("Slow", 0, 0)
        
        # fast step (k=1e12), slow step (k=0.1)
        reactions = [
            Reaction({'Fast': 1}, {'Inter': 1}, A=1e12, Ea=0),
            Reaction({'Inter': 1}, {'Slow': 1}, A=0.1, Ea=0)]
        
        # create a subclass of ChemicalSystem to inject the t_end logic 
        class StiffChemicalSystem(ChemicalSystem):
            def run_simulation(self, t_end: float = 20.0, **kwargs):
                # hardened implementation for fixed-time simulation
                y0 = self._state_to_array(self.initial_moles)
                
                # solve in one shot from 0 to t_end
                sol = solve_ivp(
                    fun=self._ode_system_adapter, t_span=(0, t_end), y0=y0, 
                    method='Radau', # use a method that can handle stiffness
                    rtol=1e-6, atol=1e-10)
                
                solutions_list = [sol]
                
                self._process_final_results(solutions_list)

        sys = StiffChemicalSystem(
            [S1, S2, S3], reactions, 
            {'Fast': 1.0, 'Inter': 0.0, 'Slow': 0.0}, 
            1.0, 300)
        
        try:
            # run for fixed 20 seconds
            sys.run_simulation(t_end=20.0)
            
            final_moles = sys.results['final_moles']
            
            # check whether the simulation finished
            if sys.results is None:
                return False, "FAIL: Simulation produced no results."

            if final_moles['Fast'] > 1e-9: # fast species should be instantly consumed (~0)
                return False, f"FAIL: Fast transient not resolved. 'Fast' remaining: {final_moles['Fast']:.2e}."

            # check the slow kinetics
            # some slow product should exist, but not necessarily equilibrium
            if final_moles['Slow'] < 1e-4:
                return False, "FAIL: Solver stuck or step too small; 'Slow' product not formed."

            return True, "PASS: Solver handled stiffness properly."

        except Exception as e:
            return False, f"CRITICAL: Solver crashed on stiff system: {str(e)}"

    def run_all_tests(self):
        """Executes the full diagnostic suite and prints a formatted report."""
        tests = [
            ("Mass conservation...", self.test_mass_conservation),
            ("Kinetic convergence...", self.test_kinetic_convergence),
            ("Thermodynamic equilibrium...", self.test_thermodynamic_equilibrium),
            ("Real gas physics...", self.test_real_gas_physics),
            ("Le Chatelier's principle...", self.test_le_chatelier_principle),
            ("Stiffness stability...", self.run_stiffness_test)]
        
        print("\n" + "="*70)
        print(f"{'SYSTEM DIAGNOSTICS REPORT':^70}")
        print("="*70)
        
        passed = 0
        for name, func in tests:
            try:
                success, message = func()
            except Exception as e:
                success = False
                message = f"ERROR: Exception during test execution: {str(e)}"
                
            status = "PASS" if success else "FAIL"
            if success: passed += 1
            print(f"{name:<30} | {status:<8} | {message}")
            
        print("-" * 70)
        print(f"OVERALL RESULT: {passed}/{len(tests)} tests passed.")
        print("="*70 + "\n")