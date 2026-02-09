import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from chemical_engine import ChemicalSystem, PerturbationSimulation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from IPython.display import display
import ipywidgets as widgets
import pandas as pd
from chemical_engine import find_all_equilibrium_pairs
from validation_toolkit import analyse_acid_base
import logging
import copy

def sanitise_results(results: dict) -> dict:
    """Removes duplicate time points to prevent divide-by-zero in rate calculations."""
    
    t = results['time']
    # keep index 0 and any index where dt > a certain amount to prevent duplicate time steps
    keep_mask = np.concatenate(([True], np.diff(t) > 1e-18))

    sanitised = results.copy()
    sanitised['time'] = t[keep_mask]
    
    for key in ['volume', 'P_real', 'P_ideal', 'temperature']:
        if key in results:
            sanitised[key] = results[key][keep_mask]
        
    # filter species data
    sanitised['species_data'] = {name: arr[keep_mask] for name, arr in results['species_data'].items()}

    # filter reaction rates
    if 'reaction_rates' in results and results['reaction_rates'] is not None:
        sanitised['reaction_rates'] = {name: arr[keep_mask] for name, arr in results['reaction_rates'].items()}
    
    # filter thermo arrays
    if 'thermo' in results and results['thermo'] is not None:
        sanitised['thermo'] = {}
        for key, val in results['thermo'].items():
            if isinstance(val, np.ndarray) and len(val) == len(t):
                sanitised['thermo'][key] = val[keep_mask]
            else:
                sanitised['thermo'][key] = val
                
    return sanitised

def find_completion_time(system: ChemicalSystem, results: dict, conc_tol=1e-7, eq_rel_tol=0.01, ignore_before: float=0.0) -> float:
    """
    Analyses simulation results to find when the reaction is complete. Returns the final time if completion is not detected.
    Completion is defined as the point where both concentration changes are negligible and all reversible equilibria have converged.
    Accepts 'ignore_before' to skip pre-perturbation "completion" times.
    """

    time = results['time']
    if len(time) < 2:
        return time[-1] if len(time) > 0 else 0.0

    # filter indices based on ignore_before time
    valid_indices_mask = time >= ignore_before
    if not np.any(valid_indices_mask):
        return time[-1] # fallback if everything is ignored
    
    # slice the arrays to match relevant_time
    slice_start_idx = np.searchsorted(time, ignore_before)
    relevant_time = time[slice_start_idx:]

    max_rate_of_change = np.zeros_like(relevant_time)
    
    for name, moles_array in results['species_data'].items():
        sliced_moles = moles_array[slice_start_idx:]
        # calculate gradient on the sliced segment
        dndt = np.gradient(sliced_moles, relevant_time)
        max_rate_of_change = np.maximum(max_rate_of_change, np.abs(dndt))
    
    stable_indices = np.where(max_rate_of_change < conc_tol)[0]
    t_stable = relevant_time[stable_indices[0]] if len(stable_indices) > 0 else time[-1]

    # check  equilibrium (convergence)
    eq_pairs = system.equilibrium_pairs
    reaction_rates = results.get('reaction_rates')
    
    if not eq_pairs or not reaction_rates:
        return t_stable

    converged_mask = np.ones_like(relevant_time, dtype=bool)

    for fwd_idx, rev_idx in eq_pairs:
        # get rates and slice them
        fwd_rate = reaction_rates.get(f"R{fwd_idx+1}")[slice_start_idx:]
        rev_rate = reaction_rates.get(f"R{rev_idx+1}")[slice_start_idx:]

        if fwd_rate is None or rev_rate is None: continue

        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.abs(fwd_rate - rev_rate) / np.maximum(fwd_rate, 1e-20)
            
            # if both are dead (0), they are converged
            dead_mask = (fwd_rate < 1e-20) & (rev_rate < 1e-20)
            relative_diff[dead_mask] = 0.0
        
        pair_converged = relative_diff < eq_rel_tol
        converged_mask = np.logical_and(converged_mask, pair_converged)

    converged_indices = np.where(converged_mask)[0]
    t_converged = relevant_time[converged_indices[0]] if len(converged_indices) > 0 else time[-1]
    
    return max(t_stable, t_converged) # return the later of the two

def align_previous_run(previous_data: dict, target_time_array: np.ndarray) -> Optional[dict]:
    """
    Aligns the data from a previous run to a new time axis for plotting comparisons. This function handles the following scenarios:
    - If the previous run is longer than the current run, its data is trimmed.
    - If the previous run is shorter, its final state is held constant (extrapolated).
    """
    if not previous_data or len(previous_data.get('time', [])) == 0: return None

    t_prev = previous_data['time']
    t_curr_max = target_time_array[-1]

    # shorten the previous run if it's longer than the current one
    cutoff_idx = np.searchsorted(t_prev, t_curr_max, side='right') # find index in the previous time array where it surpasses the current run's end time
    
    # create a new dictionary to hold the aligned (potentially trimmed) data
    aligned_data = {}
    original_len = len(t_prev)

    for key, value in previous_data.items():
        if isinstance(value, np.ndarray) and len(value) == original_len:
            aligned_data[key] = value[:cutoff_idx]
        
        elif key in ['species_data', 'reaction_rates']:
            aligned_data[key] = {
                sub_key: sub_val[:cutoff_idx]
                for sub_key, sub_val in value.items()
                if isinstance(sub_val, np.ndarray) and len(sub_val) == original_len
            }
        else: # not a time-series array, likely scalar or a dictionary
            # copy directly without slicing...
            aligned_data[key] = value

    # lengthen the previous run if it was shorter
    t_aligned = aligned_data['time']
    if t_aligned[-1] < t_curr_max:
        # append the max time of the current run to make the axes match
        aligned_data['time'] = np.append(t_aligned, t_curr_max)
        
        # for all other data arrays, append their last value 
        for key, value in list(aligned_data.items()): 
            if isinstance(value, np.ndarray) and len(value) == len(t_aligned):
                aligned_data[key] = np.append(value, value[-1])
            elif key in ['species_data', 'reaction_rates']:
                 for sub_key, sub_val in value.items():
                    if isinstance(sub_val, np.ndarray) and len(sub_val) == len(t_aligned):
                         value[sub_key] = np.append(sub_val, sub_val[-1])

    # recalculate concentrations using the newly aligned mole and volume arrays.
    vol_aligned = aligned_data.get('volume', np.full_like(aligned_data['time'], aligned_data['info']['V']))
    conc_aligned = {
        name: aligned_data['species_data'][name] / vol_aligned
        for name in aligned_data['species_data']}
    
    # return a clean dictionary with only the data needed by the plotting function
    return {'time': aligned_data['time'], 'conc': conc_aligned, 'p_real': aligned_data['P_real']}

def generate_table(current_data: dict, system_object: ChemicalSystem, previous_data: Optional[dict]=None, previous_data_label: Optional[str]= None, perturbation_window: Optional[Tuple[float, float]]=None) -> None:
    """Generates and displays a pandas DataFrame comparing the initial and final states."""

    current_data = sanitise_results(current_data)
    if previous_data: previous_data = sanitise_results(previous_data)
    
    s_names = sorted(current_data["species_data"].keys())
    min_time_threshold = perturbation_window[1] if perturbation_window else 0.0
    completion_time = find_completion_time(system_object, current_data, ignore_before=min_time_threshold)

    is_aqueous = False
    if 'H3O+' in s_names or 'H+' in s_names:
        is_aqueous = True
        
    metric_labels = [r'Volume (dm^3)', 'Temperature (K)', 'Time (s)']
    
    if is_aqueous:
        metric_labels.append('Final pH')
    else:
        metric_labels.append('P(Real) / atm')
        metric_labels.append('P(Ideal) / atm')

    for name in s_names:
        metric_labels.append(f'[{name}] init')
        metric_labels.append(f'[{name}] final')

    def get_vals(data):
        d_vol = data.get('volume', np.full_like(data['time'], data['info']['V']))
        
        vals = [f"{d_vol[-1]:.2f}",
                f"{data['info']['T']:.1f}",
                f"{completion_time:.2f}"]
        
        if is_aqueous:
            # calculate final pH
            p_name = 'H3O+' if 'H3O+' in data['species_data'] else 'H+'
            if p_name in data['species_data']:
                final_h = data['species_data'][p_name][-1] / d_vol[-1]
                # safety clamp
                final_ph = -np.log10(max(final_h, 1e-16))
                vals.append(f"{final_ph:.4f}")
            else:
                vals.append("N/A")
        else:
            # pressure
            d_pre = data.get('P_real', np.zeros_like(data['time']))
            d_ide = data.get('P_ideal', np.zeros_like(data['time']))
            vals.append(f"{d_pre[-1]:.2f}")
            vals.append(f"{d_ide[-1]:.2f}")
        
        # concentrations
        v_init = d_vol[0]
        v_final = d_vol[-1]
        c_init_factor = 1.0 / v_init if v_init > 0 else 0.0
        c_final_factor = 1.0 / v_final if v_final > 0 else 0.0

        for name in s_names:
            n_init = data['species_data'][name][0]
            n_final = data['species_data'][name][-1]
            vals.append(f"{n_init * c_init_factor:.2e}")
            vals.append(f"{n_final * c_final_factor:.2e}")
        return vals
    
    table_dict = {'Metric': metric_labels}
    
    if previous_data:
        column_header = previous_data_label if previous_data_label else "Previous Run"
        table_dict[column_header] = get_vals(previous_data)
    table_dict['Current Run'] = get_vals(current_data)
    
    print("Simulation Results:")
    # style the table
    df = pd.DataFrame(table_dict)
    display(df.style.set_table_styles(
        [{'selector': 'th, td', 'props': [('text-align', 'center')]}]
    ).hide(axis="index"))

def generate_plot(current_data: dict, system_object: ChemicalSystem, previous_data: Optional[dict]=None, previous_data_label: Optional[str]=None, perturbation_window: Optional[Tuple[float, float]]=None,
                            view_window: Optional[Tuple[float, float]]=None, log_x: bool=False, log_y: bool=False, show_analysis: bool=False,
                            title: Optional[str]=None, highlight_point: Optional[float]=None, fixed_t_end: Optional[float]=None,
                            exclude_from_yscale: List[str]=None, species_to_hide: Set[str]=None) -> None:
    # data sanitisation
    current_data = sanitise_results(current_data)
    if previous_data: previous_data = sanitise_results(previous_data)
    total_time = current_data["time"][-1]

    if species_to_hide is None: species_to_hide = set()

    if fixed_t_end: # if a fixed end time is provided by the user, override...
        display_end_time = fixed_t_end
    else:
        # calculate completion time for auto-trimming
        min_time_threshold = perturbation_window[1] if perturbation_window else 0.0
        completion_time = find_completion_time(system_object, current_data, ignore_before=min_time_threshold)
        
        # ensure the graph extends at least slightly past the perturbation window
        display_end_time = max(completion_time, min_time_threshold * 1.1)

    t_curr = current_data["time"]
    s_names = sorted(current_data["species_data"].keys())

    vol = current_data.get('volume', np.full_like(t_curr, current_data['info']['V']))
    p_curr = current_data.get('P_real', np.zeros_like(t_curr))

    conc_curr = {n: current_data['species_data'][n] / vol for n in s_names}
    curr_thermo = current_data.get('thermo', None)
    reaction_rates = current_data.get('reaction_rates', None)

    prev_plot_data = align_previous_run(previous_data, t_curr) if previous_data else None # extrapolation

    # colours
    colours = cm.tab10(np.linspace(0, 1, len(s_names)))
    colour_map = dict(zip(s_names, colours))
    rxn_names = list(reaction_rates.keys()) if reaction_rates else []
    rxn_colours = cm.viridis(np.linspace(0, 1, len(rxn_names)))
    rxn_colour_map = dict(zip(rxn_names, rxn_colours))

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex='col', gridspec_kw={'hspace': 0, 'wspace': 0.3, 'width_ratios': [3, 2]})
    ax_conc, ax_rate = axs[0, 0], axs[1, 0]
    ax_conc_detail, ax_rate_detail = axs[0, 1], axs[1, 1]

    if perturbation_window is None:
        ax_conc_detail.set_visible(False)
        ax_rate_detail.set_visible(False)
    ax_pressure_main = ax_conc.twinx()
    ax_thermo_main = ax_rate.twinx()
    ax_pressure_zoom = ax_conc_detail.twinx() if perturbation_window else None

    # y-scale limit logic
    max_y_val = 1e-9 # start with a small number
    if exclude_from_yscale is None:
        exclude_from_yscale = []
    excluded_set = set(exclude_from_yscale)

    for species_name, conc_array in conc_curr.items():
        if species_name not in excluded_set and len(conc_array) > 0:
            max_y_val = max(max_y_val, np.max(conc_array))

    # pH
    # look for H3O+ or H+ in the data keys
    ph_mode = False
    proton_key = None
    if 'H3O+' in conc_curr: proton_key = 'H3O+'
    elif 'H+' in conc_curr: proton_key = 'H+'
    
    # calculate pH array
    ph_arr = None
    if proton_key:
        ph_mode = True
        protons = conc_curr[proton_key]
        safe_protons = np.maximum(protons, 1e-16)  # safety clamp for log calculation (max pH 16)
        ph_arr = -np.log10(safe_protons)
        
        # calculate previous pH if needed
        if prev_plot_data:
            prev_protons = prev_plot_data['conc'][proton_key]
            safe_prev = np.maximum(prev_protons, 1e-16)
            prev_plot_data['ph'] = -np.log10(safe_prev)

    def style_axis(ax, ylabel, is_log_y=False, color='black', is_log_x=False): # styling helper
        ax.set_ylabel(ylabel, color=color)
        if is_log_y: ax.set_yscale('log') # handling y-axis log scale
        else: ax.set_yscale('linear')
        
        if is_log_x: ax.set_xscale('log') # x-axis
        
        for spine in ax.spines.values(): # colouring spines
            spine.set_visible(True)
            spine.set_color(color)
        ax.tick_params(axis='y', colors=color)
        
        ax.grid(True, which='major', linestyle='-', alpha=0.4) # grids
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        
    def draw_layer(ax_c: plt.Axes, ax_p: plt.Axes, ax_th: plt.Axes, t: np.ndarray, c_dict: dict, # drawing helper
                   p_arr: np.ndarray, thermo_data: Optional[dict]=None, style: str='-', alpha: float=1.0,
                   ph_data: np.ndarray=None) -> None:
        # conc & rates
        for n in s_names:
            if n in c_dict and n not in species_to_hide: # check visibility before plotting
                ax_c.plot(t, c_dict[n], color=colour_map[n], linestyle=style, alpha=alpha, lw=2.0 if style=='-' else 1.5)
        
        # pressure and pH (top graph, twin axis)
        if ph_mode and ph_data is not None:
            # pH plotting
            ax_p.plot(t, ph_data, color='navy', linestyle=':', linewidth=2, alpha=0.9*alpha)
            if style == '-': # only set labels for the main layer
                ax_p.set_ylabel("pH", color='navy')
                ax_p.tick_params(axis='y', colors='navy', labelcolor='navy')
                ax_p.spines['right'].set_color('navy')
        else:
            # pressure plotting
            ax_p.plot(t, p_arr, color='teal', linestyle=':', linewidth=2, alpha=0.9*alpha)
            if style == '-':
                ax_p.set_ylabel("Pressure / atm", color='teal')
                ax_p.tick_params(axis='y', colors='teal', labelcolor='teal')
                ax_p.spines['right'].set_color('teal')
        ax_p.spines['left'].set_visible(False); ax_p.spines['top'].set_visible(False); ax_p.spines['bottom'].set_visible(False)
        
        # thermo (bottom graph, twin axis)
        if ax_th and thermo_data:
            # Qc and Kc
            ax_th.plot(t, thermo_data['Qc'], color='purple', linestyle='--', alpha=0.8*alpha, lw=1.5)
            ax_th.plot(t, thermo_data['Kc'], color='darkviolet', linestyle='-.', alpha=0.8*alpha, lw=1.5)
            
            # Qp/Kp (only plot if they exist)
            if thermo_data.get('Qp') is not None:
                ax_th.plot(t, thermo_data['Qp'], color='deeppink', linestyle='--', alpha=0.8*alpha, lw=2.0)
            
            if thermo_data.get('Kp') is not None:
                ax_th.plot(t, thermo_data['Kp'], color='deeppink', linestyle=':', alpha=0.6*alpha, lw=1.5)

            ax_th.tick_params(axis='y', colors='purple', labelcolor='purple')
            ax_th.spines['right'].set_color('purple')

    draw_layer(ax_conc, ax_pressure_main, ax_thermo_main, t_curr, conc_curr, p_curr, curr_thermo, ph_data=ph_arr)
    if prev_plot_data:
        prev_ph = prev_plot_data.get('ph') 
        draw_layer(ax_conc, ax_pressure_main, ax_thermo_main, prev_plot_data['time'], prev_plot_data['conc'], 
                   prev_plot_data['p_real'], style='--', alpha=0.4, ph_data=prev_ph)
        
    if ph_mode:
        # standard pH scale is 0-14, only expand if the simulation goes extreme (e.g. 10M HCl -> pH -1)
        # collect all pH data to check range
        all_ph_values = ph_arr
        valid_ph = all_ph_values[all_ph_values < 15.5]
        
        if len(valid_ph) > 0:
            min_ph = np.min(valid_ph)
            max_ph = np.max(valid_ph)
        else:
            min_ph, max_ph = 7, 7
        
        lower_lim = 0.0
        upper_lim = 14.0
        # expand if needed
        if min_ph < 0: lower_lim = min_ph - 0.5
        if max_ph > 14: upper_lim = max_ph + 0.5
    
        ax_pressure_main.set_ylim(lower_lim, upper_lim) # apply to main axis
        ax_pressure_main.axhline(7.0, color='forestgreen', linestyle='--', alpha=0.6, linewidth=1.5, zorder=0)
        ax_pressure_main.text(t_curr[0], 7.1, " Neutral (pH 7)", color='forestgreen', fontsize=9, verticalalignment='bottom')
        if ax_pressure_zoom: # apply to zoom axis if it exists
            ax_pressure_zoom.set_ylim(lower_lim, upper_lim)
            ax_pressure_zoom.axhline(7.0, color='navy', linestyle=':', alpha=0.3, linewidth=1)
    
    if view_window: # view window (user is actively zooming)
        start_time, end_time = view_window
    
        # safety clamp: if start and end are too close, add a tiny offset
        if end_time - start_time < 1e-15:
            end_time = start_time + 1e-12
            
        ax_conc.set_xlim(start_time, end_time)
        ax_rate.set_xlim(start_time, end_time)
    else: # user is in default state (use the trimming)
        if log_x:
            safe_min = max(1e-6, t_curr[1] if len(t_curr) > 1 else 1e-6)
            ax_conc.set_xlim(left=safe_min, right=display_end_time)
            ax_rate.set_xlim(left=safe_min, right=display_end_time)
        else:
            ax_conc.set_xlim(left=0, right=display_end_time)
            ax_rate.set_xlim(left=0, right=display_end_time)
    
    # setting the y-axis limit
    ax_conc.set_ylim(bottom=0, top=max_y_val * 1.1)
    if log_y: # handling bottom limit for log graphs
        min_y_val = max_y_val
        for species_name, conc_array in conc_curr.items():
            if species_name not in excluded_set and len(conc_array) > 0:
                positive_vals = conc_array[conc_array > 1e-20]
                if len(positive_vals) > 0:
                    min_y_val = min(min_y_val, np.min(positive_vals))
        
        # enforce a floor for the botton limit to prevent log(0) errors
        safe_bottom = min_y_val * 0.5 if min_y_val > 1e-15 else 1e-15
        ax_conc.set_ylim(bottom=safe_bottom, top=max_y_val * 2.0)

    # apply the same logic to the detailed/zoom plot if it exists
    if ax_conc_detail:
        ax_conc_detail.set_ylim(bottom=ax_conc.get_ylim()[0], top=ax_conc.get_ylim()[1])
    
    if ax_thermo_main and curr_thermo:
        ax_thermo_main.set_ylabel("Q / K", color='purple')
        ax_thermo_main.set_yscale('log')
        ax_thermo_main.tick_params(axis='y', colors='purple', right=True)
        ax_thermo_main.spines['top'].set_visible(False); ax_thermo_main.spines['bottom'].set_visible(False); ax_thermo_main.spines['left'].set_visible(False); ax_thermo_main.spines['right'].set_visible(True)
        ax_thermo_main.spines['right'].set_color('purple')

        # only include values that exist
        thermo_arrays_to_scale = [curr_thermo['Qc'], curr_thermo['Kc']]
        if curr_thermo.get('Qp') is not None:
            thermo_arrays_to_scale.append(curr_thermo['Qp'])
        if curr_thermo.get('Kp') is not None:
            thermo_arrays_to_scale.append(curr_thermo['Kp'])
        all_thermo_values = np.concatenate(thermo_arrays_to_scale)

        positive_values = all_thermo_values[all_thermo_values > 1e-20]
        
        if positive_values.size > 0:
            min_val = np.min(positive_values)
            max_val = np.max(positive_values)
            thermo_ylim = (min_val * 0.5, max_val * 2.0)
            ax_thermo_main.set_ylim(bottom=thermo_ylim[0], top=thermo_ylim[1])
        
    # zoom logic
    if perturbation_window:
        t1, t2 = perturbation_window
        for ax in [ax_conc, ax_rate]:
            ax.axvline(t1, color='k', linestyle=':', alpha=0.5, lw=1.5)
            ax.axvline(t2, color='k', linestyle=':', alpha=0.5, lw=1.5)
        
        draw_layer(ax_conc_detail, ax_pressure_zoom, None, t_curr, conc_curr, p_curr, curr_thermo, ph_data=ph_arr)
        if prev_plot_data:
            prev_ph = prev_plot_data.get('ph') 
            draw_layer(ax_conc_detail, ax_pressure_zoom, None, prev_plot_data['time'], prev_plot_data['conc'], prev_plot_data['p_real'], style='--', alpha=0.4, ph_data=prev_ph)
        
        ax_th_zoom = ax_rate_detail.twinx()
        if reaction_rates: 
            for r_name, r_rate in reaction_rates.items():
                ax_rate_detail.plot(t_curr, r_rate, color=rxn_colour_map[r_name], linestyle='-', lw=2.0)
                
        if curr_thermo: 
            ax_th_zoom.plot(t_curr, curr_thermo['Qc'], color='purple', linestyle='--', alpha=0.8, lw=1.5, label=r'$Q_c$')
            ax_th_zoom.plot(t_curr, curr_thermo['Kc'], color='darkviolet', linestyle='-.', alpha=0.8, lw=1.5, label=r'$K_c$')

            if curr_thermo.get('Qp') is not None:
                ax_th_zoom.plot(t_curr, curr_thermo['Qp'], color='deeppink', linestyle='--', alpha=0.8, lw=1.5, label=r'$Q_p$')
                ax_th_zoom.plot(t_curr, curr_thermo['Kp'], color='deeppink', linestyle=':', alpha=0.6, lw=1.5, label=r'$K_p$')

            ax_th_zoom.set_yscale('log')
            ax_th_zoom.tick_params(axis='y', colors='purple', labelcolor='purple')
            ax_th_zoom.spines['right'].set_color('purple')
            ax_th_zoom.spines['left'].set_visible(False); ax_th_zoom.spines['top'].set_visible(False); ax_th_zoom.spines['bottom'].set_visible(False)
            ax_th_zoom.set_ylabel("Q / K", color='purple')

            if 'thermo_ylim' in locals(): ax_th_zoom.set_ylim(thermo_ylim)
            
            h1, l1 = ax_rate_detail.get_legend_handles_labels()
            h2, l2 = ax_th_zoom.get_legend_handles_labels()
            ax_rate_detail.legend(h1+h2, l1+l2, loc='upper right', fontsize='small', framealpha=0.8)

        dur = t2 - t1
        v_min, v_max = max(0, t1 - dur*0.5), min(t_curr[-1] if len(t_curr) > 0 else 0.0, t2 + dur*2.0)
        for ax in [ax_conc_detail, ax_rate_detail]:
            safe_min = max(1e-6, v_min) if log_x else v_min
            safe_max = max(safe_min * 1.01, v_max) 
            
            ax.set_xlim(safe_min, safe_max)
            ax.axvline(t1, color='k', linestyle='-', alpha=0.3)
            ax.axvline(t2, color='k', linestyle='-', alpha=0.3)
        
        ax_conc_detail.set_yscale('log')
        ax_rate_detail.set_yscale('log')

        style_axis(ax_conc_detail, r"Conc / mol dm$^{-3}$", is_log_x=log_x, is_log_y=log_y)
        ax_conc_detail.set_title("Detail")
        
        style_axis(ax_rate_detail, r"Reaction Rate / mol s$^{-1}$")
        ax_rate_detail.set_xlabel("Time / s")

        if ax_rate_detail:
            main_ylim = ax_rate.get_ylim()
            ax_rate_detail.set_ylim(main_ylim)
        
        if ph_mode:
            ax_pressure_zoom.set_ylabel("pH", color='navy')
            ax_pressure_zoom.tick_params(axis='y', labelcolor='navy', colors='navy')
            ax_pressure_zoom.spines['right'].set_color('navy')
        else:
            ax_pressure_zoom.set_ylabel("Pressure / atm", color='teal')
            ax_pressure_zoom.tick_params(axis='y', labelcolor='teal', colors='teal')
            ax_pressure_zoom.spines['right'].set_color('teal')

    # styling the main axes
    style_axis(ax_conc, r"Conc / mol dm$^{-3}$", is_log_x=log_x, is_log_y=log_y)
    if title:
        ax_conc.set_title(title, fontsize=14, pad=10)
    ax_conc.tick_params(labelbottom=False, right=False)

    style_axis(ax_rate, r"Reaction Rate / mol dm$^{-3}$ s$^{-1}$", is_log_x=log_x)
    ax_rate.set_ylim(bottom=1e-13); ax_rate.set_xlabel("Time / s"); ax_rate.tick_params(right=False)
    if ph_mode:
        ax_pressure_main.set_ylabel("pH", color='navy')
    else:
        ax_pressure_main.set_ylabel("Pressure / atm", color='teal')
 
    # legends
    plotted_species = [n for n in s_names if n not in species_to_hide]
    l1 = [Line2D([0], [0], color=colour_map[n], lw=2, label=n) for n in plotted_species]
    l1.append(Line2D([0], [0], color='teal', linestyle=':', lw=2, label='Pressure'))
    if previous_data:
        label = previous_data_label if previous_data_label else "Previous"
        short_label = label.split(':')[0] # e.g., "Run 1: Title" -> "Run 1"
        l1.append(Line2D([0], [0], color='gray', linestyle='--', label=short_label))

    ax_conc.legend(handles=l1, loc='upper right', bbox_to_anchor=(-0.07, 1.0), title="Components", frameon=True, edgecolor='black', fancybox=True)

    l2 = [
        Line2D([0], [0], color='purple', linestyle='--', lw=2, label=r'$Q_c$'),
        Line2D([0], [0], color='darkviolet', linestyle='-.', lw=2, label=r'$K_c$')]
    
    if curr_thermo and curr_thermo.get('Qp') is not None:
        l2.append(Line2D([0], [0], color='deeppink', linestyle='--', lw=2, label=r'$Q_p$'))
        l2.append(Line2D([0], [0], color='deeppink', linestyle=':', lw=2, label=r'Ideal $K_p$'))
    
    if ax_thermo_main:
        leg1 = ax_rate.legend(handles=l2, loc='upper right', bbox_to_anchor=(-0.07, 1.0), title="Equilibrium", frameon=True, edgecolor='black', fancybox=True)
        ax_rate.add_artist(leg1)

    if reaction_rates: # Y-axis scaling for reaction rates
        # this ignores massive initial spikes that can flatten the rest of the plot...
        total_duration = t_curr[-1]
        # define the initial transient phase as the first 0.1% of the simulation time.
        transient_cutoff_time = total_duration * 0.001 
        
        # find the index in the time array where the "stable" phase begins.
        stable_phase_start_index = np.searchsorted(t_curr, transient_cutoff_time, side='right')

        # find the maximum rate ONLY within the stable phase.
        stable_max_rate = 1e-9 # initialise with a small number
        for rates in reaction_rates.values():
            # slice the rates array to exclude the transient part.
            stable_rates = rates[stable_phase_start_index:]
            if stable_rates.size > 0:
                stable_max_rate = max(stable_max_rate, np.max(stable_rates))
        
        ax_rate.set_ylim(bottom=0, top=stable_max_rate * 1.1) # 10% padding

        if ax_rate_detail:
             ax_rate_detail.set_ylim(bottom=0, top=stable_max_rate * 1.1)

        num_points_to_plot = len(t_curr)
        for i, (name, rates) in enumerate(reaction_rates.items()):
            sliced_rates = rates[:num_points_to_plot]
            ax_rate.plot(t_curr, sliced_rates, color=rxn_colours[i], linestyle='-', lw=1.5, label=name)
        
        l3 = [Line2D([0], [0], color=rxn_colours[i], linestyle='-', label=name) for i, name in enumerate(reaction_rates.keys())]
        ax_rate.legend(handles=l3, loc='lower right', bbox_to_anchor=(-0.07, 0.0), title="Reaction Rates", frameon=True, edgecolor='black', fancybox=True)

    if highlight_point: # for highlighting a specific event
        if log_x: # for log scale, highlight +/- 10%
            h_min, h_max = highlight_point * 0.9, highlight_point * 1.1
        else: # for linear, highlight +/- 5s or 5% whatever is smaller
            margin = min(5.0, highlight_point * 0.05)
            h_min, h_max = max(0.0, highlight_point - margin), highlight_point + margin
            
        for ax in [ax_conc, ax_rate]:
            ax.axvspan(h_min, h_max, color='yellow', alpha=0.3, zorder=-1) # draw a yellow band
            ax.axvline(highlight_point, color='red', alpha=0.6, linestyle='-', lw=1) # draw a thin red line at exact time

    if show_analysis and system_object.system_type == 'acid_base': # for highlighting specific events for acid-base reactions
        analysis = analyse_acid_base(current_data, perturbation_window=perturbation_window)
        
        if analysis:
            # plot equivalence points (as red dots)
            for pt in analysis['eq_points']:
                if perturbation_window:
                    ax_pressure_zoom.plot(pt['time'], pt['ph'], 'ro', markersize=5, zorder=5)

                ax_pressure_main.plot(pt['time'], pt['ph'], 'ro', markersize=5, zorder=5)
                
                # add text with an arrow pointing to the dot
                ax_pressure_main.annotate(
                    f"Equivalence Point\npH {pt['ph']:.2f}, t = {pt['time']:.2f}s", 
                    xy=(pt['time'], pt['ph']), 
                    xytext=(pt['time'] + (total_time*0.05), pt['ph'] - 1.0),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9, color='darkred', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
                )

    plt.show()

def find_key_events(system: ChemicalSystem, reaction_rates: dict) -> dict:
    """Analyses simulation results to find the times of key events."""
    events = {}
    results = system.results
    time = results['time']
    
    # equilibrium convergence (R_fwd ~= R_rev)
    eq_pairs = system.equilibrium_pairs 
    for fwd_idx, rev_idx in eq_pairs:
        fwd_rate = reaction_rates[f"R{fwd_idx+1}"]
        rev_rate = reaction_rates[f"R{rev_idx+1}"]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = np.abs(fwd_rate - rev_rate)
            rel_diff = np.divide(diff, fwd_rate, out=np.ones_like(diff), where=fwd_rate > 1e-20)
            dead_mask = (fwd_rate < 1e-20) & (rev_rate < 1e-20)
            rel_diff[dead_mask] = 0.0

        convergence_indices = np.where(rel_diff < 0.01)[0]
        if len(convergence_indices) > 0:
            t_conv = time[convergence_indices[0]]
            if t_conv > 1e-9:
                events[f"Eq. convergence: R{fwd_idx+1}/R{rev_idx+1}"] = t_conv

    # intermediate peaks
    intermediates = system.find_intermediates()
    for s in intermediates:
        conc = results['species_data'][s]
        if np.max(conc) < 1e-12: continue # safety check: does it actually exist in this run?
            
        peak_idx = np.argmax(conc)
        t_peak = time[peak_idx] # find the peak time
        if t_peak > 1e-9: # only add the event if the peak happens at a valid time
            events[f"Steady state (peak {s})"] = t_peak

    # reactant(s) half-life
    for reactant, n0 in system.initial_moles.items():
        if n0 > 1e-6:
            n_t = results['species_data'][reactant]
            half_life_indices = np.where(n_t < n0 / 2.0)[0]
            if len(half_life_indices) > 0:
                events[f"Half-life: {reactant}"] = time[half_life_indices[0]]

    return events

def create_interface(system):
    style = {'description_width': '150px'}
    layout_full = widgets.Layout(width='600px')
    layout_half = widgets.Layout(width='295px')
    
    # dynamic sliders
    species_sliders = {}
    species_widgets = []
    
    for species_obj in system.species_list: # iterate over list of objects
        s_name = species_obj.name
        init_val = system.initial_moles.get(s_name, 0.0)
        # set max
        dyn_max = max(5.0, init_val * 2.0)
        
        species_slider = widgets.FloatSlider(  # a slider for each species
            value=init_val, min=0.0, max=dyn_max, step=0.01,
            description=f'Initial [{s_name}] / M', style=style, layout=layout_full)
        species_sliders[s_name] = species_slider
        species_widgets.append(species_slider)

    txt_title = widgets.Text(value="Reaction Simulation", description="Graph Title", style=style, layout=layout_full)
    vol_slider = widgets.FloatSlider(value=system.V, min=0.1, max=5.0, step=0.05, description=r'Volume (dm^3)', style=style, layout=layout_full)
    temp_slider = widgets.FloatSlider(value=system.T, min=200, max=1000, step=1, description='Temperature (K)', style=style, layout=layout_full)
    
    # analysis controls (zoom/scroll)...
    view_slider = widgets.FloatRangeSlider(value=(0, 100), min=0, max=1000, step=1e-6, description='View Window', style=style, layout=layout_full, disabled=True)
    btn_view_update = widgets.Button(description="Update View")
    chk_log_x = widgets.Checkbox(value=False, description='Log Time')
    chk_log_y = widgets.Checkbox(value=False, description='Log Concentrations')
    if system.system_type == 'acid_base':
        chk_eq_points = widgets.Checkbox(value=False, description='Show Equivalence Point', disabled=False)
        chk_eq_points.observe(lambda change: refresh_plot(), names='value')
    
    # perturbation controls....
    style_sci = {'description_width': '100px'}
    layout_sci = widgets.Layout(width='160px')
    
    # controlling start time
    t_start_mant = widgets.FloatSlider(value=1.0, min=0.0, max=9.99, step=0.01, description='Start', style=style, layout=layout_full, disabled=True)
    t_start_exp = widgets.IntSlider(value=0, min=-15, max=10, step=1, description='x 10^', style=style, layout=layout_full, disabled=True)
    
    # controlling end time
    t_end_mant = widgets.FloatSlider(value=2.0, min=0.0, max=9.99, step=0.01, description='End', style=style, layout=layout_full, disabled=True)
    t_end_exp = widgets.IntSlider(value=0, min=-15, max=10, step=1, description='x 10^', style=style, layout=layout_full, disabled=True)
    
    # grouping them for display
    t_start_box = widgets.HBox([t_start_mant, t_start_exp])
    t_end_box = widgets.HBox([t_end_mant, t_end_exp])
    
    # injection rates...
    injection_widgets = {}
    inj_ui_list = []
    for species_obj in system.species_list: # one for every species
        s_name = species_obj.name
        injection_rate_field = widgets.FloatText(
            value=0.0, step=0.01, description=f'Inject {s_name}',
            style={'description_width': '100px'}, layout=widgets.Layout(width='200px'))
        injection_widgets[s_name] = injection_rate_field
        inj_ui_list.append(injection_rate_field)
        
    inj_rows = [widgets.HBox(inj_ui_list[i:i+3]) for i in range(0, len(inj_ui_list), 3)]
    
    # buttons
    btn_run = widgets.Button(description="Run Simulation", button_style='success', layout=layout_half)
    btn_clear = widgets.Button(description="Clear Previous", button_style='warning', layout=layout_half)
    focus_button_box = widgets.HBox([])
    thermo_dropdown = widgets.Dropdown(options=[('N/A', None)], description='Target:', style=style, layout=layout_full, disabled=True)
    btn_apply = widgets.Button(description="Apply Change", button_style='danger', layout=layout_full, disabled=True)
    
    t_end_widget = widgets.FloatText(value=None, # default to None, no end time is specified
        placeholder='(e.g., 500.0)', description='End Time (s)', style=style, layout=layout_full)
    
    # control for maximum value of all concentration sliders
    conc_max_controller = widgets.FloatText(value=2.0, description='Conc. Slider Max (M)', style=style, layout=layout_full)
    def on_conc_max_change(change):
        """Updates the .max attribute of all species_sliders."""
        new_max = change['new']
        if new_max > 0: # ensure the max is a positive number
            for slider in species_sliders.values():
                # if the slider's current value is now above the new max, clamp it
                if slider.value > new_max:
                    slider.value = new_max
                slider.max = new_max
    conc_max_controller.observe(on_conc_max_change, names='value')

    widgets.HTML("<hr><b>Yield Calculator</b>"),
    
    # dropdowns for selecting species
    species_options = sorted([s.name for s in system.species_list])
    dd_yield_product = widgets.Dropdown(options=species_options, description='Product:', style=style)
    dd_yield_reactant = widgets.Dropdown(options=species_options, description='Limit Reac:', style=style)
    
    # stoichiometry input
    txt_yield_ratio = widgets.FloatText(value=1.0, description='Stoich Ratio:', style=style)

    def update_yield_stoich(*args):
        if system.overall_reaction:
            r_name = dd_yield_reactant.value
            p_name = dd_yield_product.value
            
            # get coeffs (returns None if species not in the overall equation)
            r_coeff = system.overall_reaction.reactants.get(r_name)
            p_coeff = system.overall_reaction.products.get(p_name)
            
            if r_coeff and p_coeff:
                # calculate ratio: product coeff / reactant coeff
                txt_yield_ratio.value = float(p_coeff) / float(r_coeff)
                txt_yield_ratio.description = f'Ratio ({p_coeff}:{r_coeff})'
                txt_yield_ratio.disabled = True # Lock it to show it's auto-detected
            else:
                # fallback to manual entry if species aren't in the main equation
                txt_yield_ratio.description = 'Stoich Ratio:'
                txt_yield_ratio.disabled = False
    
    dd_yield_product.observe(update_yield_stoich, names='value')
    dd_yield_reactant.observe(update_yield_stoich, names='value')
    update_yield_stoich()

    btn_calc_yield = widgets.Button(description="Calculate Yield", button_style='info', layout=layout_half)
    
    yield_box = widgets.VBox([
        widgets.HBox([dd_yield_product, dd_yield_reactant]),
        widgets.HBox([txt_yield_ratio, btn_calc_yield])
    ])

    def on_calc_yield(b): # bind the button
        prod = dd_yield_product.value
        reac = dd_yield_reactant.value
        ratio = txt_yield_ratio.value
        
        with output:
            st.curr.calculate_yield(prod, reac, ratio)
            
    btn_calc_yield.on_click(on_calc_yield)

    dd_comparison = widgets.Dropdown(
        options=[('None', None)],
        description='Compare Against:',
        style=style,
        layout=layout_full,
        disabled=True
    )
    btn_clear_history = widgets.Button(description="Clear History")
    comparison_box = widgets.HBox([dd_comparison, btn_clear_history])

    def on_comparison_select(change):
        """Called when the user selects a historical run from the dropdown."""

        selected_key = change.get('new')
        if selected_key: # retrieve the selected run's data and label
            st.comparison_results = st.run_history[selected_key]
            st.comparison_label = selected_key
        else: # user selected 'None', so clear comparison
            st.comparison_results = None
            st.comparison_label = None
        # re-draw the plot with the new comparison data
        refresh_plot()

    def on_clear_history(b):
        """Called when the user clicks the 'Clear History' button."""

        st.run_history.clear()
        st.run_counter = 0
        st.comparison_results = None
        st.comparison_label = None
        # reset and disable the dropdown
        dd_comparison.options = [('None', None)]
        dd_comparison.disabled = True
        refresh_plot()

    dd_comparison.observe(on_comparison_select, names='value')
    btn_clear_history.on_click(on_clear_history)

    # controls for y-axis scaling
    exclude_widgets = {}
    exclude_ui_list = [widgets.HTML("<b>Y-Axis Scaling:</b>")]
    for species_obj in system.species_list:
        s_name = species_obj.name
        chk = widgets.Checkbox(value=False, description=s_name, indent=False, layout=widgets.Layout(width='auto'))
        exclude_widgets[s_name] = chk
        exclude_ui_list.append(chk)
    y_scale_controls = widgets.HBox(exclude_ui_list) # arrange checkboxes horizontally

    # controls for plotting
    visibility_widgets = {}
    visibility_ui_list = [widgets.HTML("<b>Plot Visibility:</b>")]
    for species_obj in system.species_list:
        s_name = species_obj.name
        # Create a checkbox for each species, defaulting to True (visible)
        chk = widgets.Checkbox(value=True, description=s_name, indent=False, layout=widgets.Layout(width='auto'))
        visibility_widgets[s_name] = chk
        visibility_ui_list.append(chk)
    visibility_controls = widgets.HBox(visibility_ui_list)

    output = widgets.Output()

    log_handler_box = widgets.Textarea(
        value='',
        placeholder='Debug output will appear here...',
        description='Log Console:',
        layout={'width': '100%', 'height': '150px'}
    )

    # custom handler to write to the widget
    class WidgetHandler(logging.Handler):
        def __init__(self, widget):
            super().__init__()
            self.widget = widget

        def emit(self, record):
            msg = self.format(record)
            self.widget.value = f"{msg}\n{self.widget.value}"

    logging.getLogger().setLevel(logging.INFO)

    engine_logger = logging.getLogger('chemical_engine')
    engine_logger.setLevel(logging.DEBUG)
    engine_logger.propagate = False
    engine_logger.handlers.clear()
    widget_log_handler = WidgetHandler(log_handler_box)
    widget_log_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    engine_logger.addHandler(widget_log_handler)
    engine_logger.info("Log console initialised.")


    # state management
    class GUIState:
        def __init__(self):
            self.curr = None
            self.active_system = None
            self.perturb_window = None
            # history management
            self.run_history = {} # a dictionary to store past results
            self.run_counter = 0  # a simple counter for unique run IDs
            self.comparison_results = None # holds the data for the selected historical run
            self.comparison_label = None
            self.available_thermo_pairs = []
            self.selected_thermo_pair = None

    st = GUIState()

    def update_controls(sys):
        # set max times
        t_total = sys.results['time'][-1]
        # determine if a perturbation has just run by checking the GUI state
        if st.perturb_window:
            # if so, the search for the new completion time must start after the perturbation ended
            ignore_before_time = st.perturb_window[1]
        else: # otherwise, it's a normal run, so search from the beginning
            ignore_before_time = 0.0

        completion_time = find_completion_time(sys, sys.results, ignore_before=ignore_before_time)
        display_end_time = completion_time # set default view to 120% of the completion time
        if t_end_widget.value is not None and t_end_widget.value > 0:
            display_end_time = t_end_widget.value

        eq_time = sys.results.get('equilibrium_time', t_total)
        t_vis_max = eq_time * 1.2 if eq_time else t_total
        dynamic_step = max(1e-12, t_vis_max / 1000.0)
        
        # update view slider
        view_slider.max = display_end_time
        view_slider.step = dynamic_step
        view_slider.value = (0, view_slider.max)
        view_slider.disabled = False
        
        # update perturb sliders
        for w in [t_start_mant, t_start_exp, t_end_mant, t_end_exp]:
            w.disabled = False

        btn_apply.disabled = False

    def refresh_plot(perturb_window=None, highlight_point=None, fixed_t_end: Optional[float]=None):  
        with output:
            output.clear_output(wait=False)
            gui_state = st
            if not gui_state.curr: return

            current_results = gui_state.curr.results
            previous_results = gui_state.comparison_results

            # calculate thermodynamics
            if gui_state.selected_thermo_pair:
                try:
                    f_idx, r_idx = gui_state.selected_thermo_pair
                    thermo_data = gui_state.curr.calculate_thermodynamics(f_idx, r_idx, external_results=current_results)
                    current_results['thermo'] = thermo_data
                except Exception as e:
                    print(f"Thermo calculation failed: {e}")
                    current_results['thermo'] = None
            else:
                current_results['thermo'] = None

            view_window = view_slider.value
            perturbation_window = perturb_window if perturb_window else gui_state.perturb_window
            
            user_excluded_species = [name for name, chk in exclude_widgets.items() if chk.value]
            hidden_species = {name for name, chk in visibility_widgets.items() if not chk.value}
            final_excluded_species = list(set(user_excluded_species).union(hidden_species))

            if gui_state.curr.system_type=='acid_base':
                show_analysis = chk_eq_points.value
            else:
                show_analysis = False

            if len(gui_state.run_history) > 1:
                previous_run_key = dd_comparison.options[-2][1] # default to last run
                dd_comparison.value = previous_run_key

            generate_table(current_results, system_object=gui_state.curr, # generate table
                           previous_data=previous_results, previous_data_label=gui_state.comparison_label, perturbation_window=perturbation_window)
            generate_plot(current_results, system_object=gui_state.curr, previous_data=previous_results, previous_data_label=gui_state.comparison_label, # generate plot
                            perturbation_window=perturbation_window, view_window=view_window,
                            log_x=chk_log_x.value, log_y=chk_log_y.value, show_analysis=show_analysis, title=txt_title.value, highlight_point=highlight_point,
                            fixed_t_end=t_end_widget.value, exclude_from_yscale=final_excluded_species, species_to_hide=hidden_species)
            
    def update_thermo_controls(system_obj):
        """Updates the thermo dropdown based on the current system's reactions."""
        thermo_dropdown.unobserve(on_thermo_pair_change, names='value')
        
        available_pairs = find_all_equilibrium_pairs(system_obj.reactions)
        if available_pairs:
            dropdown_options = [
                (f"Eq. Pair (R{p[0]+1}, R{p[1]+1}): {system_obj.reactions[p[0]]}", p)
                for p in available_pairs]
            
            thermo_dropdown.options = dropdown_options
            thermo_dropdown.disabled = False
            
            valid_pairs = [p for _, p in dropdown_options]
            if st.selected_thermo_pair not in valid_pairs:
                st.selected_thermo_pair = dropdown_options[0][1]
            
            thermo_dropdown.value = st.selected_thermo_pair 
        else:
            thermo_dropdown.options = [("No reversible pairs found", None)]
            thermo_dropdown.disabled = True
            st.selected_thermo_pair = None
            thermo_dropdown.value = None
            
        # Start listening again
        thermo_dropdown.observe(on_thermo_pair_change, names='value')

    def on_run(b):
        concentrations = {n: s.value for n, s in species_sliders.items()}
        volume = vol_slider.value
        moles = {n: c * volume for n, c in concentrations.items()} # calculate initial moles
        if sum(moles.values()) <= 1e-9: 
            with output:
                output.clear_output()
                print("Error: total initial moles cannot be zero. Please add at least one species.")
            return
            
        with output:
            output.clear_output(wait=True)
            print("System Reactions")
            for i, r in enumerate(system.reactions): print(f"Reaction {i+1}: {r}")

        new_sys = ChemicalSystem(system.species_list, system.reactions, moles, vol_slider.value, temp_slider.value,
                                 method=system.method, rtol=system.rtol, atol=system.atol, overall_reaction=system.overall_reaction, system_type=system.system_type)

        # validation...
        initial_rates = new_sys.get_initial_rates()
        # check if the sum of the absolute values of all initial rates is effectively zero.
        if np.sum(np.abs(initial_rates)) < 1e-12:
            with output:
                output.clear_output()
                print("The initial concentrations provided will result in a zero reaction rate for all possible steps.")
                print("The system is already in a static state and will not evolve.")
                print("\nPossible reasons:")
                for i, reaction in enumerate(new_sys.reactions):
                    if abs(initial_rates[i]) < 1e-12:
                        # find which reactant is missing
                        missing_reactants = [r for r in reaction.reactants if new_sys.initial_moles.get(r, 0) < 1e-9]
                        if missing_reactants:
                            print(f"Reaction #{i+1} ({reaction}) cannot start because reactant(s) {missing_reactants} are absent.")
                print("\nPlease adjust the initial conditions and run again.")
            return # abort the simulation
        
        # read the user-defined end time (None if the box is empty)
        user_t_end = t_end_widget.value
        t_end_to_pass = user_t_end if user_t_end is not None and user_t_end > 0 else None
        new_sys.run_simulation(t_end=t_end_to_pass) # run simulation, passing user's value directly to the engine

        reaction_rates = new_sys.calculate_reaction_rates_over_time()
        new_sys.results['reaction_rates'] = reaction_rates # store for plotting
        
        key_events = find_key_events(new_sys, reaction_rates) # find the events
        
        # create a button for each event found
        focus_buttons = []
        for event_name, event_time in key_events.items():
            button = widgets.Button(description=f"{event_name} (t={event_time:.1e}s)", button_style='info')
            
            def create_handler(t):
                def handler(b):
                    start_view = max(0.0, t / 5.0) 
                    end_view = t * 5.0
                
                    max_t = new_sys.results['time'][-1]
                    end_view = min(end_view, max_t)
                    view_slider.value = (start_view, end_view) # shows context
                    refresh_plot(highlight_point=t)
                return handler
                
            button.on_click(create_handler(event_time))
            focus_buttons.append(button)
        
        # update the UI with the new buttons
        focus_button_box.children = tuple(focus_buttons)

        # handling the history
        st.run_counter += 1
        title = txt_title.value if txt_title.value else f"Untitled Simulation"
        key = f"Run {st.run_counter}: {title}"
        st.run_history[key] = copy.deepcopy(new_sys.results) # use deepcopy to ensure it won't be accidentally modified by later operations
        # update the comparison dropdown with the new history
        dropdown_options = [('None', None)] + [(key, key) for key in st.run_history.keys()] # first item is always 'None'
        dd_comparison.options = dropdown_options
        dd_comparison.disabled = False

        st.curr = new_sys 
        st.active_system = new_sys
        st.perturb_window = None

        update_thermo_controls(new_sys)
        update_controls(new_sys)
        refresh_plot(fixed_t_end=t_end_to_pass)

    def on_clear(b): st.prev = None; st.curr = None; output.clear_output()

    def on_perturb(b):
        system_to_perturb = st.active_system

        if not system_to_perturb or not system_to_perturb.results:
            with output:
                output.clear_output()
                print("Error: A successful baseline simulation must be run before a perturbation can be applied.")
            return

        # get time window
        t_start = t_start_mant.value * (10.0 ** t_start_exp.value)
        t_end = t_end_mant.value * (10.0 ** t_end_exp.value)

        if t_start >= t_end: # validation to prevent t_start > t_end
            with output: print(f"Error: Start time ({t_start:.2e}s) must be less than end time ({t_end:.2e}s).")
            return

        # collect all parameters
        target_v = vol_slider.value
        target_t = temp_slider.value

        # check if injection rates are non-zero
        rates = {}
        for name, widget in injection_widgets.items():
            if abs(widget.value) > 1e-20:
                rates[name] = widget.value
        
        user_t_end = t_end_widget.value
        if user_t_end is not None and user_t_end > 1e-12:
            if t_end > user_t_end:
                with output:
                    print(f"Error: Perturbation end time ({t_end:.2e}s) cannot exceed global end time ({user_t_end:.2e}s).")
                return
        t_global_end_param = user_t_end if user_t_end is not None and user_t_end > 1e-12 else None
        
        perturbation_sim = PerturbationSimulation(
            baseline_system=system_to_perturb,
            t_start=t_start,
            t_end=t_end,
            new_V=target_v, 
            new_T=target_t,  
            injection_rates=rates,
            t_global_end=t_global_end_param
        )

        with output:
            perturbation_sim.run_perturbation()

        if perturbation_sim.results is None: # safety check
            with output:
                print("\nCRITICAL: Simulation crashed. Injection rate too high or tolerances too tight.")
            return

        # extract the final state to create the new system object
        final_moles = perturbation_sim.results['final_moles']
        final_V = perturbation_sim.results['info']['V']
        final_T = perturbation_sim.results['info']['T']

        post_perturb_system = ChemicalSystem(
            system.species_list, system.reactions, # use original species/reaction definitions
            final_moles, final_V, final_T, method=system.method, rtol=system.rtol, atol=system.atol, overall_reaction=system.overall_reaction, system_type=system.system_type)
        # assign the complete, stitched results from the perturbation simulation to the new system object
        post_perturb_system.results = perturbation_sim.results

        st.prev = st.curr
        st.curr = post_perturb_system
        st.perturb_window = (t_start, t_end)
        update_thermo_controls(st.curr)
        update_controls(st.curr) # update all UI controls
        refresh_plot()

    btn_apply.on_click(on_perturb)
        
    def on_thermo_pair_change(change): # runs when the dropdown value changes...
        if not st.curr: return # don't do anything if no simulation has been run
        new_pair = change.get('new')
        
        # check if the new value is actually different from the current state
        if new_pair != st.selected_thermo_pair:
            st.selected_thermo_pair = new_pair
            # only replot if the change was meaningful (i.e., initiated by the user)
            refresh_plot()

    # link Buttons
    btn_run.on_click(on_run); btn_clear.on_click(on_clear)
    btn_view_update.on_click(lambda b: refresh_plot())

    thermo_dropdown.observe(on_thermo_pair_change, names='value')

    if system.system_type == 'acid_base':
        checkboxes = widgets.HBox([chk_log_x, chk_log_y, chk_eq_points], layout=widgets.Layout(gap='20px'))
    else:
        checkboxes = widgets.HBox([chk_log_x, chk_log_y], layout=widgets.Layout(gap='20px'))

    # view controls (top)
    view_controls = widgets.VBox([widgets.HTML("<b style='font-size:14px'>Graph View</b>"),
        widgets.HBox([view_slider, btn_view_update,], layout=widgets.Layout(align_items='center', gap='10px')), comparison_box,
        checkboxes, y_scale_controls, visibility_controls],
    layout=widgets.Layout(padding='12px 16px', margin='0 0 10px 0', border='1px solid #ddd',border_radius='6px', background_color='#fafafa'))

    # main (bottom)
    controls = widgets.VBox([
        widgets.HTML("<b>System Configuration</b>"),
        txt_title,
        widgets.VBox(species_widgets),
        widgets.HBox([conc_max_controller, t_end_widget]),
        widgets.HBox([vol_slider, temp_slider]),
        widgets.HBox([btn_run, btn_clear]),
        widgets.HTML("<hr>"), # separator
        widgets.HBox([thermo_dropdown, focus_button_box]),
        widgets.HTML("<hr><b>Controls</b>"),
        widgets.HTML("<i>Time Window:</i>"),
        t_start_box,
        t_end_box,
        widgets.HTML("<i>Injection Rates (mol/s):</i>"),
        widgets.VBox(inj_rows),
        btn_apply
        # yield_box
    ], layout=widgets.Layout(padding='20px', border='1px solid #ddd'))
    
    display(widgets.VBox([view_controls, output, controls, log_handler_box]))