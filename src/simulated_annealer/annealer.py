import numpy as np
from numba import njit
import pandas as pd
import time
from collections import defaultdict

import random
from typing import List, Dict, Any, Callable, Optional, Tuple

class SimulatedAnnealer:
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        initial_temp: float = 1.,
        cooling_halflife: float = 40.,
        max_iter: int = 200,
        max_attempts: int = 20,
        initial_state: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        max_neighbour_attempts: int = 10,
        var_n_param_changes: bool = False,
        n_param_proposal_poi_lam: float = 0.3,
        var_jump_size: bool = False,
        jump_size_poi_lam: float = 0.3
    ):

        # parameters
        self.param_grid = param_grid
        self.initial_temp = initial_temp
        self.lam = np.log(2) / cooling_halflife
        self.max_iter = max_iter
        self.max_attempts = max_attempts
        self.max_neighbour_attempts = max_neighbour_attempts
        self.initial_state = initial_state
        
        # for changing multiple parameters at once
        self.var_n_param_changes = var_n_param_changes
        self.n_param_proposal_poi_lam = n_param_proposal_poi_lam
        self.var_jump_size = var_jump_size
        self.jump_size_poi_lam = jump_size_poi_lam

        # set seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # initialize state
        self.state = self.get_random_state() if initial_state is None else initial_state
        self.state_coords = {k: v.index(self.state[k]) for k, v in self.param_grid.items()}
        self.histories = pd.DataFrame(columns=list(self.param_grid.keys()) + ['energy', 'temperature', 'accepted'])
        self.historical_coords = set()
        self.energy = np.inf
        self.best_state = self.state
        self.best_energy = np.inf
        self.timing_stats = defaultdict(float)
        
        
    def fit(
        self,
        evaluation: Callable[[Dict[str, Any]], float],
        evaluation_args: Optional[Dict[str, Any]] = None,
        verbose = True,
        history_jump: int = 1
        )-> Tuple[pd.DataFrame, Dict[str, Any], float]:
        '''Optimize parameters using simulated annealing.'''
        
        overall_start_time = time.time()
        
        history_data = []
        self.energy = evaluation(self.state, **(evaluation_args if evaluation_args is not None else {}))
        self.best_energy = self.energy
        self.histories.loc[len(self.histories)] = {**self.state, 'energy': self.energy, 'temperature': self.initial_temp, 'accepted': True}
        state_coords_tuple = tuple(sorted(self.state_coords.items()))
        self.historical_coords.add(state_coords_tuple)

        n = 0
        attempts = 0
        T = self.initial_temp
        accepted = 0
        
        
        while n < self.max_iter and attempts < self.max_attempts:
            
            iter_start_time = time.time()
            
            neighbor_start = time.time()
            if attempts >= self.max_neighbour_attempts:
                print("Max neighbour attempts reached, jumping to random state.")
                new_state = self.get_random_state()
            else:
                new_state = self.get_neighbour_state()
            self.timing_stats['proposal_step'] += time.time() - neighbor_start

            new_state_coords = {k: v.index(new_state[k]) for k, v in self.param_grid.items()}
            
            state_coords_tuple = tuple(sorted(new_state_coords.items()))

            if state_coords_tuple in self.historical_coords:
                attempts += 1
                if verbose:
                    print(f"Location already visited, trying again. Attempt {attempts}")
                continue
            
            attempts = 0
            energy_eval_start = time.time()
            new_energy = evaluation(new_state, **(evaluation_args if evaluation_args is not None else {}))
            self.timing_stats['energy_eval'] += time.time() - energy_eval_start
            acc_start = time.time()
            P_accept = acceptance_probability(self.energy, new_energy, T)
            self.timing_stats['acceptance_prob'] += time.time() - acc_start
            
            logic_start = time.time()
            accepted = np.random.rand() < P_accept
            if accepted:
                self.state = new_state
                self.state_coords = new_state_coords
                self.energy = new_energy
                
                if new_energy < self.best_energy:
                    self.best_state = new_state
                    self.best_energy = new_energy

            if n % history_jump == 0:
                history_data.append({**new_state, 'energy': new_energy, 'temperature': T, 'accepted': accepted})
                self.historical_coords.add(state_coords_tuple)

            self.timing_stats['logic'] += time.time() - logic_start
            
            T = cool_temperature(T, self.lam)
            
            self.timing_stats['iteration'] += time.time() - iter_start_time
            if verbose:
                print(f"Iteration {n+1:<4}, T: {T:<5.4f}, P(acc) : {P_accept:<5.4f} E: {self.energy:<5.4f} Accepted: {accepted:<1}, Best Energy: {self.best_energy:<5.4f}, Iter: {time.time() - iter_start_time:<5.2e}s")

            n += 1
            
        self.histories = pd.DataFrame(history_data)
        
        if (attempts >= self.max_attempts) and verbose:
            print("Max attempts reached, exiting. WARNING: may have gotten stuck. Try poisson update.")
        
        if verbose:
            print("\n=== FINAL TIMING SUMMARY ===")
            total_time = time.time() - overall_start_time
            print(f"Total time: {total_time:.2f}s")
            for k, v in self.timing_stats.items():
                if k == 'iteration':
                    continue
                print(f"{k:<20}: {v:.2f}s, {v/total_time*100:.2f}%")
            print("=============================\n")
            
        return self.histories, self.best_state, self.best_energy

    def get_neighbour_state(self) -> Dict[str, Any]:
        """Generate a neighboring state by changing parameters."""
        # Convert current state to array indices
        param_names = list(self.param_grid.keys())
        current_indices = np.array([self.state_coords[k] for k in param_names], dtype=np.int32)
        param_lengths = np.array([len(self.param_grid[k]) for k in param_names], dtype=np.int32)
        
        # Determine number of parameters to change
        if self.var_n_param_changes:
            n_params_to_change = np.random.poisson(self.n_param_proposal_poi_lam) + 1
            n_params_to_change = min(n_params_to_change, len(self.param_grid))
        else:
            n_params_to_change = 1
        
        # Generate new indices using numba function
        new_indices = _generate_neighbor_state(
            current_indices, 
            param_lengths, 
            n_params_to_change,
            self.var_jump_size,
            self.jump_size_poi_lam
        )
        
        # Convert back to state dictionary
        new_state = self.state.copy()
        for i, param_name in enumerate(param_names):
            new_state[param_name] = self.param_grid[param_name][new_indices[i]]
        
        return new_state

    def get_random_state(self) -> Dict[str, Any]:
        """Generate a random state from the parameter grid."""
        return {key: random.choice(values) for key, values in self.param_grid.items()}

@njit
def acceptance_probability(E: float, E_new: float, T: float) -> float:
    """Calculate the acceptance probability for a new state."""
    if E_new < E:
        return 1.0
    else:
        return np.exp((E - E_new) / T)
    
@njit
def cool_temperature(T: float, lam: float) -> float:
    return T * np.exp(-lam)

@njit
def _generate_neighbor_state(current_indices, param_lengths, n_params_to_change, var_jump_size, jump_size_lam):
    """Numba-optimized neighbor state generation"""
    new_indices = current_indices.copy()
    
    # Get indices of parameters to change (without replacement)
    params_to_change = np.random.choice(
        np.arange(len(current_indices)), 
        size=n_params_to_change, 
        replace=False
    )
    
    # Change each selected parameter
    for param_idx in params_to_change:
        # Determine jump size
        if var_jump_size:
            step_size = np.random.poisson(jump_size_lam) + 1
        else:
            step_size = 1
            
        # Choose direction (-1 or 1)
        direction = -1 if np.random.random() < 0.5 else 1
        
        # Calculate new index with boundary reflection
        new_idx = new_indices[param_idx] + direction * step_size
        
        # Apply boundary reflection
        if new_idx < 0:
            new_idx = -new_idx  # Reflect from lower boundary
        elif new_idx >= param_lengths[param_idx]:
            new_idx = 2 * (param_lengths[param_idx] - 1) - new_idx  # Reflect from upper
        
        # Update the index
        new_indices[param_idx] = new_idx
        
    return new_indices