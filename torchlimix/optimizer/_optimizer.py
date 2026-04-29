import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

FACTR = 1e7
PGTOL = 1e-7
class OptimixError(Exception):
    """Custom exception for optimization errors."""
    pass

class TorchFunction(nn.Module, ABC):
    """
    PyTorch function optimizer matching numpy optimix.Function behavior.
    """
    
    def __init__(self, name: str, composite: List[Tuple[str, nn.Module]] = None, **kwargs):
        super().__init__()
        self._name = name
        self._sign = 1.0
        self._verbose = True
        
        # Optimization state
        self._solutions: List[np.ndarray] = []
        self._flat_gradient: Optional[np.ndarray] = None
        self._flat_solution: Optional[np.ndarray] = None
        self._iteration = 0
        self._final_grad_norm: float = float('inf')
        self._n_iterations: int = 0
        
        # Variable registry: maps gradient keys to (parameter_name, module)
        self._variable_registry: Dict[str, Tuple[str, nn.Module]] = {}
        
        # Register composite modules
        if composite:
            for comp_name, comp_module in composite:
                self.add_module(comp_name, comp_module)
                self._register_composite_variables(comp_name, comp_module)
    
    def _register_composite_variables(self, prefix: str, module: nn.Module):
        """Register variables from composite modules, matching optimix naming."""
        for param_name, param in module.named_parameters(recurse=False):
            grad_key = f"{prefix}.{param_name}"
            self._variable_registry[grad_key] = (param_name, module)
        
        for child_name, child_module in module.named_children():
            for param_name, param in child_module.named_parameters(recurse=False):
                grad_key = f"{prefix}.{param_name}"
                self._variable_registry[grad_key] = (param_name, child_module)
    
    def variables(self) -> Dict[str, Tuple[torch.Tensor, nn.Module]]:
        """Return variables dict mapping gradient keys to (value, module) tuples."""
        result = {}
        for grad_key, (param_name, module) in self._variable_registry.items():
            param = getattr(module, param_name, None)
            if param is not None and isinstance(param, nn.Parameter):
                if param.requires_grad:
                    result[grad_key] = (param, module)
        return result
    
    def _get_variable_names(self) -> List[str]:
        """Get sorted list of unfixed variable names."""
        return sorted(self.variables().keys())
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name
    
    @abstractmethod
    def value(self) -> torch.Tensor:
        """Compute the function value (scalar)."""
        pass
    
    @abstractmethod
    def gradient(self) -> Dict[str, torch.Tensor]:
        """Compute gradients with respect to parameters."""
        pass
    
    def _get_variable_bounds(self, grad_key: str) -> List[Tuple[float, float]]:
        """Get bounds for a variable by its gradient key.
        
        Matches NumPy optimix logic:
            if variable.ndim == 0: bounds.append(variable.bounds)  # single tuple
            else: bounds += variable.bounds                        # list of tuples
        """
        if grad_key not in self._variable_registry:
            return [(None, None)]
        
        param_name, module = self._variable_registry[grad_key]
        param = getattr(module, param_name, None)
        
        if param is None:
            return [(None, None)]
        
        if hasattr(module, 'bounds') and module.bounds is not None:
            bounds = module.bounds
            if param.ndim == 0:
                # Scalar: bounds is a single (lo, hi) tuple
                return [bounds]
            else:
                # Array: bounds is a list of (lo, hi) tuples
                return list(bounds)
        
        # No bounds = unconstrained, use None like NumPy/SciPy expects
        return [(None, None)] * param.numel()
        
    def _initialize_flat_arrays(self, varnames: List[str]):
        """Initialize flat arrays for optimization."""
        total_size = 0
        variables = self.variables()
        for name in varnames:
            if name in variables:
                param, _ = variables[name]
                total_size += param.numel()
        
        self._flat_gradient = np.zeros(total_size, dtype=np.float64)
        self._flat_solution = np.zeros(total_size, dtype=np.float64)
    
    def _set_flat_arr(self, grad_dict: Dict[str, torch.Tensor], 
                      varnames: List[str], out: np.ndarray) -> np.ndarray:
        """Flatten gradient dictionary into output array."""
        variables = self.variables()
        offset = 0
        
        for name in varnames:
            if name not in variables:
                continue
            
            param, _ = variables[name]
            size = param.numel()
            
            if name in grad_dict and grad_dict[name] is not None:
                grad = grad_dict[name]
                # Ensure float64 for numerical stability
                arr_np = grad.detach().double().cpu().numpy().ravel()
                if arr_np.size == size:
                    out[offset:offset + size] = arr_np
                else:
                    out[offset:offset + size] = 0.0
            else:
                out[offset:offset + size] = 0.0
            
            offset += size
        
        return out
    
    def _set_var_arr(self, flat_arr: np.ndarray, varnames: List[str]):
        """Set parameters from flat array."""
        variables = self.variables()
        offset = 0
        
        for name in varnames:
            if name not in variables:
                continue
            
            param, module = variables[name]
            size = param.numel()
            
            # Always use float64 for numerical stability (matches numpy)
            new_data = torch.tensor(
                flat_arr[offset:offset + size].reshape(param.shape),
                dtype=torch.float64,
                device=param.device
            )
            param.data.copy_(new_data)
            offset += size
    
    def _get_flat_solution(self, varnames: List[str], out: np.ndarray) -> np.ndarray:
        """Get current parameter values as flat array."""
        variables = self.variables()
        offset = 0
        
        for name in varnames:
            if name not in variables:
                continue
            
            param, _ = variables[name]
            arr = param.detach().double().cpu().numpy().ravel()
            size = arr.size
            out[offset:offset + size] = arr
            offset += size
        
        return out
    
    def _approx_fprime(self, step: float = 1.49e-08) -> Dict[str, np.ndarray]:
        """Approximate gradients using finite differences."""
        self.clear_cache()
        f0 = self.value().item()
        grad = {}
        
        variables = self.variables()
        
        for name in sorted(variables.keys()):
            param, module = variables[name]
            
            if not param.requires_grad:
                continue
            
            original_data = param.data.clone()
            ndim = param.ndim
            value = param.data.flatten()
            
            grads = []
            for i in range(len(value)):
                value[i] += step
                param.data = value.reshape(param.shape)
                self.clear_cache()
                grads.append(np.asarray((self.value().item() - f0) / step))
                value[i] -= step
            
            param.data = original_data
            grad[name] = np.stack(grads, axis=-1)
            
            if ndim == 0:
                grad[name] = np.squeeze(grad[name], axis=-1)
        
        self.clear_cache()
        return grad
    
    def _check_grad(self, step: float = 1.49e-08) -> float:
        """Check analytical gradient against numerical approximation."""
        g = self.gradient()
        g = {n: np.asarray(gi.detach().double().cpu().numpy() if isinstance(gi, torch.Tensor) else gi) 
             for n, gi in g.items()}
        
        fg = self._approx_fprime(step)
        names = set(g.keys()).intersection(fg.keys())
        
        return sum(np.linalg.norm(fg[name] - g[name]) for name in names)
    
    def __call__(self, x: np.ndarray):
        x = np.atleast_1d(x).ravel()
        self._solutions.append(x.copy())

        varnames = self._get_variable_names()
        self._set_var_arr(x, varnames)

        # Match NumPy order: gradient first, then value
        grad_dict = self.gradient()
        self._set_flat_arr(grad_dict, varnames, self._flat_gradient)

        val = self._sign * self.value().item()

        # Always return, regardless of verbose
        return val, self._sign * self._flat_gradient.copy()

    def _minimize(self, verbose: bool = True):
        """Minimize function using L-BFGS-B. Matches numpy implementation."""
        self._iteration = 0
        self._verbose = verbose
        varnames = self._get_variable_names()
        
        if not varnames:
            if verbose:
                print("No parameters to optimize.")
            return
        
        # Initialize flat arrays
        self._initialize_flat_arrays(varnames)
        
        # Check initial gradient
        grad_dict = self.gradient()
        sign_grad = {name: self._sign * grad_dict[name].detach().double().cpu().numpy() 
                     for name in varnames if name in grad_dict}
        self._set_flat_arr(grad_dict, varnames, self._flat_gradient)
        
        if np.max(np.abs(self._flat_gradient)) <= PGTOL:
            if verbose:
                print("Gradient near zero before the first iteration. "
                      "Returning the current value.")
            return
        
        # Run optimization
        result = self._try_minimize(5)

        # Add this guard (matching numpy version)
        if result is None or not isinstance(result, tuple) or len(result) < 3:
            if verbose:
                print("[WARNING] Optimization did not return a valid result. "
                    "Using current parameter values.", flush=True)
            return
       
        if result[2]["warnflag"] == 1:
            raise OptimixError("L-BFGS-B: too many function evaluations or too many iterations")
        if result[2]["warnflag"] == 2:
            raise OptimixError(f"L-BFGS-B: {result[2]['task']}")
        
        # Set final parameters
        self._set_var_arr(result[0], varnames)
    
    def _maximize(self, verbose: bool = True):
        """Maximize by minimizing the negative."""
        self._sign = -1.0
        try:
            self._minimize(verbose=verbose)
        finally:
            self._sign = 1.0

    def projected_gradient(self, x, g, bounds):
        pg = g.copy()
        for i, (bmin, bmax) in enumerate(bounds):
            if x[i] <= bmin and g[i] > 0:
                pg[i] = 0.0
            elif x[i] >= bmax and g[i] < 0:
                pg[i] = 0.0
        return pg
    
    def _try_minimize(self, n: int):
        if n == 0:
            raise OptimixError("Too many bad solutions")

        varnames = self._get_variable_names()

        bounds = []
        for name in varnames:
            bounds.extend(self._get_variable_bounds(name))

        self._get_flat_solution(varnames, self._flat_solution)
        x0 = self._flat_solution.copy()

        warn = False
        result = None

        # Add the Python-side iteration counter callback
        self._iter_count = getattr(self, '_iter_count', 0)
        def _callback(x):
            self._iter_count += 1
            if self._verbose:
                val = self._sign * self.value().item()
                pg_norm = np.max(np.abs(self._flat_gradient))
                print(f"  Iter {self._iter_count}: LML = {val:.6f}  |proj g| = {pg_norm:.5E}", flush=True)

        try:
            result = fmin_l_bfgs_b(
                self, x0, bounds=bounds,
                factr=FACTR, pgtol=PGTOL, 
                disp=0, # Must be 0 to let the callback handle printing
                callback=_callback if self._verbose else None
            )
        except OptimixError:
            warn = True
        else:
            warn = result[2]["warnflag"] > 0

            if result is not None:
                self._set_var_arr(result[0], varnames)
                self._final_grad_norm = float(np.max(np.abs(result[2]['grad'])))
                self._n_iterations = result[2]['nit']
                
        if warn:
            xs = self._solutions
            if len(xs) < 2:
                raise OptimixError("Bad solution at the first iteration.")
            self._set_var_arr(xs[-2] / 2 + xs[-1] / 2, varnames)
            return self._try_minimize(n - 1)

        return result