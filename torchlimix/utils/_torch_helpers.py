import torch
import warnings

def torch_is_all_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def torch_trace(x: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Compute trace matching NumPy behavior.
    
    For 2D: standard trace
    For 3D (n, n, k): trace over dims 0,1, returning shape (k,)
    
    NumPy behavior:
    >>> x = np.random.randn(3, 3, 5)
    >>> np.trace(x).shape  # (5,)
    """
    if x.ndim == 2:
        result = torch.trace(x)
    elif x.ndim == 3:
        diag = torch.diagonal(x, dim1=0, dim2=1)  # (k, n)
        result = diag.sum(dim=1)  # (k,)
    else:
        raise ValueError(f"trace not implemented for ndim={x.ndim}")
    if debug:
        print(f"  [trace] input shape: {x.shape} -> output shape: {result.shape}")
    
    return result

def torch_vec(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten first two dimensions in Fortran (column-major) order.
    Preserves remaining dimensions.
    
    Matches NumPy: reshape(x, (-1,) + x.shape[2:], order="F")
    
    Examples:
        (m, n)    -> (m*n,)
        (m, n, k) -> (m*n, k)
        (m, n, k, l) -> (m*n, k, l)
    """
    if x.ndim < 2:
        return x.reshape(-1)
    
    # Fortran order for first two dims: transpose them, then use C-order reshape
    x_perm = x.transpose(0, 1).contiguous()
    new_shape = (-1,) + x.shape[2:]
    return x_perm.reshape(new_shape)

def torch_unvec(x: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Reverse of vec - reshape in Fortran order.
    
    Matches NumPy: reshape(x, shape, order="F")
    """
    if len(shape) < 2:
        return x.reshape(shape)
    # F-order reshape:
    swapped_shape = (shape[1], shape[0]) + shape[2:]
    x_reshaped = x.reshape(swapped_shape)
    return x_reshaped.transpose(0, 1).contiguous()

def torch_mkron(a: torch.Tensor, b: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Kronecker product matching NumPy's _mkron.
    """
    if a.ndim == 3:
        # (n, n, k) -> (k, n, n)
        a_t = a.permute(2, 0, 1).contiguous()  
        # Compute kron for each slice
        results = []
        for i in range(a_t.shape[0]):
            results.append(torch.kron(a_t[i], b))
        
        # Stack: (k, n*m, n*m)
        stacked = torch.stack(results, dim=0)
        
        # Transpose: (k, n*m, n*m) -> (n*m, n*m, k)
        result = stacked.permute(1, 2, 0)
        
        return result
    else:
        return torch.kron(a, b)

def torch_sum2diag(A, D, out=None):
    """
    Add values D to the diagonal of matrix A.
    
    Args:
        A (torch.Tensor): Left-hand side matrix
        D (torch.Tensor or float): Values to add to diagonal
        out (torch.Tensor, optional): Output tensor to store result
        
    Returns:
        torch.Tensor: Resulting matrix A + diag(D)
    """
    if out is None:
        out = A.clone()
    else:
        out.copy_(A)
    
    if not isinstance(D, torch.Tensor):
        D = torch.tensor(D, dtype=A.dtype, device=A.device)
    
    # Add D to the diagonal elements (equivalent to NumPy's einsum("ii->i", out)[:] += D)
    diagonal_indices = torch.arange(min(out.shape[-2:]), device=out.device)
    out[diagonal_indices, diagonal_indices] += D
    
    return out

def torch_safe_log(x: torch.Tensor) -> torch.Tensor:
    """
    Safe logarithm that clips input to avoid numerical issues.
    Matches the NumPy implementation structure exactly.
    
    Args:
        x: Input tensor
        
    Returns:
        Logarithm of clipped input
    """
    eps_small = torch.tensor(torch.finfo(torch.float64).eps * 1000, dtype=torch.double)
    return torch.log(torch.clamp(x, min=eps_small, max=torch.inf))

def torch_rsolve(A: torch.Tensor, b: torch.Tensor,
                              use_lstsq_fallback: bool = True) -> torch.Tensor:
    """
    Comprehensive robust solver with multiple fallback strategies.
    Attempts to replicate numpy_sugar.linalg.rsolve behavior.
    Replaces torch.linalg.solve
    """
    original_dtype = A.dtype
    rcond = torch.finfo(torch.float64).eps
    
    try:
        # First attempt: Direct solve with LU decomposition
        return torch.linalg.solve(A, b)
        
    except (torch.linalg.LinAlgError, RuntimeError):
        try:
            det = torch.linalg.det(A)
            if torch.abs(det) < 1e-14:
                warnings.warn("Matrix is singular. Using pseudo-inverse.", RuntimeWarning)
                A_pinv = torch.linalg.pinv(A, rcond=rcond)
                return A_pinv @ b
            
            if use_lstsq_fallback:
                solution = torch.linalg.lstsq(A, b, rcond=rcond).solution
                return solution
                
        except (torch.linalg.LinAlgError, RuntimeError):
            pass
    
    msg = "All solver attempts failed. Setting solution to zero."
    warnings.warn(msg, RuntimeWarning)
    return torch.zeros(A.shape[0], dtype=original_dtype, device=A.device)

def torch_mdot(*args, debug: bool = False) -> torch.Tensor:
    """Chain of torch_dot operations."""
    from functools import reduce
    if debug:
        result = args[0]
        for i, arg in enumerate(args[1:], 1):
            result = torch_dot(result, arg, debug=debug)
        return result
    return reduce(lambda a, b: torch_dot(a, b, debug=False), args)

def torch_ddot(a, b):
    """
    Equivalent to numpy_sugar.linalg.ddot using PyTorch broadcasting.
    """
    if not isinstance(a, torch.Tensor): a = torch.tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.tensor(b)

    a_is_1d = a.ndim == 1
    b_is_1d = b.ndim == 1

    if not (a_is_1d ^ b_is_1d) or (max(a.ndim, b.ndim) != 2):
        raise ValueError("Inputs must consist of one 1D tensor and one 2D tensor")

    if a_is_1d:
        # a is (N,), b is (N, M)
        return a.view(-1, 1) * b
    else:
        # a is (N, M), b is (M,).
        return a * b


def torch_dot(a: torch.Tensor, b: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Matrix multiplication matching NumPy's specific internal torch_dot behavior.
    Contracts axis min(1, a.ndim-1) of a with axis 0 of b.
    If a.ndim > b.ndim and result.ndim == 3, transpose result.
    """
    if a is None or b is None:
        return None
    
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    
    # Determine contraction axis
    a_axis = min(1, a.ndim - 1)
    
    # tensordot with dims parameter
    r = torch.tensordot(a, b, dims=([a_axis], [0]))
    
    # Apply transpose when a.ndim > b.ndim and result is 3D
    if a.ndim > b.ndim and r.ndim == 3:
        r = r.permute(0, 2, 1)
    
    if debug:
        print(f"  [torch_dot] a: {tuple(a.shape)}, b: {tuple(b.shape)} -> result: {tuple(r.shape)}, Transposed: {a.ndim > b.ndim and r.ndim == 3}")
    
    return r


def torch_lu_factor(Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs LU factorization using PyTorch but returns the result in the 
    SciPy/NumPy format: (LU_matrix, 0-based_pivots).
    
    Equivalent to the NumPy lu_factor wrapper, handling the (0, 0) edge case.
    """
    # Check for zero-sized matrix (sum(A.shape) == 0 in NumPy)
    if Z.shape[0] == 0 and Z.shape[1] == 0:
        return (torch.zeros((0, 0), dtype=Z.dtype, device=Z.device), 
                torch.zeros((0,), dtype=torch.int32, device=Z.device))

    # PyTorch lu_factor_ex returns (lu, 1-based_pivots, info)
    lu_torch, piv_torch, _ = torch.linalg.lu_factor_ex(Z)
    
    # Convert 1-based pivots (LAPACK style) to 0-based (SciPy/C style)
    piv_0based = piv_torch - 1
    
    # Return the NumPy style tuple (LU, 0-based pivots)
    return (lu_torch, piv_0based)


def torch_lu_solve(LU_and_piv: tuple, b: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    LU solve handling 1D, 2D, and 3D right-hand sides.
    
    NOTE: This function assumes the input 'pivots' (from LU_and_piv) are 
    0-based (NumPy/SciPy style) and converts them to 1-based (LAPACK style) 
    before calling the native PyTorch solver.
    """
    LU, pivots = LU_and_piv
    
    # Equivalent to NumPy check: if A[0].shape[1] == 0 and b.shape[0] == 0:
    # If the LU matrix is zero-sized, return a zero-sized result
    if LU.shape[0] == 0:
        if b.ndim == 1:
            # Result shape (0,) for 1D input
            return torch.zeros((0,), dtype=b.dtype, device=b.device)
        # Result shape (0, k) for 2D/3D input
        return torch.zeros((0, b.shape[-1]), dtype=b.dtype, device=b.device)
    
    #  Convert 0-based pivots back to 1-based (LAPACK style) 
    piv_1based = pivots + 1
    
    if debug:
        print(f"  [lu_solve] LU: {LU.shape}, b: {b.shape}, Pivots (0-based): {pivots.detach().cpu().numpy()}")
    
    if b.ndim == 1:
        # (n,) -> (n, 1) -> solve -> (n,)
        result = torch.linalg.lu_solve(LU, piv_1based, b.unsqueeze(-1)).squeeze(-1)
    elif b.ndim == 2:
        # (n, k) -> solve -> (n, k)
        result = torch.linalg.lu_solve(LU, piv_1based, b)
    elif b.ndim == 3:
        # (n, m, k) - solve for each slice along last dim
        n, m, k = b.shape
        # Reshape to (n, m*k), solve, reshape back
        b_flat = b.reshape(n, m * k)
        result_flat = torch.linalg.lu_solve(LU, piv_1based, b_flat)
        result = result_flat.reshape(n, m, k)
    else:
        raise ValueError(f"lu_solve not implemented for b.ndim={b.ndim}")
    
    if debug:
        print(f"  [lu_solve] result: {result.shape}")
    
    return result

def torch_lu_slogdet(LU_and_piv: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log-determinant from LU factorization (matching NumPy's lu_slogdet).
    
    Args:
        LU_and_piv: Tuple of (LU_matrix, 0-based_pivots) from torch_lu_factor
        
    Returns:
        (sign, logabsdet): Sign and log of absolute determinant
    """
    LU, pivots = LU_and_piv
    
    # Log determinant from diagonal of LU
    diag = torch.diagonal(LU)
    logabsdet = torch.log(torch.abs(diag)).sum()
    
    # Sign from diagonal
    sign = torch.prod(torch.sign(diag))
    
    n_exchanges = (pivots != torch.arange(pivots.size(0), device=pivots.device)).sum()
    if n_exchanges % 2 == 1:
        sign *= -1.0
    
    return sign, logabsdet

def torch_safe_logdet(M: torch.Tensor, eps: float = 1e-6, fallback: float = 0.0, raise_on_fail: bool = True) -> torch.Tensor:
    """
    Safely compute the log-determinant of a matrix using slogdet.
    Falls back or raises if matrix is not positive-definite.

    Parameters
    ----------
    M : torch.Tensor
        A square symmetric matrix (e.g. H, AᵀA).
    eps : float
        Minimum eigenvalue or jitter for stability.
    fallback : float
        Value to return if computation fails (only if raise_on_fail is False).
    raise_on_fail : bool
        Whether to raise an error if the matrix is not positive-definite.

    Returns
    -------
    torch.Tensor
        The log-determinant of M (scalar).
    """
    try:
        sign, logabsdet = torch.linalg.slogdet(M)

        if sign <= 0 or not torch.isfinite(logabsdet):
            msg = f"[safe_logdet] Matrix not PD or invalid logdet (sign={sign}, logdet={logabsdet})"
            if raise_on_fail:
                raise RuntimeError(msg)
            else:
                print(msg + f" → returning fallback={fallback}")
                return torch.tensor(fallback, device=M.device)

        return logabsdet

    except Exception as e:
        if raise_on_fail:
            raise RuntimeError(f"[safe_logdet] Failed: {e}")
        else:
            print(f"[safe_logdet] Exception: {e} → returning fallback={fallback}")
            return torch.tensor(fallback, device=M.device)