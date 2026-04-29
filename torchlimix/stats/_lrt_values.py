'Adapted from https://github.com/limix/limix/blob/master/limix/stats/_lrt.py'

import torch

def lrt_values(null_lml, alt_lmls):
    """
    Compute likelihood ratio statistics (LRTs) for a set of alternative models
    relative to a null model.

    Parameters
    ----------
    null_lml : float or torch.Tensor
        Log marginal likelihood of the null model (scalar).
    alt_lmls : array-like or torch.Tensor
        Log marginal likelihoods of alternative models (1D tensor or array).

    Returns
    -------
    torch.Tensor
        Likelihood ratio statistics (LRS) = -2 * (null_lml - alt_lmls)
    """
    lrs = -2 * null_lml + 2 * alt_lmls
    lrs = torch.clamp(lrs, min=0.0)  

    return lrs