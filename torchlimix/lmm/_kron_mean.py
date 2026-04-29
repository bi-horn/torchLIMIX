'''Torch implementation of the following multi-trait glimix-core class:
KronMean (https://github.com/limix/glimix-core/blob/master/glimix_core/mean/_kron.py)
'''

import torch
import torch as nn
from torch import nn
from torchlimix.utils._torch_helpers import torch_vec, torch_unvec
from torchlimix.optimizer._optimizer import TorchFunction

class KronMeanTorch(TorchFunction):
    """
    PyTorch version of Kronecker mean function, (A⊗X)vec(B).
    """

    def __init__(self, A: torch.Tensor, X: torch.Tensor, device: str = "cpu"):
        TorchFunction.__init__(self, "KronMean")
        self.device = device

        self.register_buffer("_A", A.double().to(device))
        self.register_buffer("_X", X.double().to(device))

        num_params = X.shape[1] * A.shape[1]
        
        self._vecB = nn.Parameter(
            torch.zeros(num_params, device=device, dtype=torch.double),
            requires_grad=False
        )
        self._nparams = num_params

    @property
    def nparams(self):
        return self._nparams

    @property
    def A(self):
        return self._A

    @property
    def X(self):
        return self._X

    @property
    def AX(self):
        return torch.kron(self.A, self.X)

    def value(self):
        return self.AX @ self._vecB

    def gradient(self):
        return {"vecB": self.AX}

    @property
    def B(self):
        return torch_unvec(self._vecB, (self.X.shape[1], self.A.shape[0]))

    @B.setter
    def B(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.double, device=self.device)
        else:
            v = v.to(dtype=torch.double, device=self.device)
        self._vecB.data = torch_vec(v)

    def __str__(self):
        tname = type(self).__name__
        msg = f"{tname}(A=..., X=...)\n"
        mat = str(self.B.detach().cpu().numpy())
        msg += "  B: " + "\n     ".join(mat.split("\n"))
        return msg