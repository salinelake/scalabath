"""
This module deals with density matrices.
"""

import torch as th
from qepsilon.operator_basis.tls import Pauli
from qepsilon.utilities import compose, ABAd, qubitconf2idx, trace, apply_to_pse, expectation_pse



class PureStatesEnsemble(th.nn.Module):
    """
    Base class for ensembles of pure states.
    """
    def __init__(self, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.nb = batchsize
        self.register_buffer("_pse", None) ## initialize later. Shape will be (self.nb, self.ns)

    ############################################################
    # Getters and setters for the pure states
    ############################################################
    def get_pse(self):
        return self._pse
    
    def set_pse(self, pse: th.Tensor):
        """
        This function sets the pure states.
        Args:
            pse: a complex tensor. Shape: (self.nb, self.ns).
        """
        if pse.dtype != th.cfloat:
            raise ValueError("Pure states must be a complex tensor (th.cfloat).")
        if pse.shape == (self.ns,):
            pse = pse.unsqueeze(0)
            if self._pse is None:
                self._pse = pse.repeat(self.nb, 1)
            else:
                self._pse = pse.repeat(self.nb, 1).to(self._pse.device)
        elif pse.shape == (self.nb, self.ns):
            if self._pse is None:
                self._pse = pse
            else:
                self._pse = pse.to(self._pse.device)
        else:
            raise ValueError("Pure states must have shape (ns) or (batchsize, ns).")
            
    @property
    def norm(self):
        return th.norm(self._pse, dim=1)

    ############################################################
    # Basic operations on pure state ensemble. Methods below do not update the stored pure states. Use setter if you want to update.
    ############################################################
    def normalize(self, pse: th.Tensor):
        """
        This function normalizes the pure states.
        Args:
            pse: the pure states to be normalized. Shape: (batchsize, ns).
        """
        return pse / self.norm[:, None]
    
    def get_expectation(self, operator: th.Tensor):
        """
        This function computes the expectation of an operator on the pure state ensemble.
        Args:
            operator: the operator to get the expectation. Shape: (ns, ns).
        Returns:
            expectation: the expectation of the operator. Shape: (batchsize).
        """
        if operator.shape == (self.ns, self.ns) or operator.shape == (self.nb, self.ns, self.ns):
            pass
        else:
            raise ValueError("Operator must have shape (ns, ns) or (batchsize, ns, ns).")
        if operator.dtype != th.cfloat:
            raise ValueError("Operator must be a complex tensor (th.cfloat).")
        return expectation_pse(self._pse, operator)



class DensityMatrix(th.nn.Module):
    """
    Base class for density matrices.
    """
    def __init__(self, num_states: int, batchsize: int = 1):
        super().__init__()
        self.ns = num_states
        self.nb = batchsize
        self.register_buffer("_rho", None) ## initialize later. Shape will be (self.nb, self.ns, self.ns)

    ############################################################
    # Getters and setters for the density matrix
    ############################################################
    def get_rho(self):
        return self._rho
    
    def set_rho(self, rho: th.Tensor):
        """
        This function sets the density matrix.
        Args:
            rho: a complex tensor. Shape: (self.nb, self.ns, self.ns).
        """
        if rho.dtype != th.cfloat:
            raise ValueError("Density matrix must be a complex tensor (th.cfloat).")
        if rho.shape == (self.ns, self.ns):
            rho = rho.unsqueeze(0)
            if self._rho is None:
                self._rho = rho.repeat(self.nb, 1, 1)
            else:
                self._rho = rho.repeat(self.nb, 1, 1).to(self._rho.device)
        elif rho.shape == (self.nb, self.ns, self.ns):
            if self._rho is None:
                self._rho = rho
            else:
                self._rho = rho.to(self._rho.device)
        else:
            raise ValueError("Density matrix must have shape (2^n, 2^n) or (batchsize, 2^n, 2^n).")
            
    @property
    def trace(self):
        return trace(self._rho)

    ############################################################
    # Basic operations on density matrices. Methods below do not update the stored density matrix. Use setter if you want to update.
    ############################################################
    def normalize(self, rho: th.Tensor):
        """
        This function normalizes the density matrix.
        Args:
            rho: the density matrix to be normalized.
        """
        return rho / trace(rho)[:, None, None]
    
