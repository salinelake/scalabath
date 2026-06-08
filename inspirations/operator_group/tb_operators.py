import torch as th
import numpy as np
from qepsilon.operator_basis.tight_binding import TightBinding
from qepsilon.system.particles import Particles
from qepsilon.operator_group.base_operators import OperatorGroup
import warnings


###########################################################################
# Base class for Tight Binding operator groups.
###########################################################################

class TightBindingOperatorGroup(OperatorGroup):
    r"""
    This class represents a group of composite Tight Binding operators on n-site systems.

    Each operator in this group is specified by a string of Tight Binding operator names.
    For example, "XXLXX" is the hopping operator :math:`| 1\rangle\langle 2 |`.
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, static: bool = False):
        self.ns = n_sites
        super().__init__(id, n_sites, batchsize, static)
        self.tb = TightBinding(n_sites)
    
    def add_operator(self, TBSequence: str, prefactor: float = 1):
        """
        Add an operator to the group. Stored as a string of Tight Binding operator names. 
        Args:
            TBSequence: str, the Tight Binding sequence. Example: "XXLXX. Contains one and only one non-`X` character, choosing from `L`, `R`, `N`.
        """
        if len(TBSequence) != self.ns:
            raise ValueError("length of TBSequence must be the number of sites")
        self._ops.append(TBSequence)
        self._prefactors.append(prefactor)
        return
    
    def sum_operators(self):
        """
        Sum up the operators in the group.
        Returns:
            total_ops: th.Tensor, the total operator matrix of shape (self.ns, self.ns).
        """
        total_ops = 0
        for op, prefactor in zip(self._ops, self._prefactors):
            total_ops += self.tb.get_composite_ops(op) * prefactor
        return total_ops


class IdentityTightBindingOperatorGroup(TightBindingOperatorGroup):
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, static: bool = True):
        super().__init__(n_sites, id, batchsize, static)
        self.add_operator("X"*n_sites)
    def _sample(self, dt: float = 1.0):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
        """
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticTightBindingOperatorGroup(TightBindingOperatorGroup):
    r"""
    This class deals with a group of operators (composite Tight Binding operators on n-site systems) and a static coefficient. 
    Each operator in this group is specified by a string of Tight Binding operator names.  For example, "XXLXX" is the hopping operator :math:`| 1\rangle\langle 2 |`.
    """
    def __init__(self, n_sites: int, id: str, batchsize: int = 1, coef: float = 1, static: bool = True, requires_grad: bool = False):
        super().__init__(n_sites, id, batchsize, static)
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef, dtype=th.float)))
        else:
            self.register_buffer("coef", th.tensor(coef, dtype=th.float))
    
    def _sample(self, dt: float = 1.0):
        """
        This function sum up the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators() 
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device) * self.coef

