import torch as th
import numpy as np
from qepsilon.operator_basis.tls import Pauli
from qepsilon.utilities import compose
from qepsilon.system.particles import Particles
from qepsilon.operator_group.base_operators import OperatorGroup
import warnings
from typing import Callable
###########################################################################
# Base class for Pauli operator groups.
###########################################################################
class PauliOperatorGroup(OperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems). 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, static: bool = False):
        self.nq = n_qubits
        ns = 2**n_qubits
        super().__init__(id, ns, batchsize, static)
        self.pauli = Pauli(n_qubits)

    def set_batch_rescaling(self, batch_rescaling: th.Tensor):
        """
        Set the batch rescaling of the operator group. The final coefficiant will be multiplied by the batch rescaling.
        Args:
            batch_rescaling: th.Tensor, the batch rescaling of shape (self.nb,).
        """
        if batch_rescaling.shape[0] != self.nb:
            raise ValueError("The shape of batch_rescaling must be (self.nb,).")
        ## check if the batch_rescaling is already a registered buffer
        if hasattr(self, "batch_rescaling"):
            raise ValueError("batch_rescaling is already set. If you want to change the batch rescaling, you need to create a new operator group.")
        else:
            self.register_buffer("batch_rescaling", batch_rescaling)

    def add_operator(self, PauliSequence: str, prefactor: float = 1):
        """
        Add an operator to the group. Stored as a string of Pauli operator names. 
        Args:
            PauliSequence: str, the Pauli sequence. Example: "XI"
        """
        if len(PauliSequence) != self.nq:
            raise ValueError("length of PauliSequence must be the number of qubits")
        self._ops.append(PauliSequence)
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
            total_ops += self.pauli.get_composite_ops(op) * prefactor
        return total_ops


class IdentityPauliOperatorGroup(PauliOperatorGroup):
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1):
        super().__init__(n_qubits, id, batchsize)
        self.add_operator("I"*n_qubits)

    def _sample(self, dt: float):
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticPauliOperatorGroup(PauliOperatorGroup):
    """
    This class deals with a group of operators (composite Pauli operators on n-qubit systems) and a static coefficient. 
    Each operator is a direct product of Pauli operators. It is specified by a string of Pauli operator names.  For example, "XI" is the 2-body operator X_1 \otimes I_2.
    """
    def __init__(self, n_qubits: int, id: str, batchsize: int = 1, coef: float = 1, static: bool = True, requires_grad: bool = False):
        super().__init__(n_qubits, id, batchsize, static)
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
        coef_scalar = self.coef
        if hasattr(self, "batch_rescaling"):
            coef = coef_scalar * self.batch_rescaling
        else:
            coef = coef_scalar * th.ones(self.nb, dtype=ops.dtype, device=ops.device)
        return ops, coef
 