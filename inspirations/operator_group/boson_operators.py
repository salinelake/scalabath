import numpy as np
import torch as th
from qepsilon.operator_basis.boson import Boson
from qepsilon.operator_group.base_operators import OperatorGroup

class BosonOperatorGroup(OperatorGroup):
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, static: bool = False):
        self.nm = num_modes  # number of modes
        self.ns = (nmax+1)**num_modes  # number of states
        super().__init__(id, self.ns, batchsize, static)
        self.boson = Boson(nmax)

    def add_operator(self, boson_sequence: str, prefactor: float = 1):
        """
        Add an operator to the group. Stored as a string of boson operator names. 
        Args:
            boson_sequence: str, the boson sequence. Example: "+-1" for creation of mode 0, annihilation of mode 1, identity of mode 2.
        """
        if len(boson_sequence) != self.nm:
            raise ValueError("length of boson_sequence must be the number of modes")
        self._ops.append(boson_sequence)
        self._prefactors.append(prefactor)
        return
    
    def sum_operators(self):
        total_ops = 0
        if len(self._ops) != len(self._prefactors):
            raise ValueError("The number of operators and prefactors do not match")
        for op, prefactor in zip(self._ops, self._prefactors):
            total_ops += self.boson.get_composite_ops(op) * prefactor
        return total_ops

class IdentityBosonOperatorGroup(BosonOperatorGroup):
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1):
        super().__init__(num_modes, id, nmax, batchsize)
        self._ops.append("I"*num_modes)
        self._prefactors.append(1.0)
    def add_operator(self, boson_sequence: str):
        raise ValueError("IdentityBosonOperatorGroup does not support adding operators")

    def _sample(self, dt: float):
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device)

class StaticBosonOperatorGroup(BosonOperatorGroup):
    """
    This class deals with a group of operators (composite boson operators on n-mode systems) and a static coefficient. 
    Each operator is a direct product of boson operators. It is specified by a string of boson operator names.  
    For example, "UDI" is the 2-body operator $$a^\dagger_0 \otimes a_1 \otimes I_2$$.
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int = 1, coef: float = 1.0, static: bool = True, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize, static)
        ## require coef is a scalar
        if not isinstance(coef, float):
            raise ValueError("coef must be a float scalar")
        if requires_grad:
            self.register_parameter("coef", th.nn.Parameter(th.tensor(coef, dtype=th.float)))
        else:
            self.register_buffer("coef", th.tensor(coef, dtype=th.float))

    def _sample(self, dt: float = 1.0):
        """
        This function returns the sum of the operators in the group.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        ops = self.sum_operators()
        return ops, th.ones(self.nb, dtype=ops.dtype, device=ops.device) * self.coef

class HarmonicOscillatorBosonOperatorGroup(BosonOperatorGroup):
    """
    static operator H = \sum_i \omega_i (a_i^\dagger a_i + 1/2)
    """
    def __init__(self, num_modes, id: str, nmax: int, batchsize: int, omega: th.Tensor, requires_grad: bool = False):
        super().__init__(num_modes, id, nmax, batchsize)
        if self.nm == 1:
            if isinstance(omega, float) is False:
                raise ValueError("omega must be a float scalar for single mode system")
            if omega <= 0:
                raise ValueError("omega must be positive")
            log_omega = th.tensor([np.log(omega)], dtype=th.float)
        else:
            if omega.shape != (self.nm,):
                raise ValueError("omega specifies the frequency of each mode. It must have shape (num_modes,)")
            if omega.dtype != th.float:
                raise ValueError("omega specifies the frequency of each mode. It must be a real float tensor.")
            if omega.min()<=0:
                raise ValueError("omega specifies the frequency of each mode. It must be all positive.")
            log_omega = th.log(omega)
        if requires_grad:
            self.register_parameter("log_omega", th.nn.Parameter(log_omega))
        else:
            self.register_buffer("log_omega", log_omega)
        for idx in range(self.nm):
            _ops = ['I'] * self.nm
            _ops[idx] = 'N'
            self._ops.append(''.join(_ops))
        self._ops.append('I'*self.nm)
        
    @property
    def omega(self):
        return th.exp(self.log_omega)
    
    def add_operator(self, boson_sequence: str):
        raise ValueError("HarmonicOscillatorBosonOperatorGroup does not support manually adding operators")

    def sum_operators(self):
        raise ValueError("HarmonicOscillatorBosonOperatorGroup does not support summing operators with `sum_operators`")

    def _sample(self, dt: float):
        """
        This function returns H = \sum_i \omega_i (a_i^\dagger a_i + 1/2) and a all-one coefficient tensor.
        Args:
            dt: float, the time step.
        Returns:
            ops: th.Tensor, the operator matrix of shape (self.ns, self.ns).
            coef: th.Tensor, the coefficient of shape (self.nb,).
        """
        total_ops = 0
        for idx, op in enumerate(self._ops[:-1]):
            total_ops += self.boson.get_composite_ops(op) * self.omega[idx]
        ## add zero-point energy
        total_ops += self.boson.get_composite_ops(self._ops[-1]) * self.omega.sum() * 0.5
        return total_ops, th.ones(self.nb, dtype=total_ops.dtype, device=total_ops.device)
    