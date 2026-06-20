"""JAX-based simulations of open quantum dynamics on lattice systems."""

from scalabath.constants import Constants
from scalabath.operators_base import boson, tight_binding_1d, tight_binding_2d, tls
from scalabath.operators_groups import (
    BosonOperatorGroup,
    ComposedOperatorGroups,
    OperatorGroup,
    SpinOperatorGroup,
    TightBindingChainOperatorGroup,
)
from scalabath.simulations_lindblad import (
    CoupledLindbladTrajectorySimulation,
    LindbladSimulation,
)
from scalabath.simulations_unitary import (
    SystemBathUnitarySimulation,
    UnitarySimulation,
)
from scalabath.systems import (
    DensityMatrixEnsemble,
    PureStatesEnsemble,
    TensorProductDensityMatrixEnsemble,
    TensorProductPureStatesEnsemble,
)
from scalabath.utilities import ABAd, compose

__version__ = "0.1.0"

__all__ = [
    "ABAd",
    "BosonOperatorGroup",
    "ComposedOperatorGroups",
    "CoupledLindbladTrajectorySimulation",
    "Constants",
    "DensityMatrixEnsemble",
    "LindbladSimulation",
    "OperatorGroup",
    "PureStatesEnsemble",
    "SpinOperatorGroup",
    "SystemBathUnitarySimulation",
    "TensorProductDensityMatrixEnsemble",
    "TensorProductPureStatesEnsemble",
    "TightBindingChainOperatorGroup",
    "UnitarySimulation",
    "__version__",
    "boson",
    "compose",
    "tight_binding_1d",
    "tight_binding_2d",
    "tls",
]
