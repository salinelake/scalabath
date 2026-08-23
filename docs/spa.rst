Stochastic phase algorithm
==========================

Purpose and scope
-----------------

The stochastic phase algorithm (SPA) targets single-quasiparticle transport on
a lattice whose sites couple to identical, independent Gaussian bosonic
environments. A direct Holstein representation assigns a separate bath to
every site, so its explicit bath Hilbert space grows exponentially with the
number of sites.

SPA reduces the spatial multiplicity of those baths. Instead of retaining one
local bath per site, it uses ``R`` shared bath copies and couples them to each
site with independent random phases. This changes the required bath Hilbert
space without treating the system–bath state semiclassically.

The method's system-size-independent error statement in the accompanying
paper relies on the single-quasiparticle sector and identical, independent
local environments. It should not be assumed for a generic many-body system,
for spatially correlated baths, or for non-identical local bath correlations.

Phase-dressed coupling
----------------------

For site projectors :math:`|i\rangle\langle i|`, bath-copy index :math:`a`,
and bath-mode index :math:`\alpha`, the phase-dressed coupling is

.. math::

   H_{\mathrm{sb},\theta}^{(R)} =
   \sum_i |i\rangle\langle i| \otimes
   \sum_{a=1}^{R}\sum_\alpha \frac{g_\alpha\omega_\alpha}{\sqrt{R}}
   \left(r_i^{(a)} b_{a\alpha}^{\dagger}
   + r_i^{(a)*} b_{a\alpha}\right),

where :math:`r_i^{(a)}=\exp(i\theta_i^{(a)})` and every phase is sampled
independently and uniformly on :math:`[0,2\pi)`. The key phase average is

.. math::

   \mathbb{E}_\theta\left[
   \frac{1}{R}\sum_{a=1}^{R}r_i^{(a)}r_j^{(a)*}
   \right]=\delta_{ij}.

Consequently, phase averaging restores the target diagonal two-point bath
correlation functions for any ``R``. Higher-order correlations are not all
reproduced at finite ``R``; convergence with the number of bath copies and the
number of phase realizations must therefore be checked.

Explicit and compressed baths
-----------------------------

``scalabath`` supports the two propagation settings used by SPA:

* :class:`~scalabath.simulations_unitary.SystemBathUnitarySimulation` evolves
  explicit, undamped bosonic modes with a tensorized second-order split step.
* :class:`~scalabath.simulations_lindblad.CoupledLindbladTrajectorySimulation`
  evolves damped auxiliary modes with stochastic quantum trajectories. It can
  accept a coupled bath Hamiltonian or groups of coupled modes.

A compressed local bath may be represented by an energy matrix ``K``, damping
matrix ``Gamma``, and coupling vector ``epsilon`` chosen so that

.. math::

   c(t) \simeq \epsilon^\dagger
   \exp[(-iK-\Gamma)t]\epsilon

over the required time window. Bath fitting is outside the current package;
the resulting matrices are inputs to the simulation classes. In the
coupled-Lindblad convention used by the accompanying work, ``Gamma`` is half
the coefficient that would appear as the conventional single-mode Lindblad
rate. Keep that convention explicit when translating fitted parameters into
jump operators.

Mapping the method to the package
---------------------------------

The package exposes solver and operator components rather than a single
high-level ``run_spa`` function. A typical workflow is:

#. Construct the lattice Hamiltonian and site projectors.
#. Choose explicit physical bath modes or provide a compressed auxiliary-bath
   representation.
#. Draw phase realizations from a recorded random seed.
#. Build one batched system–mode coupling operator per shared mode, including
   the ``1 / sqrt(R)`` factor when ``R > 1``.
#. Initialize thermal bath states or trajectory states.
#. Propagate, reduce over bath axes, and average system observables over the
   ensemble.
#. Repeat with tighter time steps, larger bosonic cutoffs, more phase samples,
   and more bath copies until the reported quantities are converged.

The complete model-building patterns used for the paper applications are in
the :doc:`examples`.
