# scalabath

scalable simulation of quantum transport in extended bosonic environment

This package provides a framework for general quantum dynamics simulations, including unitary time-evolution and Lindblad master equation time-evolution. 
More specifically, it can be used to simulate the dynamics of a quantum system coupled to a bosonic environment. The system can be a single site or a multi-site system, and the environment can be a single mode or a multi-mode environment.
It is built on top of JAX and provides a flexible and efficient way to simulate the dynamics of quantum systems in extended bosonic environments.

# Applications

One major application of the package is to simulate a reduced open quantum system where the multi-site, multi-mode environment is replaced by a single multi-mode bosonic bath through random phase approximation.
There are two scenarios: (1) the Hamiltonian case and (2) the Lindblad case.

## Hamiltonian case

Consider the following problem:

$$
\hat{H}_{B}=\sum_{k=1}^{N}\omega_{k}\hat{b}_{k}^{\dagger}\hat{b}_{k},
$$

and

$$
\hat{H}_{SB}=\sum_{j=1}^{n}\sum_{k=1}^{N}|j\rangle\langle j|\otimes (g_{k} e^{i\phi_{jk}} \hat{b}_{k} + g_{k}^{*} e^{-i\phi_{jk}^{*}} \hat{b}_{k}^{\dagger}).
$$

Then the BCF is given by:

$$
C_{jj^{\prime}}(t)=\sum_{k}g_{k}g_{k}^{*}e^{i(\phi_{jk}-\phi_{j^{\prime}k})}c_{k}(t),
$$

where $c_{k}(t)$ corresponds to the k-th mode of the bath. Of course this $C_{jj^{\prime}}(t)$ is not diagonal.

To make it diagonal, we make $\phi_{jk}$ random numbers uniformly distributed in $[0,2π)$. Then we have:

$$
\mathbb{E}(C_{jj^{\prime}}(t))=\sum_{k}g_{k}g_{k}^{*}\mathbb{E}(e^{i(\phi_{jk}-\phi_{j^{\prime}k})})(c_{k}(t))=\sum_{k}g_{k}g_{k}^{*}c_{k}(t)\delta_{jj^{\prime}}.
$$

## Coupled Lindbladian case

In the coupled Lindbladian case, we have:

$$
\partial_{t}\hat{\rho}=-i[\hat{H}_{S}+\hat{H}_{B}+\hat{H}_{SB},\hat{
\rho}]+\sum_{k,k^{\prime}}\Gamma_{kk^{\prime}}(2\hat{b}_{k}\hat{
\rho}_{k^{\prime}}^{\dagger}-\{\hat{b}_{k^{\prime}}^{\dagger}\hat{b}_{k},\hat{
\rho}\})
$$

$$
\hat{H}_{B}=\sum_{k,k^{\prime}}h_{kk^{\prime}}\hat{b}_{k}^{\dagger}\hat{b}_{k^{\prime}};
$$

and 

$$
\hat{H}_{SB}=\sum_{j=1}^{n}\sum_{k=1}^{N}|j\rangle\langle j|\otimes(g_{k}e^{i\phi_{j}}\hat{b}_{k}+g_{k}^{*}e^{-i\phi_{j}^{*}}\hat{b}_{k}^{\dagger}).
$$

where again, $\phi_{j}$ and $\phi_{j^{\prime}}$ are random numbers uniformly distributed in $[0, 2π)$. Then we have:

$$
C_{jj^{\prime}}(t)=g^{\dagger}e^{(-ih-\Gamma)t}ge^{i(\phi_{j}-\phi_{j^{\prime}})}
$$

Taking expectation, we have:

$$
\mathbb{E}(C_{jj^{\prime}}(t))=g^{\dagger}e^{(-ih-\Gamma)t}g\mathbb{E}(e^{i(\phi_{j}-\phi_{j^{\prime}})})=g^{\dagger}e^{(-ih-\Gamma)t}g\delta_{jj^{\prime}}.
$$

## Example: Carrier transport in Rubrene crystall

We simulate a 1D Holstein model of Rubrene crystal.  

## Example: Exciton transport in bio-complex
We simulate a model for BCHL. 