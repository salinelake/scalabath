import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from scalabath.simulations import UnitarySimulation
from scalabath.operators_groups import BosonOperatorGroup, SpinOperatorGroup, ComposedOperatorGroups
from scalabath.constants import Constants




parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300, help='temperature in K')
parser.add_argument('--dt', type=float, default=0.01, help='time step in fs')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon')
parser.add_argument('--sample_time', type=float, default=100, help='sampling time in fs')
parser.add_argument('--batchsize', type=int, default=1, help='batch size')
parser.add_argument('--run_id', type=int, default=0, help='run id')
args = parser.parse_args()
epsilon = args.epsilon




## load the data
data = np.loadtxt('coupling_const.csv', delimiter=',')
num_modes = 9
omega_in_cm = data[:num_modes,0]
lambda_in_cm = data[:num_modes,1]
g_factor = np.sqrt(lambda_in_cm / omega_in_cm)

## simulation parameters
ns = 1  ## number of sites
batchsize = args.batchsize
dt = args.dt * Constants.fs    # max 0.1fs
sample_time = args.sample_time * Constants.fs
sample_freq = int(Constants.fs/dt)
nsteps = int(sample_time/dt)
temperature = args.temperature
out_folder = 'T{:.0f}_epsilon{:.1f}_dt{:.3f}fs'.format(temperature, epsilon, dt/Constants.fs)
if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)

## boson bath parameters
nmax = np.array([8, 3, 1, 1, 1, 1, 1, 1, 1])[:num_modes]
omega = omega_in_cm * Constants.cm_inverse_energy
spin_boson_coupling = g_factor * omega

## derived parameters
kbT = Constants.kb * temperature  
occupation = 1/(np.exp(omega/kbT)-1)
num_boson_states = np.prod(nmax+1)
total_num_states = 2 * num_boson_states

print('nmodes={}, total_num_states={}'.format(num_modes, total_num_states))
print('g factor:', g_factor)
print('omega [cm^-1]:', omega_in_cm, 'in internal units:', omega)
print('polaron binding energy [cm^-1]:', g_factor**2 * omega_in_cm, 'sum in meV:', (g_factor**2 * omega_in_cm).sum() * Constants.cm_inverse_energy / Constants.meV)
"""
initiate the system
"""
simulation = UnitarySimulation(hilbert_dim=total_num_states, batch_size=batchsize, dt=dt)
 
############################## add the bosonic harmonic energy term to the Hamiltonian ##############################
opg_sp_id = SpinOperatorGroup(num_spins=1, id="spin_identity", batch_size=batchsize)
opg_sp_id.add_operator('I', jnp.ones(batchsize))
opg_bs_harmonic = BosonOperatorGroup(num_modes=num_modes, id="boson_harmonic", nmax=nmax, batch_size=batchsize)
opg_bs_harmonic.add_harmonic_operators(omega)
opg_tot_1 = ComposedOperatorGroups(id="total_1", operator_groups = [opg_sp_id, opg_bs_harmonic])
simulation.add_operator_group_to_hamiltonian(opg_tot_1)

############################## add the electron-boson coupling term: sigma_n \sum_i (b_i + b_i^\dagger)
opg_sp_number = SpinOperatorGroup(num_spins=1, id="spin_number", batch_size=batchsize)
opg_sp_number.add_operator('N', jnp.ones(batchsize))
opg_bs_coupling = BosonOperatorGroup(num_modes=num_modes, id="boson_coupling", nmax=nmax, batch_size=batchsize)
for i in range(num_modes):
    descriptor = ["I"] * num_modes
    descriptor[i] = "+"
    opg_bs_coupling.add_operator("".join(descriptor), spin_boson_coupling[i] * jnp.ones(batchsize))
    descriptor[i] = "-"
    opg_bs_coupling.add_operator("".join(descriptor), spin_boson_coupling[i] * jnp.ones(batchsize))
opg_tot_2 = ComposedOperatorGroups(id="total_2", operator_groups = [opg_sp_number, opg_bs_coupling])
simulation.add_operator_group_to_hamiltonian(opg_tot_2)

""" 
Set the initial state (pure state)
"""

def sample_init_state(epsilon=0.1, num_modes=9, nmax=None, omega=None, kbT=None):
    sp_init = np.zeros(2, dtype=np.complex64)
    sp_init[0] = np.sqrt(1-epsilon)
    sp_init[1] = np.sqrt(epsilon)

    bs_init_total = None
    chosen_levels = []
    for idx in range(num_modes):
        nstate = nmax[idx] + 1
        energy_levels = np.arange(nstate) * omega[idx]
        prob_levels = np.exp(-energy_levels/kbT)
        prob_levels = prob_levels / prob_levels.sum()

        bs_init = np.zeros(nstate, dtype=np.complex64)
        choose_level = np.random.choice(nstate, p=prob_levels)
        bs_init[choose_level] = 1.0
        chosen_levels.append(choose_level)
        if idx == 0:
            bs_init_total = bs_init * 1.0
        else:
            bs_init_total = np.kron(bs_init_total, bs_init)
    system_init = np.kron(sp_init, bs_init_total)
    return system_init, chosen_levels

pse_init = []
for idx in range(batchsize):
    system_init, chosen_levels = sample_init_state(epsilon=epsilon, num_modes=num_modes, nmax=nmax, omega=omega, kbT=kbT)
    pse_init.append(jnp.array(system_init))
    print('batch {} chosen levels:'.format(idx), chosen_levels)
pse_init = jnp.stack(pse_init, axis=0)
simulation.state = pse_init

"""
Define the operators we want to observe
"""
## spin number operator
opg_bs_identity = BosonOperatorGroup(num_modes=num_modes, id="boson_identity", nmax=nmax, batch_size=batchsize)
opg_bs_identity.add_operator("".join(["I"] * num_modes), jnp.ones(batchsize))
opg_sp_n = SpinOperatorGroup(num_spins=1, id="spin_number", batch_size=batchsize)
opg_sp_n.add_operator('N', jnp.ones(batchsize))
opg_sp_x = SpinOperatorGroup(num_spins=1, id="spin_x", batch_size=batchsize)
opg_sp_x.add_operator('X', jnp.ones(batchsize))
obs_1 = ComposedOperatorGroups(id="sp_number", operator_groups = [opg_sp_n, opg_bs_identity])
obs_2 = ComposedOperatorGroups(id="sp_x", operator_groups = [opg_sp_x, opg_bs_identity])

"""
Run the simulation
"""
t_list = []
sigma_n_list = []
sigma_x_list = []
niters = int(nsteps/sample_freq)
print('number of iterations:', niters, f'each iteration runs {sample_freq} steps')
for i in range(niters):
    simulation.step_AB_scheme(n_steps=sample_freq)
    sigma_n = simulation.observe(obs_1)
    sigma_x = simulation.observe(obs_2)
    print('t={:.3f}fs, sigma_n={}, sigma_x={}'.format(i*sample_freq*dt/Constants.fs, sigma_n, sigma_x))
    t_list.append(i*sample_freq*dt/Constants.fs)
    sigma_n_list.append(sigma_n)
    sigma_x_list.append(sigma_x)


"""
Post-processing
"""
sigma_n_list = np.array(sigma_n_list)  # (#samples, batchsize)
sigma_x_list = np.array(sigma_x_list)  # (#samples, batchsize)
np.save('sigma_n_run{}.npy'.format(out_folder, args.run_id), sigma_n_list)
np.save('sigma_x_run{}.npy'.format(out_folder, args.run_id), sigma_x_list)

