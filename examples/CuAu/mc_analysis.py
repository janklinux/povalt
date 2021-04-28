import pickle
import numpy as np


with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# T = 10 ... 2000 | dT = 5
# xi = -2.5 ... 2.0 | dxi = 0.05

N_T = len(data['up'])
N_xi = len(data['heat'])


def temp_integral(phi_init_in, beta_in, potential_energy):
    """
    Perform thermodynamic integration for varying T/beta

    Notes
    -----
    beta_1*phi_1 = beta_0*phi_0 + (potential_energy_0 + potential_energy_1)*0.5*(beta_1 - beta_0)

    phi_1 = phi_0*(beta_0/beta_1) + (potential_energy_0 + potential_energy_1)*0.5*(1 - (beta_0/beta_1))
    """
    phi_result = np.empty(len(beta_in))
    phi_result[0] = phi_init_in
    for ii in range(1, len(beta_in)):
        beta_ratio = beta_in[ii-1] / beta_in[ii]
        mean_epot = 0.5 * (potential_energy[ii] + potential_energy[ii-1])
        phi_result[i] = phi_result[ii-1] * beta_ratio + mean_epot * (1 - beta_ratio)
    return phi_result


def xi_integral(phi_init_in, xi_in, comp_in):
    """
    Perform thermodynamic integration for varying xi (parametric chemical potential)

    Notes
    -----
    phi_1 = phi_0 - (comp_0 + comp_1)*0.5*(xi_1 - xi_0)
    """
    phi_result = np.empty(len(xi_in))
    phi_result[0] = phi_init_in
    for ii in range(1, len(xi_in)):
        mean_comp = (comp_in[ii] + comp_in[ii-1]) * 0.5
        phi_result[ii] = phi_result[ii-1] - mean_comp * (xi[ii] - xi[ii-1])
    return phi_result


# read phi_lte for grid range: lte data ran from -3.0 -> 0.0, incr = 0.01
start = int(round((-2.2 + 3.0) / 0.01))
end = int(round((-1.31 + 3.0) / 0.01))
phi_lte = np.array(data['lte'].loc[start:end, 'phi_LTE'])

# 'phi_T_up', 'phi_xi_up', etc. store integrated free energy, but without
# checking for phase boundaries
#
# echo 'phi_*_*' is a 2d np.array,
#  row 0 is high T, last row is low T
#  col 0 is low xi, last col is high xi (parametric chemical potential)

phi_T_up = np.empty((N_T, N_xi))
phi_T_up[-1, :] = phi_lte
for i in range(N_xi):
    phi_init = phi_T_up[-1, i]
    beta = np.array(data['heat'][i]['Beta'])[:-1]
    Epot = np.array(data['heat'][i]['<potential_energy>'])[:-1]
    phi_T_up[:, i] = np.flipud(temp_integral(phi_init, beta, Epot))

phi_xi_up = np.empty((N_T, N_xi))
phi_xi_up[:, 0] = phi_T_up[:, 0]
for i in range(N_T):
    phi_init = phi_xi_up[N_T-i-1, 0]
    xi = np.array(data['up'][i]['param_chem_pot(a)'][1:])
    comp = np.array(data['up'][i]['<comp(a)>'])
    phi_xi_up[N_T-i-1, :] = xi_integral(phi_init, xi, comp)

phi_xi_down = np.empty((N_T, N_xi))
phi_xi_down[:, -1] = phi_T_up[:, -1]
for i in range(N_T):
    phi_init = phi_xi_down[N_T-i-1, -1]
    xi = np.array(data['down'][i]['param_chem_pot(a)'][:-1])
    comp = np.array(data['down'][i]['<comp(a)>'])
    phi_xi_down[N_T-i-1, :] = np.flipud(xi_integral(phi_init, xi, comp))

phi_T_down = np.empty((N_T, N_xi))
phi_T_down[0, :] = phi_xi_up[0, :]
for i in range(N_xi):
    phi_init = phi_T_down[0, i]
    beta = np.array(data['cool'][i]['Beta'])[1:]
    Epot = np.array(data['cool'][i]['<potential_energy>'])[1:]
    phi_T_down[:, i] = temp_integral(phi_init, beta, Epot)


phi = dict()
phi['T_up'] = phi_T_up
phi['T_down'] = phi_T_down
phi['xi_up'] = phi_xi_up
phi['xi_down'] = phi_xi_down
phi['lte'] = phi_lte

with open('phi.pkl', 'wb') as f:
    pickle.dump(phi, f)
