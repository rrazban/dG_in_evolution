#!/usr/bin/python

"""
Plot distribution of ddG that corresponds to fixed
mutations
useful for assessing mutation selection balance

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from dynamics import initialize_stability_space, calc_static_integrals, update_p_dG, solver_bookeeping

sys.path.append('../')
from tools import d_N, integrand, get_ddG_mean, predict_dG


def numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record): 
	p_dGs = []

	time = 0	#initialize timestep

	while time < 10**lim +1:	#solve dynamics over 10**lim timesteps
		p_dG = update_p_dG(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu)

		p_dG[p_dG<0] = 0	#permits larger mu without overflow error

		if time in record:	#record p_dG over certain timesteps
			p_dG /= np.trapz(p_dG, dx=0.1)		#normalize p_dG over the grid
			p_dGs.append((time, [float(x) for x in p_dG]))

		time+=1
	return p_dGs

def integral(ddGs, dGs, A_tot, N, fitness_function):
	ddGs_perdG = np.zeros((len(dGs), len(ddGs)))

	for i in range(len(dGs)):
		ddGs_perdG[i] = ddGs	#slightly different than original definition
	Pfix_pddG = integrand(ddGs_perdG, A_tot, dGs[:, np.newaxis], N, fitness_function)
	return np.transpose(Pfix_pddG)


def get_p_ddGs(ddGs, p_dGs, dGs, A_tot, N, fitness_function):

 	p_ddGs = []
	Pfix_pddG_T = integral(ddGs, dGs, A_tot, N, fitness_function)

	for time, p_dG in p_dGs[:]:
		p_ddG = integrate.trapz(Pfix_pddG_T * p_dG, dx=0.1, axis=1)
		plt.plot(ddGs, (p_ddG/integrate.trapz(p_ddG, dx=0.1)), label= time/10**(lim-2))

def plotout(init):
	plt.axvline(x=0.0, color = 'k', linestyle='--')
	plt.title(r'$P_{{t=0}}(\Delta G$=-{0}) = 1'.format(init))
	plt.xlabel(r'$\Delta \Delta G$ (kcal/mol)')
	plt.ylabel('probability')
	plt.legend(title = 'time ($10^{0}/\mathrm{{\mu}}$)'.format(speed+lim-2))


if __name__ == '__main__':

	fitness_function = 'MAH'	#options: MAH or FDH
	organism = 'ecoli'	
	A_tot = 100	#dG* = -4.96

	dGs, dGs_str, ddGs_perdG, p_dG = initialize_stability_space()


	Pfix_pddG_T, nofix_or_leave_factor = calc_static_integrals(ddGs_perdG, A_tot, dGs, organism, fitness_function) 

	speed, mu, lim, record = solver_bookeeping()


	dirname = 'outputs/yeast/'
	init = int(raw_input("Initial dG = -? kcal/mol: "))	#options: 0 to 15 kcal/mol

	p_dG[dGs_str.index('-{0}.0'.format(init))] = 1		#p_dG initial condition
	p_dG /= np.trapz(p_dG, dx=0.1)		#normalize p_dG over the grid

	p_dGs = numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record)

	ddGs = np.arange(-7, 7.1, 0.1)
	p_ddGs = get_p_ddGs(ddGs, p_dGs, dGs, A_tot, d_N[organism], fitness_function)

	plotout(init)
	plt.show()
