#!/usr/bin/python

"""
Perform evolutionary dynamics of dG for a protein under the 
misfolding avoidance hypothesis or flux dynamics hypothesis
for different initial conditions

"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from dynamics import initialize_stability_space, calc_static_integrals, update_p_dG

sys.path.append('../')
from tools import predict_dG


def solver_bookeeping():
	speed = 4
	mu = 10**speed	#scales how many generations a timestep corresponds to	

	lim = 10-speed	#how long to run solver

	#timesteps for which p_dG is plotted	
	record=[1]	#technically 0 but use 1 so that we can take log on xaxis
	record.extend([10**x for x in range(1, lim+1)])

	return speed, mu, lim, record


def numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, record, mode_or_mean):
	avgs = []	#output that is plotted for a given initial condition

	time = 0	#initialize timestep
	while time < 10**lim +1:	#solve dynamics over 10**lim time steps
		if time+1 in record:	#record p_dG over certain timesteps
			p_dG = p_dG/integrate.trapz(p_dG, dx=0.1)

			if mode_or_mean == 'mean':
				avgs.append(integrate.trapz(p_dG*dGs, dx=0.1))
			elif mode_or_mean == 'mode':
				avgs.append(dGs[np.argmin(p_dG*dGs)])

		p_dG = update_p_dG(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu)
		p_dG[p_dG<0] = 0	#permits larger mu without overflow error

		time+=1
	return avgs

def run_diff_initial_condition(Pfix_pddG_T, nofix_or_leave_factor, mu, record, mode_or_mean):
	for initial in range(15):
		p_dG = np.zeros(len(dGs))		#reset p_dG
		p_dG[dGs_str.index('-{0}.0'.format(initial))] = 1		#set initial condition
		p_dG /= np.trapz(p_dG, dx=0.1)

		avg_dG = numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, record, mode_or_mean)
		print initial, avg_dG
		plt.plot(record, avg_dG, label='-{0}'.format(initial))



def plotout(fitness_function, mode_or_mean):
	plt.axhline(y=predict_dG(np.log10(A_tot), organism, fitness_function), color ='k', linestyle='--')
	plt.legend(title=r'initial $\Delta G$', loc='upper right')
	plt.xlabel('time ($10^{0}/\mathrm{{\mu}}$)'.format(speed))

	if mode_or_mean == 'mode':
		plt.ylabel(r'mode of $P_t(\Delta G)$ (kcal/mol)')
	elif mode_or_mean == 'mean':
		plt.ylabel(r'$\langle \Delta G \rangle$ (kcal/mol)')


if __name__ == '__main__':

	fitness_function = 'FDH'	#options: MAH or FDH
	organism = 'ecoli'	#options: ecoli, yeast, human
	A_tot = 1			#total protein abundance
	dGs, dGs_str, ddGs_perdG, p_dG = initialize_stability_space()

	Pfix_pddG_T, nofix_or_leave_factor = calc_static_integrals(ddGs_perdG, A_tot, dGs, organism, fitness_function) 

	speed, mu, lim, record = solver_bookeeping()

	mode_or_mean = 'mode'
	run_diff_initial_condition(Pfix_pddG_T, nofix_or_leave_factor, mu, record, mode_or_mean)

	plotout(fitness_function, mode_or_mean)	#prepare figure lablels
	plt.show()
