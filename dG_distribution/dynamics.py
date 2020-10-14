#!/usr/bin/python

"""Perform evolutionary dynamics of dG for a protein under the 
misfolding avoidance hypothesis or flux dynamics hypothesis"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

sys.path.append('../')
from tools import d_N, integrand, get_ddG_mean, predict_dG
from tools import c, kT, ddG_std, ddG_m, ddG_b, b, kcatKm


def initialize_stability_space():
	dGs = np.arange(-15.0, 0.1, 0.1)	#dG space avaiable in numerical solver (kcal/mol)
										#assume dG>0 and dG<-15 kcal/mol have probability of 0

	p_dG = np.zeros(len(dGs))			#each index corresponds to a dG probability to the tenth decimal (0, 0.1, 0.2...)
	dGs_str = ["{0:0.1f}".format(dG) for dG in dGs]		#useful list of strings to have for indexing

	ddGs_perdG = ddG_space(dGs, dGs_str)
	return dGs, dGs_str, ddGs_perdG, p_dG


def ddG_space(dGs, dGs_str):
	"""
	ddG space avaiable to explore based on dG limits
	Assume that stabilities beyond dG limits have
	a probability of 0

	Returns:
	ddGs_perdG (array-like): ddGs for a given dG such
	that each column has the same dG value

	"""

	ddGs_perdG = np.zeros((len(dGs), len(dGs)))
	for i, dG in enumerate(dGs):
		ddGs_perdG[i] = np.arange(float(dGs_str[0]) - dG, float(dGs_str[-1])+0.1 - dG, 0.1)
	return ddGs_perdG


def calc_static_integrals(ddGs_perdG, A_tot, dGs, organism, fitness_function):
	"""
	static integrals and integrands that appear at each 
	timestep when iteratively solving evolutionary dynamics
	"""

	Pfix_pddG = integrand(ddGs_perdG, A_tot, dGs[:, np.newaxis], d_N[organism], fitness_function)
	integral = integrate.trapz(Pfix_pddG, dx=0.1, axis=1)
	efflux = integrate.trapz(np.triu(Pfix_pddG, k=1), dx=0.1, axis=1) + integrate.trapz(np.tril(Pfix_pddG, k=-1), dx=0.1, axis=1)	#use np.triu and np.tril to exclude ddG=0

	Pfix_pddG_T = np.transpose(Pfix_pddG)
	nofix_or_leave_factor = integral + efflux	

	return np.transpose(Pfix_pddG), nofix_or_leave_factor


def solver_bookeeping():
	speed = 4
	mu = 10**speed	#scales how many generations a timestep corresponds to	

	lim = 9-speed	#how long to run solver
	record = [10**x for x in range(lim-2, lim+1)]	#timesteps for which p_dG is plotted
	
	return speed, mu, lim, record

def update_p_dG(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu):
	"""dynamics of dG at each time step"""

	influx = integrate.trapz(Pfix_pddG_T * p_dG, dx=0.1, axis=1)
	p_dG += mu * (-nofix_or_leave_factor * p_dG + influx)
	return p_dG

def numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record): 
	time = 0	#initialize timestep

	while time < 10**lim +1:	#solve dynamics over 10**lim timesteps
		p_dG = update_p_dG(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu)

		p_dG[p_dG<0] = 0	#permits larger mu without overflow error

		if time in record:	#record p_dG over certain timesteps

			plt.plot(dGs, p_dG/integrate.trapz(p_dG, dx=0.1), label= time/10**(lim-2))

		time+=1

def plotout(organism, fitness_function, scale):
	plt.axvline(x=predict_dG(np.log10(A_tot), organism, fitness_function), color ='k', linestyle='--')
	plt.xlabel(r'$\Delta G$ (kcal/mol)')
	plt.ylabel('probability')
	plt.title(r'$P_t(\Delta G)$')
	plt.ylim([0, 0.6])	#otherwise the graph is dominated by probability of initial condition
	plt.legend(title = 'time ($10^{0}/\mathrm{{\mu}}$)'.format(scale))

def formula(dG, A_tot, organism, fitness_function):
	c1 = (ddG_m**2 + 2.*ddG_m)/ddG_std**2

	if fitness_function=='MAH':
		c2 = d_N[organism]*c*A_tot
	elif fitness_function=='FDH':
		c2 = d_N[organism]*b/(A_tot * kcatKm)
		
	constant = 0.5

	return np.exp(dG*constant + 0.5*c1*dG**2 - 2*c2*np.exp(dG/kT))

def p_dG_analytic(dGs, A_tot, organism, fitness_function):
	p_dG_analytic = np.zeros(len(dGs))
	
	for i,dG in enumerate(dGs):
		p_dG_analytic[i] = formula(dG, A_tot, organism, fitness_function)

	plt.plot(dGs, p_dG_analytic/integrate.trapz(p_dG_analytic, dx=0.1), label="theory")



if __name__ == '__main__':

	fitness_function = 'FDH'	#options: MAH or FDH
	organism = 'ecoli'	#options: ecoli, yeast, human
	A_tot = 1			#total protein abundance
	dGs, dGs_str, ddGs_perdG, p_dG = initialize_stability_space()

	Pfix_pddG_T, nofix_or_leave_factor = calc_static_integrals(ddGs_perdG, A_tot, dGs, organism, fitness_function) 

	speed, mu, lim, record = solver_bookeeping()

	p_dG[dGs_str.index('-7.0')] = 1		#p_dG initial condition
										#delta function at certain dG value at time=0 
	p_dG /= np.trapz(p_dG, dx=0.1)		#normalize p_dG over the grid
	plt.plot(dGs, p_dG, label=0)
	numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record)

	p_dG_analytic(dGs, A_tot, organism, fitness_function)

	plotout(organism, fitness_function, speed+lim-2)	#prepare figure lablels

	plt.show()
