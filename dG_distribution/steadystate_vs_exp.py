#!/usr/bin/python

"""
Compare steady state of evolutionary dynamics of dG under the 
misfolding avoidance hypothesis or flux dynamics hypothesis
with experimental dG data

"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

from dynamics import initialize_stability_space, calc_static_integrals, update_p_dG



def plotout(xlabel):
	plt.xlabel(xlabel)
	plt.ylabel('probability')
	plt.title(r'$P_t(\Delta G)$')


def get_stats(p_dG, dGs_exp):
	"""
	Sample p_dG multiple times to compare with
	dGs_exp and obtain an accurate Kolmogorov-Smirnov
	test static with its corresponding p-value	

	"""

	cumsum = np.cumsum(p_dG)

	ks = []
	pvalues = []
	for _ in range(10000):
		sample = []
		for _ in range(100):
			rand = random.uniform(0, 10)	#might need to be 1 or 10 depending on if normalized
			rand_i =  np.where(cumsum>rand)[0][0]
			sample.append(dGs[rand_i])
		k, pvalue = stats.ks_2samp(sample, dGs_exp)
		ks.append(k)
		pvalues.append(pvalue)

	ks = np.array(ks)
	pvalues = np.array(pvalues)

	mean = np.mean(ks)	#the mean KS value from all the sampling

	#determining what pvalue the mean KS value corresponds to
	#from previously recorded KS-pvalue combinations
	order = ks.argsort()
	ks_order = ks[order]
	pvalues_order = pvalues[order]
	i = np.where(ks_order > mean)[0][0]

	return mean, pvalues_order[i]

def get_experiment(protein, state_transition):
	if protein == 'folA':
		A_tot = 5

		if state_transition=='un':
			#transformed from Tm data in Bershtein et al. (2015) 
			dGs_exp = [-5.275, -4.45, -4.3, -3.7, -5.425, -5.125, -4.87, -5.65, -5.98, -4.225, -6.175, -5.275, -5.125, -3.025, -3.625, -5.125, -5.5, -3.43, -5.05, -5.08, -3.88, -4.12, -4.75, -4.825, -4.525, -4.0, -4.42, -3.13, -3.4, -4.525, -4.525, -4.075, -4.975]	
			xlabel = r'$\Delta G$ (kcal/mol)'

		elif state_transition=='ui':
			#transformed from ANS data in Bershtein et al. (2013) 
			dGs_exp = [-5.6425, -5.6425, -5.5675, -4.2925, -4.7425, -4.8925, -6.691, -3.4705, -5.4865, -5.944, -4.8115, -4.8925, -6.691, -3.6175, -4.9705, -6.0175, -3.6175, -4.2925, -5.647, -4.816, -4.3675, -5.4145, -4.2925, -3.4675, -5.7175, -4.0645, -6.394, -6.2425, -4.5115, -4.5895, -5.6425, -6.019, -5.5675, -6.019, -5.0425, -6.8395, -6.094]
			xlabel = r'$\Delta G_{\mathrm{ANS}}$ (kcal/mol)'

	#	dGs_exp = np.array(dGs_exp)	-1		#add stability effect of binding to NADP+
		label = 'Shakhnovich et al. 2015'

	elif protein == 'rnhA':
		A_tot = 1 

		#all experimental data from Lim et al. (2016) for only mesophilic ancestors and E.coli
		if state_transition=='un':
			dGs_exp = [-10.4, -8.7, -9.1, -9.7, -9.4, -9.9]	#overall unfolding of mesophilic ancestors
			xlabel = r'$\Delta G$ (kcal/mol)'
		
		elif state_transition=='ui':
			dGs_exp = [-5.2, -2.4, -3.8, -3.9, -3.6, -3.5]	#UI results of mesophilic ancestors	
			xlabel = r'$\Delta G_{\mathrm{UI}}$ (kcal/mol)'

		elif state_transition=='in':
			dGs_exp = np.array(dGs_exp_unf)-np.array(dGs_exp_ui)
			xlabel = r'$\Delta G_{\mathrm{IN}}$ (kcal/mol)'

	return A_tot, dGs_exp, xlabel


def solver_bookeeping():
	speed = 4
	mu = 10**speed	#scales how many generations a timestep corresponds to	

	lim = 9-speed	#how long to run solver

	record = [10**x for x in range(lim, lim+1)]

	return speed, mu, lim, record


def numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record, fitness_function): 
	time = 0	#initialize timestep

	while time < 10**lim +1:	#solve dynamics over 10**lim timesteps
		p_dG = update_p_dG(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu)

		p_dG[p_dG<0] = 0	#permits larger mu without overflow error

		if time in record:	#record p_dG over certain timesteps

			p_dG /= np.trapz(p_dG, dx=0.1)		#normalize p_dG over the grid
			plt.plot(dGs, p_dG, label= fitness_function)

#			label+= r'$s$ prefactor / {0}'.format(A_tot))	# r'$\mathrm{{A}}^{{\mathrm{{tot}}}}$={0}'.format(A_tot))
		time+=1
	return p_dG


if __name__ == '__main__':

	fitness_function = 'MAH'	#options: MAH or FDH
	organism = 'ecoli'	
	dGs, dGs_str, ddGs_perdG, p_dG = initialize_stability_space()

	protein = 'folA'	#options: rnhA or folA
	state_transition = 'ui'	#options: un, ui
	A_tot, dGs_exp, xlabel = get_experiment(protein, state_transition)

	Pfix_pddG_T, nofix_or_leave_factor = calc_static_integrals(ddGs_perdG, A_tot, dGs, organism, fitness_function) 

	speed, mu, lim, record = solver_bookeeping()

	p_dG[dGs_str.index('-7.0')] = 1		#p_dG initial condition
										#delta function at certain dG value at time=0 
	p_dG /= np.trapz(p_dG, dx=0.1)		#normalize p_dG over the grid

	p_dG = numerical_solver(p_dG, Pfix_pddG_T, nofix_or_leave_factor, mu, lim, record, fitness_function)

	plt.hist(dGs_exp, range = (-10, 0), normed=True, label = '{0} orthologs'.format(protein))
	plotout(xlabel)

	ks, pvalue = get_stats(p_dG, dGs_exp)	#statistically compare exp vs theory dG distributions
	plt.legend(title='KS = {0:.2f} ({1:.2f})'.format(ks, pvalue))
	plt.show()
