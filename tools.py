#!/usr/bin/python

"""
Definitions of fitness functions and population dynamics necessary
to perform evolutionary dynamics and predict abundance-evolutionary rate
relationship

"""

import sys
import numpy as np
from scipy.special import lambertw
from scipy.stats import norm


#experimentally measured parameters (Table 1)
kT = 0.59	#thermal energy (kcal/mol)

#MAH parameter
c = 8*10**(-7)	#cellular cost of a misfolded protein

#FDH parameters
b = 0.1			#effects of enzymatic chain for folA. includes division by A (set Atot = 1)	
				#1/(s*mM)-1 
kcatKm = 8.1 	#catalytic activity [1/(s*mM)]

#distribution of attempted stability effects
ddG_std = 1.7	#kcal/mol
dGddG_anticorrelation = True	#required to be True for A-ER correlation to be present

#if dGddG_anticorrlation = True
#ddG_mean = ddG_m * dG + ddG_b
ddG_m = -0.13	
ddG_b = 0.23	#kcal/mol	

#if dGddG_anticorrlation = False
ddG_mean_constant = 1


#Probability of fixation of an attempted mutation, parameter
d_N = {'ecoli':10**8, 'yeast':10**7, 'human':10**4}


def log_fitness_FDH(dG, A_tot):
	A = A_tot/(1+np.exp(dG/kT))
	return np.log(A*kcatKm/(b + A*kcatKm))

def log_fitness_MAH(dG, A_tot):	
	return -c*A_tot/(1+np.exp(-dG/kT))	#already logged

def selection(ddG, A_tot, dG, fitness_function):
	if fitness_function=='MAH':
		return log_fitness_MAH(dG+ddG, A_tot) - log_fitness_MAH(dG, A_tot)
	elif fitness_function=='FDH':
		return log_fitness_FDH(dG+ddG, A_tot) - log_fitness_FDH(dG, A_tot)
	#	return np.log(log_fitness_FDH(dG+ddG, A_tot)) - np.log(log_fitness_FDH(dG, A_tot))

def Pfix(ddG, A_tot, dG, N, fitness_function):
	s = selection(ddG, A_tot, dG, fitness_function)
	Pfix = (1-np.exp(-2*s))/(1-np.exp(-2*N*s))

	if type(Pfix)==np.float64:		#for running this code
		return Pfix
	else: 					#for dynamics.py, need to correctly treat s=0 case
		Pfix[np.isnan(Pfix)] = 1./N	#s=0, Pfix=nan	
		Pfix[Pfix==0] = 1./N		#some s are not 0 but ~E-27, giving Pfix = 0
		return Pfix 

def get_ddG_mean(dGs):
	"""enable correct handling of ddG mean depending on dGddG_anticorrelation"""

	if dGddG_anticorrelation:
		ddG_mean = -0.13*dGs + 0.23
	else:
		if type(dGs)==np.float64:
			ddG_mean = ddG_mean_constant
		else:	#for array
			ddG_mean = np.zeros(len(dGs))
			ddG_mean.fill(ddG_mean_constant)
	return ddG_mean

def integrand(ddG, A_tot, dG, N, fitness_function):
	ddG_mean = get_ddG_mean(dG)
	return Pfix(ddG, A_tot, dG, N, fitness_function)* 1 / (ddG_std * np.sqrt(2*np.pi)) * np.exp(-(ddG - ddG_mean)**2 / (2*ddG_std**2))


def predict_dG(A_tots, organism, fitness_function):
	N = d_N[organism]
	As = np.power(10, A_tots)

	if fitness_function=='MAH':
		R = c*As
	elif fitness_function=='FDH':
		R = b/(kcatKm*As)

	if dGddG_anticorrelation:
		constant = kT/(R*N*ddG_std**2)	
		content = -np.exp(-ddG_b/(ddG_m*kT))/(constant*ddG_m*kT)

		dGs = -ddG_b/ddG_m - kT*lambertw(content) 
		return dGs.real	
	else:
		return -kT*(np.log(N)+np.log(As)+np.log(c)+np.log(ddG_std**2/(kT*ddG_mean_constant)))
