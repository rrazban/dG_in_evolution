#!/usr/bin/python

"""
Numerically solve for evolutionary rate as a function of abundance
Compare analytical expressions with numerical solution.

"""

import sys
from scipy.integrate import dblquad, quad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from scipy.special import lambertw
from matplotlib import cm
from scipy.stats import norm
from scipy import special

sys.path.append('../')
from tools import d_N, integrand, get_ddG_mean, predict_dG
from tools import ddG_std, ddG_m, kT

sys.path.append('../dG_distribution')
from dynamics import formula 


def full_integrand(dG, ddG, A_tot, N, norm):
	p_dG = formula(dG, A_tot, organism, 'MAH')/norm

	return p_dG*integrand(ddG, A_tot, dG, N, 'MAH')


def numerics(A_tots, dGs, organism):
	"""3-dimensional surface relating all three protein variables according to Eqn 6"""

	ERs = np.zeros(len(A_tots))

	for j, A_tot in enumerate(A_tots):
		dG_min = -20		#numerics breaks down beyond -20 kcal/mol 
					#p_dG analytical solution no longer valid					
		dG_max = 0 

		norm = quad(formula, dG_min, dG_max, args=(10**A_tot, organism, 'MAH'))[0]	
 		ERs[j] = dblquad(full_integrand, -np.inf, np.inf, dG_min, dG_max, args=(10**A_tot, d_N[organism], norm))[0]
	return d_N[organism]*ERs


def predict_ER(dGs, A_tot):
	ddG_mean = get_ddG_mean(np.array(dGs))
	term = -1/ddG_std**2 - 1./3. * (ddG_mean/ddG_std**2)**2 - 1./kT * ddG_mean/ddG_std**2

	fudge = -ddG_m/2.
	return norm(ddG_mean, ddG_std).pdf(0)*(np.sqrt(2*np.pi/abs(term))) + fudge	

def plotout(A_tots, ERs, organism):

	plt.scatter(A_tots, ERs, color = 'b', label='numerical solution')

	ERs = predict_ER(predict_dG(A_tots, organism, 'MAH'), 10**np.array(A_tots))
	plt.plot(A_tots, ERs, color = 'r', label='theoretical prediction')

	plt.xlabel('log10(total abundance)'), plt.ylabel(r'$\langle \mathrm{evolutionary \ rate} \rangle$')
	plt.ylim([0, 1])
	plt.legend()
	plt.show()

if __name__ == '__main__':
	organism = 'ecoli'	#options: ecoli, yeast, human

	A_tots = np.linspace(0, 7, 50)	#in log10 scale
	dGs = np.linspace(-15, 0, 100)

	ERs = numerics(A_tots, dGs, organism) 
	plotout(list(A_tots), list(ERs), organism)
