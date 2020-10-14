#!/usr/bin/python

"""
Numerically solve for evolutionary rate subset without 
including p_dG as a function of abundance and dG.
Compare analytical expressions with numerical solution.

"""

import sys
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from scipy.special import lambertw
from matplotlib import cm
from scipy.stats import norm


sys.path.append('../')
from tools import d_N, integrand, get_ddG_mean, predict_dG
from tools import ddG_std, kT

def numerics(A_tots, dGs, N):
	"""3-dimensional surface relating all three protein variables"""

	ERs = np.zeros((len(dGs), len(A_tots)))

	for i, dG in enumerate(dGs):
		for j, A_tot in enumerate(A_tots):
 			ERs[i][j] = quad(integrand, -np.inf, np.inf, args=(10**A_tot, dG, N,'MAH'))[0]
	return N*ERs

def min_values(A_tots, dGs, ERs):
	"""
	Obtain minimum values along molecular clock surface.
	Corresponds to values predicted by derived analytical expressions.

	"""

	A_mins, dG_mins, ER_mins = [], [], []
	for dG_i, ER_a in enumerate(ERs):
		A_min = np.argmin(ER_a)
		A_mins.append(A_tots[A_min])
		dG_mins.append(dGs[dG_i])
		ER_mins.append(ER_a[A_min])
#		print A_tots[A_min], dGs[dG_i], ER_a[A_min]
	return A_mins, dG_mins, ER_mins

def predict_ER(dGs):
	ddG_mean = get_ddG_mean(np.array(dGs))
	term = -1/ddG_std**2 - 1./3. * (ddG_mean/ddG_std**2)**2 - 1./kT * ddG_mean/ddG_std**2
	return norm(ddG_mean, ddG_std).pdf(0)*(np.sqrt(2*np.pi/abs(term)))	

def plotout_min_2d(A_mins, dG_mins, ER_mins, organism):
	if ER_mins==0:
		plt.scatter(A_mins, dG_mins, color = 'b', label = 'numerical solution')
		plt.plot(A_mins, predict_dG(A_mins, organism, 'MAH'), color = 'r', label='theoretical prediction')
		plt.xlabel('log10(total abundance)'), plt.ylabel(r'$\Delta G^{\mathrm{opt}}$ (kcal/mol)')
	elif A_mins==0:
		plt.scatter(dG_mins, ER_mins, color = 'b', label='numerical solution')
		plt.plot(dG_mins, predict_ER(dG_mins), color = 'r', label='theoretical prediction')
		plt.xlabel(r'$\Delta G^{\mathrm{opt}}$ (kcal/mol)'), plt.ylabel('integral')
		plt.ylim([0, 1])
	elif dG_mins==0:
		plt.scatter(A_mins, ER_mins, color = 'b', label='numerical solution')
		plt.plot(A_mins, predict_ER(predict_dG(A_mins, organism, 'MAH')), color = 'r', label='theoretical prediction')
		plt.xlabel('log10(total abundance)'), plt.ylabel('integral')
		plt.ylim([0, 1])

	plt.legend()
	plt.show()


def plotout(A_tots, dGs, ERs, A_mins, dG_mins, ER_mins):
	ERs[ERs>2] = np.nan	#for visualization purposes, cutoff ER>2
	hf = plt.figure()
	ha = hf.add_subplot(111, projection='3d')

	A_tots, dGs = np.meshgrid(A_tots, dGs)	#makes A_tots and dGs into 2d matrix so that plot surface can read it	
	ha.plot_surface(A_tots, dGs, ERs)

	ha.scatter(A_mins[::2], dG_mins[::2], ER_mins[::2], c='b', marker='o')
	ha.set_xlabel('log10(total abundance)'), ha.set_ylabel(r'$\Delta G^{\mathrm{opt}}$ (kcal/mol)'), ha.set_zlabel(r'integral')
	plt.show()



if __name__ == '__main__':
	organism = 'yeast'	#options: ecoli, yeast, human

	A_tots = np.linspace(0, 7, 50)	#in log scale
	dGs = np.linspace(-15, 0, 100)

	ERs = numerics(A_tots, dGs, d_N[organism]) 
	A_mins, dG_mins, ER_mins = min_values(A_tots, dGs, ERs)

	i = 24	#remove unsmooth part of 3d surface	
	j = -13

	A_mins, dG_mins, ER_mins = A_mins[i:j], dG_mins[i:j], ER_mins[i:j]
	plotout_min_2d(A_mins, dG_mins, 0, organism)
	plotout_min_2d(0, dG_mins, ER_mins, organism)
	plotout_min_2d(A_mins, 0, ER_mins, organism)
	plotout(A_tots, dGs, ERs, A_mins, dG_mins, ER_mins)
