#!/usr/bin/python

"""
Numerically solve for evolutionary rate as a function of abundance and dG (Eqn 6).
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


#experimentally measured parameters (Table 1)
d_N = {'ecoli':10**8, 'yeast':10**7, 'human':10**4}
c = 8*10**(-7)
kT = 0.59
ddG_mean_constant = 1
ddG_std = 1.7
ddG_m = -0.13
ddG_b = 0.23

dGddG_anticorrelation = True	#required to be True for A-ER correlation to be present
print 'dGddG_anticorrelation = {0}'.format(dGddG_anticorrelation)


def selection(ddG, A_tot, dG):	#Eqn S5
	return -c*A_tot*(1/(1+np.exp(-(dG+ddG)/kT)) - 1/(1+np.exp(-dG/kT)))	

def Pfix(ddG, A_tot, dG, N):	#Eqn S6
	s = selection(ddG, A_tot, dG)
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

def integrand(ddG, A_tot, dG, N):	#Eqn 6
	ddG_mean = get_ddG_mean(dG)
	return Pfix(ddG, A_tot, dG, N)* 1 / (ddG_std * np.sqrt(2*np.pi)) * np.exp(-(ddG - ddG_mean)**2 / (2*ddG_std**2))


def numerics(A_tots, dGs, N):
	"""3-dimensional surface relating all three protein variables according to Eqn 6"""

	ERs = np.zeros((len(dGs), len(A_tots)))

	for i, dG in enumerate(dGs):
		for j, A_tot in enumerate(A_tots):
 			ERs[i][j] = quad(integrand, -np.inf, np.inf, args=(10**A_tot, dG, N))[0]
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

def predict_dG(A_tots, organism):
	N = d_N[organism]
	As = np.power(10, A_tots)
	if dGddG_anticorrelation:
		constant = kT/(c*N*ddG_std**2)	
		content = -As*np.exp(-ddG_b/(ddG_m*kT))/(constant*ddG_m*kT)
		dGs = -ddG_b/ddG_m - kT*lambertw(content) 

		#approximations for Lambert W function	
#		dGs = -ddG_b/ddG_m -kT*(np.log(content)-np.log(np.log(content)))	#looks good!
#		dGs = -ddG_b/ddG_m -kT*(np.log(content))	#off, log approx not good enough
		return dGs.real	
	else:
		return -kT*(np.log(N)+np.log(As)+np.log(c)+np.log(ddG_std**2/(kT*ddG_mean_constant)))
def predict_ER(dGs):
	ddG_mean = get_ddG_mean(np.array(dGs))
	term = -1/ddG_std**2 - 1./3. * (ddG_mean/ddG_std**2)**2 - 1./kT * ddG_mean/ddG_std**2
	return norm(ddG_mean, ddG_std).pdf(0)*(np.sqrt(2*np.pi/abs(term)))	

def plotout_min_2d(A_mins, dG_mins, ER_mins):
#	if not dGddG_anticorrelation:
#		plt.title('no dG-ddG anticorr')

	if ER_mins==0:
		plt.scatter(A_mins, dG_mins, color = 'b', label = 'numerical solution')
		plt.plot(A_mins, predict_dG(A_mins, organism), color = 'r', label='theoretical prediction')
	#	plt.title(r'$W(x) \approx \ln x - \ln (\ln x)$')
		plt.xlabel('log10(abundance)'), plt.ylabel(r'$\Delta G^{\mathrm{wt}}$ (kcal/mol)')
	elif A_mins==0:
		plt.scatter(dG_mins, ER_mins, color = 'b', label='numerical solution')
		plt.plot(dG_mins, predict_ER(dG_mins), color = 'r', label='theoretical prediction')
		plt.xlabel(r'$\Delta G^{\mathrm{wt}}$ (kcal/mol)'), plt.ylabel(r'$\langle \mathrm{evolutionary \ rate} \rangle$')
		plt.ylim([0, 1])
	elif dG_mins==0:
		plt.scatter(A_mins, ER_mins, color = 'b', label='numerical solution')
		plt.plot(A_mins, predict_ER(predict_dG(A_mins, organism)), color = 'r', label='theoretical prediction')
		plt.xlabel('log10(abundance)'), plt.ylabel(r'$\langle \mathrm{evolutionary \ rate} \rangle$')
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
	ha.set_xlabel('log10(abundance)'), ha.set_ylabel(r'$\Delta G^{\mathrm{wt}}$ (kcal/mol)'), ha.set_zlabel(r'$\langle \mathrm{evolutionary \ rate} \rangle$')
	plt.show()



if __name__ == '__main__':
#	organism = str(raw_input("Which organism: "))
	organism = 'yeast'

	A_tots = np.linspace(0, 7, 50)	#in log scale
	dGs = np.linspace(-15, 0, 100)

	ERs = numerics(A_tots, dGs, d_N[organism]) 
	A_mins, dG_mins, ER_mins = min_values(A_tots, dGs, ERs)

	i = 24	#remove unsmooth part of 3d surface	
	j = -13

	A_mins, dG_mins, ER_mins = A_mins[i:j], dG_mins[i:j], ER_mins[i:j]
	plotout_min_2d(A_mins, dG_mins, 0)
	plotout_min_2d(0, dG_mins, ER_mins)
	plotout_min_2d(A_mins, 0, ER_mins)
	plotout(A_tots, dGs, ERs, A_mins, dG_mins, ER_mins)
