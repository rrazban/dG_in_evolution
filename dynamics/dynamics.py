#!/usr/bin/python

"""Perform evolutionary dynamics of dG uner MAH"""

import sys, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

sys.path.append('../theory')
from accuracy import d_N, integrand


def all_possible_ddGs(dGs):
	"""ddG space avaiable to explore based on dGs"""

	ddGs_perdG = np.zeros((len(dGs), len(dGs)))
	for i, dG in enumerate(dGs):
		ddGs_perdG[i] = np.arange(min(dGs) - dG, max(dGs)+0.1 - dG, 0.1)
	return ddGs_perdG

def update_p_dG(p_dG):
	"""dynamics of dG at each time step (Eqn 4)"""

	muL = 1	#no effect on distribution shape, only time at which distribution appears (Eqn S7)
		#10^5 too big, relationship breaks down

	influx = integrate.trapz(Pfix_pddG_T * p_dG, dx=0.1, axis=1)	#third line of Eqn 4
	p_dG += muL * (-nofix_or_leave_factor * p_dG + influx)
	return p_dG#/integrate.trapz(p_dG, dx=0.1)	#no need to normalize between runs, results differ by only a scaling constant

def plotout():
	plt.axvline(x = -4.9, color = 'k', linestyle='--')
	plt.xlabel(r'$\Delta G$ (kcal/mol)')
	plt.ylabel('probability')
	plt.ylim([0, 0.6])


if __name__ == '__main__':
	start = datetime.datetime.now().replace(microsecond=0)

	organism = 'yeast'	#str(raw_input("Which organism: "))
	A_tot = 100		#dG* = -4.96 kcal/mol according to Eqn 7
	dGs = np.arange(-10.0, 0.1, 0.1)	#dG space avaiable in the simulation. assume dG>0 and dG<-10 have probability of 0
	dGs_str = ["{0:0.1f}".format(dG) for dG in dGs]	
	ddGs_perdG = all_possible_ddGs(dGs)

	p_dG = np.zeros(len(dGs))		#each index corresponds to a dG probability to the tenth decimal (0, 0.1, 0.2...)
	p_dG[dGs_str.index('-7.0')] = 1		#delta function at certain dG value at time=0 
	p_dG /= np.trapz(p_dG, dx=0.1)


	Pfix_pddG = integrand(ddGs_perdG, A_tot, dGs[:, np.newaxis], d_N[organism])
	integral = integrate.trapz(Pfix_pddG, dx=0.1, axis=1)
	efflux = integrate.trapz(np.triu(Pfix_pddG, k=1), dx=0.1, axis=1) + integrate.trapz(np.tril(Pfix_pddG, k=-1), dx=0.1, axis=1)	#exclude ddG=0

	Pfix_pddG_T = np.transpose(Pfix_pddG)
	nofix_or_leave_factor = integral + efflux	#first and second line of Eqn 4	

	lim = 5
	record = [10**x for x in range(0, lim+1)]
#	lim = 9 
#	record = [0, 10**7, 10**8, 5*10**8, 10**9]

	time = 0
	plotout()
	while time < 10**lim +1:	#solve dynamics over 10**lim time steps
		p_dG = update_p_dG(p_dG)
		if time in record:	#record p_dG over certain time steps
			check = datetime.datetime.now().replace(microsecond=0)
			print "{0:.0E}".format(time), list(p_dG), check-start
			plt.plot(dGs, p_dG/integrate.trapz(p_dG, dx=0.1), label="{0:.0E}".format(time))
			plt.legend(title = 'time points')
			plt.pause(.0001)
		time+=1

	end = datetime.datetime.now().replace(microsecond=0)
	print 'duration of simulation: {0}'.format(end-start)
	plt.show()	#prevents figure from immediately closing
