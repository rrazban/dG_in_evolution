#!/usr/bin/python

"""Plot fixing p_ddG from previously run evolutionary dynamics simulation of dG uner MAH deposited in outputs/"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from plt_p_dG import readin
sys.path.append('../theory')
from accuracy import d_N, integrand



def integral():
	ddGs_perdG = np.zeros((len(dGs), len(ddGs)))

	for i in range(len(dGs)):
		ddGs_perdG[i] = ddGs
	Pfix_pddG = integrand(ddGs_perdG, A_tot, dGs[:, np.newaxis], d_N[organism])
	return np.transpose(Pfix_pddG)


def get_p_ddGs(p_dGs):
	"""Calculate fixing p_ddG according to Eqn 5"""

 	p_ddGs = []
	Pfix_pddG_T = integral()

	for time, p_dG in p_dGs[:]:
		p_ddG = integrate.trapz(Pfix_pddG_T * p_dG, dx=0.1, axis=1)
		p_ddGs.append((time, p_ddG))
	return p_ddGs 

def plotout(init):
	plt.axvline(x=0.0, color = 'k', linestyle='--')
	if init==0:
		plt.title(r'$p_{{t=0}}(\Delta G$={0}) = 1'.format(init))
	else:
		plt.title(r'$p_{{t=0}}(\Delta G$=-{0}) = 1'.format(init))
	plt.xlabel(r'$\Delta \Delta G$ (kcal/mol)')
	plt.ylabel('probability')
	plt.legend(title = r'time ($10^7$)'.format(scale))
	plt.ylim([-0.025, 0.45])


if __name__ == '__main__':
	print 'make sure s=0 case accounted for in molecular_clock_surface_MAH.py'

	organism = 'yeast'	#str(raw_input("Which organism: "))
	A_tot = 100	#dG* = -4.96
	dGs = np.arange(-10, 0.1, 0.1)	
	ddGs = np.arange(-7, 7.1, 0.1)

	dGs_str = ["{0:0.1f}".format(dG) for dG in dGs]	
	ddGs_str = ["{0:0.1f}".format(ddG) for ddG in ddGs]

	dirname = 'outputs/yeast/'
	init = int(raw_input("Initial dG = -(0, 3, 7): "))
	p_dGs = readin(dirname, init)
	p_ddGs = get_p_ddGs(p_dGs)

	scale = 10**7
	p_ddGs.pop(-2)	#remove time = 5*10**8 result
	for time, p_ddG in p_ddGs[:]:
		plt.plot(ddGs, (p_ddG/integrate.trapz(p_ddG, dx=0.1)), label="{0}".format(int(time/scale)))
	plotout(init)
	plt.show()
