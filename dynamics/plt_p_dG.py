#!/usr/bin/python

"""Plot p_dG from previously run evolutionary dynamics simulation of dG uner MAH deposited in outputs/"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def readin(dirname, init):
	"""read in p_dGs from previously run dynamics.py simulation"""

	p_dGs = []

	with open('{0}/dG0{1}.txt'.format(dirname, init), 'r') as rfile:
		for line in rfile:
			words = line.split(' [')
			time = float(words[0])

			pre_dist = words[1].split('] ')
			dist = pre_dist[0].split(',')
		
			p_dGs.append((time, [float(x) for x in dist]))
	return p_dGs

def plotout(init):
	plt.axvline(x= -4.9, color = 'k', linestyle='--')
	if init==0:
		plt.title(r'$p_{{t=0}}(\Delta G$={0}) = 1'.format(init))
	else:
		plt.title(r'$p_{{t=0}}(\Delta G$=-{0}) = 1'.format(init))
	plt.xlabel(r'$\Delta G$ (kcal/mol)')
	plt.ylabel('probability')
	plt.ylim([0, 0.6])
	plt.legend(title = r'time ($10^7$)'.format(scale))


if __name__ == '__main__':
	organism = 'yeast'	#str(raw_input("Which organism: "))
	A_tot = 100		#dG* = -4.96
	dGs = np.arange(-10, 0.1, 0.1)

	dirname = 'outputs/yeast/'
	init = int(raw_input("Initial dG = -(0, 3, 7): "))
	p_dGs = readin(dirname, init)

	scale = 10**7
	p_dGs.pop(-2)	#remove time = 5*10**8 result
	for time, p_dG in p_dGs[:]:
		plt.plot(dGs, p_dG/integrate.trapz(p_dG, dx=0.1), label="{0}".format(int(time/scale)))	#make sure each p_dG is normalized
	plotout(init)
	plt.show()
