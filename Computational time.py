# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:59:32 2022

@author: Yi Ming Chang
"""

import matplotlib.pyplot as plt

MPI_numba = [75.38, 40.68, 22.01,11.88,6.42,4.82]
processes = [1,2,4,8,16,23]

nodes = [1,2,3,4]

MPI_23 = [4.82,2.99,2.39,1.96]
MPI_16 = [6.42,3.71,2.87,2.43]
MPI_8 = [11.88,6.27,4.7,3.66]
MPI_4 = [22.01,11.48,7.94,6.29]
MPI_2 = [40.68,20.85,14.38,10.8]


plt.rcParams['figure.figsize']=(6,4) # figure size 7, 4.5
plt.rcParams.update({'font.size':20}) # font size of the figures
plt.rc('text', usetex=True) # font in LaTeX form
plt.rc('font', family='serif')

plt.figure()
plt.plot(processes, MPI_numba, label = 'MPI+numba')
# plt.plot(processes, numba, label = 'numba')
plt.text(-2.5,75,"(a)")
plt.legend()
plt.xlabel('Number of Processes')
plt.ylabel('Computational Time (s)')
plt.savefig("Computational Time.pdf", format="pdf",dpi=1200,bbox_inches ="tight")
plt.show()

plt.figure()
plt.plot(nodes, MPI_2, label = 'nodes=2')
plt.plot(nodes, MPI_4, label = 'nodes=4')
plt.plot(nodes, MPI_8, label = 'nodes=8')
plt.plot(nodes, MPI_16, label = 'nodes=16')
plt.plot(nodes, MPI_23, label = 'nodes=23')
plt.text(0.5,43,"(b)")
# plt.plot(processes, numba, label = 'numba')
plt.xlabel('Number of Nodes')
plt.ylabel('Computational Time (s)')
plt.legend(fontsize=12)
plt.savefig("ComputationalTime_nodes.pdf", format="pdf",dpi=1200,bbox_inches ="tight")
plt.show()