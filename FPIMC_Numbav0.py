# -*- coding: utf-8 -*-
"""
@author: Yi Ming Chang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit # may as well speed up 1d code as well
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 18}) #24
plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18)
import timeit
import numba
import random
import math 

#%%
"2D Path Integral variational approach with Monte Carlo Simulation on ground state energy"
"and isotropic harmonic oscillator"

"2D Harmonic Oscillator"
@numba.jit(nopython=True) 
def V(x,y,m,omega):
    return 1/2*m*omega**2*(x**2+y**2)  # 1/2*m*omega**2*(x**2+y**2)

"Action, corresponding Eq (14)"
@numba.jit(nopython=True) 
def Action(V,x2,x1,x,y2,y1,y,m,omega):
    S = m/(2)*((x2-x)**2+(x-x1)**2+(y2-y)**2+(y-y1)**2) + V(x,y,m,omega) # 1/2*m((r_{i+1}-r_i)^2+(r_i-x_{i-1})^2) + V(r_i)
    return S

"""
Calculation/main code/Path Integral Monte Carlo approach with Metropolis algorithm

Path Integral MC2d takes: 
    path_x: initial path generate randomly x [-0.5,0.5)
    path_y: initial path generate randomly y [-0.5,0.5)
    Nsteps: time slices in a path, x(t), y(t)
    h: step size change per timestep, initial h = 0.1
    V: potential energy, ex: harmonic oscillator
    m: mass depend on \Delta t
    omega: angular frequency depdend on \Delta t
    
    the subroutine returns:
        a path x(t), a path y(t), and h after all of time slices in x been updated.    
"""

@numba.jit(nopython=True) 
def Path_integral_MC2d(path_x,path_y,Nsteps,h,V,m,omega):        
    change_rate = 0.            
    
    index = np.arange(0,Nsteps) # index of path links
    random.shuffle(index)       # randomly shuffle the index to update the path links
    
    for i in range(Nsteps):
        t = index[i]
        t_lower = (t + Nsteps - 1)% Nsteps # periodic boundary conditions [t - 1]
        t_upper = (t + 1) % Nsteps         # periodic boundary conditions [t + 1]
        
        # changed step of x, y, z
        alpha_x = h * (np.random.rand() - 0.5) # randomly generate changing step of [-0.5h, 0.5h)
        alpha_y = h * (np.random.rand() - 0.5) # randomly generate changing step of [-0.5h, 0.5h)
        # alpha_z = 0 # set z-aixs to zero; h * (np.random.rand() - 0.5) 
        
        # new paths
        x_new = path_x[t] + alpha_x
        y_new = path_y[t] + alpha_y
        # z_new = path_z[t] + alpha_z
        
        # use Boltman factor to weight the changes, exp(-S_new)/exp(-S_old)
        S_old = Action(V,path_x[t_upper],path_x[t_lower],path_x[t],path_y[t_upper],path_y[t_lower],path_y[t],m,omega)
        S_new = Action(V,path_x[t_upper],path_x[t_lower],x_new,path_y[t_upper],path_y[t_lower],y_new,m,omega)
        
        # Metropolis algorithm
        if S_new-S_old < 0 or np.exp(-(S_new-S_old))>np.random.rand():
            path_x[t] = x_new
            path_y[t] = y_new
            # path_z[t] = z_new
            change_rate += 1/Nsteps  
            
        
    h = h * change_rate/idrate 
    return path_x,path_y,h #path_z


@numba.jit(nopython=True) 
def Paths_generator2d(path_x,path_y,N_paths,Nsteps,timesteps,h,V,m,omega): # path_z
    # m = 1 * dt
    # omega = 1 * dt
    for i in range(timesteps):
        path_x,path_y,h = Path_integral_MC2d(path_x,path_y,Nsteps,h,V,m,omega)

    
    thermalized_path_x = path_x # this will be used as initial path 
    thermalized_path_y = path_y # this will be used as initial path 
    # thermalized_path_z = path_z # this will be used as initial path 
    
    "storing all the path links"
    path_arr_x = np.zeros((N_paths,Nsteps)) 
    path_arr_y = np.zeros((N_paths,Nsteps)) 
    # path_arr_z = np.zeros((N_paths,Nsteps)) 
    
    # simulating N paths with Nsteps of path links for a number of timesteps
    for i in range(N_paths):# N paths
        h = 0.1
        path_x = thermalized_path_x # initial
        path_y = thermalized_path_y # initial
        # path_z = thermalized_path_z # initial
        
        for j in range(timesteps): # simulation
            path_x,path_y,h = Path_integral_MC2d(path_x,path_y,Nsteps,h,V,m,omega)
        path_arr_x[i,:] = path_x[:]
        path_arr_y[i,:] = path_y[:] 
        # path_arr_z[i,:] = path_z[:] 
    
    return path_arr_x,path_arr_y,h # path_arr_z


"simple animation of change of path links for a run of simulation"
def graph(): 
    plt.clf() # close each time for new update graph
    
    ax = plt.axes(projection='3d')
    ax.plot(path_x,path_y,time_arr,color='black',marker='o')
    ax.plot(np.zeros((len(time_arr))),np.zeros((len(time_arr))),time_arr, color='r', linestyle='--') 
    # ax.contour3D(path_x,path_y,time_arr, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'Imaginary Time index')
    ax.view_init(0, 40)
    # ax.view_init(15, 40)
    
    plt.show()
    plt.pause(time_pause) # pause sensible value to watch what is happening
    if save == True:
        plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')

#%%

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.01

"Quick and dirty graph to save"
livegraph_flag = 0 # update graphs on screen every cycle (1)
cycle = 1     # 0.1
save = False
filename = " PathIntegral.pdf" # save at final time slice (example)

"Parameters"
dt = 0.2 # time step of x(t)
Nsteps = 200 # Number of time slices $x(\tau)$, also number of links for a path
timesteps = 100 # number of timesteps
h = 0.1 # spacing of each lattice or set to 0.1
m = 1 * dt # mass
omega = 1 * dt # angular frequency
idrate = 0.8 # or 1?
time_arr = np.arange(0,Nsteps,1) # time array of paths or index of paths, 
path_x = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, x(t) 
path_y = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, y(t) 
path_z = np.zeros((Nsteps)) # np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, z(t) 

N_paths = 10000    # number of paths to run
timesteps = 100 # each path runs simulation 20 timesteps
Nsteps = 200


start = timeit.default_timer() 

path_arr_x,path_arr_y,h = Paths_generator2d(path_x,path_y,N_paths,Nsteps,timesteps,h,V,m,omega)

stop = timeit.default_timer()
print ("Time time for solver", stop - start)


path_arr_histx = np.array(path_arr_x).flatten() # create an array in a size of len(N*Nsteps)
path_arr_histy = np.array(path_arr_y).flatten() # create an array in a size of len(N*Nsteps)
# path_arr_histz = np.array(path_arr_z).flatten() # create an array in a size of len(N*Nsteps)

# path_arr_histr = np.sqrt(path_arr_histx**2+path_arr_histy**2)
# plt.hist(path_arr_histr,bins=100,density=True,color='blue',histtype='bar',label='Numerical') # or bins = len(xs)

#%% plot 3-dimensional with original scale of x and y

hist, xedges, yedges = np.histogram2d(path_arr_histx, path_arr_histy,bins=40,density = True)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# hist = hist.T
# Construct arrays with the dimensions for the 16 bars.
dx = dy = np.ones_like(zpos)
dz = hist.ravel()

cmap = cm.get_cmap('viridis') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 


fig = plt.figure()
ax = plt.axes(projection='3d')
im = ax.bar3d(xpos, ypos, zpos,dx,dy,dz,color=rgba,label='Numerical')
# cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'$|\psi_0|^2$')
ax.view_init(20, 40)
# plt.savefig("Numerical_G3d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()

#%% plot 3-dimensional with double scale of x and y 

"The following plot is double space of x and y position, this might be more clear to see"

fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = np.ones_like(zpos)
dz = hist.ravel()

cmap = cm.get_cmap('viridis') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
ax.view_init(20, 40)
# plt.savefig("Numerical_G3d_2" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()
#%% "Analytical solution of probability density of ground state" 

"Analytical solution of probability distribution of 2d harmonic oscillator wavefunction (ground state)"
xs = np.arange(min(path_arr_histx),max(path_arr_histx),0.01)
ys = np.arange(min(path_arr_histy),max(path_arr_histy),0.01)

hbar = 1
gamma = m*omega/hbar

# @numba.jit(nopython=True) 
def f(xs,ys):
    psi0_2 = gamma/np.pi*np.exp(-gamma*(xs**2+ys**2))
    return psi0_2
    # (xs**2+ys**2)*np.exp(-gamma*(xs**2+ys**2)/2) #
    
    # np.sqrt(m*omega/np.pi/hbar) * np.exp(-gamma*(xs**2+ys**2)/2)

X, Y = np.meshgrid(xs, ys)
psi0_2 = f(X,Y)

print("probability density of analytical 2d harmonic oscillator wavefunction: ",
      np.trapz(np.trapz(psi0_2,xs),ys))


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, psi0_2,  cmap='viridis', edgecolor='none')
ax.view_init(20, 40)
ax.set_xlabel('x')
ax.set_ylabel('y')
# cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_zlabel(r'$|\psi_0|^2$')
# plt.savefig("Analytical_G3d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()
#%%
"Comparison of Numerical and Analytical probability distribution"
# Plot histogram using pcolormeshim = NonUniformImage(ax, interpolation=interp

"Analytical"
fig, ax = plt.subplots(figsize=(5,5))
im = ax.pcolormesh(X,Y,psi0_2,shading='auto') # , interpolation='nearest', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
# plt.savefig("Analytical_G2d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()

"Numerical"
X1, Y1 = np.meshgrid(xedges, yedges)
fig, ax = plt.subplots(figsize=(5,5)) 
im = ax.pcolormesh(X1,Y1,hist,shading='auto') # , interpolation='nearest', origin='lower')
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('histogram2d')
# plt.savefig("Numercial_G2d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()


