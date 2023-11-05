# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:40:56 2022

@author: Yi Ming Chang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 24})
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24)
import timeit
import numba
import random

#%%
"3D Path Integral variational approach with Monte Carlo Simulation on ground state energy"
"and isotropic harmonic oscillator"

"3D Harmonic Oscillator"
@numba.jit(nopython=True) 
def V(r,m,omega):
    return 1/2*m*omega**2*(r**2)  # 1/2*m*omega**2*(x**2+y**2+z**2)

"Calcution radius/distance"
@numba.jit(nopython=True) 
def R(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

"Action, corresponding Eq (16)"
@numba.jit(nopython=True) 
def Action(V,r2,r1,r,m,omega):
    S = m/(2)*((r2-r)**2+(r-r1)**2) + V(r,m,omega) # 1/2*m((r_{i+1}-r_i)^2+(r_i-x_{i-1})^2) + V(r_i)
    return S



"""
Calculation/main code/Path Integral Monte Carlo approach with Metropolis algorithm

Path Integral MC3d takes: 
    path_x: initial path generate randomly x [-0.5,0.5)
    path_y: initial path generate randomly y [-0.5,0.5)
    path_z: initial path generate randomly z [-0.5,0.5)                                              
    Nsteps: time slices in a path, x(t), y(t), z(t)
    h: step size change per timestep, initial h = 0.1
    V: potential energy, ex: harmonic oscillator
    m: mass depend on \Delta t
    omega: angular frequency depdend on \Delta t
    
    the subroutine returns:
        a path x(t), a path y(t), and h after all of time slices in x been updated.    
"""
@numba.jit(nopython=True) 
def Path_integral_MC(path_x,path_y,path_z,Nsteps,h,V,m,omega):        
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
        alpha_z = 0 # set z-aixs to zero; h * (np.random.rand() - 0.5) 
        
        # new paths
        x_new = path_x[t] + alpha_x
        y_new = path_y[t] + alpha_y
        z_new = path_z[t] + alpha_z
        
        # update r
        r2 = R(path_x[t_upper],path_y[t_upper],path_z[t_upper])
        r1 = R(path_x[t_lower],path_y[t_lower],path_z[t_lower])
        r  = R(path_x[t],path_y[t],path_z[t])
        r_new = R(x_new,y_new,z_new)
        
        # use Boltman factor to weight the changes, exp(-S_new)/exp(-S_old)
        S_old = Action(V,r2,r1,r,m,omega)
        S_new = Action(V,r2,r1,r_new,m,omega)
        
        # Metropolis algorithm
        if S_new-S_old < 0 or np.exp(-(S_new-S_old))>np.random.rand():
            path_x[t] = x_new
            path_y[t] = y_new
            path_z[t] = z_new
            change_rate += 1/Nsteps  
            
        # S_old = S_new
    h = h * change_rate/idrate 
    return path_x,path_y,path_z, h


# @numba.jit(nopython=True) 
# def Paths_generator(path,h,N_paths,Nsteps,timesteps,m,omega,dt):
#     # m = 1 * dt
#     # omega = 1 * dt
#     for i in range(timesteps):
#         path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)

#     thermalized_path = path

#     path_arr = np.zeros((N_paths,Nsteps)) 
    
#     # simulating N paths with Nsteps of path links for a number of timesteps
#     for i in range(N_paths):
#         h = 0.1 # boundary of changing steps
#         path = thermalized_path
#         for j in range(timesteps):
#             path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)
#         path_arr[i,:] = path[:]
    
#     return path_arr, h




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
filename = "PathIntegral.pdf" # save at final time slice (example)

# "Parameters"
dt = 0.3 # time step of x(t)
Nsteps = 1000 # Number of time slices $x(\tau)$, also number of links for a path
timesteps = 100 # number of timesteps
h = 0.1 # spacing of each lattice or set to 0.1
m = 1 * dt # mass
omega = 1 * dt # angular frequency
idrate = 0.8 #0.8    # or 0.8?
time_arr = np.arange(0,Nsteps,1) # time array of paths or index of paths, 
path_x = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, x(t) 
path_y = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, y(t) 
path_z = np.zeros((Nsteps)) # np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, z(t) 

# path_x = np.zeros(Nsteps)
# path_y = np.zeros(Nsteps)
# path_z = np.zeros(Nsteps)


if livegraph_flag==1: 
    fig = plt.figure(figsize=(10,8))
    
start = timeit.default_timer() # timing

"n timesteps of simulation with simple animation"
for i in range(timesteps):
    path_x,path_y,path_z,h = Path_integral_MC(path_x,path_y,path_z,Nsteps,h,V,m,omega)
 
    # Animation of path links changing
    if livegraph_flag == 1 and i%cycle ==0:
        graph()
        
stop = timeit.default_timer()
print ("Time time for solver", stop - start)

thermalized_path_x = path_x # this will be used as initial path 
thermalized_path_y = path_y # this will be used as initial path 
thermalized_path_z = path_z # this will be used as initial path 

#%%

N_paths = 5000    # number of paths to run
timesteps = 100 # each path runs simulation 20 timesteps
Nsteps = 1000

"storing all the path links"
path_arr_x = np.zeros((N_paths,Nsteps)) 
path_arr_y = np.zeros((N_paths,Nsteps)) 
path_arr_z = np.zeros((N_paths,Nsteps)) 


start = timeit.default_timer() 

for i in range(N_paths):# N paths
    h = 0.1
    path_x = thermalized_path_x # initial
    path_y = thermalized_path_y # initial
    path_z = thermalized_path_z # initial
    
    for j in range(timesteps): # simulation
        path_x,path_y,path_z,h = Path_integral_MC(path_x,path_y,path_z,Nsteps,h,V,m,omega)
    path_arr_x[i,:] = path_x[:]
    path_arr_y[i,:] = path_y[:] 
    path_arr_z[i,:] = path_z[:] 

    
stop = timeit.default_timer()
print ("Time time for solver", stop - start)


#%%
path_arr_histx = np.array(path_arr_x).flatten() # create an array in a size of len(N*Nsteps)
path_arr_histy = np.array(path_arr_y).flatten() # create an array in a size of len(N*Nsteps)
path_arr_histz = np.array(path_arr_z).flatten() # create an array in a size of len(N*Nsteps)

path_arr_histr = np.sqrt(path_arr_histx**2+path_arr_histy**2)

#%%
# hist, xedges, yedges = np.histogram2d(path_arr_histx, path_arr_histy,bins=(100,100), 
#                                       range = [[min(path_arr_histx),max(path_arr_histx)],
#                                                [min(path_arr_histy),max(path_arr_histy)]],density = True)

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# # hist = hist.T
# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = np.ones_like(zpos)

# dz = hist.ravel()
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.bar3d(xpos, ypos, zpos,dx,dy,dz,cmap='viridis',edgecolor='none',label='Numerical')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel(r'$|\psi_0|^2$');
# ax.view_init(10, 40)
# plt.savefig("Numerical_G3d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
# plt.show()

#%%
# Plot histogram using pcolormeshim = NonUniformImage(ax, interpolation=interp

hist, xedges, yedges = np.histogram2d(path_arr_histx, path_arr_histy,bins=100, 
                                     density = True)
X, Y = np.meshgrid(xedges, yedges)

fig, ax = plt.subplots(figsize=(5,5)) 
im = plt.pcolormesh(X,Y,hist) # , interpolation='nearest', origin='lower')
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.savefig("Numercial_G2d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()
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


fig = plt.figure(figsize=(3,3))
plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
ax = plt.axes(projection='3d')
im = ax.bar3d(xpos, ypos, zpos,dx,dy,dz,color=rgba,label='Numerical')
# cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.text(-20,-20,0.02,"(a)")
ax.set_zlabel(r'$|\psi_0|^2$')
# ax.view_init(20, 40)
# plt.savefig("Numerical_G3d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()
#%%


fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

#make histogram stuff - set bins - I choose 20x20 because I have a lot of data
# hist, xedges, yedges = np.histogram2d(x, y, bins=)

hist, xedges, yedges = np.histogram2d(path_arr_histx, path_arr_histy,bins=400, 
                                     density = True)

xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

cmap = cm.get_cmap('viridis') # Get desired colormap - you can change this!
max_height = np.max(dz)   # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/max_height) for k in dz] 

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
plt.show()

#%%

"Analytical solution of probability density of frist excitation wavefunction \psi_(1,0)*\psi_(0,1)"
xs = np.arange(min(path_arr_histx),max(path_arr_histx),0.01)
ys = np.arange(min(path_arr_histy),max(path_arr_histy),0.01)

hbar = 1
gamma = m*omega/hbar

def f(xs,ys):
    return  (gamma/np.pi)*(xs**2+ys**2)*np.exp(-gamma*(xs**2+ys**2)) #
    # np.sqrt(m*omega/np.pi/hbar) * np.exp(-gamma*(xs**2+ys**2)/2)

X, Y = np.meshgrid(xs, ys)
psi0_2 = f(X,Y)


fig, ax = plt.subplots(figsize=(5,5),ncols=1)
im = ax.pcolormesh(X,Y,psi0_2) # , interpolation='nearest', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
# plt.savefig("Analytical_G2d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()


fig = plt.figure(figsize=(3,3),)
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




