"""
ENPH/PHYS 479/879: High Performance Computational Physics, Winter Term, 2022
Project: Feynman's Quantum Path Integration variational approach to 
                the ground state wavefuction with Quantum oscillator
@author: Yi-Ming Chang
Student ID: 20296862
Email: 21ymc@queensu.ca
Instructor: Prof. S. Hughes
"""

"""
Please apply "%matplotlib auto" to the console before run the code.

The following code is unparallelized 1d and 2d Feynman's Quantum path integral Monte Carlo approach, solved
            ground state wavefunction with harmonic oscillator. The computational 
            method used Metropolis Algorithm.
            
            
            The main subroutines are "Path_Integral_MC", "Paths_generator" and "Path_Integral_MC2d"
            Each function has a description on the top and has listed out which equation it
            corresponding inside the report.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 24})
import timeit
import random
import numba

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#%% 1d ground state Feynman's Quantum path integral formulation of Quantum Mechanics

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.01

"Quick and dirty graph to save"
livegraph_flag = 1 # update graphs on screen every cycle (1)
cycle = 1     # 0.1
save = False
filename = "PathIntegral.pdf" # save at final time slice (example)

"1D Quantum oscillator"
@numba.jit(nopython=True) 
def V(x,m,omega): 
    return 1/2*m*omega**2*x**2 # m*g*np.abs(x)

"Calculating action, S at x_i (Eq 9 in the report)"
@numba.jit(nopython=True)
def Action(V,x1,x2,x,m,omega): # where x1 is next path link and x2 is previous path link
    S = m/(2)* ((x2-x)**2+(x-x1)**2) + V(x,m,omega)  # 1/2*m((x_{i+1}-x_i)^2+(x_i-x_{i-1})^2) + V(x_i)
    return S


# @numba.jit(nopython=True) 
# def action_sum(path): # in here x=x(t) it is the path from partile A to B
#     S_sum = 0.
#     for i in range(Nsteps):
#         S_sum += 1/2 * m *(path[(i+1)%Nsteps]-path[i])**2 + V(path[i],m,omega) # periodic boundary condition
#     return S_sum


"Simple animation of change of path links for a run of simulation"
def graph(): 
    plt.clf() # close each time for new update graph
  
    plt.plot(time_arr,path,'black')
    plt.axhline(y=0, color='r', linestyle='--') 
    plt.xlabel(r'Imaginary Time index')
    plt.ylabel(r'Position')

    plt.show()
    plt.pause(time_pause) # pause sensible value to watch what is happening
    if save == True:
        plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')

#%% Main code section

if livegraph_flag==1: 
    fig = plt.figure(figsize=(10,8))

"""
Calculation/main code/Path Integral Monte Carlo approach with Metropolis algorithm

Path Integral MC takes: 
    path: initial path generate randomly x [-0.5,0.5)
    Nsteps: time slices in a path, x(t)
    h: step size change per timestep, initial h = 0.1
    V: potential energy, ex: harmonic oscillator
    m: mass depend on \Delta t
    omega: angular frequency depdend on \Delta t
    
    the subroutine returns:
        a path x(t) and h after all of time slices in x been updated.                
"""

@numba.jit(nopython=True) 
def Path_integral_MC(path,Nsteps,h,V,m,omega,dt):        
    change_rate = 0.            
    
    index = np.arange(0,Nsteps) # index of path links
    random.shuffle(index)       # randomly shuffle the index to update the path links
    
    for i in range(Nsteps):
        t = index[i]
        t_lower = (t + Nsteps - 1)% Nsteps # periodic boundary conditions [t - 1]
        t_upper = (t + 1) % Nsteps         # periodic boundary conditions [t + 1]
        
        # new path 
        alpha = h * (np.random.rand() - 0.5) # randomly generate changing step of [-0.5h, 0.5h)
        x_new = path[t] + alpha
        
        # use Boltman factor to weight the changes, exp(-dS)
        S_old = Action(V,path[t_upper],path[t_lower],path[t],m,omega)
        S_new = Action(V,path[t_upper],path[t_lower],x_new,m,omega)
        
        # Metropolis algorithm
        if S_new-S_old < 0 or np.exp(-(S_new-S_old))>np.random.rand():
            path[t] = x_new
            change_rate += 1/Nsteps  
            
        # S_old = S_new
    h = h * change_rate/idrate # update h for every timesteps
    return path, h
        
"Parameters"
dt = 0.3 
Nsteps = 100 # Number of links in $x(\tau)$, or time slices
timesteps = 100 # number of timesteps
h = 0.1 # spacing of each lattice 
m = 1 * dt # mass
omega = 1 * dt # angular frequency
idrate = 0.8  # or 1
time_arr = np.arange(0,Nsteps,1) # time array of paths or index of paths, 
path = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, x(t) 
# or path0 = np.zeros(Nsteps)
# path = path0  # load initial path links


start = timeit.default_timer() # timing

"n timesteps of simulation with simple animation"
for i in range(timesteps):
    path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)
    
    # Animation of path links changing
    if livegraph_flag == 1 and i%cycle ==0:
        graph()
        
stop = timeit.default_timer()
print ("Time time for solver", stop - start)
        
thermalized_path = path # this will be used as initial path 
#%% Monte carlo simulation section

N_paths = 5000  # number of paths to run
timesteps = 100 # each path runs simulation 20 timesteps
Nsteps = 100
path_arr = np.zeros((N_paths,Nsteps)) # storing all the path links


start = timeit.default_timer() 

"Monte Carlo simulation use a thermalized path for N number of paths"
for i in range(N_paths):# N paths
    h = 0.1
    path = thermalized_path # initial
    for j in range(timesteps): # simulation
        path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)
    path_arr[i,:] = path[:] 
    
stop = timeit.default_timer()
print ("Time time for solver", stop - start)

path_arr_hist = np.array(path_arr).flatten() # create an array in a size of len(N*Nsteps)
#%%

"Analytical solution of probability density of ground state"
hbar = 1
xs = np.arange(min(path_arr_hist),max(path_arr_hist),0.01)
psi0_2 = np.exp(- m*omega * xs**2) * np.sqrt(m*omega/np.pi/hbar) 

print("Probability density of analytical solution: ", np.trapz(psi0_2,xs))

#%%
"Comparison of Numerical and Analytical probability distribution"
plt.figure()
plt.hist(path_arr_hist,bins=100,density=True,color='blue',histtype='bar',label='Numerical') # or bins = len(xs)
plt.plot(xs,psi0_2,label="Analytical",color="black")
plt.xlabel("x")
plt.ylabel(r"$|\psi_0|^2$")
plt.legend(handlelength=1.7, loc='upper center', 
           bbox_to_anchor=(0.5, 1.23),
           ncol=2, prop={'size': 20})
# plt.savefig("Harmonic_oscillator_ground_state" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()

"One single paths"
plt.figure()
plt.plot(path_arr[N_paths-1], time_arr)
plt.xlabel("x")
# plt.xlim((-3,3))
plt.ylabel("Imaginary Time Index")
# plt.savefig("One_path" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()

#%% Calculation of expectation values; <x>, <x^2>, <x^3>, <x^4>

"""
Paths_generator combines Main code section and Monte Carlo simulation section:
    
Path generator takes: 
    path: initial path generate randomly x [-0.5,0.5)
    h: step size change per timestep, initial h = 0.1
    N_paths: number of paths to simulate
    Nsteps: time slices in a path, x(t)
    timesteps: number of loops to update a path
    m: mass depend on \Delta t
    omega: angular frequency depdend on \Delta t
    
    It returns:
        a matrix of path (N_paths,Nsteps), h
        
    One can simulate N paths together using a same thermalized path
"""
@numba.jit(nopython=True) 
def Paths_generatorMC(path,h,N_paths,Nsteps,timesteps,m,omega,dt):
    # m = 1 * dt
    # omega = 1 * dt
    for i in range(timesteps):
        path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)

    thermalized_path = path

    path_arr = np.zeros((N_paths,Nsteps)) 
    
    # simulating N paths with Nsteps of path links for a number of timesteps
    for i in range(N_paths):
        h = 0.1 # boundary of changing steps
        path = thermalized_path
        for j in range(timesteps):
            path, h = Path_integral_MC(path,Nsteps,h,V,m,omega,dt)
        path_arr[i,:] = path[:]
    
    return path_arr, h

"Parameters"
N_paths = 5000  # number of paths
timesteps = 100   # number of time loops
Nsteps = 100 # Number of links/steps
dt = 0.3
m = 1 * dt
omega = 1 * dt
h = 0.1
idrate = 0.8
"initial conditions/inputs"
path = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, x(t)

start = timeit.default_timer()

"Simulating N paths with 2500 path links for 50 timesteps "
path_arr, h = Paths_generatorMC(path,h,N_paths,Nsteps,timesteps,m,omega,dt)
        
stop = timeit.default_timer()
print ("Time time for solver", stop - start)

#%%

"Calculation of expectation values of x, x^2, x^3 and x^4, n is the degree number"
@numba.jit(nopython=True) 
def Expectation_value(func,n,path_arr,Nsteps,N_paths,dt):
    s = 0
    for i in (path_arr):
        for j in i: # read through all the x values
            s += func(j,n,dt)
    s = s/(Nsteps*N_paths) # Normalized it
    return s

"Expectation value of various degree of x"
@numba.jit(nopython=True) 
def Expectation_x(x,n,dt):
    return (x*dt)**n


#%%

dt = 0.3

Expt_x_1 = Expectation_value(Expectation_x,1,path_arr,Nsteps,N_paths,dt)
print("Expectation value ⟨X⟩ =  ", Expt_x_1)

Expt_x_2 = Expectation_value(Expectation_x,2,path_arr,Nsteps,N_paths,dt)
print("Expectation value ⟨X^2⟩ =  ", Expt_x_2)

Expt_x_3 = Expectation_value(Expectation_x,3,path_arr,Nsteps,N_paths,dt)
print("Expectation value ⟨X^3⟩ =  ", Expt_x_3)

Expt_x_4 = Expectation_value(Expectation_x,4,path_arr,Nsteps,N_paths,dt)
print("Expectation value ⟨X^4⟩ =  ", Expt_x_4)


#%% 
"ground state energy calculation"


"""
Calculate ground state energy of harmonic oscillator using Eq (12) inside the report

E_0  =  m*\omega^2(<x^2> - <x>)
"""    
@numba.jit(nopython=True) 
def ground_state_energy(path_arr,m,omega,Nsteps,N_paths,dt):
    m = m / dt # make mass to nature unit 1
    omega = omega/ dt # make angaular frequency to 1
    x = Expectation_value(Expectation_x,1,path_arr,Nsteps,N_paths,dt)
    x2 = Expectation_value(Expectation_x,2,path_arr,Nsteps,N_paths,dt)
    return m*(omega**2)*(x2 - x**2)  # mw^2(<x^2>-<x>**2)

"""
Calcuate ground state energy of harmonic oscillator with a vector of path arr

One can calculate N paths for different times with different dt and h to see the ground state energy changes
with different value of dt.
"""
@numba.jit(nopython=True)
def Energy(Paths_generator,ground_state_energy,dt_arr,h_arr,N_arr,Nsteps,timesteps,m,omega):
    E0_arr = np.zeros((len(N_arr)))
    for j in range(len(N_arr)):
        dt = dt_arr[j]
        m = 1 * dt
        omega = 1 * dt
        path0 = np.zeros(Nsteps)
        path_arr, h= Paths_generator(path0,h_arr[j],N_arr[j],Nsteps,timesteps,m,omega,dt)
        E0 = ground_state_energy(path_arr,m,omega,Nsteps,N_arr[j],dt)
        E0_arr[j] = E0
    return E0_arr

m = 1
omega = 1
dt_arr = np.linspace(0.1,1,8) # testing various delta t from 0.1 to 1 with
N_arr = np.array([1000]*8) #[500,500,500,500,500,500,500,500]
h_arr = np.array([0.1]*8)
timesteps = 100
Nsteps = 5000
idrate = 0.8

start = timeit.default_timer() # timing

E0_arr = Energy(Paths_generatorMC,ground_state_energy,dt_arr,h_arr,N_arr,Nsteps,timesteps,m,omega)
    
stop = timeit.default_timer()
print ("Time time for solver", stop - start)

#%%
print(E0_arr)

plt.figure()
plt.plot(dt_arr,E0_arr,marker="o")
plt.xlabel(r"$\Delta t$")
plt.ylabel(r"$E_0$")
# plt.savefig("Ground_state_energy " + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()

#%% Propagator (Haven't figure out how to plot numerical green's function yet)

"Analytical solution of propagator, Green's function, <x|exp(-H*T)|x>"

N = 100
Xmax = 101

T0 = 2*np.pi
periods = 2
tmax = periods*T0 # 4
dt = tmax/N
E0 = 1/2
xs = np.arange(0,2,2/N)
hbar = 1
A = (m/(2*np.pi*dt*hbar))**(N/2) # normalized factor

# calculate analytical solution of propagator, <x|exp(-H*T)|x>, where T = t_f - t_i
def propagator(x):
    return (np.exp(-x**2/2)/np.pi**(1/4))**2*np.exp(-E0*tmax)


fig=plt.figure(figsize=(8,6),dpi=100)  

ax = fig.add_axes([.15, .15, .6, .6])
ax.plot(xs,propagator(xs)) # label='Analytical solution'
plt.xlabel('x')
plt.ylabel(r"$\left\langle x|e^{-HT}|x \right\rangle$")#(r"$\braket{$")
ax.set_title('Propagator')
# plt.legend(handlelength=1.7, loc='upper center', 
#            bbox_to_anchor=(0.5, 1.15),
#            ncol=1, prop={'size': 16})
# plt.savefig("Propagator " + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()


#%%

"""
2D Path Integral variational approach with Monte Carlo Simulation on ground state energy
on isotropic harmonic oscillator

isotropic harmonic oscillator: Omega_x = Omega_y = Omega_z = Omega
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit # may as well speed up 1d code as well
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 24}) #24
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24)
import timeit
import numba
import random
import math 

"2D Harmonic Oscillator"
@numba.jit(nopython=True) 
def V(x,y,m,omega):
    return 1/2*m*omega**2*(x**2+y**2)  # 1/2*m*omega**2*(x**2+y**2+z**2)

"Action, corresponding Eq (14)"
@numba.jit(nopython=True) 
def Action(V,x2,x1,x,y2,y1,y,m,omega):
    S = m/(2)*((x2-x)**2+(x-x1)**2+(y2-y)**2+(y-y1)**2) + V(x,y,m,omega) # 1/2*m((r_{i+1}-r_i)^2+(r_i-x_{i-1})^2) + V(r_i)
    return S

"""
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
def Path_integral_MC2d(path_x,path_y,path_z,Nsteps,h,V,m,omega):        
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
        
        # # update r for three dimensional
        # r2 = R(path_x[t_upper],path_y[t_upper],path_z[t_upper])
        # r1 = R(path_x[t_lower],path_y[t_lower],path_z[t_lower])
        # r  = R(path_x[t],path_y[t],path_z[t])
        # r_new = R(x_new,y_new,z_new)
        
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
    return path_x,path_y,path_z, h


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
    ax.view_init(7, 40)
    # ax.view_init(15, 40)
    
    plt.show()
    plt.pause(time_pause) # pause sensible value to watch what is happening
    if save == True:
        plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')

#%%

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.01
    
"Quick and dirty graph to save"
livegraph_flag = 1 # update graphs on screen every cycle (1)
cycle = 1     # 0.1
save = False
filename = " PathIntegral.pdf" # save at final time slice (example)

"Parameters"
dt = 0.2 # time step of x(t)
Nsteps = 100 # Number of time slices $x(\tau)$, also number of links for a path
timesteps = 100 # number of timesteps
h = 0.1 # spacing of each lattice or set to 0.1
m = 1 * dt # mass
omega = 1 * dt # angular frequency
idrate = 0.8 # or 1?
time_arr = np.arange(0,Nsteps,1) # time array of paths or index of paths, 
path_x = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, x(t) 
path_y = np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, y(t) 
path_z = np.zeros((Nsteps)) # np.array([np.random.rand()-0.5 for i in range(Nsteps)]) # random generate path links, z(t) 

# path_x = np.zeros(Nsteps)
# path_y = np.zeros(Nsteps)
# path_z = np.zeros(Nsteps)


if livegraph_flag== 1: 
    fig = plt.figure(figsize=(4,4))
    
start = timeit.default_timer() # timing

"n timesteps of simulation with simple animation"
for i in range(timesteps):
    path_x,path_y,path_z,h = Path_integral_MC2d(path_x,path_y,path_z,Nsteps,h,V,m,omega)
 
    # Animation of path links changing
    if livegraph_flag == 1 and i%cycle ==0:
        graph()
        
stop = timeit.default_timer()
print ("Time time for solver", stop - start)

thermalized_path_x = path_x # this will be used as initial path 
thermalized_path_y = path_y # this will be used as initial path 
thermalized_path_z = path_z # this will be used as initial path 

#%%

N_paths = 10000    # number of paths to run
timesteps = 100 # each path runs simulation 20 timesteps
Nsteps = 100

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
        path_x,path_y,path_z,h = Path_integral_MC2d(path_x,path_y,path_z,Nsteps,h,V,m,omega)
    path_arr_x[i,:] = path_x[:]
    path_arr_y[i,:] = path_y[:] 
    path_arr_z[i,:] = path_z[:] 

    
stop = timeit.default_timer()
print ("Time time for solver", stop - start)


#%%
path_arr_histx = np.array(path_arr_x).flatten() # create an array in a size of len(N*Nsteps)
path_arr_histy = np.array(path_arr_y).flatten() # create an array in a size of len(N*Nsteps)
path_arr_histz = np.array(path_arr_z).flatten() # create an array in a size of len(N*Nsteps)


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
xs = np.arange(min(path_arr_histx),max(path_arr_histx),0.1)
ys = np.arange(min(path_arr_histy),max(path_arr_histy),0.1)

hbar = 1
gamma = m*omega/hbar

# @numba.jit(nopython=True) 
"Analytical solution corresponding to Eq (15)"
def f(xs,ys):
    psi0_2 = gamma/np.pi*np.exp(-gamma*(xs**2+ys**2))
    return psi0_2
    # (xs**2+ys**2)*np.exp(-gamma*(xs**2+ys**2)/2) #
    
    # np.sqrt(m*omega/np.pi/hbar) * np.exp(-gamma*(xs**2+ys**2)/2)

X, Y = np.meshgrid(xs, ys)
psi0_2 = f(X,Y)

print("probability density of analytical 2d harmonic oscillator wavefunction: ",
      np.trapz(np.trapz(psi0_2,xs),ys))

fig = plt.figure(figsize=(3,3))
plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)

ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, psi0_2,  cmap='viridis', edgecolor='none')
# ax.view_init(20, 40)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.text(-20,-20,0.02,"(b)")
# cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_zlabel(r'$|\psi_0|^2$')
# plt.savefig("Analytical_G3d" + filename,format='pdf',dpi=1200,bbox_inches = 'tight')
plt.show()
#%%
"Comparison of Numerical and Analytical probability distribution"
plt.rcParams.update({'font.size': 18})
plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18)

"Analytical"
fig, ax = plt.subplots(figsize=(4,4))
im = ax.pcolormesh(X,Y,psi0_2) # , interpolation='nearest', origin='lower')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.text(-24,14,"(b)")
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
# plt.savefig("Analytical_G2d" + filename,format='pdf',dpi=1200,bbox_inches = 'tight')
plt.show()

"Numerical"
X1, Y1 = np.meshgrid(xedges, yedges)
fig, ax = plt.subplots(figsize=(4,4)) 
im = ax.pcolormesh(X1,Y1,hist) 
ax.text(-24,14,"(a)")
cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel(r'$|\psi_0|^2$')
ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('histogram2d')
# plt.savefig("Numercial_G2d" + filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.show()





