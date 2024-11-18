# Path_integral_approach_with_monte_carlo
Feynman’s Path Integral Monte Carlo approach with Metropolis and Lattice Quantum Mechanics algorithm:

For the details of the project please read the PDF file: **"Research_project_of_879.pdf"**.

There are five Python files:

Please apply "%matplotlib auto" to the console before run the following code.
**FPIMC_1d2dv3.py:** It is unparallelized 1d and 2d Feynman's Quantum path integral Monte Carlo (FPIMC) approach, solved
          	 ground state wavefunction with harmonic oscillator. The computational method used Metropolis Algorithm.
		 The 1d FPIMC has validated with the analytical solution of ground state probability distribution and 
		 values of $\braket{x}$, $\braket{x^2}$, $\braket{x^3}$ and $\braket{x^4}$ and ground state energy of harmonic oscillator.
		
**FPIMC_3dv0.py:** It is unparallelized 3d Feynman's Quantum path integral Monte Carlo (FPIMC) approach, solved
               ground state wavefunction with harmonic oscillator. It hasn't validate with any analytical solutions. 
	       The three dimensional python has not complete since, I am not sure how to plot 4 dimensional $|\psi(\mathbf{r},t)|^2$, 
	       but I think that I accidentally plot the first excitation probability distribution from it. I am not sure if it is 
	       correct, so I did not report it in my final report. 

**FPIMC_Numbav0.py:** Apply only numba to 2d ground state wavefunction with Feynman's Quantum path Integral Monte Carlo (FPIMC).

**FPIMC_MPI2dv0.py:** Parallel simulation (MPI4py) code with numba to 2d ground state wavefunction with Feynman's Quantum path Integral Monte Carlo (FPIMC).

**Computational time.py:** Plot the simulation time from 2D Path Integral MC with various cores and nodes with numba

Excel file is just to record the simulation time. 

[View the PDF](Harmonic_oscillator_ground_statePathIntegral.pdf)

[View the PDF](Numerical_G2d_PathIntegral.pdf)

[View the PDF](Numerical_G3d_PathIntegral.pdf)

