#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_run_MD.py

Main program for running molecular dynamics simulations using Lennard-Jones particles.
Initializes the system, runs the integrator loop, records energy and trajectory data, 
and visualizes results.

Author: Bettina Keller
Created: May 28, 2025

Modified by: Luka Jurečič, Charlotte Schuster
Date: July 10, 2026 #TODO

Modifications include:
#TODO

This script imports all classes and functions from md_simulation.py and controls
the simulation workflow.

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt

import time
from datetime import datetime


from LJ_gas import(
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    calculate_force,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure,
    simulate_leapfrog_step,
    B_step
    )

#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------
# Define tic and toc functions
def tic():
    """Start a timer."""
    global _tic_time
    _tic_time = time.time()

def toc():
    """Stop the timer and return the elapsed time in seconds."""

    elapsed_time = None
    
    if '_tic_time' in globals():
        elapsed_time = time.time() - _tic_time
    
    else:
        print("Error: tic() was not called before toc()")
    
    return elapsed_time


#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.1                # ps
n_steps = 1000 
temperature = 300       # K
box_length = 100        # nm
tau_thermostat = 1      # thermostat coupling constant in 1/ps
rij_min = 1e-2          # nm
NVT = False              # switch to decide between NVT and NVE
LEAPFROG = True        # switch to decide if the script uses the velocity Verlet or Leapfrog integrator
RSeed = True            # switch to decide if the random seed is used or not
random_seed = 42        # a random seed (RSeed = True) is necessary for comparisons between integrators

# output
if LEAPFROG == True:
    file_name_base = "my_simulation_NVE_leapfrog"   # file name for all output files using 
                                                    # leapfrog integrator
elif NVT == True:
    file_name_base = "my_simulation_NVT_Langevin"   # file name for all output files using 
                                                    # Langevin integrator (BAOAB splitting) in NVT ensemble

else:
    file_name_base = "my_simulation_NVE_vVerlet"    # file name for all output files using 
                                                    # velocity Verlet integrator in NVE ensemble

#----------------------------------------------------------------
#   P R O G R A M
#----------------------------------------------------------------
# start the timer
tic()
#
# initialize simulation parameters
#
sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps, 
                           temperature = temperature, 
                           box_length = box_length, 
                           tau_thermostat = tau_thermostat,
                           rij_min=rij_min
                           )

#
# initialize ParticleSystem 
#
ps = ParticleSystem(n_particles)

# fill in the parameters for argon
for i in range(n_particles): 
    ps.set_parameters(i, mass=mass_argon, sigma=sigma_argon, epsilon=epsilon_argon)

# Switch random seed or off:
if RSeed == True:
    np.random.seed(random_seed)

# set initial positions     
initialize_positions(ps, sim.box_length)

# set initial velocities     
initialize_velocities(ps, sim.temperature)

# calculate force according to initial positions
calculate_force(ps, sim)

# calculate box density
rho = density(ps, sim)

# calculate initial values of variable properties
E_pot_init = potential_energy(ps, sim)
E_kin_init = kinetic_energy(ps)
T_init = instantaneous_temperature(ps)
P_init = ideal_gas_pressure(ps, sim)


# initialize position trajectory
position_trajectory = np.zeros((sim.n_steps+1, n_particles, 3))
position_trajectory[0,:,:] = ps.position # initial position

# initialize energy trajectory
energy_trajectory = np.zeros((sim.n_steps+1, 4))
energy_trajectory[0,0] = potential_energy( ps, sim)       # potential energy
energy_trajectory[0,1] = kinetic_energy(ps)               # kinetic energy
energy_trajectory[0,2] = instantaneous_temperature(ps)    # instantaneous temperature
energy_trajectory[0,3] = ideal_gas_pressure(ps, sim)      # ideal gas pressure

# modification
# Initialise storage of velocity
velocity_trajectory = np.zeros((sim.n_steps+1,n_particles, 3))
velocity_trajectory[0,:,:] = ps.velocity #initial velocity is stored






#--------------------------------------------------
#  The acutal MD simulation
#--------------------------------------------------

# For the leapfrog algoritm we need to do a half B step forward before 
# entering the function simulate_leapfrog_step(ps, sim)
# (to have the velocities from v(t = 0) --> v(t = 0+1/2 delta_t)):
if LEAPFROG == True:
    B_step(ps, sim, half_step=True)



for i in range(sim.n_steps):
    '''
    This loop computes potential and kintetic energiy updates, instantaneous temperature updates and ideal gas pressure updates
    for every simulation step;
    with respect to the Leapfrog algorithm, the Langevin integrator or the velocity Verlet integrator
    '''
    if LEAPFROG == True:
        '''
        This is the loop for the Leapfrog algorithm
        For every time step, the function simulate_leapfrog_step(ps, sim) is called
        '''
        v_before = ps.velocity.copy()                           # store the velocities before the loop, 
                                                                # so if t=0, it stores v(0 + 0.5 * delta_t)
        simulate_leapfrog_step(ps, sim)                         # simulates one leapfrog step
                                                                # first loop: positions at t = 1, velocities at t = 1.5 * delta_t                
        v_after = ps.velocity.copy()                            # store the velocities after the loop, so if i=0 (which means t=1)
                                                                # it stores v(1.5 * delta_t)
        v_sync = 0.5 * (v_before + v_after)                     # averaged velocity at integer time step
                                                                # for the first loop v(t = 1)
        
        


        # Using synchronized velocity for energy calculation
        v_actual = ps.velocity.copy()                           # store the actual velocity (at half time step)
        ps.velocity = v_sync                                    # temporarily replacing the velocity with the synchronized velocity 
        
        #modification
        velocity_trajectory[i+1,:,:] = ps.velocity              # we want to store the integer velocity to plot


        #Now calculating the energies with the synchronized velocity (at integer time step)
        energy_trajectory[i+1,0] = potential_energy(ps,sim)     # calculate the potential energy with the synchronized velocity
        energy_trajectory[i+1,1] = kinetic_energy(ps)           # calculate the kinetic energy with the synchronized velocity
        energy_trajectory[i+1,2] = instantaneous_temperature(ps)# calculate the instantaneous temperature with the synchronized velocity
        energy_trajectory[i+1,3] = ideal_gas_pressure(ps, sim)  # calculate the ideal gas pressure with the synchronized velocity

        ps.velocity = v_actual                                  # replace the velocity back to the actual velocity (at half time step)
        


    else:
        if NVT==True:
            '''
            This is the loop for the Langevin integrator (BAOAB splitting) in NVT ensemble
            For every time step, the function simulate_NVT_step(ps, sim) is called
            '''
            simulate_NVT_step(ps, sim)
            
        else: 
            '''
            This is the loop for the velocity Verlet integrator in NVE ensemble.
            For every time step, the function simulate_NVE_step(ps, sim) is called
            '''
            simulate_NVE_step(ps, sim)
        
        # store updated energies, temperature and pressure
        energy_trajectory[i+1,0] = potential_energy( ps, sim)     # potential energy
        energy_trajectory[i+1,1] = kinetic_energy(ps)             # kinetic energy
        energy_trajectory[i+1,2] = instantaneous_temperature(ps)  # instantaneous temperature
        energy_trajectory[i+1,3] = ideal_gas_pressure(ps, sim)    # ideal gas pressure

        #modification
        velocity_trajectory[i+1,:,:] = ps.velocity                # we want to store the integer velocity to plot
    
        
        
    # store updated positions
    position_trajectory[i+1,:,:] = ps.position # store updated positions

#--------------------------------------
# W R I T E    T R A J E C T O R I E S 
#--------------------------------------
# write position trajectory to file
write_xyz_trajectory(file_name_base + "_pos.xyz", position_trajectory, atom_symbol="Ar")
# write energy trajectory to file (binary and text)
np.save(file_name_base + "_ene.npy", energy_trajectory)
np.savetxt(file_name_base + "_ene.dat", energy_trajectory, fmt="%.6e", header="#E_pot  E_kin  T  P", comments='')


#----------------------------------------------------
# P L O T   E N E R G Y   T R A J E C T O R I E S
#----------------------------------------------------
# set time axis
time_ps = np.arange(sim.n_steps + 1) * sim.dt

#
# potential energy
# 
E_pot_min = np.mean(energy_trajectory[:,0]) - 1   # lower limit of E_pot axis
E_pot_max = np.mean(energy_trajectory[:,0]) + 1   # upper limit of E_pot axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,0]) 
plt.ylim(E_pot_min, E_pot_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_pot [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Epot.png", dpi=300, bbox_inches='tight')
plt.show()

#
# kinetic energy
# 
E_kin_min = np.mean(energy_trajectory[:,1]) - 100   # lower limit of E_kin axis
E_kin_max = np.mean(energy_trajectory[:,1]) + 100   # upper limit of E_kin axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,1]) 
plt.ylim(E_kin_min, E_kin_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_kin [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Ekin.png", dpi=300, bbox_inches='tight')
plt.show()




#
# Total Energy
#
E_tot = energy_trajectory[:,0] + energy_trajectory[:,1]         # sum of pot and kin energy
E_tot_min = np.mean(E_tot) - 5                                # lower limit of E_tot axis
E_tot_max = np.mean(E_tot) + 5                               # upper limit of E_tot axis

plt.figure(figsize=(8, 6))
plt.plot(time_ps, E_tot) 
plt.ylim(E_tot_min, E_tot_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_tot [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Etot.png", dpi=300, bbox_inches='tight')
plt.show()


#
# temperature
# 
T_min = np.mean(energy_trajectory[:,2]) - 100   # lower limit of T axis
T_max = np.mean(energy_trajectory[:,2]) + 100   # upper limit of T axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,2]) 
plt.ylim(T_min, T_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("T [K]", fontsize=14)

plt.savefig(file_name_base + "_T.png", dpi=300, bbox_inches='tight')
plt.show()

#
# pressure
# 
P_min = np.mean(energy_trajectory[:,3]) - 200   # lower limit of P axis
P_max = np.mean(energy_trajectory[:,3]) + 200   # upper limit of P axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,3]) 
plt.ylim(P_min, P_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("P [Pa]", fontsize=14)

plt.savefig(file_name_base + "_P.png", dpi=300, bbox_inches='tight')
plt.show()



"""

# modification
#
# plot 3D trajectory of multiple particles
#

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for p in [0, 1, 2, 3, 4, 150]:

    x = position_trajectory[:,p,0]
    y = position_trajectory[:,p,1]
    z = position_trajectory[:,p,2]

    for i in range(len(x)-1):

        # checking whether PBC jump took place
        if (abs(x[i+1]-x[i]) > sim.box_length/2 or
            abs(y[i+1]-y[i]) > sim.box_length/2 or
            abs(z[i+1]-z[i]) > sim.box_length/2):

            linestyle = "--"
            color="gray"

    
        else:
            linestyle = "-"
            color="C0"

        ax.plot(x[i:i+2],
                y[i:i+2],
                z[i:i+2],
                linestyle = linestyle,
                color = color)

ax.set_xlim(0, sim.box_length)
ax.set_ylim(0, sim.box_length)
ax.set_zlim(0, sim.box_length)

# draw simulation box 
L = sim.box_length

# untere Fläche
ax.plot([0,L],[0,0],[0,0],'k')
ax.plot([L,L],[0,L],[0,0],'k')
ax.plot([L,0],[L,L],[0,0],'k')
ax.plot([0,0],[L,0],[0,0],'k')

# obere Fläche
ax.plot([0,L],[0,0],[L,L],'k')
ax.plot([L,L],[0,L],[L,L],'k')
ax.plot([L,0],[L,L],[L,L],'k')
ax.plot([0,0],[L,0],[L,L],'k')

# vertikale Kanten
ax.plot([0,0],[0,0],[0,L],'k')
ax.plot([L,L],[0,0],[0,L],'k')
ax.plot([L,L],[L,L],[0,L],'k')
ax.plot([0,0],[L,L],[0,L],'k')

ax.set_xlabel(r"$x$ (nm)", fontsize=14)
ax.set_ylabel(r"$y$ (nm)", fontsize=14)
ax.set_zlabel(r"$z$ (nm)", fontsize=14)

plt.savefig(file_name_base + "_pos_traj_xyz.png", dpi=300, bbox_inches='tight')
plt.show()
"""

# modification
#
# plotting velocities along x axis
#

plt.figure(figsize=(8, 6))
plt.plot(time_ps[:], velocity_trajectory[:, 0, 0]) 
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel(r"$v_x$ ($10^{3}$ m/s)", fontsize=14)

plt.savefig(file_name_base + "_v_x.png", dpi=300, bbox_inches='tight')
plt.show()



# modification
#
# plotting vector magtitude of the 3D-velocities vector for the first particle
#



plt.figure(figsize=(8, 6))
#for p in [0, 1, 2, 3, 4, 150]:
for p in range(100):
    velocity_traj_magnitude = np.sqrt(
        velocity_trajectory[:, p, 0]**2
        + velocity_trajectory[:, p, 1]**2
        + velocity_trajectory[:, p, 2]**2
    )

    plt.plot(time_ps, velocity_traj_magnitude, label=f"Particle {p}")

plt.xlabel("time [ps]", fontsize=14)
plt.ylabel(r"$|\vec{v}|$ ($10^{3}$ m/s)", fontsize=14)
plt.legend()

plt.savefig(file_name_base + "_v_magnitude.png", dpi=300, bbox_inches='tight')
plt.show()



#--------------------------------------
# O U T P U T 
#--------------------------------------
elapsed_time = toc()   # stop the timer
output_lines = []

output_lines.append("")
output_lines.append("----------------------------------------------------------")
output_lines.append("Simulation parameters ")    
output_lines.append("----------------------------------------------------------")
output_lines.append(f"{'Number of particles:':<30}{ps.n:>10.0f} ")
output_lines.append(f"{'Box length:':<30}{sim.box_length:>10.3e} nm")
output_lines.append(f"{'Box volume:':<30}{sim.box_length**3:>10.3e} nm^3")
output_lines.append(f"{'Density:':<30}{rho:>10.3e} g/cm^3")
output_lines.append("")   
output_lines.append(f"{'Time step:':<30}{sim.dt:>10.3f} ps")
output_lines.append(f"{'Number of time steps:':<30}{sim.n_steps:>10.0f}")
output_lines.append(f"{'Simulation time:':<30}{sim.n_steps * sim.dt :>10.3e} ps")
output_lines.append("")   
if NVT==True: 
    output_lines.append(f"{'Ensemble:':<30}{'NVT':>10}")
    output_lines.append(f"{'Thermostat temperature:':<30}{sim.temperature:>10.0f} K")
    output_lines.append(f"{'Thermostat coupling:':<30}{sim.tau_thermostat:>10.3e} ps")
else: 
    output_lines.append(f"{'Ensemble:':<30}{'NVE':>10}")
    output_lines.append(f"{'Initial velocities:':<30}{sim.temperature:>10.0f} K")

output_lines.append("")     
output_lines.append(f"{'Lower cutoff radius:':<30}{sim.rij_min:>10.3f} nm")
output_lines.append("----------------------------------------------------------")
if elapsed_time: 
    time_per_time_step = elapsed_time/sim.n_steps
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append(f"{'Elapsed time:':<30}{elapsed_time:>10.3f} s")   
    output_lines.append(f"{'Elapsed time per time step:':<30}{time_per_time_step:>10.3f} s")
    output_lines.append(f"{'Time stamp:':<30}{now} s")
output_lines.append("----------------------------------------------------------")
output_lines.append("END")  
output_lines.append("----------------------------------------------------------")

# Print to screen
for line in output_lines:
    print(line)
  
# Write to file
with open(file_name_base + ".out", "w") as f:
    for line in output_lines:
        f.write(line + "\n")    