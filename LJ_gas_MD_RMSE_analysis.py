"""
LJ_gas_MD_RMSE_analysis.py

This is a supporting program to the LJ_gas.py and LJ_gas_run_MD.py programs 
to analyze the simulated trajectories and calculate the Root Mean Square Error (RMSE) 
to analyze trajectory stability of the ensembles and integrators (plotting). 

For the RMSE calculation the program requires
and input of binary .npy trajectory files (reference and he one for analysis).

IMPORTANT: To get valid comparisons the seed (np.random.seed(42)) has to be 
reset for each run. Before running this script run the LJ_gas_run_MD.py script
and toggle on and off the desired integrators/ensembles. This means that the
same number of particles have to be initialized at the same positions in both
of the runs to be compared.

MANDATORY INPUT FILES (in .npy format):
- trajectory for analysis
- reference trajectory (e.g. vVerlet with small dt)
- parameters file of the trajectory for analysis
- parameters file of the reference trajectroy

Author: Luka Jurecic
Created: July 11th, 2026

Modified by:
Date: 
"""
#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt


#----------------------------------------------------------------
"""
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
"""
#defining plotting parameters
plt.rcParams.update({
    'font.size': 22,              # Standardgröße
    'axes.labelsize': 24,         # Achsenbeschriftung
    'xtick.labelsize': 22,        # X-Achse Werte
    'ytick.labelsize': 22,        # Y-Achse Werte
    'legend.fontsize': 22,        # Legendentext
    'figure.titlesize': 12        # Falls Titel
})
#----------------------------------------------------------------

#----------------------------------------------------------------
#   FILE LOADINGS AND NAMES 
#----------------------------------------------------------------
# Change the following string variables to load the desired .npy files
file_name_analysis = "sim_NVE_leapfrog_50nm_200particles_dt0p1_pos.npy" # Trajectory file name of the simulation to compare
file_name_ref = "sim_NVE_vVerlet_50nm_200particles_dt0p1_pos.npy" # Trajectory file name for the reference simulation (e.g. vVerlet)

file_name_params_analysis = "sim_NVE_leapfrog_50nm_200particles_dt0p1_params.npy" # File name of the simulation parameters
file_name_params_ref = "sim_NVE_vVerlet_50nm_200particles_dt0p1_params.npy" # File name of the reference simulation parameters

# Change the variable names for the title of RMSE graph:
title_analysis = "NVE_Leapfrog, dt = 0.1 ps_200part_50nm"
title_reference = "NVE_vVerlet, dt = 0.1 ps_200part_50nm"

# Instead of using the txt files we can use the .npy files
# since they are binary and can be loaded quickly, without a parser
# Position .npy files have a shape of (n_steps+1, n_particles, 3) and contain the xyz positions of all particles at each timestep
traj_file = np.load(file_name_analysis) # load trajectory file to be compared to reference
traj_file_ref = np.load(file_name_ref) # load reference trajectory file

# Loading the parameter file for calculations and plotting
# The parameters should be the same for both runs!
params_analysis = np.load(file_name_params_analysis, allow_pickle=True).item() # load parameters from analysis run
params_ref = np.load(file_name_params_ref, allow_pickle=True).item() # load parameters from reference run

continue_ = str(input("Did you change all of the parameters and title correctly? (y/n) "))
if continue_ != "y":
    print("Check again the parameters!")
    raise SystemExit
#----------------------------------------------------------------
#   CALCULATING THE RMSE OF POSITIONS
#----------------------------------------------------------------
n_particles_ = params_analysis["n_particles"] # number of particles in the simulation

# Cross-checking if the number or particles are matching
# The number of particles from the file has to match the number of particles in the parameters dictionary
if n_particles_ != traj_file.shape[1] or n_particles_ != traj_file_ref.shape[1]:
    raise ValueError("Number of particles in the parameter file does not match the number of particles in the trajectory files.")

#TODO:
# Subsampling the trajectory with more time steps to match the number of frames (times steps)
# in the reference trajectory.

# This part raises a warning if the reference trajectory has lower number of time steps than 
# the trajectory to be compared to the reference.
if traj_file.shape[0] > traj_file_ref.shape[0]: # Axis 0 is the number of time steps
    raise ValueError("The reference trajectory file has a lower number of time steps than the trajectory file to be compared to the reference.")
else:
    dt_ref = params_ref["dt"]
    dt_analysis = params_analysis["dt"]
    stride = round(dt_analysis / dt_ref) # calculates how many frames from reference trajectory per frame in analysis trajectory

    if stride > 1:
        traj_file_ref = traj_file_ref[::stride] # subsampling the reference file if it has more time steps than reference file
    elif stride < 1:
        traj_file = traj_file[::round(dt_ref / dt_analysis)] # subsampling the analysis file if it has more time steps than reference file
        raise ValueError("The trajectory file has a lower number of time steps than the trajectory file to be compared to the reference.")


# Calculating the difference in xyz position
L = params_analysis["box_length"] 
difference = traj_file - traj_file_ref # shape of (n_steps+1, n_particles, 3)
difference -= L * np.rint(difference / L) # Minimum image convention, rounds to nearest integer
diff_sum_xyz = np.sum(difference**2, axis = 2) # sum over xyz, shape (n_steps+1, n_particles)
diff_sum_particles = np.sum(diff_sum_xyz, axis = 1) # sum over particles, shape (n_steps+1,)
rmse = np.sqrt(diff_sum_particles/n_particles_)
np.save(f"{title_analysis}_{title_reference}_RMSE.npy", rmse) 
"""
print(difference.shape)
print(diff_sum_xyz.shape)
print(diff_sum_particles.shape)
print(rmse.shape)
"""
#----------------------------------------------------------------
#   PLOTTING THE RMSE
#----------------------------------------------------------------

dt = params_analysis["dt"]
n_steps = params_analysis["n_steps"]

time_ps = np.arange(n_steps + 1) * dt
plt.figure(figsize=(8, 6))
plt.plot(time_ps, rmse) 
plt.xlabel(r"$t\ \mathrm{[ps]}$")
plt.ylabel(r"$RMSE\ \mathrm{[nm]}$")
#plt.title("RMSE of " + title_analysis + ", reference simulation " + title_reference)
plt.tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True)
plt.savefig(f"{title_analysis}_{title_reference}_RMSE.png", dpi=300, bbox_inches='tight')
plt.show()