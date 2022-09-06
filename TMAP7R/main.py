# Pol de Dalmau Huguet
# Created 17/05/2022
# Using python version 3.9.10
# Before running this script, make sure to use the system terminal (not another shell like powershell) 
# and run t7 from it (maybe also run a tmap7 inp file from the terminal).
# Only then will this program work properly.
import sys, os
sys.path.append(os.path.join(sys.path[0],'Pol_de_Dalmau_MSc', 'Pol_code')) 

import numpy as np
import matplotlib.pyplot as plt
from time import time
import Analyser_and_Writer_Pol as an

current_dir = os.getcwd()

# filename of initial parameter guess simply for comparison after descent
filename_initial = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\spdg_tester_initial"
# filename of .inp file that will be run over and over.
filename1 = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\spdg_tester"
# filename of theta and S history (theta is important to record as it is useful to pick up a fit where it was "left off" previously.)
filename_theta = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\theta_hist.txt"
filename_S = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\S_hist.txt"
filename_S_int = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\S_intervals_hist.txt"

# Generated or experimental data
generated_data_filename = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\Generated" 

Fe_nbrden = 8.42794668519528e28 # [atoms/m^3] lattice number density of (pure) iron (7.874*1e3/(55.845*1.67e-27)
sample_thic = 0.001 # [m] sample thickness
Area = 0.015**2 # [m^2]
heating_rate = 2 # [K/s]

# Parameters are fit withing a range only. This allows to match each peak properly.
# Temperatures are given in Kelvin
room_temp = 300       # K
final_temp = 1200     # K
total_temp_interval = [room_temp, final_temp]
trap1_start_T = room_temp
trap1_end_T = 610
trap2_start_T = 610
trap2_end_T = 757
trap3_start_T = 757
trap3_end_T = 1200
retrp_start_T = room_temp
retrp_end_T = 450		  

trap_intervals = [[trap1_start_T, trap1_end_T], 
				  [trap2_start_T, trap2_end_T], 
				  [trap3_start_T, trap3_end_T], 
				  [retrp_start_T, retrp_end_T]]

# Instantiations
tmap_w = an.tmap_writer(room_temp, final_temp, Fe_nbrden, sample_thic, Area, heating_rate)
tds = an.tds_exp()
out = an.out_file_reader()
mini = an.minimizer()

current_dir = os.getcwd()

filename_my_tds = current_dir + r"\Pol_de_Dalmau_MSc\Code\TDS_data\TDS\raw_data"
filename_viviam_tds = current_dir + r"\Pol_de_Dalmau_MSc\Code\TDS_data\desorption_data_viviam.csv"

def saveplot(T_data, flux_data, T_tmap, flux_tmap,T_tmap_init, flux_tmap_init, num_iterations):
	""" Saves a plot for each iteration"""
	title = "Generated Data SPGD test"
	file_savefig1 = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\All_iterations\Generated_data_fit" + str(num_iterations) + ".pdf"
	file_savefig2 = current_dir + r"\Pol_de_Dalmau_MSc\Poltest\SPGD_tester\All_iterations\Generated_data_fit" + str(num_iterations) + ".png"
	out.plot_init_n_final(T_data, flux_data, T_tmap, flux_tmap,T_tmap_init, flux_tmap_init, title, num_iterations, file_to_save_plot = file_savefig1)
	out.plot_init_n_final(T_data, flux_data, T_tmap, flux_tmap,T_tmap_init, flux_tmap_init, title, num_iterations, file_to_save_plot = file_savefig2)


N1 = 4e+17                
N2 =  1e+17               
N3 =  4e+16              
e_1 =  1.2 # [eV]
e_2 =  1.5 # [eV]
e_3 =  1.8 # [eV]
e_ret =  0.28 # [eV]
imp_depth =  7e-07 # [m]

# Get generated and initially guessed data
T_data, dNdt_data = tmap_w.get_flux(generated_data_filename)
T_init, dNdt_init = tmap_w.get_flux(filename_initial)

# add noise to generated data:
T_data = T_data[::2]
dNdt_data = dNdt_data[::2]
noise = []
for i in range(len(dNdt_data)):
	noise.append(np.random.normal(0, 300000 * np.sqrt(dNdt_data[i])))
dNdt_data = dNdt_data + np.array(noise)

true_theta = np.array([N1, N2, N3, e_1, e_2, e_3, e_ret, imp_depth])


# Loop in which iterations are done and gradient descent is implemented:
first_iteration = True # Should only be true if the following files are empty:
    # S_hist.txt 
    # S_intervals_hist
    # theta_hist.txt (only first row with initial values of theta, i.e. the first guess)
# It records the first S. If the files already have entries, you will repeat values.
theta_initial = out.file_to_np(filename_theta)[0]
theta = out.file_to_np(filename_theta)[-1]
num_iterations = len(theta)

print("Most recent theta: \n", theta)
SS_total_spgd = []

# SPGD:
# A pretty good routine: (e_1, N_1, e_2, N_2, e_3, N_3, e_ret)
list_parameter_indices = [[1, 5], [1, 2], [1, 6], [1, 3], [1, 7], [1, 4], [1, 8]]

num_iterations_per_parameter = 2
num_loops = 20
j_max = num_iterations_per_parameter * num_loops * len(list_parameter_indices)

list_parameter_indices_final = mini.get_param_list(num_iterations_per_parameter, list_parameter_indices, num_loops) # SPDG
dtheta_frac = 0.0007 # Derivative is calculated with a dtheta = theta * 0.07% WARNING: IF YOU CHANGE THIS VALUE, YOU WILL HAVE TO FIND THE OPTIMAL ALPHAS AGAIN!!!!
perturb = mini.get_perturbation_matrix(theta_initial, dtheta_frac)
alpha = np.array([5, 15, 15, 0.1, 0.1, 0.1, 0.4, 0])



if first_iteration: j = 0
if not first_iteration: j = num_iterations

total_runs_spgd = 0
tic = time()
for param_indices in list_parameter_indices_final:
    print("current param indices:", param_indices)
    SS_res = np.zeros((len(theta_initial)+1))
    trap_interval = trap_intervals[mini.get_interval_index_tmap(param_indices[-1])]
    for i in param_indices: 
        tmap_w.inp_write(filename1, *(theta + perturb[i-1,:]))
        mini.run(filename1)
        T_tmap, dNdt_tmap = tmap_w.get_flux(filename1)
        if first_iteration and j == 0 and i == 1:
            # The first S is recorded here.
            out.record_values(filename_S, tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *total_temp_interval))
            S_int = np.array([tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[0]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[1]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[2]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[3])])
            out.record_values(filename_S_int, S_int)
        SS_res[i-1] = tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_interval)
        total_runs_spgd += 1
    SS_total_spgd.append(tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *total_temp_interval))
    theta = theta - alpha * mini.dSS(SS_res[0], SS_res[1:]) * theta
    out.record_values(filename_theta, theta)
    out.record_values(filename_S, SS_total_spgd[-1])
    S_int = np.array([tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[0]),
                tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[1]),
                tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[2]),
                tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[3])])
    out.record_values(filename_S_int, S_int)
    if first_iteration:		saveplot(T_data, dNdt_data, T_tmap, dNdt_tmap, T_init, dNdt_init, j+1)
    if not first_iteration: saveplot(T_data, dNdt_data, T_tmap, dNdt_tmap, T_init, dNdt_init, j)
    print("iteration", j - num_iterations, "(",j, ")", "/", j_max, "SS", SS_total_spgd[-1])
    j += 1
tmap_w.inp_write(filename1, *(theta))
mini.run(filename1)
T_tmap, dNdt_tmap = tmap_w.get_flux(filename1)
SS_total_spgd.append(tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *total_temp_interval))
out.record_values(filename_S, SS_total_spgd[-1])
S_int = np.array([tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[0]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[1]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[2]),
                        tmap_w.SS(T_tmap, dNdt_tmap, T_data, dNdt_data, *trap_intervals[3])])
out.record_values(filename_S_int, S_int)
total_runs_spgd += 1
print("time elapsed (hours)", (time()-tic)/ 3600 )
print("differences in theta befor [%]:", (np.array(theta_initial)/np.array(true_theta) - 1) * 100)
print("differences in theta after [%]:", (np.array(theta)/np.array(true_theta) - 1) * 100)
print("total_runs_spgd", total_runs_spgd)