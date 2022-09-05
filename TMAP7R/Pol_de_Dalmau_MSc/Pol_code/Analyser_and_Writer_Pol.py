# Pol de Dalmau Huguet (PDDH) 2021-2022
# A script to read and write .inp and .plt files for TMAP7 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from scipy.interpolate import interp1d
import os
from scipy.integrate import odeint

class tmap_writer(object):
    """
    TMAP7 writer and reader class.
    """

    # User specified parameters (All parameters are automatically taken care of in the script EXCEPT FOR NBRDEN. Please make sure to include that manually!!):
    room_temp = int() # [K]. Must match the starting temperature of the TMAP7 simulation.
    final_temp = int() # [K]
    Fe_nbrden = float() # [atoms/m^3] 
    sample_thic = float() # [m] sample thickness
    Area = float() # [m^2]

    def __init__(self, room_temp, final_temp,
                 Fe_nbrden, sample_thic, Area, heating_rate):
        self.room_temp = room_temp
        self.final_temp = final_temp
        self.Fe_nbrden = Fe_nbrden
        self.sample_thic = sample_thic
        self.Area = Area
        self.heating_rate = heating_rate

    def del_x_generator(self, imp_depth):
        """
        imp_depth: implantation depth [m]
        sample_thic: sample thickness [m]
        returns string that can be written in .inp file
        Given the desired implantation depth, segments are generated as follows:
            - 50 "small" segments each with size imp_depth/10
            - 20 "medium" segments with size imp_depth * 4/10
            - 9 with size imp_depth
            - 9 "large" segments with size imp_depth * 10
            - 10 equally large segments to fill up the remaining space to reach the desired length of sample_thic
        """
        index_impdepth = 10
        small = imp_depth/index_impdepth * 1e6
        small = round(small, 8)
        medium = (imp_depth * 4)/10 * 1e6
        medium = round(medium, 8)
        large = 10*imp_depth * 1e6

        s = "0.0," + "50*" + str(small) + "e-6" + ",20*" + str(medium) + "e-6" + ",9*" + str(imp_depth) + ",9*" + str(large) + "e-6" 

        s_dummy = (s.split(",end")[0]).strip(" ").split(",")
        segs = out_file_reader.split_segs(out_file_reader, s_dummy)[0]

        size = segs[-1]
        rest = (self.sample_thic - size)
        s = s + ",10*" + str(rest/10 * 1e6) + "e-6"
        num_segs = len(segs) +1
        return "delx=" + s + ",0.0,end\n", num_segs, index_impdepth

    def trap_i_set(self, N1, N2, N3, index_imp_depth, num_segs, imp_depth):
        """
        returns string that can be written in .inp file. The string describes the trap
        population as a function of depth given the number of trapped D atoms initially
        in traps 1 (N1), 2 (N2) and 3 (N3).
        """
        frac_1 = round(N1/(self.Area*self.Fe_nbrden*imp_depth), 8) # N = nbrden * volume; with volume = Area * imp_depth
        frac_2 = round(N2/(self.Area*self.Fe_nbrden*imp_depth), 8)
        frac_3 = round(N3/(self.Area*self.Fe_nbrden*imp_depth), 8)

        one_or_zero1 = "*1.0,"
        one_or_zero2 = "*1.0,"
        one_or_zero3 = "*1.0,"
        if N1 == 0: # if trap is not populated, make it disappear altogether (also faster and avoids retrapping so it really is quantitatively different!
            one_or_zero1 = "*0.0,"
        if N2 == 0: one_or_zero2 = "*0.0,"
        if N3 == 0: one_or_zero3 = "*0.0,"

        string = "trapping=ttyp,1,tconc,const," + str(frac_1) + "\n"\
            "              tspc,d,alphr,equ,4,alpht,equ,3\n"\
            "              ctrap,0.0," + str(index_imp_depth) + one_or_zero1 + str(num_segs-index_imp_depth-2) + "*0.0,0.0\n"\
            "         ttyp,2,tconc,const," + str(frac_2)+ "\n"\
            "              tspc,d,alphr,equ,5,alpht,equ,3\n"\
            "              ctrap,0.0," + str(index_imp_depth) + one_or_zero2 + str(num_segs-index_imp_depth-2) + "*0.0,0.0\n"\
            "         ttyp,3,tconc,const," + str(frac_3) + "\n"\
            "              tspc,d,alphr,equ,6,alpht,equ,3\n"\
            "              ctrap,0.0," + str(index_imp_depth) + one_or_zero3 + str(num_segs-index_imp_depth-2) + "*0.0,0.0,end\n"
        return string

    def trap_i_energie_set(self, e_1, e_2, e_3):
        """
        Returns string of equations that describe trap energies.
        In TMAP: alphr
        """
        string = "$ (4) PDDH: Alphr for trap 1 in 0.3% Y2O3 12 Cr ODS (1/s)\n"\
            "     y=8.4e12*exp(-" + str(round(e_1,8)) + "/8.625e-5/temp),end\n"\
            "$ (5) Alphr for trap 2 in 0.3% Y2O3 12 Cr ODS (1/s)\n"\
            "     y=8.4e12*exp(-" + str(round(e_2,8)) + "/8.625e-5/temp),end\n"\
            "$ (6) Alphr for trap 3 in 0.3% Y2O3 12 Cr ODS (1/s)\n"\
            "     y=8.4e12*exp(-" + str(round(e_3,8)) + "/8.625e-5/temp),end\n"
        return string

    def retrap_nrgy_set(self, e_ret):
        """
        String for tmap describing retrapping.
        In TMAP: alpht
        """
        string = "$ (3) Alpht for d in 0.3% Y2O3 12 Cr ODS (1/s) PDDH: retrapping\n"\
            "	  y=2.9e12*exp(-" + str(round(e_ret,8)) + "/8.625e-5/temp),end\n"
        return string

    def retrap_nrgy_i_set(self, e_ret, int_trap):
        """
        String for tmap describing retrapping.
        In TMAP: alpht
        """
        equ_num = ["3", "9" , "10"]
        string = "$ (" + equ_num[int_trap - 1] + ") Alpht for d in 0.3% Y2O3 12 Cr ODS trap " + str(int_trap) + " (1/s) PDDH: retrapping\n"\
            "	  y=2.9e12*exp(-" + str(round(e_ret,8)) + "/8.625e-5/temp),end\n"
        return string

    def inp_write(self, filename, N1, N2, N3, e_1, e_2, e_3, e_ret, imp_depth):
        """
        Opens given file in argument (without extension!),
        and writes all relevant information.
        """
        ## open file to read:
        f =  open(filename + ".inp", "r")
        read = f.readlines()
        read = np.array(read)
        # 1. generate strings for: delx, trap concentrations, trap energies, retrap energies, 
        # Area, atomic number density, (temperature table -> maybe in the future)...
        delx_string, num_segs, index_imp_depth = self.del_x_generator(imp_depth)
        trap_string = self.trap_i_set(N1, N2, N3, index_imp_depth, num_segs, imp_depth)
        nrgy_string = self.trap_i_energie_set(e_1, e_2, e_3)
        retr_string = self.retrap_nrgy_set(e_ret)
        surf_string = 'surfa=' + str(self.Area) + ',end $ PDDH: Surface Area: ' + str(np.sqrt(self.Area) * 1000) + ' x ' + str(np.sqrt(self.Area) * 1000) + ' mm^2'
        nbrden_string = "nbrden= " + str(self.Fe_nbrden) + ",end                   $ PDDH: lattice number density of (pure) iron in atoms/m3 (7.874*1e3/(55.845*1.67e-27))"

        # The below code could appear in a for-loop to be more compact, however, I find it clearer this way.
        # 1.1 delx actual string
        keyword_1 = "delx="
        line_nr_1 = np.where((np.char.find(read, keyword_1)) == 0)[0][0]
        # 1.2 from delx part, infer segnds and write.
        keyword_2 = "segnds"
        line_nr_2 = np.where((np.char.find(read, keyword_2)) == 0)[0][0]
        # 2 traps:
        # 2.1 trap concentration profiles
        keyword_3 = "trapping"
        line_nr_3 = np.where((np.char.find(read, keyword_3)) == 0)[0][0]
        # 2.2 trap energies
        keyword_4 = "$ (4) PDDH: Alphr for trap 1 in 0.3% Y2O3 12 Cr ODS (1/s)"
        line_nr_4 = np.where((np.char.find(read, keyword_4)) == 0)[0][0]
        # 2.3 retrapping energy
        keyword_5 = "$ (3) Alpht for d in 0.3% Y2O3 12 Cr ODS"
        line_nr_5 = np.where((np.char.find(read, keyword_5)) == 0)[0][0]
        # 2.4 Surface Area
        keyword_6 = "surfa="
        line_nr_6 = np.where((np.char.find(read, keyword_6)) == 0)[0][0]
        # 2.5 number density
        keyword_7 = "nbrden="
        line_nr_7 = np.where((np.char.find(read, keyword_7)) == 0)[0][0]
        # Change the file (still just in memory at this point):
        read[line_nr_1] = delx_string
        read[line_nr_2] = keyword_2 + "=" + str(num_segs) + ",end\n"
        for i in range(9): # 9 is the number of lines in the trap statement. See trap_i_set function and the string it prints.
            read[line_nr_3 + i] = trap_string.split("\n")[i] + "\n"
        for i in range(6): # same reasoning as above loop.
            read[line_nr_4 + i] = nrgy_string.split("\n")[i] + "\n"
        for i in range(2): # same reasoning as above loop.
            read[line_nr_5 + i] = retr_string.split("\n")[i] + "\n"
        read[line_nr_6] = surf_string + "\n"
        read[line_nr_7] = nbrden_string + "\n"
        # Finally, write it to the specified file.
        with open(filename + ".inp", 'w') as f:
            f.writelines(read)

    def inp_write_spec_ret(self, filename, N1, N2, N3, e_1, e_2, e_3, e_ret1, e_ret2, e_ret3, imp_depth):
        """
        Opens given file in argument (without extension!),
        and writes all relevant information.
        variation of inp_write in that it has specific retrapping energies for
        each type of trap.
        """
        ## open file to read:
        f =  open(filename + ".inp", "r")
        read = f.readlines()
        read = np.array(read)
        # 1. generate strings for: delx, trap concentrations, trap energies, retrap energies, Area, atomic number density, (temperature table)...
        delx_string, num_segs, index_imp_depth = self.del_x_generator(imp_depth)
        trap_string = self.trap_i_set_spec_ret(N1, N2, N3, index_imp_depth, num_segs, imp_depth)
        nrgy_string = self.trap_i_energie_set(e_1, e_2, e_3)
        retr1_string = self.retrap_nrgy_i_set(e_ret1, 1)
        retr2_string = self.retrap_nrgy_i_set(e_ret2, 2)
        retr3_string = self.retrap_nrgy_i_set(e_ret3, 3)
        surf_string = 'surfa=' + str(self.Area) + ',end $ PDDH: Surface Area: ' + str(np.sqrt(self.Area) * 1000) + ' x ' + str(np.sqrt(self.Area) * 1000) + ' mm^2'
        nbrden_string = "nbrden= " + str(self.Fe_nbrden) + ",end                   $ PDDH: lattice number density of (pure) iron in atoms/m3 (7.874*1e3/(55.845*1.67e-27))"

        # The below code could appear in a for-loop to be more compact, however, it's probably clearer this way.
        # 1.1 delx actual string
        keyword_1 = "delx="
        line_nr_1 = np.where((np.char.find(read, keyword_1)) == 0)[0][0]
        # 1.2 from delx part, infer segnds and write.
        keyword_2 = "segnds"
        line_nr_2 = np.where((np.char.find(read, keyword_2)) == 0)[0][0]
        # 2 traps:
        # 2.1 trap concentration profiles
        keyword_3 = "trapping"
        line_nr_3 = np.where((np.char.find(read, keyword_3)) == 0)[0][0]
        # 2.2 trap energies
        keyword_4 = "$ (4) PDDH: Alphr for trap 1 in 0.3% Y2O3 12 Cr ODS (1/s)"
        line_nr_4 = np.where((np.char.find(read, keyword_4)) == 0)[0][0]
        # 2.3 retrapping energies
        keyword_51 = "$ (3) Alpht for d in 0.3% Y2O3 12 Cr ODS"
        line_nr_51 = np.where((np.char.find(read, keyword_51)) == 0)[0][0]
        keyword_52 = "$ (9) Alpht for d in 0.3% Y2O3 12 Cr ODS"
        line_nr_52 = np.where((np.char.find(read, keyword_52)) == 0)[0][0]
        keyword_53 = "$ (10) Alpht for d in 0.3% Y2O3 12 Cr ODS"
        line_nr_53 = np.where((np.char.find(read, keyword_53)) == 0)[0][0]
        # 2.4 Surface Area
        keyword_6 = "surfa="
        line_nr_6 = np.where((np.char.find(read, keyword_6)) == 0)[0][0]
        # 2.5 number density
        keyword_7 = "nbrden="
        line_nr_7 = np.where((np.char.find(read, keyword_7)) == 0)[0][0]
        # Change the file (still just in memory at this point):
        read[line_nr_1] = delx_string
        read[line_nr_2] = keyword_2 + "=" + str(num_segs) + ",end\n"
        for i in range(9): # 9 is the number of lines in the trap statement. See trap_i_set function and the string it prints.
            read[line_nr_3 + i] = trap_string.split("\n")[i] + "\n"
        for i in range(6): # same reasoning as above loop.
            read[line_nr_4 + i] = nrgy_string.split("\n")[i] + "\n"
        for i in range(2): # same reasoning as above loop.
            read[line_nr_51 + i] = retr1_string.split("\n")[i] + "\n"
        for i in range(2): # same reasoning as above loop.
            read[line_nr_52 + i] = retr2_string.split("\n")[i] + "\n"
        for i in range(2): # same reasoning as above loop.
            read[line_nr_53 + i] = retr3_string.split("\n")[i] + "\n"
        read[line_nr_6] = surf_string + "\n"
        read[line_nr_7] = nbrden_string + "\n"
        # Finally, write it to the earlier specified file.
        with open(filename + ".inp", 'w') as f:
            f.writelines(read)

    def plt_reader(self, filename):
        """
        This function reads a given plt file (specific to this problem), 
        selects the text only from where the first empty line is (line containing only a linebreak).
        It also removes the first line of numbers because it sometimes contains special characters
        instead of numbers, i.e. "1.#QNBE" (weird bug again idk...).  returns array.  
        ----------------------------------------------------------------------------------------------------------------------------------
        WARNING:        
        If the error below appears, it means there is a problem in the plt file currently being read (line with too many columns, 
        some timesteps are skipped for some reason. Maybe tmap running out of memory idk...). The solution I recommend is to
        just re-run the simulation. Most times it just works
        
        Shown Error:
        Users\...\Pol_code\Analyser_and_Writer_Pol.py:357:
        VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of 
        lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you
        must specify 'dtype=object' when creating the ndarray. 
        read = np.array(read)    
        Traceback (most recent call last): 
        File "Users\polde\OneDrive\Desktop\TU_Delft\Second_Year\Thesis\Thesis_Preparation\TMAP7R\tmap_SPGD_tester.py", line 153, in <module>
        T_tmap, dNdt_tmap = tmap_w.get_flux(filename1)                                                                                     
        File "Users\polde\OneDrive\Desktop\TU_Delft\Second_Year\Thesis\Thesis_Preparation\TMAP7R\Pol_code\Analyser_and_Writer_Pol.py", line 371, in get_flux 
        read = self.plt_reader(filename)       File "Users\polde\OneDrive\Desktop\TU_Delft\Second_Year\Thesis\Thesis_Preparation\TMAP7R\Pol_code\Analyser_and_Writer_Pol.py", line 358, in plt_reader 
        read = read[1:, 1:]                  IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed 
        
        Possible solution:
        Maybe this can be treated with a try and catch method that simply reruns the most recent inp file. 
        I have been doing it manually so far but that's not the best...
        """
        f =  open(filename + ".plt", "r")
        read = f.readlines()
        read = np.array(read)
        start = np.where(read == "\n")[0][0] + 1
        read = read[start:]
        read = [text_line.replace("\n", "").replace("   ", "  ").split("  ") for text_line in read]
        read = np.array(read)
        read = read[1:, 1:]
        read = read.astype(float) # Take only from second line onwards because the first entry of the flux is sometimes "1.#QNBE" which cannot be turned into float... Strange tmap error I suppose...
        return read

    def get_flux(self, filename): 
        """
        filename: name of file (without extension)
        gets flux data from .plt file. The flux is taken
        to be the flux leaving the front surface and is 
        assumed to appear on the third column, hence -read[:, 2].
        Negative because flux going to the left is negative. Index 2 
        to get third column.
        """
        read = self.plt_reader(filename)
        dNdt_tmap = np.array(-read[:, 2]*self.Area)
        t = read[:, 0]
        heating_rate = (self.final_temp-self.room_temp)/t.max() # K/s
        T_tmap = t * heating_rate + self.room_temp
        return T_tmap, dNdt_tmap

    def SS_res(self, T_tmap, dNdt_tmap, T_data, dNdt_data):
        """
        interpolates tmap data to make it compatible with experimental one. Finally, calculates sum of squared residuals divided by .
        """
        # Make experimental and tmap data compatible for comparison 
        f = interp1d(T_tmap, dNdt_tmap, kind="quadratic")
        T_data = np.array(T_data)
        T_data_copy = np.copy(T_data)
        dNdt_data_copy = np.copy(dNdt_data)
        dNdt_data_copy = dNdt_data_copy[(T_data_copy >= self.room_temp) & (T_data_copy <= self.final_temp)] # Points outside range will be removed.
        T_data_copy = T_data_copy[(T_data_copy >= self.room_temp) & (T_data_copy <= self.final_temp)]
        e_i = f(T_data_copy) - dNdt_data_copy
        SS_res = np.dot(e_i,e_i) # sum of squares of residuals
        dNdt_mean = np.mean(dNdt_data_copy)
        SS_tot = np.dot((dNdt_data_copy - dNdt_mean), (dNdt_data_copy - dNdt_mean))
        #R_squared = 1 - SS_res/SS_tot
        return SS_res/SS_tot

    def SS(self, T_model, dNdt_model, T_data, dNdt_data,
          init_temp, fin_temp, plot_bool = False, bool_weights = False):
        """
        Should be moved to class: minimizer !!!!!
        interpolates model (tmap or kissinger) data (dNdt_model) to make
        it compatible with experimental one (dNdt_data). 
        Finally, calculates sum of squared residuals (S).
        """
        f = interp1d(T_model, dNdt_model, kind="cubic")
        T_data = np.array(T_data)
        T_data_copy = np.copy(T_data)
        dNdt_data_copy = np.copy(dNdt_data)
        dNdt_data_copy = dNdt_data_copy[(T_data_copy>=init_temp) & (T_data_copy<=fin_temp)] # selects data only in given range.
        T_data_copy = T_data_copy[(T_data_copy>=init_temp) & (T_data_copy<=fin_temp)]
        e_i = f(T_data_copy) - dNdt_data_copy
        if bool_weights:
            f_mod = np.copy(f(T_data_copy))
            f_mod[f(T_data_copy) < 1] = 1 # if f is zero, dividing will give an error.
            #SS_res = np.dot(e_i,e_i*(np.sqrt(f_mod))) # sum of squares of residuals with sqrt(exp_data) weights 
            SS_res = np.dot(e_i,e_i*(f_mod))    # If this if statement is set to true, we actually get the chi^2.
                                                # with exp_data as weights
        else: # standard case
            SS_res = np.dot(e_i,e_i) # sum of squares of residuals
            MSS = np.mean(SS_res) / (dNdt_data_copy.max())**2 # normalize by square of maximum from data to get reasonable magnitude.
        # option to plot is given here so you can select the range appropriately.
        if plot_bool: 
            plt.scatter(T_data_copy, dNdt_data_copy,
                       label = "Experimental", facecolors='none',edgecolors='r', s=10)
            plt.plot(T_data_copy, f(T_data_copy), label = "Model", linestyle='dashed')
            plt.legend()
            plt.show()
        return MSS


class out_file_reader(object):
    """ Extracts data from .out and .plt files generated by TMSAP7"""
 
    def get_indices(self, keyword, read):
        """
        extracts indices in read where keyword appears..
        """
        pt = np.char.rfind(read, keyword)
        ind =  np.where(pt > -1) 
        return ind[0]

    def file_to_np(self, filename):
        """
        Given a txt file with one line of values, 
        this function returns the values as an np.array(floats)
        """
        ## open file to read:
        f =  open(filename, "r")
        read = f.readlines()
        read = np.array(read)
        read = [text_line.replace("\n", "").split("  ") for text_line in read]
        read = np.array(read).astype(float)
        read = read.astype(np.float64)
        return read

    def split_segs(self, s):
        """
        Given the string from the .inp file, this function returns two lists:
        delx: thickness of each segment.
        segs: cumulative sum of delx, i.e. the x-coordinate in the sample thickness.
        For example: given s = "2*4,3*1", returns: [4,8,9,10,11], [4,4,1,1,1]
        """
        delta_x = []
        for i in s:
            if i.find("*") != -1:
                i = i.split("*")
                occur = int(i[0])
                val = float(i[1])
                [delta_x.append(val) for k in range(occur)]
            else:
                delta_x.append(float(i))
        segs = np.cumsum(delta_x)
        return segs, delta_x

    def get_segs(self, read):
        """
        Finds the line in the .inp file containing information
        about the parameter delx. Refines it for the function
        split_segs to do its thing.
        """
        row = self.get_indices("delx", read)
        s = read[row][0] + read[row + 1][0] # For now, assume that the desired data is 
                                            # contained in two lines, hence the + read[row + 1] statement.
                                            # If one line, only left term
        s = ((s.split("delx="))[1].split(",end")[0]).strip(" ").replace("\n", ",").replace(" ", "").split(",")
        return self.split_segs(s)
        
    def record_values(self, filename, array):
        """
        records a given array or just float! Simply needs to match the 
        dimensions of the already written data. Don't wory about overwriting. 
        It's taken care of. If there is no data, it writes from scratch.
        """
        file_is_empty =  os.path.isfile(filename) and os.path.getsize(filename) == 0
        if file_is_empty:
            np.savetxt(filename, [array], fmt='%.6e', delimiter='   ')
        else: 
            current_values = self.file_to_np(filename)
            merged_array = np.vstack((current_values, np.transpose(array)))
            np.savetxt(filename, merged_array, fmt='%.6e', delimiter='   ')

    def plot_init_n_final(self, T_data, flux_data, T_tmap, flux_tmap,T_tmap_init, flux_tmap_init, title, num_iterations, file_to_save_plot = ""):
        """
        plots initial (suffix _tmap_init) and final data (suffix _tmap) from tmap (ie, before and after SPGD)
        on top of experimental or generated data (suffix _data)
        Saves plot in file_to_save_plot
        """        
        #plt.plot(T1, flux1, label ="TMAP7 before gradient descent", color = "k", linestyle='dashed')
        plt.xlabel("Temperature (K)", fontsize = 16)
        plt.ylabel(r"Desorption rate ($\times 10^{15}$ D.s$^{-1}$)", fontsize = 16)
        plt.plot(T_tmap, flux_tmap/1e15, label ="TMAP7 after " + str(num_iterations) + " iterations", color = "k")
        plt.plot(T_tmap_init, flux_tmap_init/1e15, label ="TMAP7 before  gradient descent", color = "gray", linestyle='dashed')
        plt.scatter(T_data, flux_data/1e15, label="Generated data with noise", facecolors='none',edgecolors='r', s=10)
        plt.legend()
        plt.title(title, fontsize = 18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if file_to_save_plot != "":
            plt.savefig(file_to_save_plot)
            plt.clf()
        else:
            plt.show()


class minimizer(object):

    def run(self, tmap7_inpfile):
        """
        Allows running tmap files from a script. 
        Basic step to automatize tmap.
        Before this function can work properly, you need 
        to open the terminal yourself, change to your TMAP7 directory,
        and run the following: 
            >> tmap7
            >> t7 "path\to\some\TMAP7\input\file\without\the\inp\extension"
        Input filename (with full path to it) without .inp extension
        """
        run_tmap = "t7 " + tmap7_inpfile
        subprocess.run(run_tmap, shell=True)

    def get_perturbation_matrix(self, theta_initial, fraction_dtheta = 0.01):
        """
        Matrix where rows are only nonzero at the position where 
        theta_i will be varied to compute derivatives. First row is 
        zero since we also need the starting point for the derivative.
        Works for theta of any dimension as long as it is a vector.
        """
        delta_theta = fraction_dtheta * theta_initial
        basis = np.identity(len(theta_initial))
        perturb = (delta_theta * basis)
        perturb = np.vstack([np.zeros(len(perturb)), perturb]) # include zeros for case 
                                                               # where theta is unchanged 
                                                               # (needed to compute derivative!)
        return perturb

    def get_param_list(self, num_iterations_per_parameter, list_parameter_indices, num_loops):
        """
        Generates list of list for iterations in minimizer.
        """
        list_params = []
        for i in list_parameter_indices:
            list_params.extend([i] * num_iterations_per_parameter)
        list_params.extend(list_params * (num_loops-1))
        return(list_params)

    def get_interval_index(self, param_indices):
        """
        Trap intervals are given in a list of lists.
        We want to know which trap we need to consider:
        (trap 1: index 0, trap 2: index 1, trap 3: index2)
        given the parameter index i. Only valid for Kissinger with 3 traps for now.
        """
        i_to_index = {2:0, 5:0, 3:1, 6:1, 4:2, 7:2, 8:0, 9:1, 10:2}
        i = param_indices[-1]
        return i_to_index[i]

    def get_interval_index_tmap(self, param_index):
        """
        Trap intervals are given in a list of lists.
        We want to know which trap we need to consider:
        (trap 1: index 0, trap 2: index 1, trap 3: index2, other: index 3)
        given the parameter index param_index.
        """
        i_to_index = {2:0, 5:0, 3:1, 6:1, 4:2, 7:2, 8:3}# , 9:3} # 9 corresponds to imp_depth. Not used.
        return i_to_index[param_index]

    def get_interval_index_tmap_spec_ret(self, param_index):
        """
        Trap intervals are given in a list of lists.
        We want to know which trap we need to consider:
        (trap 1: index 0, trap 2: index 1, trap 3: index2, other: index 3)
        given the parameter index param_index.
        This is the form: 
        trap_intervals = [[trap1_start_T, trap1_end_T],
				  [trap2_start_T, trap2_end_T], 
				  [trap3_start_T, trap3_end_T], 
				  [retrp1_start_T, retrp1_end_T],
				  [retrp2_start_T, retrp2_end_T],
				  [retrp3_start_T, retrp3_end_T]]
        """
        i_to_index = {2:0, 5:0, 3:1, 6:1, 4:2, 7:2, 8:3, 9:4,10:5}# , 9:3} # 9 corresponds to imp_depth. Not used.
        return i_to_index[param_index]

    def get_alpha_index(self, param_indices):
        """
        alpha has a different optimal value if we are considering 
        the trap energy, trap population (even the order of magnitude of it), or other parameters.
        This function overcomes this issue.
        trap population: index 0
        trap energy: index 1
        imp_depth... -> not sure we'll use that anymore. Too insensitive. Does not really converge...
        """
        i_to_index = {2:0,3:0,4:0, 5:1, 6:1, 7:1}
        i = param_indices[-1]
        return i_to_index[i]

    def dSS(self, SS_res_thet, SS_res_thetpdthet):
        """Computes change in error.
        SS_res_thetpdthet: Sum of Squares, theta plus dtheta
        The derivative is taken to be zero if the given error is zero,
        this will result in the respective theta to remain unchanged
        """
        deriv = (SS_res_thetpdthet - SS_res_thet)
        deriv[SS_res_thetpdthet == 0] = 0
        return deriv


class tds_exp(object):

    def data_reader(self, sample_case, csv_file_path, return_all_bool = False):
        """
        Returns all experimental TDS data for all four tds spectra taken by Viviam (1,2,3,4)
        and eventually from PDDH's project's measurements (5, ...).
        """

        if sample_case <= 4:
            df = pd.read_csv (csv_file_path)
        # AS-1
        if sample_case == 1:
            title = "AS-1"
            T_data = df.AS1_T
            t_data = df.AS1_t
            dNdt_data = df.AS1_dNdt 
            if return_all_bool:
                E_d = [0.5,0.5,0.56,0.61,0.68,0.74,0.81,0.93] # detrapping energy in eV
                A = [2.5E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E4] # frequency factor in s^-1
                peak_T = ["450", "480", "535", "580", "627", "650", "690", "760", "870"] # K
                N_0 = [1E16 , 5.5E16 ,5E16 ,2.5E16 ,2E16 ,1.7E16 ,1.5E16 ,3.5E16] # N(t=0), initial number of trapped D atoms.

        # AS-25
        if sample_case == 2:
            title = "AS-25"
            T_data = df.AS25_T
            t_data = df.AS25_t
            dNdt_data = df.AS25_dNdt
            if return_all_bool:
                E_d = [0.61,0.68,0.74,0.81,0.93] # detrapping energy in eV
                A = [1E4, 1E4, 1E4, 1E4, 1E4] # frequency factor in s^-1
                N_0 = [10E16, 7E16, 3.5E16, 1.5E16, 2E16] # frequency factor in s^-1
                peak_T = ["580", "627", "650", "690", "760", "870"] # K
    
        # ANN-1
        if sample_case == 3:
            title = "1573 K-1"
            T_data = df.T1573K1_T
            t_data = df.T1573K1_t
            dNdt_data = df.dNdt1573K1_dNdt
            if return_all_bool:
                E_d = [0.5, 0.5, 0.56, 0.61, 0.68, 0.68, 0.74, 0.81, 0.93] # detrapping energy in eV
                A = [2.5E4, 1E4, 1E4, 1E4, 1E4, 0.6E4, 1E4, 1E4, 1E4] # frequency factor in s^-1
                peak_T = ["450", "480", "535", "580", "627", "650", "690", "760", "870"] # K
                N_0 = [0.5E16 , 4.5E16 ,13.3E16 ,5.5E16 ,9.2E16 ,10.0E16 ,6.0E16 ,3.5E16 ,2.7E16] # N(t=0), initial number of trapped D atoms.

        # ANN-25
        if sample_case == 4:
            title = "1573 K-25"
            T_data = df.T1573K25_T
            t_data = df.T1573K25_t
            dNdt_data = df.dNdt1573K25_dNdt
            if return_all_bool:
                E_d = [0.68, 0.74, 0.81, 0.93] # detrapping energy in eV
                A = [1.8E4, 1E4, 1E4, 1.4E4] # frequency factor in s^-1
                peak_T = ["627-650", "690", "760", "870"] # K
                N_0 = [12E16 , 6.5E16 ,3.5E16 ,5E16] # N(t=0), initial number of trapped D atoms according to Kissinger fit.

        # Below we consider the data that was gathered during my study.
        def text_to_np(file_path):
            f = open(file_path, "r")
            read = f.readlines()
            read = np.array(read)
            read = [text_line.replace("\n", "").split("  ") for text_line in read]
            read = np.array(read)
            read = read.astype(float)
            return read.reshape((len(read),))
        # AS-12
        if sample_case == 5:
            title = "AS-14 (this study)"
            T_data = text_to_np(csv_file_path + r"\Temp_list.txt")
            dNdt_data = text_to_np(csv_file_path + r"\desorpDe_list.txt")
            t_data = text_to_np(csv_file_path + r"\timeList.txt")

            #plt.scatter(T_data, dNdt_data)
            
            #T_data = [1]
            #t_data = [9]
            #T_data = [8]
            #dNdt_data = [3]

        t_data = [x for x in t_data if str(x) != 'nan'] # different measurements will have different lengths, these lines correct for that.
        T_data = [x for x in T_data if str(x) != 'nan']
        dNdt_data = [x for x in dNdt_data if str(x) != 'nan']
        if not return_all_bool:
            return title, np.array(t_data), np.array(T_data), np.array(dNdt_data)
        if return_all_bool:
            return title, t_data, E_d, A, peak_T, T_data, dNdt_data, N_0

    def plot_layout(self, sample_case):
        if sample_case < 3:
            color = 'k'
        else: color = 'red'
        if sample_case % 2 == 0:
            markerstyle = 'o'
            facecolors = 'none'
        else: 
            markerstyle = 's'
            facecolors = color
        if sample_case == 5:
            markerstyle = '^'
            color = 'green'
            facecolors = color
        edgecolors = color
        return markerstyle, facecolors, color, edgecolors


class kissinger(object):
    k = 8.617333262145E-5 # Boltzmann in eV/K
    deb_freq = 8.4e12 # Debye frequency in s^-1

    def modelT(self, N, T_model, freq, E_def, heating_rate):
        dNdT =  - 1/heating_rate * freq * N * np.exp(-E_def/(self.k*T_model))
        return dNdT

    def three_peaks(self, N1, N2, N3, e_1, e_2, e_3, T, heating_rate):
        freq_kgr_1 = 8.4e12
        # trap 1
        N_kgr_1 = odeint(self.modelT, N1, T, args=(freq_kgr_1, e_1, heating_rate))
        flux1 = - heating_rate * np.diag(self.modelT(N_kgr_1, T, freq_kgr_1, e_1, heating_rate)) # somehow, the solution is given as a matrix with the values on the diagonal.
        # trap 2
        N_kgr_2 = odeint(self.modelT, N2, T, args=(freq_kgr_1, e_2, heating_rate))
        flux2 = - heating_rate * np.diag(self.modelT(N_kgr_2, T, freq_kgr_1, e_2, heating_rate))
        # trap 3
        N_kgr_3 = odeint(self.modelT, N3, T, args=(freq_kgr_1, e_3, heating_rate))
        flux3 = - heating_rate * np.diag(self.modelT(N_kgr_3, T, freq_kgr_1, e_3, heating_rate))

        allflux = flux1+flux2+flux3
        return flux1, flux2, flux3, allflux

    def three_peaks_with_freq(self, N1, N2, N3, e_1, e_2, e_3, freq_kgr_1, freq_kgr_2, freq_kgr_3, T, heating_rate):
        # trap 1
        N_kgr_1 = odeint(self.modelT, N1, T, args=(freq_kgr_1, e_1, heating_rate))
        flux1 = - heating_rate * np.diag(self.modelT(N_kgr_1, T, freq_kgr_1, e_1, heating_rate)) # somehow, the solution is given as a matrix with the values on the diagonal.
        # trap 2
        N_kgr_2 = odeint(self.modelT, N2, T, args=(freq_kgr_2, e_2, heating_rate))
        flux2 = - heating_rate * np.diag(self.modelT(N_kgr_2, T, freq_kgr_2, e_2, heating_rate))
        # trap 3
        N_kgr_3 = odeint(self.modelT, N3, T, args=(freq_kgr_3, e_3, heating_rate))
        flux3 = - heating_rate * np.diag(self.modelT(N_kgr_3, T, freq_kgr_3, e_3, heating_rate))

        allflux = flux1+flux2+flux3
        return allflux