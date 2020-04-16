README

Filename: 191217_AllData_100g500i10m50c
Directory: /pine/scr/a/e/aeallen/Longleaf/Publication/191217_AllData_100g500i10m50c
Data file: /pine/scr/a/e/aeallen/Longleaf/experimental_v2.pickled

Generations: 100
Individuals: 500
Mutation rate: 0.1
Crossover rate: 0.5




#!/usr/bin/env python
#!/usr/bin/env python3 #iammoresentient shebang!

# Author: Amy Allen
# Date: 170618
# Notes: python35, killdevil

# This script will run an EA given input data
# from command line: ./scriptname.py
# it will check for / create new working dir name datafilename
# all output will be sequentually saved in dir datafilename

###################################################################
#IMPORT PACKAGES
###################################################################
import numpy as np
from scipy.integrate import odeint
from deap import base, creator, tools, algorithms
import os
import sys
import pickle
import time as timeski
import math as math
import time as timeapp

#import cProfile, pstats, io
##from pstats import SortKey
#pr = cProfile.Profile()
#pr.enable()

start = timeapp.time()

###########################################################################
#LOAD EXPERIMENTAL DATA
###########################################################################
#filename = '/Users/AmyAllen/Documents/ThesisWork/ModelSelection/Mating-Model/Longleaf/experimental_v2.pickled' #local
filename = '/pine/scr/a/e/aeallen/Longleaf/experimental_v2.pickled' #longleaf
data_to_score_against = filename
with open(filename, 'rb') as f:
    experimental = pickle.load(f)
times_50constant, data_50constant, times_50pulse, data_50pulse, times_50periodic, data_50periodic, times_10periodic, data_10periodic, times_mutpulse, data_mutpulse = experimental

###################################################################
#EA PARAMS
###################################################################
number_of_runs = 1000
number_of_generations = 100
number_of_individuals = 500
mutation_rate = 0.1
crossover_rate = 0.5

###################################################################
#MATRIX FOR VARIABLES TO INTERP AND EXPONENTIATE 
###################################################################
def make_conversion_matrix():
    # want easily savable matrix to hold this info
    # interp boolean, interp range (min,max), power boolean, power number (y)
    len_ind = 28
    arr_IandP = np.zeros((5,len_ind))
    # Set all interp booleans to 1 - everything is going to be interpreted
    arr_IandP[0,:] = 1
    # Set all power booleans to 1 - everything is in the form of powers
    arr_IandP[3,:] = 1
    # Set all power numbers to 10 - everything has a base of 10
    arr_IandP[4,:] = 10
    # Set minimums and maximums for all parameters. Parameters are in the following order: ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D
    minimums = [-3.9,-4.3,-3.2,-5,-5,-3.6,-3.7,-4.3,-3.2,-3.7,-5.4,-3.2,-5.1,-2.8,-5.4,-5.4,-4.3,-4.3,-3.2,-5,-4.4,-2.7,-3.7,-4.3,-4.3,-3.2,-2.4,-3.7]
    maximums = [-0.7,-0.3,1,-1,-1,-1.5,-0.9,-0.3,1,-1.5,1.4,0,0.3,1,0.3,1.4,-0.4,-0.3,1,-1,0.3,-0.5,-0.5,-0.4,1.4,-1.6,-1.6,-1.5]
    for i in range(len(minimums)):
        arr_IandP[1,i] = minimums[i] #interp_range_min
        arr_IandP[2,i] = maximums[i] #interp_range_max
    return arr_IandP

arr_conversion_matrix = make_conversion_matrix()
#############################################################################
#PARAMETERS
#############################################################################
time = np.linspace(0,600,6001)

#############################################################################
#EA FUNCTIONS
#############################################################################

def convert_individual(ea_individual, conversion_matrix):
    # use conversion matrix to convert interp and exponentiate individual:
    # conversion matrix has 5 rows of arrs of size len(individual):
    # conversion_matrix[0] = interp_bool
    # conversion_matrix[1] = interp_range_min
    # conversion_matrix[2] = interp_range_max
    # conversion_matrix[3] = power_bool
    # conversion_matrix[4] = base_val
    
    # copy and get len of individual
    arr_params_conv = np.zeros(28)
    len_ind = len(ea_individual)
    
    # Interp:
    for idx in np.nonzero(conversion_matrix[0])[0]:
        ea_val = ea_individual[idx]
        r_min = conversion_matrix[1][idx]
        r_max = conversion_matrix[2][idx]
        arr_params_conv[idx] = np.interp(ea_val, (0,1), (r_min, r_max))
    
    # Exponentiate:
    for idx in np.nonzero(conversion_matrix[3])[0]:
        ea_val = arr_params_conv[idx]
        base_val = conversion_matrix[4][idx]
        arr_params_conv[idx] = np.power(base_val, ea_val)
    
    return arr_params_conv

 # signal - single pulse
def signal(signal_number,t,slope_on,slope_off):
    p = signal_number
    if (p)*slope_on < 1:
        maxs = (p)*slope_on
    else:
        maxs = 1
        
    if signal_number == 0:
        return 0
    elif signal_number == 1:
        if t < 1/slope_on:
            return t*slope_on
        else:
            return 1
    else :
        if t < p:
            if t <= p and t < (1/slope_on):
                return ((t)*slope_on)
            else:
                return 1
        else:
            return 0
# signal - periodic
def periodic_signal(signal_number,t,slope_on,slope_off):
    p = signal_number
    if (p/2)*slope_on < 1:
        maxs = (p/2)*slope_on
    else:
        maxs = 1
    
    if signal_number == 0:
        return 0
    elif signal_number == 1:
        if t < 1/slope_on:
            return t*slope_on
        else:
            return 1
    else :
        it = math.floor(t/signal_number)
        if t>=it*p and t < it*p+p/2:
            if t >= it*p and t < it*p+(1/slope_on):
                return ((t-it*p)*slope_on)
            else:
                return 1
        else:
            return 0

def scorefxn1(arr_parameters, time):
    mse_total = 0
    arr_conversion_matrix = make_conversion_matrix()
    arr_params_IP = convert_individual(arr_parameters, arr_conversion_matrix)
    # parameters to be learned
    ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_params_IP
    # parameters to be kept constant
    kdegGFP = 10**-1
    hc = 2
    slope_off = 0
    
    ##### WT #####

    def DE(y,t,signal_number,slope_on,slope_off):
        GFP, Fus3, ppFus3, Ste12, Ste12Digs, Far1, pFar1 = y

        s=signal(signal_number,t,slope_on,slope_off)

        # GFP
        dGFPdt = ksynGFP+(ka4*Ste12**hc)/(KGFP+Ste12**hc)-kdegGFP*GFP
        # Fus3
        dFus3dt = ksynF3+(kfb1*Ste12**hc)/(KFus3+Ste12**hc)-kp1*s*Fus3+kp2*ppFus3-kdegF3*Fus3
        # ppFus3
        dppFus3dt = kp1*s*Fus3-kp2*ppFus3
        # Ste12
        dSte12dt = ksynS12+(kfb2*Ste12**hc)/(KSte12+Ste12**hc)-kdegS12*Ste12*(1+(kff1*pFar1)/(km1+pFar1))-ka1*Ste12*(DigsT-Ste12Digs)+(ka2*ppFus3+ka3)*Ste12Digs
        # Ste12Digs
        dSte12Digsdt = ka1*Ste12*(DigsT-Ste12Digs)-(ka2*ppFus3+ka3)*Ste12Digs-kdegS12D*Ste12Digs
        # Far1
        dFar1dt = ksynF1+(ka5*Ste12**hc)/(KFar1+Ste12**hc)+kp3*pFar1-kp4*ppFus3*Far1-kdegF1*Far1
        # pFar1
        dpFar1dt = kp4*ppFus3*Far1-kp3*pFar1-kdegPF1*pFar1

        return [dGFPdt, dFus3dt, dppFus3dt, dSte12dt, dSte12Digsdt, dFar1dt, dpFar1dt]

    def DE_periodic(y,t,signal_number,slope_on,slope_off):
        GFP, Fus3, ppFus3, Ste12, Ste12Digs, Far1, pFar1 = y

        s=periodic_signal(signal_number,t,slope_on,slope_off)

        # GFP
        dGFPdt = ksynGFP+(ka4*Ste12**hc)/(KGFP+Ste12**hc)-kdegGFP*GFP
        # Fus3
        dFus3dt = ksynF3+(kfb1*Ste12**hc)/(KFus3+Ste12**hc)-kp1*s*Fus3+kp2*ppFus3-kdegF3*Fus3
        # ppFus3
        dppFus3dt = kp1*s*Fus3-kp2*ppFus3
        # Ste12
        dSte12dt = ksynS12+(kfb2*Ste12**hc)/(KSte12+Ste12**hc)-kdegS12*Ste12*(1+(kff1*pFar1)/(km1+pFar1))-ka1*Ste12*(DigsT-Ste12Digs)+(ka2*ppFus3+ka3)*Ste12Digs
        # Ste12Digs
        dSte12Digsdt = ka1*Ste12*(DigsT-Ste12Digs)-(ka2*ppFus3+ka3)*Ste12Digs-kdegS12D*Ste12Digs
        # Far1
        dFar1dt = ksynF1+(ka5*Ste12**hc)/(KFar1+Ste12**hc)+kp3*pFar1-kp4*ppFus3*Far1-kdegF1*Far1
        # pFar1
        dpFar1dt = kp4*ppFus3*Far1-kp3*pFar1-kdegPF1*pFar1

        return [dGFPdt, dFus3dt, dppFus3dt, dSte12dt, dSte12Digsdt, dFar1dt, dpFar1dt]

    def simulate_single_experiment1(arr_parameters, time, signal_val,SS):
        # parameters to be learned
        ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_parameters
        # parameters to be kept constant
        kdegGFP = 10**-1
        #solve odes:
        odes = odeint(DE, SS, time, args=(signal_val,slope_on,slope_off,))
        # return array of individual cell counts:
        return odes

    def simulate_single_experiment_per(arr_parameters, time, signal_val,SS):
        # parameters to be learned
        ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_parameters
        # parameters to be kept constant
        kdegGFP = 10**-1
        #solve odes:
        odes = odeint(DE_periodic, SS, time, args=(signal_val,slope_on,slope_off,))
        # return array of individual cell counts:
        return odes

    #### Constant ####
    
    # Solve steady state
    IC = [0,0,0,0,0,0,0]
    t  = np.linspace(0,40000,100001)
    odes = odeint(DE, IC, t, args=(0,100,100,))
    TE=0
    for i in range(len(IC)):
        TE+=abs(odes[100000,i]-odes[100000-1,i])
    #print TE
    SS = odes[100000,:]

    expX = simulate_single_experiment1(arr_params_IP, time, 1,SS)
        
    # get index of time points closest
    idx_closest_time_points = []
    for each_time in times_50constant[0][~np.isnan(times_50constant[0])]:
        closest_idx = np.abs(time - each_time).argmin()
        idx_closest_time_points.append(closest_idx)
        
    # use indexes of time points to get data points to score against
    expX_scorefxn_data = expX[[idx_closest_time_points]]
    norm_max = max(expX_scorefxn_data[:,0])
        
    #SCORE IT! using MSE
    expX_mse = (np.abs(data_50constant[0][~np.isnan(data_50constant[0])] - expX_scorefxn_data[:,0]/norm_max)).mean()
    #print('MSE Exp' + str(idx+1) + ': ', expX_mse)
    mse_total += expX_mse
    
    #### Single Pulse ####

    # loop through different periods
    signal_numbers = [45,60,75,90,200]
    for i in range(len(signal_numbers)):
        expX = simulate_single_experiment1(arr_params_IP, time, signal_numbers[i],SS)
        
        # get index of time points closest
        
        # use indexes of time points to get data points to score against
        
        #SCORE IT! using MSE
        if signal_numbers[i] != 200:
            idx_closest_time_points = []
            for each_time in  times_50pulse[i][~np.isnan(times_50pulse[i])]:
                closest_idx = np.abs(time - each_time).argmin()
                idx_closest_time_points.append(closest_idx)
            expX_scorefxn_data = expX[[idx_closest_time_points]]
            expX_mse = (np.abs(data_50pulse[i][~np.isnan(data_50pulse[i])] - expX_scorefxn_data[:,0]/norm_max)).mean()
        else:
            idx_closest_time_points = []
            for each_time in  times_50pulse[i+1][~np.isnan(times_50pulse[i+1])]:
                closest_idx = np.abs(time - each_time).argmin()
                idx_closest_time_points.append(closest_idx)
            expX_scorefxn_data = expX[[idx_closest_time_points]]
            expX_mse = (np.abs(data_50pulse[i+1][~np.isnan(data_50pulse[i+1])] - expX_scorefxn_data[:,0]/norm_max)).mean()
        mse_total += expX_mse

        ### MAPK ACTIVATION ###
        if signal_numbers[i] == 90:
        # get index of time points closest
            idx_closest_time_points = []
            for each_time in [0,2,15,30,60,90,97,102,105,120,150,180]:
                closest_idx = np.abs(time - each_time).argmin()
                idx_closest_time_points.append(closest_idx)
        
            # use indexes of time points to get data points to score against
            expX_scorefxn_data = expX[[idx_closest_time_points]]
        
            #SCORE IT! using MSE
            expX_mse = (np.abs([0.05,0.37,0.38,0.62,0.76,0.85,1.00,0.58,0.32,0.15,0.13,0.21] - expX_scorefxn_data[:,2]/expX_scorefxn_data[5,2])).mean()
            #print('MSE Exp' + str(idx+1) + ': ', expX_mse)
            mse_total += expX_mse
        

    #### Periodic ####


    # loop through different periods
    signal_numbers = [90,120,150,180,240]
    for i in range(len(signal_numbers)):
        expX = simulate_single_experiment_per(arr_params_IP, time, signal_numbers[i],SS)
        
        # get index of time points closest
        idx_closest_time_points = []
        for each_time in times_50periodic[i][~np.isnan(times_50periodic[i])]:
            closest_idx = np.abs(time - each_time).argmin()
            idx_closest_time_points.append(closest_idx)
        
        # use indexes of time points to get data points to score against
        expX_scorefxn_data = expX[[idx_closest_time_points]]
        
        #SCORE IT! using MSE
        expX_mse = (np.abs(data_50periodic[i][~np.isnan(data_50periodic[i])] - expX_scorefxn_data[:,0]/norm_max)).mean()
        
        mse_total += expX_mse

    ##### FAR1 DELETE #####

    kp4_,ksynF1_,ka5_,kp3_ = [kp4,ksynF1,ka5,kp3]
    kp4,ksynF1,ka5,kp3 = [0,0,0,0]
    mut_params = [ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D]

    def DE(y,t,signal_number,slope_on,slope_off):
        GFP, Fus3, ppFus3, Ste12, Ste12Digs, Far1, pFar1 = y

        s=signal(signal_number,t,slope_on,slope_off)

        # GFP
        dGFPdt = ksynGFP+(ka4*Ste12**hc)/(KGFP+Ste12**hc)-kdegGFP*GFP
        # Fus3
        dFus3dt = ksynF3+(kfb1*Ste12**hc)/(KFus3+Ste12**hc)-kp1*s*Fus3+kp2*ppFus3-kdegF3*Fus3
        # ppFus3
        dppFus3dt = kp1*s*Fus3-kp2*ppFus3
        # Ste12
        dSte12dt = ksynS12+(kfb2*Ste12**hc)/(KSte12+Ste12**hc)-kdegS12*Ste12*(1+(kff1*pFar1)/(km1+pFar1))-ka1*Ste12*(DigsT-Ste12Digs)+(ka2*ppFus3+ka3)*Ste12Digs
        # Ste12Digs
        dSte12Digsdt = ka1*Ste12*(DigsT-Ste12Digs)-(ka2*ppFus3+ka3)*Ste12Digs-kdegS12D*Ste12Digs
        # Far1
        dFar1dt = ksynF1+(ka5*Ste12**hc)/(KFar1+Ste12**hc)+kp3*pFar1-kp4*ppFus3*Far1-kdegF1*Far1
        # pFar1
        dpFar1dt = kp4*ppFus3*Far1-kp3*pFar1-kdegPF1*pFar1

        return [dGFPdt, dFus3dt, dppFus3dt, dSte12dt, dSte12Digsdt, dFar1dt, dpFar1dt]

    def simulate_single_experiment1(arr_parameters, time, signal_val,SS):
        # parameters to be learned
        ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_parameters
        # parameters to be kept constant
        kdegGFP = 10**-1
        #solve odes:
        odes = odeint(DE, SS, time, args=(signal_val,slope_on,slope_off,))
        # return array of individual cell counts:
        return odes

    # Solve steady state
    IC = [0,0,0,0,0,0,0]
    t  = np.linspace(0,40000,100001)
    odes = odeint(DE, IC, t, args=(0,100,100,))
    TE=0
    for i in range(len(IC)):
        TE+=abs(odes[100000,i]-odes[100000-1,i])
    #print TE
    SS = odes[100000,:]

    expX = simulate_single_experiment1(mut_params, time, 1,SS)
    # get index of time points closest
    idx_closest_time_points = []
    for each_time in times_50constant[2][~np.isnan(times_50constant[2])]:
        closest_idx = np.abs(time - each_time).argmin()
        idx_closest_time_points.append(closest_idx)

    # use indexes of time points to get data points to score against
    expX_scorefxn_data = expX[[idx_closest_time_points]]
    
    #SCORE IT! using MSE
    expX_mse = (np.abs(data_50constant[2][~np.isnan(data_50constant[2])] - expX_scorefxn_data[:,0]/norm_max)).mean()
    
    mse_total += expX_mse

     ##### BROKEN FEEDBACK #####
    
    kp4,ksynF1,ka5,kp3 = [kp4_,ksynF1_,ka5_,kp3_]
    kfb2 = 0
    mut_params = [ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D]

    def DE(y,t,signal_number,slope_on,slope_off):
        GFP, Fus3, ppFus3, Ste12, Ste12Digs, Far1, pFar1 = y

        s=signal(signal_number,t,slope_on,slope_off)

        # GFP
        dGFPdt = ksynGFP+(ka4*Ste12**hc)/(KGFP+Ste12**hc)-kdegGFP*GFP
        # Fus3
        dFus3dt = ksynF3+(kfb1*Ste12**hc)/(KFus3+Ste12**hc)-kp1*s*Fus3+kp2*ppFus3-kdegF3*Fus3
        # ppFus3
        dppFus3dt = kp1*s*Fus3-kp2*ppFus3
        # Ste12
        dSte12dt = ksynS12+(kfb2*Ste12**hc)/(KSte12+Ste12**hc)-kdegS12*Ste12*(1+(kff1*pFar1)/(km1+pFar1))-ka1*Ste12*(DigsT-Ste12Digs)+(ka2*ppFus3+ka3)*Ste12Digs
        # Ste12Digs
        dSte12Digsdt = ka1*Ste12*(DigsT-Ste12Digs)-(ka2*ppFus3+ka3)*Ste12Digs-kdegS12D*Ste12Digs
        # Far1
        dFar1dt = ksynF1+(ka5*Ste12**hc)/(KFar1+Ste12**hc)+kp3*pFar1-kp4*ppFus3*Far1-kdegF1*Far1
        # pFar1
        dpFar1dt = kp4*ppFus3*Far1-kp3*pFar1-kdegPF1*pFar1

        return [dGFPdt, dFus3dt, dppFus3dt, dSte12dt, dSte12Digsdt, dFar1dt, dpFar1dt]

    def simulate_single_experiment1(arr_parameters, time, signal_val,SS):
        # parameters to be learned
        ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_parameters
        # parameters to be kept constant
        kdegGFP = 10**-1
        #solve odes:
        odes = odeint(DE, SS, time, args=(signal_val,slope_on,slope_off,))
        # return array of individual cell counts:
        return odes

    # Solve steady state
    IC = [0,0,0,0,0,0,0]
    t  = np.linspace(0,40000,100001)
    odes = odeint(DE, IC, t, args=(0,100,100,))
    TE=0
    for i in range(len(IC)):
        TE+=abs(odes[100000,i]-odes[100000-1,i])
    #print TE
    SS = odes[100000,:]

    expX = simulate_single_experiment1(mut_params, time, 1,SS)
    # get index of time points closest
    idx_closest_time_points = []
    for each_time in times_50constant[1][~np.isnan(times_50constant[1])]:
        closest_idx = np.abs(time - each_time).argmin()
        idx_closest_time_points.append(closest_idx)

    # use indexes of time points to get data points to score against
    expX_scorefxn_data = expX[[idx_closest_time_points]]
    
    #SCORE IT! using MSE
    expX_mse = (np.abs(data_50constant[1][~np.isnan(data_50constant[1])] - expX_scorefxn_data[:,0]/norm_max)).mean()
    
    mse_total += expX_mse

    print(mse_total)
    
    return mse_total

def scorefxn_helper(individual):
    # just a helper function that pulls all of scorefxn1 dependencies together
    # note the (), <--using single optimization in DEAP for now
    # scorefxn1 is taking care of the multiple optimizations for now
    return scorefxn1(individual, time),

###################################################################
#CHECK FOR / CREATE DIR FOR DATA
###################################################################
def strip_filename(fn):
    #input = full path filename
    #output = filename only
    #EX input = '/home/iammoresentient/phd_lab/data/data_posnegfb_3cellsum.pickled'
    #EX output = 'data_posnegfb_3cellsum'
    if '/' in fn:
        fn = fn.split('/')[-1]
    fn = fn.split('.')[0]
    return fn


def add_info(fn, gens, inds, mutationrate, crossoverrate):
    #input = filename only
    #output = date + filename + EA data
    # EX input = 'data_posnegfb_3cellsum'
    # EX output = '170327_data_posnegfb_3cellsum_#g#i#m#c'
    
    #get current date:
    cur_date = timeski.strftime('%y%m%d')
    # setup EA data:
    ea_data = str(gens) + 'g' + str(inds) + 'i' + str(int(mutationrate*100)) + 'm' + str(int(crossoverrate*100)) + 'c'
    #put it all together:
    #new_fn = cur_date + '_' + fn + '_' + ea_data
    new_fn = cur_date + '_' + os.path.basename(__file__).split('.')[0].split('_')[-1] + '_' + ea_data
    return new_fn
    
stripped_name = strip_filename(data_to_score_against)
informed_name = add_info(stripped_name, number_of_generations, number_of_individuals, mutation_rate, crossover_rate)
fn_to_use = informed_name
dir_to_use = os.getcwd() + '/' + informed_name

#check if dir exists and make 
if not os.path.isdir(dir_to_use):
    os.makedirs(dir_to_use)
    print('Made: ' + dir_to_use)
    # and make README file:
    fn = dir_to_use + '/' + 'README.txt'
    file = open(fn, 'w')
    
    # write pertinent info at top
    file.write('README\n\n')
    file.write('Filename: ' + fn_to_use + '\n')
    file.write('Directory: ' + dir_to_use + '\n')
    file.write('Data file: ' + data_to_score_against + '\n\n')
    file.write('Generations: ' + str(number_of_generations) + '\n')
    file.write('Individuals: ' + str(number_of_individuals) + '\n')
    file.write('Mutation rate: ' + str(mutation_rate) + '\n')
    file.write('Crossover rate: ' + str(crossover_rate) + '\n')
    file.write('\n\n\n\n')

    #write script to file
    #script_name = os.getcwd() + '/' + 'EA_1nf1pf.py'
    script_name = os.path.basename(__file__)
    open_script = open(script_name, 'r')
    write_script = open_script.read()
    file.write(write_script)
    open_script.close()

    file.close()
    
    


###################################################################
#LOOP: EVOLUTIONARY ALGORITHM + SAVE DATA 
###################################################################
for i in range(number_of_runs):
    ###################################################################
    #EVOLUTIONARY ALGORITHM
    ###################################################################
    #TYPE
    #Create minimizing fitness class w/ single objective:
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    #Create individual class:
    creator.create('Individual', list, fitness=creator.FitnessMin)

    #TOOLBOX
    toolbox = base.Toolbox()
    #Register function to create a number in the interval [1-100?]:
    #toolbox.register('init_params', )
    #Register function to use initRepeat to fill individual w/ n calls to rand_num:
    toolbox.register('individual', tools.initRepeat, creator.Individual, 
                     np.random.random, n=48)
    #Register function to use initRepeat to fill population with individuals:
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    #GENETIC OPERATORS:
    # Register evaluate fxn = evaluation function, individual to evaluate given later
    toolbox.register('evaluate', scorefxn_helper)
    # Register mate fxn = two points crossover function 
    toolbox.register('mate', tools.cxTwoPoint)
    # Register mutate by swapping two points of the individual:
    toolbox.register('mutate', tools.mutPolynomialBounded, 
                     eta=0.1, low=0.0, up=1.0, indpb=0.2)
    # Register select = size of tournament set to 3
    toolbox.register('select', tools.selTournament, tournsize=3)

    #EVOLUTION!
    pop = toolbox.population(n=number_of_individuals)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(key = lambda ind: [ind.fitness.values, ind])
    stats.register('all', np.copy)

    # using built in eaSimple algo
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, 
                                       mutpb=mutation_rate, 
                                       ngen=number_of_generations, 
                                       stats=stats, halloffame=hof, 
                                       verbose=False)


    ###################################################################
    #MAKE LISTS
    ###################################################################
    # Find best scores and individuals in population 
    arr_best_score = []
    arr_best_ind = []
    for a in range(len(logbook)):
        scores = []
        for b in range(len(logbook[a]['all'])):
            scores.append(logbook[a]['all'][b][0][0])
        #print(a, np.nanmin(scores), np.nanargmin(scores))
        arr_best_score.append(np.nanmin(scores))
        #logbook is of type 'deap.creator.Individual' and must be loaded later
        #don't want to have to load it to view data everytime, thus numpy
        ind_np = np.asarray(logbook[a]['all'][np.nanargmin(scores)][1])
        ind_np_conv = convert_individual(ind_np, arr_conversion_matrix)
        arr_best_ind.append(ind_np_conv)
        #arr_best_ind.append(np.asarray(logbook[a]['all'][np.nanargmin(scores)][1]))


    #print('Best individual is:\n %s\nwith fitness: %s' %(arr_best_ind[-1],arr_best_score[-1]))

    ###################################################################
    #PICKLE
    ###################################################################
    arr_to_pickle = [arr_best_score, arr_best_ind]

    def get_filename(val):
        filename_base = dir_to_use + '/' + fn_to_use + '_'
        if val < 10:
            toret = '000' + str(val)
        elif 10 <= val < 100:
            toret = '00' + str(val)
        elif 100 <= val < 1000:
            toret = '0' + str(val)
        else:
            toret = str(val)
        return filename_base + toret + '.pickled'

    counter = 0
    filename = get_filename(counter)
    while os.path.isfile(filename) == True:
        counter += 1
        filename = get_filename(counter)

    pickle.dump(arr_to_pickle, open(filename,'wb'))
    #print('Dumped data to file here: ', filename)'

#pr.disable()
#s = io.StringIO()
#ps = pstats.Stats(pr, stream=s)
#ps.print_stats()
#print(s.getvalue())

end = timeapp.time()
print(end - start)

