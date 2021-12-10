# Guide to the Evolutionary Algorithm 

## Importing packages
This code requires the following packages and functions:
* numpy
* os
* sys
* pickle
* time
* math
* odeint from the scipy package
* base, creator, tools, and algorithms from the deap package 

```
import numpy as np
from scipy.integrate import odeint
from deap import base, creator, tools, algorithms
import os
import sys
import pickle
import time as timeski
import math as math
import time as timeapp
``` 

## Loading experimental data 
The evolutionary algorithm is used to fit a model to experimental data. In this case, the experimental data being fit is stored in the file `experimental_v2.pickled`. You can find this in the Data/Experimental folder in this repository. In this pickle file there are 10 lists (time and normalized data) for 5 data sets. Inlcuding: 
* Response of WT to constant stimulus of 50 nM pheromone (`50constant`)
* Response of WT to single pulses of varying druations of 50 nM pheromone (`50pulse`)
* Response of WT to periodic pulses of varying durations of 50 nM pheromone (`50pulse`)
* Response of WT to periodic pulses of varying durations of 10 nM pheromone (`10 periodic`)
* Response of pathways mutants to single pulses and constant stimulation with 50 nM pheromone (`mutpulse`) 
```
filename = '/pine/scr/a/e/aeallen/Longleaf/experimental_v2.pickled' #longleaf
data_to_score_against = filename
with open(filename, 'rb') as f:
    experimental = pickle.load(f)
times_50constant, data_50constant, times_50pulse, data_50pulse, times_50periodic, data_50periodic, times_10periodic, data_10periodic, times_mutpulse, data_mutpulse = experimental
```

## EA Hyperparameters
There are five defined hyperparameters for the EA. They include:
* Number of runs: How many times the EA will be initialized. _I do not suggest 1000 unless you're running final simulations for a paper_
* Number of generations: How many times crossover and mutation will occur. _Again, start with <100 generations to make sure that the code is working_
* Number of inidividuals: How many parameter sets will be simulated each generation. _Start small, work bigger_
* Mutation rate: How frequently a parameter will randomly mutate. _Keep this small but non-zero. The mutation rate is used to help escape local minima_
* Crossover rate: The fraction of the "mated" parameter set that came from one of the "parent" parameter sets. _I think 0.5 makes the most sense here_ 
```
number_of_runs = 1000
number_of_generations = 100
number_of_individuals = 500
mutation_rate = 0.1
crossover_rate = 0.5
```

## Conversion matrix functions
The conversion matrix is one of the less *elegant* parts of this code. It is used to convert parameters from being 0-1 to a scale that is appropriate for each parameter. The appropriate parameter range is determined by the user and inputted into a matrix, `arr_IandP` that takes the following form:

| Parameter | Interpreted? | Minimum | Maximum | Power? | Base |
|-----------|--------------|---------|---------|--------|------|
| p1        | 1            | -3.9    | -0.7    | 1      | 10   |
| ...       | ...          | ...     | ...     | ...    | ...  |
| pn        | 1            | -3.7    | -1.5    | 1      | 10   |

For example, this means that parameter `p1` should be interpreted, and it can range from 10^-3.9 to 10^-0.7. 

This is a part of the code that should be modified for your model's parameters and appropriate parameter ranges. 

### Defining the conversion matrix 
```
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
```

### Conversion function 
The conversion function takes the conversion matrix in the previous step and converts it to the parmater set given by the EA, which by default ranges from 0-1, to a parameter with the appropriate range for the model. I do not recommend messing with this function much. 
```
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
 ```
 
 ## Model functions
 This part is for functions specific to the model. This includes the differential equations, the signal input functions, and the score function. 
 
 ### Signal inputs 
 Because the model is trained on multiple different temporal stimulus profiles there are two signal input functions, one for a single pulse and one for periodic stimulus. Both can handle a constant stimulus (`signal_number == 1`) or no stimulus (`signal_number == 0`). 
 ```
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
```

### Score function
I will be honest and say this could be more elegant. I'm going to break it down into sections to try make more sense of it, but it's some of the ugliest code I've ever seen/written. 

Here I set the time I solve across (600 minutes / 10 hours) and start defining the function. In this part of the function I:
* set the varying parameters equal to the value determined by the EA and the conversion matrix. 
* Set the parameters that stay contant (degredation rate for GFP, the hill coeffiecient, and the slope_off rate for signal function) 
* set multiple functions including the system of differential equations for single pulses of pheromone and periodic pulses of pheromone. Between these function the differentiall equations themselves don't change, only the signal input function changes. 
* define functions that solve the differential equations using `odeint` from the scipy package. 

```
time = np.linspace(0,600,6001)

ef scorefxn1(arr_parameters, time):
    mse_total = 0
    arr_conversion_matrix = make_conversion_matrix()
    arr_params_IP = convert_individual(arr_parameters, arr_conversion_matrix)
    # parameters to be learned
    ksynF3,kfb1,KFus3,kp1,kp2,kdegF3,ksynS12,kfb2,KSte12,kdegS12,kff1,km1,ka1,DigsT,ka2,ka3,ksynF1,ka5,KFar1,kp3,kp4,kdegF1,kdegPF1,ksynGFP,ka4,KGFP,slope_on,kdegS12D = arr_params_IP
    # parameters to be kept constant
    kdegGFP = 10**-1
    hc = 1
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
```

In this next part of the score function I:
* Solve the steady state of the model when there is no stimulus. This will be the value at t=0 for other stimulations. 
* Calculuate the model outcome for constant, single pulse, and period stimulus and compare the simuluation to the experimental data for WT cells using absolute error. 
```
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
```        

In this next part I do the same thing but for the _far1D_ and broken feedback mutants as well. In both cases the differential equations are redefined for the new scenario (_this honestly might not be necessary, this code is a hot mess_) and the steady state without stimulus is calculated for each mutant. 
```
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
```
This function returns the total absolute error, which the EA will then attempt to minimize. 

### Score function helper
This function is less ugly. It just passes time to the score function as well as the individual parameter set.
```
def scorefxn_helper(individual):
    # just a helper function that pulls all of scorefxn1 dependencies together
    # note the (), <--using single optimization in DEAP for now
    # scorefxn1 is taking care of the multiple optimizations for now
    return scorefxn1(individual, time),
```

## Data management
This makes systematically named file folders and the README.txt (that I pulled this code from. Yay for reproducible coding practices!)
```
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
```

## Actual EA Stuff
This is where the evolutationary algorithm, implemented in deap does its thing. _This is another section I wouldn't mess with too much_ 
```
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

    #MAKE LISTS
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
```

## Output data
Here data is outputted to a pickle file. It is outputted every generation so even if your run errors out later, you should be able to see the early generations' parameter sets. 
```
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
```
