import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
import emcee
import batman
import corner
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing as mp
import pickle
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde
from superprofile_mcmc_helper import run_superprofile_mcmc, extract_info_superprof
from vanilla_mcmc_helper import run_vanilla_mcmc, extract_info_vanilla
from miniprofile_mcmc_helper import run_miniprofile_mcmc, extract_info_miniprof 
from plotting_final import alpha_plotting, beta_plotting, psi_plotting

### Background Info
# This script runs three different MCMC's on an inputted set of lightcurves. The different MCMC's are: vanilla MCMC, mini-profile MCMC, and profile MCMC.
# In these documents, "parameters" refer to all the parameters of the MCMC. depending on the version of algorithm, this is a combination of the regression coefficients (betas), the golden search parameter (psi) or the fitted parameter, or the alpha parameters (which are walked over but not optimized/fitted). A complete set of batman parameters are referred to as "theta".
###


# region: kepler True parameter values
### True Parameter values for comparison and seed

#initial parameters, currently true values, used for comparison and to seed mcmc from true values 

p_0 = .07485 #p, radius radius in units stellar radii
b_0 = .106 #impact parameter
rho_s_0 = 1304 #set stellar density, in SI
q1_0 = .45 #quadratic limb darkening coefficient 1
q2_0 = .323 #quadratic limb darkening coefficient 2
tmidr_0 = 0 #original relative offset from tmid0, center of first transit reference tmid

beta0_0 = 1 
beta1_0 = 0
beta2_0 = 0
beta0_1 = 1 
beta1_1 = 0
beta2_1 = 0
beta0_2 = 1
beta1_2 = 0
beta2_2 = 0


#convert to log
ln_rho_s_0 = np.log(rho_s_0)

#save "true" parameters, note need to manually adjust these if changing number of betas or order of regression
theta0 = np.array([p_0, b_0, ln_rho_s_0, q1_0, q2_0, tmidr_0]) #true variables sent to batman for lightcurve0, relative tmid (about 0), use for seeding mcmc
betas0 = np.array([beta0_0, beta1_0]) #true betas for lightcurve0, use for seeding mcmc. include beta2 if using quadratic or higher order regression
#betas0 = np.array([beta0_0, beta1_0, beta2_0]) #true betas for lightcurve0, use for seeding mcmc. include beta2 if using quadratic or higher order regression
params0 = np.array([p_0, b_0, ln_rho_s_0, q1_0, q2_0, tmidr_0, beta0_0, beta1_0]) #parameters that set fake data, includes betas

# endregion: kepler True parameter values  


def load_data(show=True): #function to input lightcurves, load curves dictionary, and precompute info

#     lc_dir = Path("/Users/johnnymoynihan/Summer25Project/lightcurves") #folder of lightcurve's location

#     def load_curves(lc_dir, pattern="curve_*.csv"):
#         data = []
#         for csv in sorted(lc_dir.glob(pattern)):
#             df = pd.read_csv(csv)
#             data.append((df["t"].to_numpy(), df["flux"].to_numpy()))
#         return data

#     data = load_curves(lc_dir)
    


    transit_name = "kepler1513"

    lc_dir_real = Path("/Users/johnnymoynihan/Downloads/kepler1513_for_john/kepler1513_undetrended_kepler_pdc.csv")
    df = pd.read_csv(lc_dir_real)
    time = df['time'].values
    flux = df['flux'].values
    flux_err = df['flux_err'].values

    plt.plot(time, flux, '.')
    plt.xlabel("Time")
    plt.ylabel("Relative Flux")
    plt.title("Kepler 1513 Light Curve")

    plt.show()


    "divide data into different transits based off the tmid and period"
    period = 160.884088
    halfperiod = period / 2
    transit_duration = .5 #duration of transit in days
    cushion = 3 * transit_duration #cushion around transit to include in analysis. 

    tmax = np.max(time)
    #estimate tmids of each transit
    tmid0 = 277.50639 #starting tmid of first transit, used as original reference tmid0
    tmids = [tmid0 + i * period for i in range(0, int((tmax - (tmid0 - halfperiod)) // period) + 1)] #calculate estimated tmids of each transit, given tmid0 and period
    print(f"Estimated tmids of transits: {tmids}")
    data = []
    for i in range(len(tmids)): #note 1 for less curves
        bound_min = tmids[i] - cushion
        bound_max = tmids[i] + cushion
        filter = (time >= bound_min) & (time < bound_max) #only look at local data, times close to transit
        lc_i = time[filter]
        fl_i = flux[filter]
        ferr_i = flux_err[filter]
        plt.figure(figsize=(6,4))
        plt.plot(lc_i, fl_i, '.')
        plt.xlabel("Time")
        plt.ylabel("Relative Flux")
        plt.title(f"Transit {i + 1}")
        plt.show()
        data.append((lc_i, fl_i, ferr_i))

    data.pop(2) #remove third transit which is incomplete)
    print("Removing Transit 3 due to bad data")

    
    ncurves = len(data)

    data_info = { #dictionary of data info to be passed into create_info
        "ncurves" : ncurves,
        "tmid0" : tmid0,
        "period" : period,
        "halfperiod" : halfperiod,
        "transit_duration" : transit_duration,
        "cushion" : cushion,
        "transit_name" : transit_name,
        "tmids" : tmids
    }

    #load and precompute info for each lightcurve. lightcurves will be iterated over in log_post and current curve's dictionary passed through
    curves = [] #empty list of lightcurves
    for t, flux, ferr in data: #loop over each lightcurve

        integer = np.round((np.median(t) - tmid0)/period)
        tmidi = tmid0 + integer * period #calculate tmidi for each lightcurve, what we would expect for that lc tmid given tmid0 and period
        trel = t - tmidi #calculate relative times, these are tvalues centered about 0
        tmid_bounds = (-.7 * cushion, .7*cushion)
        
        
        flux_med_i = np.median(flux) #calculate mean flux of current lightcurve, used to scale betas accordingly
        
        curve = { #build dictionary of current curve, to be sent to golden_search/log_likelihood etc. this is how those functions will get current curve data
            "t" : t,
            "trel" : trel, 
            "tmidi" : tmidi,
            "tmid_bounds" : tmid_bounds,

            "flux" : flux,
            "flux_med_i" : flux_med_i,

            "sigma" : ferr  #calculate relative sigma, this is the error of the relative flux of the current trial lightcurve, right now, keep the same
        }
        curves.append(curve)

        if show: plt.scatter(t, flux, s=1, label=f"Curve {len(curves)}") #plot each lightcurve if show is True

    print(f"Loaded {ncurves} lightcurves")

    return curves, data_info

 ###set info

def create_info(data_info): #create info dictionary for algorithm

    #unpack data_info, move to general info dictionary
    ncurves = data_info["ncurves"]
    tmid0 = data_info["tmid0"]
    period = data_info["period"]
    cushion = data_info["cushion"]
    transit_name = data_info["transit_name"]

    #set bounds of parameters 
    bounds = {
        "p" : (1e-3, .3), #planet‑radius 
        "b" : (1e-4, 1.0), #impact parameter
        "ln_rho_s" : (np.log(1200), np.log(2000)), #stellar‑density 
        "q1" : (0.1, 1.0), #limb‑darkening q1
        "q2" : (0.1, 1.0), #limb‑darkening q2
        "tmid" : (-.8*cushion, .8*cushion), #relative bounds of tmid, this will enforced about the tmidi of the current lightcurve
        "beta0" : (.98, 1.02), #bounds for beta0
        "beta1" : (-.01, .01), #bounds for beta1
        "beta2" : (-.01, .01), #bounds for beta2
    }

   


    #set golden search info
    psi_name = "tmid" #set psi parameter for golden search and fitting, other variables will be walked over in the mcmc
    tol_loglikelihood_gs = .01 #set the tolerance of the log_likelihood for the golden search. once this tolerance is met, the golden search function will settle on a psi value
    maxiter_gs = 50 #set max number of iterations for golden search. tol_loglikelihood_gs will usually terminate first 

    #set regression info
    beta_num = 2 #set number of beta coefficients in regression of noise. 2 is linear or (beta0 + beta1*t). 3 is quadratic or (beta0 + beta1*t + beta2*t**2) etc...
    tot_betas = ncurves * beta_num #total number of betas, this is the number of beta parameters walked over in the vanilla mcmc

    #set mcmc info
    mcmcpoints = 300000 #set number of points to test
    thin = 10 #set thin
    burnin = 250000#set burnin


    mcmcpoints_van = 500000
    burnin_van = 400000


    # region: organize info  
    theta_ordered_names = ["p", "b", "ln_rho_s", "q1", "q2", "tmid"] #fixed order of parameters, code should reflect this order
    alpha_names = [name for name in theta_ordered_names if name != psi_name] #ordered theta_mcmc that are walked over in profile, no golden search/psi parameter included

    ndim_superprof = len(alpha_names) #number of dimensions walked over in super profile mcmc
    ndim_miniprof = len(alpha_names) + ncurves #number of dimensions
    ndim_van = ncurves + len(alpha_names) + tot_betas #number of dimensions in vanilla mcmc

    nwalkers_superprof = 2 * ndim_superprof + 2#number of walkers in super profile mcmc
    nwalkers_miniprof = 2 * ndim_miniprof + 2 #number of walkers in mini profile mcmc
    nwalkers_van = 2 * ndim_van + 2


    param_indices = {name: idx for idx, name in enumerate(theta_ordered_names)} #dictionary of parameter indices for easy access, this reflects the order in theta_ordered

    psi_idx = theta_ordered_names.index(psi_name) #index of psi parameter for easy access

    #create information dictionary for algorithm. this dictionary is passed into the mcmc functions
    info = {    
        "transit_name" : transit_name,

        "beta_num" : beta_num,
        "tot_betas" : tot_betas,


        "tol_loglikelihood_gs" : tol_loglikelihood_gs, 
        "maxiter_gs" : maxiter_gs, 

        "params0" : params0, #initial guess of all parameters, used to seed mcmc ([p_0, b_0, ln_rho_s_0, q1_0, q2_0, tmid_0, beta0_0, beta1_0 ...])
        "theta0" : theta0, #initial guess of theta parameters, used to seed ([p_0, b_0, ln_rho_s_0, q1_0, q2_0, tmid_0])
        "betas0" : betas0, #initial guess of betas, used to seed vanilla mcmc ([beta0_0, beta1_0 ...])

        "bounds" : bounds,
        "theta_ordered_names" : theta_ordered_names, #fixed order of parameters for batman, code should reflect this order
        "param_indices" : param_indices, #indices of parameters for easy access
        "psi_name" : psi_name,
        "psi_idx" : psi_idx,
        "psi_bounds" : bounds[psi_name],
        "alpha_names" : alpha_names, #ordered theta list without psi parameter, these are the parameters walked over in super profile mcmc

        "mcmcpoints" : mcmcpoints,
        "mcmcpoints_van": mcmcpoints_van,
        "burnin_van": burnin_van,

        "nwalkers_superprof" : nwalkers_superprof,
        "nwalkers_miniprof" : nwalkers_miniprof,
        "nwalkers_van" : nwalkers_van,

        "thin" : thin,
        "burnin" : burnin,
        "ndim_superprof" : ndim_superprof,
        "ndim_miniprof" : ndim_miniprof,
        "ndim_van" : ndim_van,
        "ncurves" : ncurves,

        "ncurves" : ncurves,
        "tmid0" : tmid0,
        "period" : period,
        }
    # endregion: organize info

    return info

def precompute_function(curves, info): #function to precompute helpful info for each curve
    beta_num = info["beta_num"]

    for curve in curves:
        trel = curve["trel"]
        if beta_num == 2:
            X = np.column_stack([np.ones_like(trel), trel])
        if beta_num == 3:
            X = np.column_stack([np.ones_like(trel), trel, trel**2])

        XtX_inv = np.linalg.inv(X.T @ X)
        curve["X"] = X
        curve["XtX_inv"] = XtX_inv

def main():
    
    
    #load data
    print("loading lightcurve data")
    print("plotting lightcurve data")
    curves, data_info = load_data(show=True)
    
    #create info
    print("creating algorithm info dictionary")
    info = create_info(data_info)

    #precompute info
    print("precomputing curve info")
    precompute_function(curves, info)
    
    #run super profile MCMC
    print("running super Profile Likelihood MCMC")
    superprof_sampler, samples = run_superprofile_mcmc(curves, info)
    print("Profilelikelihood MCMC complete")

    #run mini profile mcmc
    print("running mini Profile Likelihood MCMC")
    miniprof_sampler, mini_samples = run_miniprofile_mcmc(curves, info)
    print("mini Profilelikelihood MCMC complete")

    #run vanilla mcmc
    print("running vanilla MCMC")
    van_sampler, van_samples = run_vanilla_mcmc(curves, info)
    print("vanilla MCMC complete")
    
    #extract results
    nonlinear_results  = extract_info_superprof(superprof_sampler, curves, info)
    linear_results   = extract_info_miniprof(miniprof_sampler, curves, info)
    vanilla_results = extract_info_vanilla(van_sampler, curves, info)


    #create comparison plots
    print("creating comparison plots")  
    
    
    # alpha_plotting(info, nonlinear_results, linear_results, vanilla_results)
    # beta_plotting(info, nonlinear_results, linear_results, vanilla_results)
    # psi_plotting(info, nonlinear_results, linear_results, vanilla_results)


    #save results
    with open("datainfo_kepler1513.pkl", "wb") as f:
        pickle.dump({
            "info": info,
            "nonlinear_results": nonlinear_results,
            "linear_results": linear_results,
            "vanilla_results": vanilla_results,
            "curves" : curves,
            "data_info": data_info
            }, f,)

        

        return

if __name__ == "__main__":
    main()

