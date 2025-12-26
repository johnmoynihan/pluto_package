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
from matplotlib.ticker import MaxNLocator
import multiprocessing as mp
from tqdm import tqdm
import arviz as az


    





def batman_func(theta_batman, curve, info): #generic function to calculate lightcurve, expects [p, b, ln_rho_s, q1, q2, tmid], curve, info
    t = curve["t"] #this are the times of the current trial lightcurve

    p, b, ln_rho_s, q1, q2, tmid = theta_batman #this batman function expects rho_s to be in log

    ecc = 0
    periastron_long = 90
    period = info["period"] #period in days, this is fixed for all curves
    ### convert input variables to batman variables ###
    rho_s = np.exp(ln_rho_s)
    
    P_seconds = period * 86400 
    u1 = 2*math.sqrt(q1)*q2
    u2 = math.sqrt(q1) * (1-(2*q2))
    a_batman = ((rho_s * (6.67e-11) * P_seconds**2)/(3*math.pi))**(1/3) #calculate semi major axis (in units of stellar radii)
    x = np.clip(b / a_batman, -1 + 1e-12, 1 - 1e-12) #clip to avoid divs by zero or invalid values
    inc = 180/np.pi * np.arccos(x)

    ### set batman parameters ###
    params = batman.TransitParams() #initialize object to store parameters
    params.t0 = tmid  #time of inferior conjunction
    params.per = period #period in days
    params.rp = p #planet radius in units of stellar radii
    params.a = a_batman #semi major axis (in units of stellar radii)
    params.inc = inc #orbital inclination (in degrees)
    params.ecc = ecc #eccentricity
    params.w = periastron_long #longitude of periastron
    params.limb_dark = "quadratic" #limb darkening model
    params.u = [u1, u2] #limb darkening coefficients (u1, u2)


    m = batman.TransitModel(params, t) #initializes model, uses the t values of the current trial curve
    flux =  m.light_curve(params) #perform batman

    return flux

def final_flux_func(theta_batman_trial, beta_trial, curve, info): #function to complete flux calculation for inputted trial flux
    beta_num = info["beta_num"]
    t = curve["t"]
    t_rel = curve["trel"] #retrieve relative times from curve, this is the time of inferior conjunction of the current trial lightcurve
    flux_data = curve["flux"]


    batman_trial_data = batman_func(theta_batman_trial, curve, info) #calculate batman flux for given theta_batman parameters

    b0_trial = beta_trial[0] #beta0 of current mcmc trial step
    b1_trial = beta_trial[1] #beta1 of current mcmc trial step
    if beta_num == 2:
        flux_trial = batman_trial_data * (b0_trial + b1_trial*t_rel)

    if beta_num == 3:
        b2_trial = beta_trial[2]
        flux_trial = batman_trial_data * (b0_trial + b1_trial*t_rel + b2_trial*(t_rel**2)) ### compute trial flux

    return flux_trial

def log_likelihood_van(batman_trial, beta_trial, curve, info):
    sigma = curve["sigma"] #get sigma from info
    flux_data = curve["flux"]
    flux_trial = final_flux_func(batman_trial, beta_trial, curve, info) #call the predicted flux function, this returns the flux as the first object, betas as second, errors as last
    neg_chi2 = -.5 * np.sum((flux_data - flux_trial)**2 / (sigma**2)) #calculate chi2, using constant sigma
    return neg_chi2

def log_prior_van(params_van_mcmc, curves, info): #function to return log_prior, note that params_van_mcmc has psi_params, alpha_params, and beta_params
    ndim_van = len(params_van_mcmc) #number of dimensions in vanilla mcmc, this is the number of parameters walked over in the mcmc
    beta_num = info["beta_num"]
    tot_betas = info["tot_betas"]
    ncurves = info["ncurves"] #number of curves

    beta_startidx = ndim_van - tot_betas

    psi_params = params_van_mcmc[:ncurves]
    alpha_params = params_van_mcmc[ncurves:beta_startidx]
    betas = params_van_mcmc[beta_startidx:] #get betas from params_van_mcmc, these are the parameters that are walked over in the mcmc
    psi_bounds = info["psi_bounds"] #bounds on psi, the variable being fitted over each curve
    bounds = info["bounds"]

    alpha_names = info["alpha_names"] #ordered theta_mcmc that are walked over, no golden search. this is a list of strings

    psi_name = info["psi_name"]

    logprior = 0.0 #initialize log prior

    #check if psi parameter trials are within bounds, if psi is tmid, bounds are relative (ie. -0.5, 0.5), if psi is not tmid, bounds are absolute (ie. 0, 10)
    for i in range(ncurves):
        psi_val = psi_params[i]
        curve = curves[i]
        psi_min, psi_max = psi_bounds
        if psi_name == "tmid":
            psi_min, psi_max = curve["tmid_bounds"]
        
        if psi_val < psi_min or psi_val > psi_max:
            return -np.inf
        logprior += -np.log(psi_max - psi_min) #right now, using same bounds as gs, plus flat prior
        
    #check if alpha parameters in bounds
    for i, param_name in enumerate(alpha_names):
        param_val = alpha_params[i]
        param_min, param_max = bounds[param_name]
        if param_val < param_min or param_val > param_max:
            return -np.inf
        if param_name == "ln_rho_s": #special case as sampled in log
            logprior += -np.log(np.exp(param_max) - np.exp(param_min)) + param_val #flat prior in log space
        else:
            logprior += -np.log(param_max - param_min) #flat prior in linear space
        

    #check beta parameters
    beta0_min, beta0_max = bounds["beta0"]
    beta1_min, beta1_max = bounds["beta1"]
    beta2_min, beta2_max = bounds["beta2"]

    for i in range(ncurves): #loop over the betas for each curve
        betas_i = betas[i*beta_num:(i+1)*beta_num] #get the beta parameters for each curve
        beta0 = betas_i[0] #beta0 is the first parameter
        if not (beta0_min <= beta0 <= beta0_max):
            return -np.inf
        logprior += -np.log(beta0_max - beta0_min)
        

        beta1 = betas_i[1] #beta1 is the second parameter
        if not (beta1_min <= beta1 <= beta1_max):
            return -np.inf
        logprior += -np.log(beta1_max - beta1_min)

        if beta_num == 3: #if there is a third beta parameter, check it
            beta2 = betas_i[2]
            if not (beta2_min <= beta2 <= beta2_max):
                return -np.inf
            logprior += -np.log(beta2_max - beta2_min)

    return logprior

def log_post_van(params_van_mcmc, curves, info):
    ndim_van = len(params_van_mcmc) #number of dimensions in vanilla mcmc, this is the number of parameters walked over in the mcmc
    psi_idx = info["psi_idx"] #index of psi parameter for easy access
    psi_name = info["psi_name"] #name of psi parameter for easy access
    beta_num = info["beta_num"]
    ncurves = info["ncurves"] #number of curves
    tot_betas = info["tot_betas"]


    psi_params = params_van_mcmc[:ncurves] #get the fitted parameters from the inputted params_van_mcmc. these will be walked over for each lightcurve
    alpha_params = params_van_mcmc[ncurves: ndim_van - tot_betas] #get the other theta parameters from the inputted params_van_mcmc
    beta_params = params_van_mcmc[ndim_van - tot_betas:] #get the beta parameters from the inputted params_van_mcmc. these are in form (beta0,1, beta1,1, beta2,1, beta0,2, beta1,2, beta2,2, ...)

    #check priors for step
    logprior_van = log_prior_van(params_van_mcmc, curves, info) #calculate the log prior
    if not np.isfinite(logprior_van):         
            return -np.inf 

    ll_curves = np.empty(ncurves) #shape (ncurves,)
    for i, curve in enumerate(curves): #loop over each curve
        

        alpha_params_i = alpha_params.copy() #copy the other parameters for the current curve

        #adjust tmid trial to current curve's tmidi
        tmidi = curve["tmidi"] #retrieve tmidi from curve, this is the time of inferior conjunction of the current trial lightcurve
        if psi_name == "tmid":
            psi_trial_param = psi_params[i] + tmidi

        else: #if psi is not tmid, adjust relative tmid trial to current curve's tmidi
            tmid_idx = len(alpha_params) - 1 #if tmid is not psi, it is the last parameter in alpha_params (so index len(alpha_params) - 1 always works)
            alpha_params_i[tmid_idx] += tmidi #adjust tmid trial to current curve's tmidi
            psi_trial_param = psi_params[i] 
    
        #finalize the trial parameters
        batman_trial_total = np.insert(alpha_params_i, psi_idx, psi_trial_param) #insert the fitted parameter into the trial parameters at the psi index
        beta_trial = beta_params[i*beta_num:(i+1)*beta_num] #get the beta parameters for the current curve
        ll_curves[i] = log_likelihood_van(batman_trial_total, beta_trial, curve, info) #calculate the log likelihood for the current parameters and curve
    ll_total = np.sum(ll_curves)

    return logprior_van + ll_total #return the log posterior, which is the sum of the log prior and the log likelihood

def run_vanilla_mcmc(curves, info): #run the vanilla mcmc, 
    #MCMC and data info
    psi_idx = info["psi_idx"]
    beta_num = info["beta_num"]
    nwalkers = info["nwalkers_van"] #number of walkers in the mcmc
    mcmcpoints = info["mcmcpoints_van"] #number of points to sample
    ncurves = info["ncurves"] #number of curves



    #set starting walker positions, note vanilla seed includes psi_params, alpha_params, and beta_params. there are (ncurves) # of psi_params and (ncurves * beta_num) # of beta_params
    theta0 = info["theta0"] #true variables to seed mcmc, defined in script.py
    betas0 = info["betas0"] #true betas to seed mcmc, defined in script.py
    psi0 = theta0[psi_idx]
   
    alpha_seed = np.delete(theta0, psi_idx) #seed from true values
    betas_seed = np.tile(betas0, ncurves) #create seed for betas, which are walked over in the mcmc. this tiles the betas0 array ncurves times, so if betas0 = [1, 0, 0] and ncurves = 3, betas_seed = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    psi_seed = np.repeat(psi0, ncurves) #create seed for psi parameter, which is walked over in the mcmc

    vanilla_seed = np.concatenate((psi_seed, alpha_seed, betas_seed)) #concatenate to create full seed for mcmc: [psi_params, alpha_params, beta_params]


    ndim_van = len(vanilla_seed) #number of dimensions in vanilla mcmc, this is the number of parameters walked over in the mcmc
    print(f"ndim_van: {ndim_van}") #print number of dimensions in vanilla mcmc

    pos_van = [np.array(vanilla_seed) + 1e-5 *np.random.randn(ndim_van) for i in range(nwalkers)] #set starting position for the walkers, seed from true
    print(f"starting positions of walkers: {pos_van[0]}") #print starting position of first walker for reference
    ###run mcmc ###
    print(f"Running Vanilla MCMC with {nwalkers} walkers for {mcmcpoints} points with {ndim_van} dimensions...")
    sampler_van = emcee.EnsembleSampler(nwalkers, ndim_van, log_post_van, args = (curves, info))

    samples_van = sampler_van.run_mcmc(pos_van, mcmcpoints, progress = True) #start at pos, points, show progress

    print("Vanilla MCMC Complete")

    return sampler_van, samples_van




def extract_info_vanilla(sampler_van, curves, info): #function to extract results from vanilla mcmc sampler
    transit_name = info["transit_name"]
    burnin = info["burnin_van"] #number of points to discard as burnin
    thin = info["thin"] #thinning factor
    psi_idx = info["psi_idx"]
    psi_name = info["psi_name"]
    beta_num = info["beta_num"]
    ncurves = info["ncurves"]
    tot_betas = info["tot_betas"]
    alpha_names = info["alpha_names"]
    theta_ordered_names = info["theta_ordered_names"] #ordered theta_mcmc that are walked over, includes psi parameter


    #create list of variable names:
    psi_names = [f"{psi_name}{i}" for i in range(ncurves)]
    beta_names = [f"beta_{j}_{i}" for i in range(ncurves) for j in range(beta_num)]
    alpha_names = list(info["alpha_names"])
    var_names = psi_names + alpha_names + beta_names


    ### extract arviz results and save to arviz
    idata = az.from_emcee(sampler_van, var_names = var_names)

    #drop args so arviz can interpret file, these (mainly info, will be sent to arvix file seperately)
    for group in ["observed_data", "constant_data"]:
        if hasattr(idata, group):
            ds = getattr(idata, group)
            bad_vars = [v for v in ds.data_vars if ds[v].dtype == "O"]
            if bad_vars:
                setattr(idata, group, ds.drop_vars(bad_vars))

    idata.to_netcdf(f"arviz_{transit_name}_vanilla.nc")


    ### extract results
    flat_samples_van = sampler_van.get_chain(flat = True, discard = burnin, thin=thin) #get flatchain, shape is (# of points, # of theta)
    log_prob_van = sampler_van.get_log_prob(flat=True,  discard = burnin, thin=thin) #extract log prob as well, shape is (# of points)

    ndim_van = flat_samples_van.shape[1] #number of dimensions in vanilla mcmc, this is the number of parameters walked over in the mcmc
    post_max_idx_van = np.argmax(log_prob_van) #index of the highest log prob
    post_max_van = log_prob_van[post_max_idx_van] #the highest log prob

    #determine slices of flatchain
    psi_slice = slice(0, ncurves) #slice for psi parameters
    alpha_slice = slice(ncurves, ndim_van - tot_betas) #slice for theta parameters
    beta_slice = slice(ndim_van - tot_betas, ndim_van) #slice for beta parameters


    #convert ln_rho_s samples to rho_s samples for easier interpretation
    vanilla_samples_lin = flat_samples_van.copy()
    rho_s_idx = info["param_indices"]["ln_rho_s"]  #index of ln_rho_s in theta_ordered_names
    rho_s_vanilla_idx = ncurves + (rho_s_idx - 1 if psi_idx < rho_s_idx else rho_s_idx)  # ln_rho_s is at index 2 in theta_ordered
    rho_s_samps = np.exp(vanilla_samples_lin[:, rho_s_vanilla_idx])
    vanilla_samples_lin[:, rho_s_vanilla_idx] = rho_s_samps

    #extract different samples
    psi_samps = vanilla_samples_lin[:, psi_slice] #samples of the fitted parameter, equivalent to psi in profilelikelihood
    alpha_samps = vanilla_samples_lin[:, alpha_slice] #theta_mcmc samples
    beta_samps = vanilla_samples_lin[:, beta_slice] #beta_mcmc samples. format: (beta0,1, beta1,1, beta2,1, beta0,2, beta1,2, beta2,2, ...)
    theta_samps_log = flat_samples_van[:, alpha_slice] #theta_mcmc samples in log space

    #extract best sample
    best_lin_samp_van = vanilla_samples_lin[post_max_idx_van] #best sample in log space
    best_psi_samps_van = best_lin_samp_van[psi_slice] #best psi sample, this is the fitted parameter
    best_alpha_samp_van_log = best_lin_samp_van[alpha_slice] #best theta sample in log space
    best_beta_samp_van = best_lin_samp_van[beta_slice]
    best_beta_samp_van2d = best_beta_samp_van.reshape(ncurves, beta_num) #reshape to 2D array, shape (ncurves, beta_num)

    ncurves = info["ncurves"] #number of curves



    #calculate predicted lightcurve of best parameters
    best_flux_van = np.empty(ncurves, dtype=object) #shape (ncurves, n_times)
    for i, curve in enumerate(curves):
        trel = curve["trel"] #retrieve relative time from curve, this is the relative time of the current trial lightcurve
        best_betas = best_beta_samp_van2d[i] #extract betas for current curve

        best_batman_params = np.insert(best_alpha_samp_van_log, psi_idx, best_psi_samps_van[i]) #insert best psi into theta_mcmc

        if psi_name != "tmid":
            tmid_idx_full = theta_ordered_names.index("tmid")
            best_batman_params[tmid_idx_full] += curve["tmidi"]
        else:
            best_batman_params[psi_idx] += curve["tmidi"]

        best_batman_results = batman_func(best_batman_params, curve, info) #calculate best batman results
        poly_base = np.polynomial.polynomial.polyval(trel, best_betas)
        best_flux_van[i]= best_batman_results * poly_base


    vanilla_results = {
        "best_sample": best_lin_samp_van,
        "log_prob_van" : log_prob_van,
        "best_psi_sample_van": best_psi_samps_van,
        "best_alpha_sample_van": best_alpha_samp_van_log,
        "best_beta_sample_van": best_beta_samp_van,
        "best_flux_van": best_flux_van,

        "psi_samps_van": psi_samps,
        "alpha_samps_van": alpha_samps,
        "beta_samps_van": beta_samps,
    }



    return vanilla_results