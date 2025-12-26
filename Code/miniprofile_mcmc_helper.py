import math, numpy as np, os, sys, emcee, matplotlib.pyplot as plt, corner, batman, math, scipy, pandas as pd
from pathlib import Path
import multiprocessing as mp
import arviz as az

### Calclulate lightcurve with batman
def batman_func_miniprof(theta_f, curve, info): #generic function to calculate lightcurve, expects [p, b, ln_rho_s, q1, q2, tmid], curve, info
    t = curve["t"] #this are the times of the current trial lightcurve
    
    p, b, ln_rho_s, q1, q2, tmid = theta_f #this batman function expects p and rho_s to be in log 

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

### Calculate trial flux with regression on data ###
def flux_predicted_miniprof(theta_trial_f, curve, info): #expects full batman variables
    t = curve["t"] #retrieve t from curve, this is the time of the current trial lightcurve
    flux_data = curve["flux"] #retrieve flux_data from curve, this is the flux of the current trial lightcurve

    ncurves = info["ncurves"] #number of curves
    X = curve["X"] #retrieve X from curve, this is the design matrix of the current trial lightcurve
    XtX_inv = curve["XtX_inv"] #retrieve XtX_inv from curve, this is the inverse of the design matrix of the current trial lightcurve
    
    beta_num = info['beta_num'] #retrieve beta_num from info, this is the number of beta parameters to use in regression
    sigma = curve['sigma'] #retrieve sigma from info, this is the error of the

    #extract relative time and flux for this lightcurve, independent of t and flux scale
    trel = curve["trel"] #retrieve trel from curve, this is the relative time of the current trial lightcurve

    batman_trial_data = batman_func_miniprof(theta_trial_f, curve, info) #call batman function to calculate trial flux corresponding to the inputted theta_trial_f, this uses absolute time
    bat = np.clip(batman_trial_data, 1e-12, None) #clip to avoid division by zero

    ### calculate flux_data/batman_trial fraction
    regr_frac = flux_data / bat #this is the the flux_data(the flux and tilt) divided the trial batman, the outcome is just the "tilt"

    ### perform regression and compute covar matrix
    betas, residuals, rank, s = scipy.linalg.lstsq(X, regr_frac) #perform least squares regression on the "tilt", giving noise/nuisance parameters, uses relative time

    poly_base = np.polynomial.polynomial.polyval(trel, betas)   #creates (beta0 + beta1*t + beta2*(t**2) + ...) depending on # of beta, usually go up to beta1 or beta2, uses relative time
    flux_trial = batman_trial_data * poly_base
    
    #calculate errors
    sigma2 = float(np.median(sigma))**2
    covar = sigma2 * XtX_inv
    beta_var = np.diag(covar)

    return flux_trial, betas, beta_var #return trial flux and corresponding beta info

### calculate log_likelihood ###
def loglikelihood_miniprof(theta_trial_f, curve, info): #function to return log_likelihood, expects full batman variables
    flux_data = curve["flux"] #retrieve flux_data from curve, this is the flux of the current trial lightcurve
    sigma = curve['sigma'] #retrieve sigma from info, this is the error of the current trial lightcurve

    flux_predicted, betas, beta_var = flux_predicted_miniprof(theta_trial_f, curve, info) #call flux_predicted function to calculate trial flux corresponding to the inputted theta_trial_f, this uses absolute time

    neg_chi2 = -.5 * np.sum((flux_data - flux_predicted)**2 / (sigma**2)) #calculate neg chi squared
    return neg_chi2, betas, beta_var #return neg chi squared and corresponding betas


### calculate prior
def log_prior_miniprof(trial_params, info, curves): #function to return log_prior, note that trial params have psi parameters and alpha parameters, no betas
    ncurves = info["ncurves"] #number of curves
    beta_num = info["beta_num"]
    psi_name = info["psi_name"]
    psi_index = info["psi_idx"]
    alpha_names = info["alpha_names"] #ordered theta_mcmc that are walked over, no psi parameter included

    full_psi_trial = trial_params[:ncurves]  #extract psi values from theta_trial, this is the variable that is fit for each curve
    params_trial = trial_params[ncurves:] #extract other params from theta_trial, these are the variables that are fit for all curves


    psi_bounds = info["psi_bounds"] #bounds on psi, the variable being fitted over each curve
    bounds = info["bounds"]
    #check if psi params are within bounds
    #check if non-psi params are within bounds

    logprior = 0.0 #initialize log prior
    
    for i in range(ncurves):
            psi_val = full_psi_trial[i]
            psi_min, psi_max = psi_bounds
            curve = curves[i]

            if psi_name == "tmid":
                psi_min, psi_max = curve["tmid_bounds"]
            if psi_val < psi_min or psi_val > psi_max: #confirm psi is in bounds
                return -np.inf #-inf log prior if out of bounds, enforces zero loglikelihood for this theta_trial
            logprior += -np.log(psi_max - psi_min) #right now, using same bounds as gs, plus flat prior


    for i, param_name in enumerate(alpha_names):
        param_min, param_max = bounds[param_name]
        param_val = params_trial[i]

        if param_val < param_min or param_val > param_max: #check to confirm param is in bounds
            return -np.inf #-inf log prior if out of bounds, enforces zero loglikelihood for this theta_trial
        
        if param_name == "ln_rho_s": #special case as sampled in log
            logprior += -np.log(np.exp(param_max) - np.exp(param_min)) + param_val #flat prior in log space
        else:
            logprior += -np.log(param_max - param_min) #flat prior in linear space
        
    return logprior


#calculate log posterior
def log_post_miniprof(trial_params, curves, info): #function to return log_posterior. this is the function the script calls for each theta_trial
    ncurves = info["ncurves"] #number of curves
    beta_num = info["beta_num"]
    psi_name = info["psi_name"]
    psi_index = info["psi_idx"]

    full_psi_trial = trial_params[:ncurves]  #extract psi values from trial_params, this is the variable that is fit for each curve
    alpha_trial = trial_params[ncurves:] #extract other params from trial_params, these are the variables that are fit for all curves

    ll_curves = np.empty(ncurves) #shape (ncurves,)
    betas = np.empty((beta_num, ncurves)) #shape (beta_num, ncurves)
    beta_var = np.empty((beta_num, ncurves)) #same (beta_num, ncurves)

    logprior = log_prior_miniprof(trial_params, info, curves) #calculate log prior for this trial_params
    if not np.isfinite(logprior): #confirm theta_trial are within prior bounds
        return -np.inf, np.zeros((beta_num, ncurves)), np.zeros((beta_num, ncurves)) #has to match shape of return/blobs


    for i, curve in enumerate(curves): #loop through each curve
        psi_trial = full_psi_trial[i] #extract psi value for this curve
        tmidi = curve["tmidi"] #extract original tmid for this curve, this is the offset from zero
        tmid_idx = info["param_indices"]["tmid"] #index of tmid in full batman params

        theta_trial_f = np.insert(alpha_trial, psi_index, psi_trial)

        theta_trial_f[tmid_idx] += tmidi #adjust psi such that it fits the current curve's time values


    
        ll_curves[i], betas[:, i], beta_var[:, i] = loglikelihood_miniprof(theta_trial_f, curve, info) #calculate log likelihood for this curve, store corresponding betas for this curve

    loglikelihood_total = np.sum(ll_curves) #sum log likelihoods of all curves

    logposterior = loglikelihood_total + logprior #calculate log posterior

    return logposterior, betas, beta_var


#run mini profile mcmc
def run_miniprofile_mcmc(curves, info): #run the mini_profile mcmc, this is the funciton the script will call
    #MCMC and data info
    mcmcpoints = info["mcmcpoints"]
    nwalkers = info["nwalkers_miniprof"]
    ncurves = info["ncurves"]
    theta0 = info["theta0"]

    #setup blob dtype for the specific number of curves and desired blobs
    beta_num = info["beta_num"]
    blob_dtype_miniprof = np.dtype([
        ("betas",     np.float64,  (beta_num, ncurves)),
        ("beta_var",  np.float64,  (beta_num, ncurves)),
    ])

    #set walker starting positions 
    psi_idx = info["psi_idx"]
    psi0 = theta0[psi_idx] #true values of psi, these are defined at top of .py file
    alpha_seed_miniprof = np.delete(theta0, psi_idx) #seed from true values, these are defined at top of .py file
    psi_seed_miniprof = np.repeat(psi0, ncurves) #seed from true values, these are defined at top of .py file. if psi is tmid, these are relative tmid values (about t = 0)

    params_seed_miniprof = np.concatenate((psi_seed_miniprof, alpha_seed_miniprof)) #combine seed values
    ndim_miniprof = len(params_seed_miniprof) #number of dimensions being walked over
    pos_miniprof = [np.array(params_seed_miniprof) + 1e-5 * np.random.randn(ndim_miniprof) for i in range(nwalkers)] #starting position of walkers with small random perturbations
    
    print(f"starting positions of walkers: {pos_miniprof[0]}")
    print(f"Running Mini-Prof MCMC with {nwalkers} walkers for {mcmcpoints} points with {ndim_miniprof} dimensions...")
    #run mcmc
    mp.set_start_method("spawn", force=True)
    with mp.get_context("spawn").Pool() as pool:
        sampler_miniprof = emcee.EnsembleSampler(
            nwalkers, ndim_miniprof, log_post_miniprof,
            args=(curves, info),
            blobs_dtype=blob_dtype_miniprof,
            pool=pool
        )
        samples_miniprof = sampler_miniprof.run_mcmc(pos_miniprof, mcmcpoints, progress=True)

    print("Mini-Prof MCMC Complete")

    return sampler_miniprof, samples_miniprof

def extract_info_miniprof(sampler_miniprof, curves, info):
    theta_ordered_names = info["theta_ordered_names"]
    #define basic info
    transit_name = info["transit_name"]
    burnin = info["burnin"]
    thin = info["thin"]
    psi_name = info["psi_name"]
    psi_idx = info["psi_idx"]
    ncurves = info["ncurves"]


    #create list of variable names:
    psi_names = [f"{psi_name}{i}" for i in range(ncurves)]
    alpha_names = list(info["alpha_names"])
    var_names = psi_names + alpha_names

    #extract arviz results
    ### Extract arviz results and save to arviz
    idata = az.from_emcee(sampler_miniprof, var_names = var_names)

    #drop args so arviz can interpret file, these (mainly info, will be sent to arvix file seperately)
    for group in ["observed_data", "constant_data"]:
        if hasattr(idata, group):
            ds = getattr(idata, group)
            bad_vars = [v for v in ds.data_vars if ds[v].dtype == "O"]
            if bad_vars:
                setattr(idata, group, ds.drop_vars(bad_vars))

    idata.to_netcdf(f"arviz_{transit_name}_linear.nc")

    #extract results
    samples_miniprof = sampler_miniprof.get_chain(discard=burnin, flat=True, thin=thin) 
    blobs_miniprof = sampler_miniprof.get_blobs(discard=burnin, flat=True, thin=thin) 
    logprob_miniprof = sampler_miniprof.get_log_prob(discard=burnin, flat=True, thin=thin) 
    
    
    #extract chains of blobs
    beta_chains = blobs_miniprof["betas"] 
    var_chains = blobs_miniprof["beta_var"]

    #convert ln_rho_s samples to rho_s samples for easier interpretation
    miniprof_samples_lin = samples_miniprof.copy()
    rho_s_idx = info["param_indices"]["ln_rho_s"]  #index of ln_rho_s in theta_ordered_names
    rho_s_miniprof_idx = ncurves + (rho_s_idx - 1 if psi_idx < rho_s_idx else rho_s_idx)  # ln_rho_s is at index 2 in theta_ordered
    rho_s_samps = np.exp(miniprof_samples_lin[:, rho_s_miniprof_idx])
    miniprof_samples_lin[:, rho_s_miniprof_idx] = rho_s_samps

    #organize final samps
    psi_samps = miniprof_samples_lin[:, :ncurves] 
    alpha_samps = miniprof_samples_lin[:, ncurves:] 
    beta_samps = beta_chains

    #organize best sample
    post_max_idx = np.argmax(logprob_miniprof) #index of the highest log prob, or best parameters
    best_alpha_samp = alpha_samps[post_max_idx] #best alpha sample corresponding to highest log prob
    best_psi_samp = psi_samps[post_max_idx] #best psi sample corresponding to highest log prob
    best_betas_samp = beta_samps[post_max_idx] #best beta sample corresponding to highest log prob


    #calculate predicted lightcurve of best parameters
    best_flux_prof = np.empty(ncurves, dtype=object) #shape (ncurves, n_times)
    for i, curve in enumerate(curves):
        trel = curve["trel"] #retrieve relative time from curve, this is the relative time of the current trial lightcurve
        best_betas = best_betas_samp[:, i] #extract betas for current curve

        best_batman_params = np.insert(best_alpha_samp, psi_idx, best_psi_samp[i]) #insert best psi into theta_mcmc

        if psi_name != "tmid":
            tmid_idx_full = theta_ordered_names.index("tmid")
            best_batman_params[tmid_idx_full] += curve["tmidi"]
        else:
            best_batman_params[psi_idx] += curve["tmidi"]

        best_batman_params[rho_s_idx] = np.log(best_batman_params[rho_s_idx])  #convert back to ln as batman expects ln_rho_s

        best_batman_results = batman_func_miniprof(best_batman_params, curve, info) #calculate best batman results
        poly_base = np.polynomial.polynomial.polyval(trel, best_betas)
        best_flux_prof[i]= best_batman_results * poly_base

    #compute percentiles
    q_low, q_high = 2.5, 97.5
    percentiles = np.percentile(alpha_samps, [q_low, q_high], axis=0).T  #shape (n_params, 2)
    #create ranges for plotting, add padding
    ranges = []
    for lo, hi in percentiles:
        width = hi - lo
        pad   = 0.05 * width #padding of 5% of the width
        ranges.append((lo - pad, hi + pad))


    linear_mcmc_results = { #build dictionary of profile_likelihood mcmc results
        "linear_logprob" : logprob_miniprof,
        "linear_alpha_samples" : alpha_samps, #these are the profile 
        "linear_betas" : beta_samps,
        "linear_var" : var_chains, 
        "linear_psi_samps" : psi_samps,
        "linear_best_alpha_samp" : best_alpha_samp, #best sample in linear space
        "linear_bestpsi" : best_psi_samp,
        "linear_bestbetas" : best_betas_samp,
        "linear_bestflux" : best_flux_prof,
        "linear_post_max" : logprob_miniprof[post_max_idx],
        "linear_percentiles" : percentiles,
        "linear_ranges" : ranges,
    }           

    return linear_mcmc_results