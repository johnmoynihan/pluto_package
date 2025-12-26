#this code, when golden_search is called, returns the maximum log_likelihood of inputted theta_trial and psi, the golden search parameter
#the code calls functions in order: log_post() -> log_prior() -> golden_search() -> log_likelihood() -> flux_predicted() -> batman_func() 


import math, numpy as np, os, sys, emcee, matplotlib.pyplot as plt, corner, batman, math, scipy, pandas as pd
from pathlib import Path
import multiprocessing as mp
import arviz as az


### Calculate lightcurve with batman
def batman_func_prof(theta_f, curve, info): #generic function to calculate lightcurve, expects [p, b, ln_rho_s, q1, q2, tmid], curve, info
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
def flux_predicted_prof(theta_trial_f, curve, info): #expects full batman variables
    t = curve["t"] #retrieve t from curve, this is the time of the current trial lightcurve
    flux_data = curve["flux"] #retrieve flux_data from curve, this is the flux of the current trial lightcurve

    ncurves = info["ncurves"] #number of curves

    X = curve["X"] #retrieve X from curve, this is the design matrix of the current trial lightcurve
    XtX_inv = curve["XtX_inv"] #retrieve XtX_inv from curve, this is the inverse of the design matrix of the current trial lightcurve
    
    beta_num = info['beta_num'] #retrieve beta_num from info, this is the number of beta parameters to use in regression
    sigma = curve['sigma'] #retrieve sigma from info, this is the error of the

    #extract relative time and flux for this lightcurve, independent of t and flux scale
    trel = curve["trel"] #retrieve trel from curve, this is the relative time of the current trial lightcurve

    batman_trial_data = batman_func_prof(theta_trial_f, curve, info) #call batman function to calculate trial flux corresponding to the inputted theta_trial_f, this uses absolute time
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
    

### Golden Search
def golden_search_prof(theta_trial, curve, info): #function that uses golden search to find the best psi given the inputted theta_mcmc and trial curve. gs theta_trial has shape (n_params_prof,)
    phi = (math.sqrt(5) - 1) / 2

    t = curve["t"] #retrieve t from curve, this is the time of the current trial lightcurve
    tmidi = curve["tmidi"] #retrieve tmidi from curve, this is the time of inferior conjunction of the current trial lightcurve



    flux_data = curve["flux"] #retrieve flux_data from curve, this is the flux of the current trial lightcurve

    sigma = curve['sigma'] #retrieve sigma from info, this is the error of the flux_data
    #sigma_rel = sigma / curve["flux_adj"] #calculate relative sigma, this is the error of the relative flux of the current trial lightcurve

    psi_bounds0 = info['psi_bounds'] #retrieve psi bounds from info, this is a tuple of (left, right) bounds for psi
    psi_idx     = info["psi_idx"] #retrieve psi_idx from info, this is the index of psi of batman's expected input
    maxiter = info['maxiter_gs'] #retrieve maxiter from info, this is the maximum number of iterations for golden search
    tol_loglikelihood = info['tol_loglikelihood_gs'] #retrieve tol_loglikelihood from info
    psi_name = info["psi_name"] #retrieve psi name from info, this is the name of the psi parameter

 
    def log_likelihood_prof(psi_val): #computes log_likelihood of theta_trial and the inputted psi
        theta_trial_f = np.insert(theta_trial, psi_idx, psi_val) #insert psi_val at psi_idx in theta_trial, creates array matching the expected batman input
        flux_trial, betas, beta_var = flux_predicted_prof(theta_trial_f, curve, info) #call the predicted flux function, this returns the flux as the first object, betas as second, errors as last    
        neg_chi2 = -.5 * np.sum((flux_data - flux_trial)**2 / (sigma**2)) #calculate neg chi squared. the data, flux_tc, are the flux measurements of the current trial lightcurve
        if psi_name == "tmid": psi_val = psi_val - tmidi  #if psi is tmid, we need to adjust the tmid value to the current curve's tmidi. this returns a relative tmid value
        return neg_chi2, betas, beta_var, psi_val
    

    left, right = psi_bounds0 #use bounds of psi as initial golden ratio bounds. if tmid is psi: these are relative bounds
    width = abs(right - left) #calculate width of the bounds
    left = left - (.01 * width * np.random.random()) #add small random offset to left bound to avoid numerical issues
    right = right + (.01 * width * np.random.random()) #add small random offset to right bound to avoid numerical issues
    if psi_name == "tmid": #if psi is tmid, we need to adjust the bounds to be dependent on the tmidi of the current lightcurve
        left, right = curve["tmid_bounds"]
        width = abs(right - left) #calculate width of the bounds
        left = left - (.01 * width * np.random.random()) #add small random offset to left bound to avoid numerical issues
        right = right + (.01 * width * np.random.random()) #add small random offset to right bound to avoid numerical issues 
        left = tmidi + left #lower bound of psi, depending on the tmid of the current lightcurve
        right = tmidi + right #upper bound of psi, depending on the tmid of the current lightcurve

    #calculate initial interior points c and d within the bounds using golden ratio
    c = right - phi * (right - left)  #c is point closer to left bound
    d = left + phi * (right - left)   #d is point closer to right bound

    #calculate log_likelihoods at c and d
    f_c = log_likelihood_prof(c)[0] 
    f_d = log_likelihood_prof(d)[0]

    iteration = 0
    while iteration < maxiter:
        if f_c > f_d:
            #f(c) > f(d) means c yields a higher likelihood, so maximum lies in [left, d]
            right = d            #move the right bound to d
            d = c                #shift c to d position
            f_d = f_c            #carry over f_c, efficient golden search
            c = right - phi * (right - left)  #new point c in [left, right]
            f_c = log_likelihood_prof(c)[0]
        else:
            # f(d) >= f(c) means d yields a higher likelihood, so maximum lies in [c, right]
            left = c             #move the left bound to c
            c = d                #shift d to c position
            f_c = f_d            #carry over f_d, efficient golden search
            d = left + phi * (right - left)   #new point d in [left, right]
            f_d = log_likelihood_prof(d)[0]
        best_f = max(f_c, f_d)
        if iteration > 0 and abs(f_c - f_d) < tol_loglikelihood: #test if points are within log_likelihood tolerance
            break 
        iteration += 1
        
    best_psi = (left + right) / 2.0 #average of final points

    return log_likelihood_prof(best_psi)


### Calculate prior
#this prior calls the bounds from info. if parameter is psi, it is assigned np.nan and ignored. if any param step outside bounds, returns -np.inf.
# np.nan is assigned to psi to allow for easy changes to psi parameter without having to change the prior function. 
def log_prior_prof(theta_trial, info): #function to return log_prior, note that theta_trial has only alpha parameters, no psi or betas
    ncurves = info["ncurves"] #number of curves
    beta_num = info["beta_num"]
    psi_name = info["psi_name"]
    psi_index = info["psi_idx"]
    alpha_names = info["alpha_names"] #ordered theta_mcmc that are walked over, no psi parameter included

    bounds = info["bounds"]

    logprior = 0.0 #initialize logprior
    for i, param_name in enumerate(alpha_names):
        param_min, param_max = bounds[param_name]
        param_val = theta_trial[i]

        if param_val < param_min or param_val > param_max: #check to confirm param is in bounds
            return -np.inf #-inf log prior if out of bounds, enforces zero loglikelihood for this theta_trial

        if param_name == "ln_rho_s": #special case as sampled in log
            logprior += -np.log(np.exp(param_max) - np.exp(param_min)) + param_val #flat prior in log space
        else:
            logprior += -np.log(param_max - param_min) #flat prior in linear space

    return logprior


### calculate log_posterior, to be called by mcmc
def log_post_prof(theta_trial, curves, info): #function to return log_posterior. this is the function the script calls for each theta_trial
    ncurves = info["ncurves"] #number of curves
    beta_num = info["beta_num"]
    psi_name = info["psi_name"]

    logprior = log_prior_prof(theta_trial, info) #calculate priors for given step
    if not np.isfinite(logprior): #confirm theta_trial are within prior bounds
        return -np.inf, np.zeros((beta_num, ncurves)), np.zeros((beta_num, ncurves)), np.full(ncurves, np.nan) #has to match shape of return/blobs
    
    #calculate log_likelihood for each curve and sum
    ll_curves = np.empty(ncurves) #shape (ncurves,)
    betas = np.empty((beta_num, ncurves)) #shape (beta_num, ncurves)
    beta_var = np.empty((beta_num, ncurves)) #same (beta_num, ncurves)
    psi_val = np.empty(ncurves) #shape (ncurves,)

    for i, curve in enumerate(curves):
        theta_trial_curve = theta_trial.copy()
        tmidi = curve["tmidi"] #retrieve tmidi from curve, this is the time of inferior conjunction of the current trial lightcurve

        if psi_name != "tmid": #if psi is not tmid, adjust tmid trial to current curve's tmidi. note, if psi is tmid, it is adjusted in the golden search function
            alpha_names = info["alpha_names"] #retrieve ordered theta list from info
            tmid_idx = alpha_names.index("tmid") #find index of tmid in alpha_names
            theta_trial_curve[tmid_idx] += tmidi  #adjust tmid trial to current curve's tmidi

        ll_curves[i], betas[:, i], beta_var[:, i], psi_val[i] = golden_search_prof(theta_trial_curve, curve, info) #golden search finds ll of alpha_trial and corresponding best psi, betas and beta_var

    ll_total = np.sum(ll_curves)

    logposterior = logprior + ll_total #calculate log_posterior for given alpha_trial
    return logposterior, betas, beta_var, psi_val

def run_superprofile_mcmc(curves, info): #run the prof mcmc
    #MCMC and data info
    mcmcpoints = info["mcmcpoints"]
    nwalkers = info["nwalkers_superprof"]
    ndim_superprof = info["ndim_superprof"]
    ncurves = info["ncurves"]

    #get blob dtype for the specific number of curves
    beta_num = info["beta_num"]
    blob_dtype_prof = np.dtype([
        ("betas",     np.float64,  (beta_num, ncurves)),
        ("beta_var",  np.float64,  (beta_num, ncurves)),
        ("psi_val",   np.float64,  (ncurves,)),
    ])
    
    #set walker starting positions
    theta0 = info["theta0"] #true variables to seed mcmc, defined in script.py

    psi_idx = info["psi_idx"]
    theta_seed_prof = np.delete(theta0, psi_idx) #remove psi from seed values, profile mcmc only walks over alpha parameters
        
    pos = [np.array(theta_seed_prof) + 1e-3 * np.random.randn(ndim_superprof) for i in range(nwalkers)] #starting position of walkers
    print(f"starting positions of walkers: {pos[0]}")
    print(f"Running Super Profile MCMC with {nwalkers} walkers for {mcmcpoints} points with {ndim_superprof} dimensions...")

    #run MCMC
    mp.set_start_method("spawn", force=True)
    with mp.get_context("spawn").Pool() as pool:
        sampler_superprof = emcee.EnsembleSampler(
            nwalkers, ndim_superprof, log_post_prof,
            args=(curves, info),
            blobs_dtype=blob_dtype_prof,
            pool=pool
        )
        samples_superprof = sampler_superprof.run_mcmc(pos, mcmcpoints, progress=True)

    print("Super Profile MCMC Complete")

    return sampler_superprof, samples_superprof


def extract_info_superprof(sampler_superprof, curves, info): #function to extract information from the sampler and samples, returns a dictionary of results
    theta_ordered_names = info["theta_ordered_names"]

    transit_name = info["transit_name"]

    #define basic info
    burnin = info["burnin"]
    thin = info["thin"]
    psi_name = info["psi_name"]
    psi_idx = info["psi_idx"]

    #create variable names
    alpha_names = list(info["alpha_names"])
    var_names = alpha_names


    ### Extract arviz results and save to arviz
    idata = az.from_emcee(sampler_superprof, var_names = var_names)

    #drop args so arviz can interpret file, these (mainly info, will be sent to arvix file seperately)
    for group in ["observed_data", "constant_data"]:
        if hasattr(idata, group):
            ds = getattr(idata, group)
            bad_vars = [v for v in ds.data_vars if ds[v].dtype == "O"]
            if bad_vars:
                setattr(idata, group, ds.drop_vars(bad_vars))

    idata.to_netcdf(f"arviz_{transit_name}_nonlinear.nc")



    ### Extract results

    flat_samples = sampler_superprof.get_chain(flat = True, discard = burnin, thin=thin) #get flatchain, shape is (# of points, # of theta)
    log_prob = sampler_superprof.get_log_prob(flat=True,  discard = burnin, thin=thin) #extract log prob as well, shape is (# of points)
    blobs = sampler_superprof.get_blobs(flat=True,  discard = burnin, thin=thin) #extract blobs, which keep track of non-walked over variables, must be set earlier

    #extract chains of blobs
    beta_chains = blobs["betas"] 
    var_chains = blobs["beta_var"]
    psi_samps = blobs["psi_val"]
    if psi_name.startswith("ln"): psi_samps = np.exp(psi_samps) #if psi was initially in log, put in linear space for better reference

    #convert log variables to linear space
    rho_s_idx = info["param_indices"]["ln_rho_s"] #index of rho_s in theta_ordered_names
    full_samples = np.insert(flat_samples, psi_idx, np.nan, axis = 1) #add dummy column so array follows rho_idx order
    full_lin_samples = full_samples.copy()
    full_lin_samples[:,rho_s_idx] = np.exp(full_lin_samples[:,rho_s_idx])

    #organize final samples
    alpha_samps = np.delete(full_lin_samples, psi_idx, axis = 1) #shape (n_samples, n_alpha), these are linear
    psi_samps = psi_samps #shape (n_samples, ncurves)
    beta_samps = beta_chains #shape (n_samples, beta_num, ncurves)

    #extract sample corresponding to highest likelihood
    post_max_idx = np.argmax(log_prob) #index of the highest log prob, or best parameters
    best_betas_samp = beta_chains[post_max_idx, :, :] #betas corresponding to highest likelihood
    var_mcmc = var_chains[post_max_idx, :] #beta_var corresponding to highest likelihood
    best_alpha_samp = alpha_samps[post_max_idx] #the theta corresponding to the highest log prob
    best_psi_samp = psi_samps[post_max_idx] #the best psi value corresponding to the highest log prob
    post_max = log_prob[post_max_idx] #the highest log prob
    

    ### calculate predicted lightcurve of best parameters
    ncurves = info["ncurves"] #number of curves
    best_flux_prof = np.empty(ncurves, dtype=object) #shape (ncurves, empty)
    for i, curve in enumerate(curves):
        trel = curve["trel"] #retrieve relative time from curve, this is the relative time of the current trial lightcurve
        best_betas = best_betas_samp[:, i] #extract betas for current curve
        t = curve["t"]
        flux_data = curve["flux"]
        best_batman_params = np.insert(best_alpha_samp, psi_idx, best_psi_samp[i]) #insert best psi into theta_mcmc

        if psi_name != "tmid":
            tmid_idx_full = theta_ordered_names.index("tmid")
            best_batman_params[tmid_idx_full] += curve["tmidi"]
        else:
            best_batman_params[psi_idx] += curve["tmidi"]
        best_batman_params[rho_s_idx] = np.log(best_batman_params[rho_s_idx])  #convert back to ln as batman expects ln_rho_s

        best_batman_results = batman_func_prof(best_batman_params, curve, info) #calculate best batman results
        poly_base_best = np.polynomial.polynomial.polyval(trel, best_betas)
        best_flux_prof[i]= best_batman_results * poly_base_best
        plt.figure(figsize=(6, 4))
        plt.scatter(t, flux_data, s=5, alpha=0.6, label="Data")
        plt.plot(t, best_flux_prof[i], lw=1.5, label="Best MCMC model")
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title(f"Curve {i}: data vs best model")
        plt.legend()
        plt.tight_layout()


        


    

    #compute percentiles
    q_low, q_high = 2.5, 97.5
    percentiles = np.percentile(alpha_samps, [q_low, q_high], axis=0).T  # shape (n_params, 2)
    #create ranges for plotting, add padding
    ranges = []
    for lo, hi in percentiles:
        width = hi - lo
        pad   = 0.05 * width #padding of 5% of the width
        ranges.append((lo - pad, hi + pad))

    nonlinear_mcmc_results = {             #build dictionary of profile_likelihood mcmc results
        "nonlinear_logprob" : log_prob,
        "nonlinear_alpha_samples" : alpha_samps, #these are the profile likelihood mcmc walked over samples, no psi, no betas
        "nonlinear_betas" : beta_samps,
        "nonlinear_var" : var_chains, 
        "nonlinear_psi_samps" : psi_samps,
        "nonlinear_best_alpha_samp" : best_alpha_samp, #best sample in linear space
        "nonlinear_post_max" : post_max,
        "nonlinear_percentiles" : percentiles,
        "nonlinear_ranges" : ranges,
        "nonlinear_best_psi_samp": best_psi_samp,
        "best_flux_prof" : best_flux_prof, #best flux of the profile likelihood mcmc

    }

    return nonlinear_mcmc_results

