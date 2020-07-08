import matplotlib.pyplot as plt
import numpy             as np
import batman            as bat
import pprint            as pp
import emcee
import warnings

class TransitFit():
    def __init__(self, phase, lc, err, airmass, crpa, common_mode_array, \
                 psf_width, psf_width_ratio, psf_yposition, shift_position,
                 x1_array, prior_string, priorsigmas):
        """Return a TransitFit object with set phase and light curve photometry.
            All inputs are 1-D arrays and should have the same size as 'lc'
            (except the last two)! If any of
            these is not aplicable, replace with a 1-D array of size len(lc)
            filles with ones. All arguments are required, but phase, lc and err
            cannot be replaced with dummy arrays of 1s.
        Args:
            phase (float array): The 1-D array of observed orbital phases/JD times.
            lc (float array): The 1-D array of observed light curve (normalized to unity).
            err (float array): The 1-D array of uncertainties for
                                each parametric point (normalized
                                to by the same factor as lc).
            airmass (float array): 1-D array of airmasses of the observation.
            crpa (float array): 1-D array of CRPA angles of the observation.
            common_mode_array (float array): 1-D array of the common mode
                    decorrelation function.
            psf_width (float array): 1-D array of the target PSF widths
                    during the observation.
            psf_width_ratio (float array): 1-D array of the ratio of target
                    over reference star PSF width.
            psf_yposition (float array): 1-D array of the spectral centroid
                    position in the cross-dispersion direction.
            shift_position (float array): 1-D array of the shift of the
                    spectrum in the dispersion direction.
            x1_array (float array): 1-D array of a generic decorrelation function.
            prior_string (string array): 1-D array of size number of
                    parameters, describing the shape of the value:
                    't' for tophat,
                    'g' for Gaussian
            priorsigmas (float array): 1-D array of size number of parameters.
                    If the corresponding element of prior_string is 'g', then
                    the value in priorsigmas is the standard deviation of the
                    Gaussian prior (the peak location is the initial guess value).
                    If the orresponding element of prior_string is 't', then
                    the value here does nothing. In this case,
                    prior is determined by the max and min values allowed
                    for this parameter by running set_par_range().
        """
        self.phase = phase
        self.lc = lc
        self.err= err
        self.airmass = airmass
        self.crpa = crpa
        self.common_mode_array = common_mode_array
        self.psf_width = psf_width
        self.psf_width_ratio = psf_width_ratio
        self.psf_yposition = psf_yposition
        self.shift_position = shift_position
        self.x1_array = x1_array ## generic additional correction.
        self.prior_string = prior_string ## prior shape, 'gaus', 'tophat'
        self.priorsigmas = priorsigmas

    def set_par_range(self, mins, maxs, frozen):
        """ Set the minimum and maximu allowed values for the fitted parameters.
        """
        self.parmins = mins
        self.parmaxs = maxs
        self.pars_frozen = frozen
        return
    def set_par_names(self, par_names):
        self.parnames = par_names
        return

    def mcmc_fit(self, nwalkers, nsteps, threads, live_dangerously, bat_pars, fac, tmod, xsquared, slope, yintercept, \
                 airmass_coeff, crpa_coeff, crpa_ang0, common_mode_strength, \
                 psf_width_strength, psf_widthratio_strength, psf_yposition_strength, shift_strength, \
                 x1_strength, \
                 a0_exp_coeff, a1_exp_coeff, tau1_exp_coeff, a2_exp_coeff, tau2_exp_coeff ):
        """
        Perform the actual fit using the emcee package.
        Return a tuple with the emcee output blobs, chain, and flatchain.
        Args:
            nwalkers (int): number of emcee MCMC walkers.
            nsteps (int): number of MCMC steps per walker.
            threads (int): number of CPU threads to use (should be <= number of CPUs)
            live_dangerously (bool): emcee's live_dangerously parameter.
            bat_pars (object): a batman object with the transit parameters.
            fac (float): value for the batman fac parameter.
            tmod (object): batman transit object.
            ----- initial guesses for the transit and systematics free parameters: ------
            xsquared, slope, yintercept (floats): a*phase**2 + b*phase + yintercept
                terms potentially caused by systematics on the transit LC.
            airmass_coeff (float): airmass multiplicative amplitude parameter.
            crpa_coeff, crpa_ang0 (floats): crpa correction amplitude par and phase angle parameter.
            common_mode_strength (float): common mode correction amplitude parameter.
            psf_width_strength (float): PSF width correction amplitude parameter.
            psf_widthratio_strength (float): amplitude parameter for the ratios
                of target over reference PSF width.
            psf_yposition_strength (float): amplitude parameter for the location
                of the spectrum in cross-dispersion direction
            shift_strength (float): amplitude parameter for the shift
                of the spectrum in dispersion direction
            x1_strength (float): the amplitude parameter for a generic
                correction function.
            a0_exp_coeff, a1_exp_coeff, tau1_exp_coeff,
            a2_exp_coeff, tau2_exp_coeff (floats): Agol et al. 2010 double exponential
                LC ramp parameters.
        Returns:
            array: the emcee blobs output, an array of all sampled models.
            array: the emcee chain output, a chain of all attempted
                parameter vectors.
            array: the emcee flatchain output. A flattened chain.
        """
        # set the initial guess array for emcee:
        self.ig = ig = np.asarray([bat_pars.t0, bat_pars.rp, bat_pars.a, bat_pars.u[0], bat_pars.u[1], bat_pars.u[2], bat_pars.u[3], \
                                   xsquared, slope, yintercept, \
                                   airmass_coeff, crpa_coeff, crpa_ang0, common_mode_strength, \
                                   psf_width_strength, psf_widthratio_strength, psf_yposition_strength, shift_strength,\
                                   x1_strength, \
                                   a0_exp_coeff, a1_exp_coeff, tau1_exp_coeff, a2_exp_coeff, tau2_exp_coeff])
        self.fac = fac # batman scale factor for the step size <-- keep fixed to speed up calculations; if not supplied, batman will try to optimize it, which slows down computation.
        if np.any(self.ig > self.parmaxs):
            raise ValueError('Initial guess parameters over allowed bound: ' + str([self.parnames[i] for i in np.where(self.ig > self.parmaxs)[0]]) + ' = ' + str([self.ig[i] for i in np.where(self.ig > self.parmaxs)[0]]) )
        if np.any(self.ig < self.parmins):
            raise ValueError('Initial guess parameters under allowed bound: ' + str([self.parnames[i] for i in np.where(self.ig < self.parmins)[0]]) + ' = ' + str([self.ig[i] for i in np.where(self.ig < self.parmins)[0]]) )

        ndim = len(ig)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=(self.phase, self.lc, self.err, bat_pars, tmod), threads=threads, live_dangerously=live_dangerously, a=2)
        ig_ball = [0.001 * ig * np.random.randn( ndim ) + ig for i in range(nwalkers)] # a list of arrays...
        # But, make sure that the frozen parameters are frozen to one value:
        if max(self.pars_frozen) > 0.0:
            frozen = np.where(np.asarray(self.pars_frozen) > 0) # for some reason a tuple with empty second entry
            for indx in frozen[0].tolist():                     # for some reason a tuple with empty second entry
                for jndx in np.arange(nwalkers):
                    ig_ball[jndx][indx] = ig[indx]

        sampler.run_mcmc(ig_ball, nsteps)
        print('EMCEE :Mean acceptance fraction: {0:.3f}'.format(np.mean(sampler.acceptance_fraction)))
        return sampler.blobs, sampler.chain, sampler.flatchain

    def lnlike(self, theta, x, y, yerr, bat_pars, tmod):
        """ The log-likelihood calculator function for emcee.
            Contains the actual transit+systematics model generation
            Args:
                theta (float array): the parametervalues array.
                x (float array): array of times.
                y (float array): array of flux measurements (the LC measurements)
                yerr (float array): uncertainties for y.
                bat_pars (object): batman parameters object.
                tmod (object): batman transit object.
            Returns:
                float: log-likelihood of the trial fit.
                array: the trial fit model light curve.
        """
        bat_pars.t0 = theta[0]
        bat_pars.rp = theta[1]
        bat_pars.a  = theta[2]
        bat_pars.u[0] = theta[3]
        bat_pars.u[1] = theta[4]
        bat_pars.u[2] = theta[5]
        bat_pars.u[3] = theta[6]
        tmod = bat.TransitModel(bat_pars, x, fac = self.fac)
        trial_fit = tmod.light_curve(bat_pars) * \
                    (theta[9] + theta[8] * (x - x[0]) + theta[7]*((x-x[0])**2)) * \
                    (1 + theta[10] * self.airmass) * \
                    (1 + theta[11] * np.cos(self.crpa + theta[12])) * \
                    (1 + theta[14] * self.psf_width) * \
                    (1 + theta[15] * (self.psf_width_ratio - 1.0) ) * \
                    (1 + theta[16] * self.psf_yposition ) * \
                    (1 + theta[17] * self.shift_position ) * \
                    (1 + theta[18] * self.x1_array ) * \
                    ( theta[13] * self.common_mode_array ) * \
                    ( theta[19] - theta[20] * np.exp(-(x-x[0])/theta[21]) - theta[22] * np.exp(-(x-x[0])/theta[23]) )

        pdf = np.sum( -0.5*((trial_fit - y)**2/yerr**2) -0.5*np.log(2*np.pi*np.square(yerr)) )
        return pdf, trial_fit

    # We can now set informative (as opposed to flat) priors for EMCEE
    def lnprior(self, theta):
        """ The log-prior likelihood calculator function for emcee.
            Args:
                theta (float array): the parameter values array.
            Returns:
                float: log-prior probability of the trial fit.
        """
        return_lp = 0.0
        # ensure parameter limits are kept and frozen parameters are frozen.
        for t in np.arange(self.ig.size):
            if self.prior_string[t] == 't': ## for tophat
                if self.pars_frozen[t] == 1 and theta[t] != self.ig[t]: ## if the parameter is frozen and the trial value is not the starting value
                    print('ERROR: theta[',t,'] is frozen and the trial value is not the starting value')
                    return_lp = return_lp -np.inf
                elif theta[t] < self.parmins[t] or theta[t] > self.parmaxs[t]: ## if the parameter isn't frozen, but the trial value is outside limits
                    return_lp = return_lp -np.inf
                else:
                    return_lp = return_lp + 0.0 # 0, because log of 1 is 0.
            if self.prior_string[t] == 'g':   ## for gaussian -- get informative prior value
                if (self.pars_frozen[t] == 1 and theta[t] != self.ig[t]) or (theta[t] < self.parmins[t] or theta[t] > self.parmaxs[t]):
                    return_lp = return_lp -np.inf
                elif self.pars_frozen[t] == 1 and theta[t] == self.ig[t]:
                    return_lp = return_lp + 0.0
                else:
                    #gaussian prior on a
                    mu = self.ig[t]
                    sigma = self.priorsigmas[t]
                    return_lp = return_lp + (np.log(1.0/( sigma * np.sqrt(2*np.pi) ) )  -  0.5*(theta[t] - mu)**2/sigma**2)
        return return_lp

    def lnprob(self, theta, x, y, yerr, bat_pars, tmod):
        """ The log-probability (combining the prior and the frequentist
            likelihoods) calculator function for emcee.
            Args:
                theta (float array): the parameter values array.
                x (float array): array of times.
                y (float array): array of flux measurements (the LC measurements)
                yerr (float array): uncertainties for y.
                bat_pars (object): batman parameters object.
                tmod (object): batman transit object.
            Returns:
                float: log-prior + log-likelihood of the trial fit.
                array: the trial fit model light curve.

            Returns:
                float: log-prior probability of the trial fit.
        """
        lp = self.lnprior(theta)
        if not np.isfinite(lp): # if the prior is infinitely small, return -infinity without calculating the model to save computational time.
            return -np.inf, list(0.0 for xx in x)

        likelihood, trial_fit = self.lnlike(theta, x, y, yerr, bat_pars, tmod)
        return lp + likelihood, trial_fit # the second argument goes in the blobs variable and saves all trial fits this way.
