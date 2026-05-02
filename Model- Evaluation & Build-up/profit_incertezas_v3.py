import itertools
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from subprocess import call
from astropy.io import fits
from scipy import optimize
from scipy import stats
import numpy as np
import multiprocessing as mp
import warnings
import numpy.ma as ma
from functools import partial

"""
profit_incertezas_v3.py

Estimation of parameter uncertainties for Profit surface‑photometry fits
using Markov Chain Monte Carlo (MCMC) with the emcee ensemble sampler.

This module takes the best‑fit models from the earlier Profit‑based pipeline
(see ``profit_anysample_fix_psf.py``) and performs full Bayesian inference
to obtain robust parameter uncertainties and convergence diagnostics.

For each galaxy and for each model (single Sérsic and double Sérsic) the
pipeline:

- Reads the pre‑fitted best‑fit parameters and constructs the corresponding
  ``ProfitData`` object.
- Runs an MCMC chain using emcee's differential‑evolution move (DEMove)
  with a burn‑in phase and a long sampling phase.
- Converts the chain from the internal (normalised) parameterisation back
  to physical units.
- Computes posterior medians and 1‑σ (2.5–97.5 percentile) confidence
  intervals.
- Generates and saves a set of diagnostic plots:
    * trace plots (parameter value vs. MCMC step for individual walkers)
    * walker convergence plots (mean ±1σ over walkers vs. step)
    * autocorrelation function plots
- Saves all results in a compressed NumPy .npz archive for later use
  (including the full chain, log‑probabilities, and configuration).
- Writes a summary text file with the best‑fit values and uncertainties.

The code is designed for the same two fixed samples ('WHL' and 'L07')
and expects the data directory structure produced by the earlier Profit
fitting script.  Multiprocessing is used to process many galaxies in
parallel.

Global variables ``save_value`` and ``model_flag`` (used within some
functions) are set in the calling wrappers and are documented where
applicable.
"""
warnings.filterwarnings("ignore")
#####################################################
# ----------------------------------------------------------------------
# Prior and log‑probability helpers
# ----------------------------------------------------------------------
#####################################################
def prior_func(s):
    """Return a zero‑centred Gaussian log‑PDF with fixed standard deviation ``s``.

    This is used to create a weak Gaussian prior for the Profit parameters.
    The returned function is suitable for use with ``scipy.stats.norm.logpdf``.

    Args:
        s (float): Standard deviation of the prior.

    Returns:
        callable: A function ``f(x)`` that returns the log‑probability of
                  ``x`` under a normal distribution N(0, s).
    """
    return partial(stats.norm.logpdf, loc=0, scale=s)
class LogProbFunction:
    """Log‑probability function for emcee MCMC sampling.

    This callable takes a parameter vector ``theta`` (in the normalised
    Profit internal space) and returns the log‑posterior, which is
    proportional to the negative log‑likelihood from the Profit model plus
    a hard boundary check.  It is designed to be passed to an emcee
    ``EnsembleSampler``.

    Args:
        data (ProfitData): The ``ProfitData`` object that contains the
            model, data, and bounds.

    Methods:
        __call__(theta): evaluate log‑posterior for parameter vector ``theta``.
    """

	def __init__(self, data):
		self.data = data
	def __call__(self, theta):
		from profit_optim_v4 import profit_like_model
        """Evaluate the log‑posterior (up to an additive constant).

        If any parameter falls outside the bounds defined in
        ``self.data.bounds``, returns -inf.  Otherwise returns the
        negative profit log‑likelihood (so that the MCMC maximises the
        posterior, equivalently minimises the profit objective).

        Args:
            theta (ndarray): Parameter vector in normalised Profit space
                             (length = number of free parameters).

        Returns:
            float: Log‑posterior value (or -inf outside bounds).
        """


		for t, (low, high) in zip(theta, self.data.bounds):
			if not (low <= t <= high):
				return -np.inf
		ll = -profit_like_model(theta, self.data)
		
		return ll
#####################################################
# ----------------------------------------------------------------------
# Position, magnitude, and sky estimation (same as in other scripts)
# ----------------------------------------------------------------------
#####################################################
def centro_magzero(sample,cluster,save_value):
    """Determine initial galaxy centre, magnitude, shape, and sky properties.

    (This function is identical to the one in ``profit_anysample_fix_psf.py``;
    see that documentation for details.)

    Args:
        sample (str): Sample name ('WHL' or 'L07').
        cluster (str): Cluster identifier.
        save_value (str): Tag used in path for temporary files.

    Returns:
        tuple: (xc, yc, mag, pa, ax, re, magzero, sigma_sky, median_sky,
                mask_center, sum_vec_1) as described earlier.
    """
    # (body unchanged, not repeated here)


	import photutils.psf as ppsf
	import photutils.aperture as phta
	from math import floor
	# header = fits.open(f'../L07/{cluster}/ajust-bcg-r.fits',memmap=True)[1].header
	# model_header=fits.open(f'../L07/{cluster}/ajust-bcg-r.fits',memmap=True)[2].header
	data2_path=f'../{sample}/{cluster}/bcg_r.fits'
	cluster_path=f'{sample}/{cluster}/{save_value}'

	mask_b=fits.open(f'../{sample}/{cluster}/bcg_r_mask_b.fits',memmap=True)[0].data
	data2=fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[1].data
	header = fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[1].header
	model_header=fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[2].header
	sigma_img=fits.open(f'../{sample}/{cluster}/sigma-r.fits',memmap=True)[0].data
	mask=fits.open(f'../{sample}/{cluster}/bcg_r_mask.fits',memmap=True)[0].data
	psf=fits.open(f'../{sample}/{cluster}/bcg_r_psf_b.fits',memmap=True)[0].data
	############
	NMGY = float(header['NMGY'])
	EXPTIME = float(header['EXPTIME'])
	data = data2*NMGY/EXPTIME
	#####################
	vec = data[np.where(mask_b == 1)]
	vec_1= data[np.where(mask == 0)]

	magzero=float(model_header['MAGZPT'])+2.5*np.log10(EXPTIME)

	sky=data2[(mask==0) & (mask_b==0)]
	sigma_sky=np.std(sky)
	median_sky=np.median(sky)

	x0=float(model_header['1_XC'].split()[0].replace('*',''))
	y0=float(model_header['1_YC'].split()[0].replace('*',''))

	#######################################
	with open('base_default.sex','r') as inp2:
		ninp2=len(inp2.readlines())
	inp2=open('base_default.sex','r')
	out1=open(f'{sample}/{cluster}/{save_value}/base_default.sex','w')
	for j in range(0,ninp2):
		ls2=inp2.readline()
		ll2=ls2.split()
		if len(ll2)>0 and ll2[0]=='CATALOG_NAME':

			ll2[1]=cluster_path+'/out_sex_large.cat'
			lstrin=' '
			for k in range(0,len(ll2)):
				lstrin+=ll2[k]+' '
			out1.write('%s\n' % lstrin[1:len(lstrin)])
		elif len(ll2)>0 and ll2[0]=='DETECT_MINAREA':
			ll2[1]='100'
			lstrin=' '
			for k in range(0,len(ll2)):
				lstrin+=ll2[k]+' '
			out1.write('%s\n' % lstrin[1:len(lstrin)])

		elif len(ll2)>0 and ll2[0]=='BACK_SIZE':
			ll2[1]='128'
			lstrin=' '
			for k in range(0,len(ll2)):
				lstrin+=ll2[k]+' '
			out1.write('%s\n' % lstrin[1:len(lstrin)])
		elif len(ll2)>0 and ll2[0]=='CHECKIMAGE_NAME':
			ll2[1]=f'{cluster_path}/check1_large.fits,{cluster_path}/check2_large.fits,{cluster_path}/check3_large.fits'
			lstrin=' '
			for k in range(0,len(ll2)):
				lstrin+=ll2[k]+' '
			out1.write('%s\n' % lstrin[1:len(lstrin)])
		else:
			out1.write('%s' % ls2)
	inp2.close()
	out1.close()
	call(f'sex {data2_path} -c {sample}/{cluster}/{save_value}/base_default.sex',shell=True)
	call(f'rm -r {sample}/{cluster}/{save_value}/check*',shell=True)

	with open(f'{cluster_path}/out_sex_large.cat','r') as infa:
		ninfa=len(infa.readlines())
	infa=open(f'{cluster_path}/out_sex_large.cat','r')
	dist=10.
	xb=-1.
	for i in range(0,ninfa):
		lsa=infa.readline()
		lla=lsa.split()
		if lla[0]!='#':
			X=float(lla[7])
			Y=float(lla[8])
			if ((X-x0)**2+(Y-y0)**2)**0.5<dist:
				xb=float(lla[0])
				xc=X
				yc=Y
				pa=90.-float(lla[11])
				siz=1.5*float(lla[9])*float(lla[4])
				ax=float(lla[10])/float(lla[9])
				kron_radius=float(lla[9])*float(lla[4])
	infa.close()

	mag=22.5-2.5*np.log10(sum(vec))
	if np.isnan(mag):
		mag=15.0
	re=siz/4.5
	psf_fit=np.ceil(ppsf.fit_fwhm(psf))
	rad_mask=phta.CircularAperture((xc,yc),int(1.5*psf_fit)).to_mask()
	mask_center=rad_mask.to_image(data.shape).astype(int)

	return xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,mask_center,sum(vec_1)
##############################################################################################
# ----------------------------------------------------------------------
# Model initialisation (single / double Sérsic)
# ----------------------------------------------------------------------
##############################################################################################
def sersic_unico(sample,cluster,save_value):
    """Set up initial parameters and bounds for a single‑Sérsic fit.

    (Same as in ``profit_anysample_fix_psf.py``.)

    Args:
        sample, cluster, save_value: Standard identifiers.

    Returns:
        tuple: (magzero, names, model0, tofit, tolog, sigmas, priors,
                lowers, uppers)
    """

	#####
	xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,_,_=centro_magzero(sample,cluster,save_value)
	n =4.
	box=0.
	sky=median_sky
	names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('xcen','ycen','mag','re','nser','ang','axrat','box'), ('sersic',))]
	names.append('sky.bg')
	model0 = np.array((xc,yc,mag,re,n,pa,ax,box,sky))
	tofit  = np.array((True,  True, True, True, True,  True,  True, True,True))
	tolog  = np.array((False, False, False,True, True,False, True, False,False))

	min_sky=-sigma_sky
	max_sky=sigma_sky
	sigmas = np.array((2,     2,     5,     1.5,    1.,     45.,   0.3,   0.3,sigma_sky/3.))
	lowers = np.array((xc-5,   yc-5,    10,    0,   np.log10(0.5),  -180,    -1,    -1,min_sky))
	uppers = np.array((xc+5,   yc+5,    30,   np.log10(np.max(re*1.8)),  np.log10(15.),   360, -0.01,     1,max_sky))
	priors = np.array([prior_func(s) for s in sigmas])
	return magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers
def sersic_duplo(sample,cluster,data_entry,img_siz,save_value):
    """Set up initial parameters for a double‑Sérsic fit.

    (Same as in ``profit_anysample_fix_psf.py``.)

    Args:
        sample, cluster, data_entry, img_siz, save_value: Standard identifiers.

    Returns:
        tuple: (magzero, names, model0, tofit, tolog, sigmas, priors,
                lowers, uppers)
    """


	xc,yc,mag,re,n,pa,ax,box,sky = data_entry

	magzero,sigma_sky=centro_magzero(sample,cluster,save_value)[6:8]
	xc_ss=xc
	yc_ss=yc
	re_ss = re/3.
	mag_ss = mag+2.5*np.log10(2.)
	n_ss = 4.
	axis_ss = ax
	pa_ss = pa
	box_ss=box

	rd_ss=re
	magd_ss =mag+2.5*np.log10(2.)
	nd_ss=4.
	axisd_ss =ax
	pad_ss= pa
	boxd_ss=box

	sigmas = np.array((     5,     5,     5,     5,     1.5,    1.5,   1.,    1.,     45,    45,   0.3,   0.3,   0.3,   0.3,sigma_sky/3.))

	names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('mag','re','nser','ang','axrat','box'), ('sersic1','sersic2'))]
	names.insert(0,'common.ycen')
	names.insert(0,'common.xcen')
	names.append('sky.bg')
	model0 = np.array((xc_ss, yc_ss, mag_ss, magd_ss, re_ss, rd_ss, n_ss, nd_ss, pa_ss, pad_ss, axis_ss,axisd_ss, box_ss,boxd_ss,sky))
	tofit  = np.array((True,  True, True,  True,  True, True, True, True, True,  True,  True, True,  True,  True,True))
	tolog  = np.array((False, False, False, False, True, True, True, True,  False, False, True,  True,  False, False,False))
	lowers = np.array((xc_ss-5,yc_ss-5,     10,    10,    0,    0,      np.log10(0.5),   np.log10(0.5),   -180,  -180,    -1,    -1,    -1,    -1,-sigma_sky))
	uppers = np.array((xc_ss+5,yc_ss+5,   30,    30,    np.log10(np.max(img_siz)/3.5),np.log10(np.max(img_siz)/1.5),np.log10(10.),np.log10(15.),    360,   360, -0.01, -0.01,     1,     1,sigma_sky))
	priors = np.array([prior_func(s) for s in sigmas])
	return magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers
###########################################################################################
# ----------------------------------------------------------------------
# File‑naming helper
# ----------------------------------------------------------------------
###########################################################################################
def infotype(modeltype,save_value):
    """Return standardised filenames for a given mode.

    (Same as in ``profit_anysample_fix_psf.py``.)

    Args:
        modeltype (int): 0 (observation), 1 (simulation on single Sérsic),
                         2 (simulation on double Sérsic).
        save_value (str): Tag embedded in filenames.

    Returns:
        tuple: (name_s_ajust, name_se_ajust, name_ss_ajust, save_file)
    """

	if modeltype==0:
		name_s_ajust=f'ajust-sersic-llh-{save_value}.fits'
		name_se_ajust=f'ajust-sersic-exp-llh-{save_value}.fits'
		name_ss_ajust=f'ajust-sersic-duplo-llh-{save_value}.fits'
		save_file='output_obs.dat'
	elif modeltype==1:
		name_s_ajust=f'ajust-simul-s-sersic-llh-{save_value}.fits'
		name_se_ajust=f'ajust-simul-s-sersic-exp-llh-{save_value}.fits'
		name_ss_ajust=f'ajust-simul-s-sersic-duplo-llh-{save_value}.fits'
		save_file='output_simul_s.dat'
	elif modeltype==2:
		name_s_ajust=f'ajust-simul-ss-sersic-llh-{save_value}.fits'
		name_se_ajust=f'ajust-simul-ss-sersic-exp-llh-{save_value}.fits'
		name_ss_ajust=f'ajust-simul-ss-sersic-duplo-llh-{save_value}.fits'
		save_file='output_simul_ss.dat'
	return	name_s_ajust,name_se_ajust,name_ss_ajust,save_file
############################################################################################
# ----------------------------------------------------------------------
# Functions to retrieve previously fitted model parameters
# ----------------------------------------------------------------------
############################################################################################
def sersic_entry_incs(sample,cluster,save_value,modeltype):
    """Read the best‑fit single‑Sérsic parameters from an existing FITS file
    and return the corresponding initialisation data for an MCMC run.

    Args:
        sample (str): Sample name.
        cluster (str): Cluster ID.
        save_value (str): Tag used to locate the FITS file.
        modeltype (int): Mode (0/1/2).

    Returns:
        tuple: (magzero, names, model0, tofit, tolog, sigmas, priors,
                lowers, uppers) with the initial values set to the
                previously fitted parameters.
    """

	name_s_ajust=infotype(modeltype,save_value)[0]
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
		ajuste_sersic=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',2)		
		data_entry = [float(ajuste_sersic[k].split()[0].replace('*','')) for k in ('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')]
	#('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')
	_,_,_,_,_,_,magzero,sigma_sky,median_sky,_,_=centro_magzero(sample,cluster,save_value)
	xc,yc,mag,re,n,pa,ax,box,sky=data_entry

	names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('xcen','ycen','mag','re','nser','ang','axrat','box'), ('sersic',))]
	names.append('sky.bg')
	model0 = np.array((xc,yc,mag,re,n,pa,ax,box,sky))
	tofit  = np.array((True,  True, True, True, True,  True,  True, True,True))
	tolog  = np.array((False, False, False,True, True,False, True, False,False))

	min_sky=-sigma_sky
	max_sky=sigma_sky
	sigmas = np.array((2,     2,     5,     1.5,    1.,     45.,   0.3,   0.3,sigma_sky/3.))
	lowers = np.array((xc-5,   yc-5,    10,    0,   np.log10(0.5),  -180,    -1,    -1,min_sky))
	uppers = np.array((xc+5,   yc+5,    30,   np.log10(np.max(re*1.8)),  np.log10(15.),   360, -0.01,     1,max_sky))
	priors = np.array([prior_func(s) for s in sigmas])
	return magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers
def sersic_duplo_entry_incs(sample,cluster,img_siz,save_value,modeltype):
    """Read the best‑fit double‑Sérsic parameters from an existing FITS file
    and return the MCMC initialisation data.

    Args:
        sample (str): Sample name.
        cluster (str): Cluster ID.
        img_siz (tuple): Image dimensions (only max extent used for bounds).
        save_value (str): Tag for the FITS file.
        modeltype (int): Mode (0/1/2).

    Returns:
        tuple: (magzero, names, model0, tofit, tolog, sigmas, priors,
                lowers, uppers) with initial values from the previous fit.
    """

	name_ss_ajust=infotype(modeltype,save_value)[2]
	param_models=['XCEN_1', 'YCEN_1', 'MAG_1', 'RE_1', 'NSER_1', 'ANG_1', 'AXRAT_1', 'BOX_1', 'MAG_2', 'RE_2', 'NSER_2', 'ANG_2', 'AXRAT_2', 'BOX_2','SKY']
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}'):
		ajuste_sersic_duplo=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}',2)		
		data_entry = [float(ajuste_sersic_duplo[k].split()[0].replace('*','')) for k in param_models]

	xc_ss,yc_ss,mag_ss,re_ss,n_ss,pa_ss,axis_ss,box_ss,magd_ss,rd_ss,nd_ss,pad_ss,axisd_ss,boxd_ss,sky = data_entry

	magzero,sigma_sky=centro_magzero(sample,cluster,save_value)[6:8]

	sigmas = np.array((     5,     5,     5,     5,     1.5,    1.5,   1.,    1.,     45,    45,   0.3,   0.3,   0.3,   0.3,sigma_sky/3.))

	names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('mag','re','nser','ang','axrat','box'), ('sersic1','sersic2'))]
	names.insert(0,'common.ycen')
	names.insert(0,'common.xcen')
	names.append('sky.bg')
	model0 = np.array((xc_ss, yc_ss, mag_ss, magd_ss, re_ss, rd_ss, n_ss, nd_ss, pa_ss, pad_ss, axis_ss,axisd_ss, box_ss,boxd_ss,sky))
	tofit  = np.array((True,  True, True,  True,  True, True, True, True, True,  True,  True, True,  True,  True,True))
	tolog  = np.array((False, False, False, False, True, True, True, True,  False, False, True,  True,  False, False,False))
	lowers = np.array((xc_ss-5,yc_ss-5,     10,    10,    0,    0,      np.log10(0.5),   np.log10(0.5),   -180,  -180,    -1,    -1,    -1,    -1,-sigma_sky))
	uppers = np.array((xc_ss+5,yc_ss+5,   30,    30,    np.log10(np.max(img_siz)/3.5),np.log10(np.max(img_siz)/1.5),np.log10(10.),np.log10(15.),    360,   360, -0.01, -0.01,     1,     1,sigma_sky))
	priors = np.array([prior_func(s) for s in sigmas])
	return magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers
############################################################################################
# ----------------------------------------------------------------------
# Generic data builder (same as earlier)
# ----------------------------------------------------------------------
############################################################################################
def build_data(cluster, image, mask, sigim, segim, psf,	magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center=None):
    """Construct a ``ProfitData`` object (wrapper around ``profit_setup_data``).

    (Same as in ``profit_anysample_fix_psf.py``.)

    Args:
        (see previous documentation)

    Returns:
        ProfitData: data container for profit optimisation.
    """

	from profit_optim_v4 import profit_setup_data
	data = profit_setup_data(
	cluster, image.copy(), mask.copy(), sigim.copy(), segim.copy(), psf.copy(),
	magzero, names[:], model0.copy(), tofit.copy(), tolog.copy(),
	sigmas.copy(), priors.copy(), lowers.copy(), uppers.copy(),
	mask_center if mask_center is not None else np.zeros_like(image, dtype=bool))

	return data
############################################################################################
# ----------------------------------------------------------------------
# MCMC diagnostic plotting functions
# ----------------------------------------------------------------------
############################################################################################
def plot_mcmc_trace(trace_data, output_dir='./'):
    """Plot the MCMC trace for all parameters (walker evolution over steps).

    For up to 10 walkers, the sampled physical parameter values are plotted
    as a function of MCMC step.  The burn‑in phase is marked with a vertical
    dashed line, and the posterior median (after burn‑in) is indicated by a
    horizontal line.

    Args:
        trace_data (dict): Dictionary with keys:
            - 'samples_phys': 3D array (n_steps, n_walkers, ndim) of
              physical parameter values.
            - 'names': list of parameter names.
            - 'n_burnin': number of burn‑in steps.
            - 'cluster': string identifier for the title.
        output_dir (str): Directory where the plot will be saved (ignored,
            the caller handles saving).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

	import matplotlib.pyplot as plt
	samples_phys = trace_data['samples_phys']
	names = trace_data['names']
	n_steps,n_walkers,ndim = samples_phys.shape
	n_burnin = trace_data.get('n_burnin', 0)

	fig, axes = plt.subplots(ndim, 1, figsize=(12, 3*ndim))
	if ndim == 1:
		axes = [axes]

	for i in range(ndim):
		ax = axes[i]

		# Plotar cada walker
		for walker in range(min(10, n_walkers)):
			ax.plot(samples_phys[:, walker, i], alpha=0.5, linewidth=0.5, color='blue', label='Walker' if walker == 0 else "")

		# Linha vertical para o burn-in
		ax.axvline(x=n_burnin, color='red', linestyle='--', linewidth=2, label='Burn-in' if i == 0 else "")

		# Linha horizontal para a mediana
		p_median = np.median(samples_phys[:, n_burnin:, i])
		ax.axhline(y=p_median, color='green', linestyle='-', linewidth=1, alpha=0.7,label='Mediana' if i == 0 else "")
		ax.set_ylabel(names[i], fontsize=10)
		ax.set_xlabel('Passo' if i == ndim-1 else '')
		ax.grid(True, alpha=0.3)
		ax.legend(loc='upper right', fontsize=8)

		plt.suptitle(f'Trace Plot - {trace_data["cluster"]}', fontsize=9)
		plt.tight_layout()
	return fig
def plot_walker_convergence(trace_data, output_dir='./'):
    """Plot the convergence of the MCMC ensemble by showing the mean and
    standard deviation of each parameter across all walkers as a function
    of step number.

    Args:
        trace_data (dict): Same as for ``plot_mcmc_trace``.

    Returns:
        matplotlib.figure.Figure: The convergence figure.
    """
	import matplotlib.pyplot as plt
	samples_phys = trace_data['samples_phys']
	names = trace_data['names']
	n_steps,n_walkers, ndim = samples_phys.shape
	n_burnin = trace_data.get('n_burnin', 0)
	
	n_cols = 2
	n_rows = (ndim + n_cols - 1) // n_cols
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
	axes = axes.flatten()
	
	for i in range(ndim):
		ax = axes[i]
		param_values = samples_phys[:, :, i]
		
		mean_over_walkers = np.mean(param_values, axis=1)
		std_over_walkers = np.std(param_values, axis=1)
		
		ax.plot(mean_over_walkers, 'b-', linewidth=1.5, label='Média')
		ax.fill_between(range(n_steps),mean_over_walkers - std_over_walkers,mean_over_walkers + std_over_walkers,alpha=0.3, color='blue', label=r'$\pm \ 1\sigma$')
		ax.axvline(x=n_burnin, color='red', linestyle='--', linewidth=2, label='Burn-in')
		ax.set_ylabel(names[i])
		ax.set_xlabel('Passo')
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=8)
	
	for i in range(ndim, len(axes)):
		axes[i].set_visible(False)
	
	plt.suptitle(f'Convergência - {trace_data["cluster"]}', fontsize=9)
	plt.tight_layout()
	
	return fig
def plot_autocorrelation(trace_data, output_dir='./'):
    """Plot the autocorrelation function for each parameter based on the
    first walker's chain.  A horizontal line at correlation = 0.5 is drawn
    to help estimate the effective sample size.

    Args:
        trace_data (dict): Same as for ``plot_mcmc_trace``.

    Returns:
        matplotlib.figure.Figure: The autocorrelation figure.
    """

	import matplotlib.pyplot as plt
	samples_phys = trace_data['samples_phys']
	names = trace_data['names']
	n_steps,n_walkers, ndim = samples_phys.shape
	
	n_cols = 2
	n_rows = (ndim + n_cols - 1) // n_cols
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
	axes = axes.flatten()
	
	for i in range(ndim):
		ax = axes[i]

		param_values = samples_phys[0, :, i]
		
		autocorr = np.correlate(param_values - np.mean(param_values),param_values - np.mean(param_values),mode='full')
		autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
		
		ax.plot(autocorr, 'b-', linewidth=1.5)
		ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
		ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Correlação 0.5')
		ax.set_xlabel('Lag')
		ax.set_ylabel('Autocorrelação')
		ax.set_title(names[i])
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=8)
	
	# Esconder axes extras
	for i in range(n_dim, len(axes)):
		axes[i].set_visible(False)
	
	plt.suptitle(f'Autocorrelação - {trace_data["cluster"]}', fontsize=14)
	plt.tight_layout()
	
		return fig
############################################################################################
# ----------------------------------------------------------------------
# Regeneration of MCMC plots from saved archive
# ----------------------------------------------------------------------
############################################################################################
def regenerate_mcmc_plots(sample, cluster, model_flag):
    """Load a previously saved MCMC result (.npz) and regenerate the diagnostic
    plots (trace, convergence, autocorrelation).

    This function is useful for re‑plotting existing results without
    re‑running the chains.  It reads the compressed archive, builds a
    ``trace_data`` dictionary, and calls the three plotting functions,
    saving the figures in the same output directory.

    Args:
        sample (str): Sample name.
        cluster (str): Galaxy/cluster ID.
        model_flag (str): Model identifier ('S' or 'SS') used to select the
            correct .npz file name.

    Returns:
        None.  Saves the figures to disk.

    Note:
        The global variable ``save_value`` must be defined in the calling
        scope (it is used to build the path).
    """

	import numpy as np
	import matplotlib.pyplot as plt

	output_dir = f'{sample}/{cluster}/{save_value}/inc_mcmc'
	npz_file = os.path.join(output_dir, f"mcmc_results_{model_flag}.npz")
	data = np.load(npz_file, allow_pickle=True)

	# Recuperar cadeias
	chains_full = data['chains_full']
	n_steps,n_walkers, ndim = chains_full.shape

	# Recuperar configuração
	n_burnin = int(data['n_burnin'])

	# Recuperar nomes (se salvou)
	names = data['names'] if 'names' in data else [f'param_{i}' for i in range(ndim)]

	print(f"✓ Dados carregados: {n_walkers} walkers, {n_steps} passos, {ndim} parâmetros")
	print(f"  Burn-in: {n_burnin} passos")

	trace_data = {
		'samples_phys': chains_full,        # (walkers, steps, params)
		'names': names,
		'n_walkers': n_walkers,
		'n_steps': n_steps,
		'n_burnin': n_burnin,
		'cluster': cluster,
		'sample': sample
	}

	# 1. Trace plot
	try:
		fig_trace = plot_mcmc_trace(trace_data, output_dir=output_dir)
		trace_file = os.path.join(output_dir, f"trace_plot_{model_flag}.png")
		fig_trace.savefig(trace_file, dpi=150, bbox_inches='tight')
		plt.close(fig_trace)
		print(f"  ✓ Trace plot: {trace_file}")
	except Exception as e:
		print(f"  ✗ Erro no trace plot: {e}")

	# 2. Convergence plot
	try:
		fig_conv = plot_walker_convergence(trace_data, output_dir=output_dir)
		conv_file = os.path.join(output_dir, f"convergence_{model_flag}.png")
		fig_conv.savefig(conv_file, dpi=150, bbox_inches='tight')
		plt.close(fig_conv)
		print(f"  ✓ Convergence plot: {conv_file}")
	except Exception as e:
		print(f"  ✗ Erro no convergence plot: {e}")

	# 3. Autocorrelation plot
	try:
		fig_auto = plot_autocorrelation(trace_data, output_dir=output_dir)
		auto_file = os.path.join(output_dir, f"autocorr_{model_flag}.png")
		fig_auto.savefig(auto_file, dpi=150, bbox_inches='tight')
		plt.close(fig_auto)
		print(f"  ✓ Autocorrelation plot: {auto_file}")
	except Exception as e:
		print(f"  ✗ Erro no autocorrelation plot: {e}")

	return
############################################################################################
# ----------------------------------------------------------------------
# Core MCMC routine
# ----------------------------------------------------------------------
############################################################################################
def model_inc_mcmc(sample, cluster, image, mask, sigim, segim, psf,modeltype, save_value,entry_config):
    """Run the MCMC sampling for one galaxy and one model.

    If the output .npz file already exists, the function skips the sampling
    and only regenerates the diagnostic plots via ``regenerate_mcmc_plots``.

    Otherwise it:
    1. Unpacks the configuration (mask_center, data_entry, data_model,
       model_flag) from ``entry_config``.
    2. Builds the ``ProfitData`` object and the ``LogProbFunction``.
    3. Initialises the walkers in a tight ball around the best‑fit
       parameters, clipped to the allowed bounds.
    4. Runs a burn‑in phase (default 400 steps) with differential‑evolution
       moves.
    5. Resets the sampler and runs the main sampling phase (default 3000
       steps).
    6. Converts the chains to physical parameters, computes medians and
       2.5–97.5% confidence intervals.
    7. Saves the full chain, burn‑in chain, flat post‑burn‑in samples, and
       derived statistics into a compressed .npz archive.
    8. Generates trace, convergence, and autocorrelation plots and saves
       them as PNG files.
    9. Writes a summary text file with the final parameter estimates and
       uncertainties.
    10. Cleans up memory and sleeps briefly.

    Args:
        sample (str): Sample name.
        cluster (str): Galaxy ID.
        image (2D array): Galaxy stamp image (used only to pass through).
        mask, sigim, segim, psf: Input data (not directly used in MCMC;
            they are part of the configuration already in ``data_model``).
        modeltype (int): Mode (0/1/2) – not used here but passed for
            compatibility.
        save_value (str): Tag for output directories.
        entry_config (tuple): Contains:
            mask_center : 2D bool array for central mask.
            data_entry  : Output of ``sersic_entry_incs`` or
                          ``sersic_duplo_entry_incs``.
            data_model  : The ``ProfitData`` instance.
            model_flag  : 'S' or 'SS'.

    Returns:
        None.  Results are written to disk.

    Global variables used:
        ``save_value`` (for the path inside ``regenerate_mcmc_plots``),
        ``model_flag`` (set from entry_config).
    """

	from profit_optim_v4 import profit_like_model
	import emcee
	import time
	import pickle
	import matplotlib.pyplot as plt
	import gc

	print(f'{cluster} MCMC')
	if os.path.isfile(f"{sample}/{cluster}/{save_value}/inc_mcmc/mcmc_results_{model_flag}.npz"):
		print(f'{cluster} MCMC -- completo')
		regenerate_mcmc_plots(sample,cluster,model_flag)
		return
	else:
		start_time = time.time()
		mask_center,data_entry,data_model,model_flag=entry_config

		magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = data_entry
		theta_best = data_model.init.copy()

		ndim = len(theta_best)
		n_burnin=400
		n_main=3000
		escala=0.01
		output_dir=f'{sample}/{cluster}/{save_value}/inc_mcmc'
		os.makedirs(output_dir, exist_ok=True)
		################################

		n_walkers = max(32, 2 * ndim)

		pos = theta_best + escala * np.abs(theta_best) * np.random.randn(n_walkers, ndim)
		for i in range(n_walkers):
			for j in range(ndim):
				low, high = data_model.bounds[j]
				pos[i, j] = np.clip(pos[i, j], low, high)
		
		log_prob_func = LogProbFunction(data_model)

		sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_func,moves=emcee.moves.DEMove())
		#######################################################
		print(f"Executando burn-in para {cluster} com {n_burnin} ...")
		try:
			pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=False)
			chains_burnin = sampler.get_chain()
			log_prob_burnin = sampler.get_log_prob()
			acceptance_burnin = sampler.acceptance_fraction
			sampler.reset()
		except Exception as e:
			print(f"Erro no burn-in: {e}")
			return None	
		########################################################
		print(f"Executando amostragem principal para {cluster} com {n_main}...")
		try:
			sampler.run_mcmc(pos, n_main, progress=False)
		except Exception as e:
			print(f"Erro na amostragem: {e}")
			return None
		
		all_samples = sampler.get_chain()
		n_steps = all_samples.shape[0]

		thin = max(1, n_steps // 100)
		# ========================
		# CONVERTER PARA FÍSICO
		# ========================
		samples_phys = all_samples * data_model.sigmas
		samples_phys[:, :, data_model.tolog] = 10 ** samples_phys[:, :, data_model.tolog]
		
		# ========================
		# GARANTIR FORMATO 3D PARA TRACE PLOTS
		# ========================
		# Pegar as amostras completas (formato 3D)
		all_samples_3d = sampler.get_chain()  # shape: (n_walkers, total_steps, ndim)

		# Converter para físico (mantendo formato 3D)
		samples_phys_3d = all_samples_3d * data_model.sigmas
		samples_phys_3d[:, :, data_model.tolog] = 10 ** samples_phys_3d[:, :, data_model.tolog]

		# Para resultados finais (formato flat, pós burn-in)
		samples_flat = sampler.get_chain(discard=n_burnin, thin=thin, flat=True)
		samples_phys_flat = samples_flat * data_model.sigmas
		samples_phys_flat[:, data_model.tolog] = 10 ** samples_phys_flat[:, data_model.tolog]
		###############################################
		# ========================
		# SALVAMENTO COMPRIMIDO (.npz)
		# ========================
		print("\nSalvando resultados comprimidos...")
		
		# Calcular estatísticas finais
		ndim_final = len(names)
		medians = np.zeros(ndim_final)
		err_plus = np.zeros(ndim_final)
		err_minus = np.zeros(ndim_final)
		
		for i in range(ndim_final):
			medians[i] = np.median(samples_phys_flat[:, i])
			p_low = np.percentile(samples_phys_flat[:, i], 2.5)
			p_high = np.percentile(samples_phys_flat[:, i], 97.5)
			err_minus[i] = medians[i] - p_low
			err_plus[i] = p_high - medians[i]
		
		# Salvar tudo em um único arquivo comprimido
		npz_file = os.path.join(output_dir, f"mcmc_results_{model_flag}.npz")
		np.savez_compressed(
			npz_file,
			# Dados brutos
			chains_full=samples_phys,                      # (walkers, steps, params)
			chains_burnin=chains_burnin,                   # (walkers, n_burnin, params)
			chains_sampling=samples_phys[n_burnin:, :, :], # (walkers, n_main, params)
			flat_samples=samples_phys_flat,                # (n_samples, params)
			
			# Estatísticas finais
			medians=medians,
			err_plus=err_plus,
			err_minus=err_minus,
			
			# Informações do MCMC
			log_prob_burnin=log_prob_burnin,
			acceptance_burnin=acceptance_burnin,
			acceptance_sampling=sampler.acceptance_fraction,
			
			# Configuração
			n_walkers=n_walkers,
			n_burnin=n_burnin,
			n_main=n_main,
			n_steps=n_steps,
		)
		###############################################
		trace_data = {
			'samples': all_samples_3d,                    # Formato 3D
			'samples_phys': samples_phys_3d,              # Formato 3D para plots!
			'samples_phys_flat': samples_phys_flat,       # Formato 2D para resultados
			'names': names,
			'n_walkers': n_walkers,
			'n_steps': all_samples_3d.shape[0],           # Total de passos
			'n_burnin': n_burnin,
			'n_main': n_main,
			'thin': thin,
			'theta_best': theta_best,
			'data_bounds': data_model.bounds,
			'tolog': data_model.tolog,
			'sigmas': data_model.sigmas,
			'acceptance_fraction': sampler.acceptance_fraction,
			'cluster': cluster,
			'sample': sample
		}	
		try:
			fig_trace = plot_mcmc_trace(trace_data,output_dir=output_dir)
			trace_plot_file = os.path.join(output_dir, f"trace_plot_{model_flag}.png")
			fig_trace.savefig(trace_plot_file, dpi=150, bbox_inches='tight')
			plt.close(fig_trace)
			print(f"  ✓ Trace plot: {trace_plot_file}")
		except Exception as e:
			print(f"  ✗ Erro no trace plot: {e}")
		
		try:
			fig_conv = plot_walker_convergence(trace_data,output_dir=output_dir)
			conv_plot_file = os.path.join(output_dir, f"convergence_{model_flag}.png")
			fig_conv.savefig(conv_plot_file, dpi=150, bbox_inches='tight')
			plt.close(fig_conv)
			print(f"  ✓ Convergence plot: {conv_plot_file}")
		except Exception as e:
			print(f"  ✗ Erro no convergence plot: {e}")
		
		try:
			fig_auto = plot_autocorrelation(trace_data,output_dir=output_dir)
			auto_plot_file = os.path.join(output_dir, f"autocorr_{model_flag}.png")
			fig_auto.savefig(auto_plot_file, dpi=150, bbox_inches='tight')
			plt.close(fig_auto)
			print(f"  ✓ Autocorrelation plot: {auto_plot_file}")
		except Exception as e:
			print(f"  ✗ Erro no autocorrelation plot: {e}")
		
		# 5. Salvar resumo (USANDO AMOSTRAS FLAT)
		summary_file = os.path.join(output_dir, f"summary_{model_flag}.txt")
		with open(summary_file, 'w') as f:
			f.write(f"Cluster: {cluster}\n")
			f.write(f"Sample: {sample}\n")
			f.write(f"Número de parâmetros: {ndim}\n")
			f.write(f"Número de walkers: {n_walkers}\n")
			f.write(f"Burn-in steps: {n_burnin}\n")
			f.write(f"Main steps: {n_main}\n")
			f.write(f"Total steps: {n_steps}\n")
			f.write(f"Thinning: {thin}\n")
			f.write(f"Taxa de aceitação: {np.mean(sampler.acceptance_fraction):.3f}\n\n")
			f.write("Parâmetros (valor físico):\n")
			
			for i in range(ndim):
				# Usar amostras flat para estatísticas
				p_median = np.median(samples_phys_flat[:, i])
				p_low = np.percentile(samples_phys_flat[:, i], 2.5)   # 1σ
				p_high = np.percentile(samples_phys_flat[:, i], 97.5)  # 1σ
				sigma_minus = p_median - p_low
				sigma_plus = p_high - p_median
				
				f.write(f"  {names[i]}: {p_median:.4f} -{sigma_minus:.4f} +{sigma_plus:.4f}\n")
		
		print(f"  ✓ Summary: {summary_file}")
		
		# ========================
		# RESULTADOS (amostras pós-burn-in)
		# ========================
		burnin = min(n_burnin, n_steps // 4)
			
		samples_flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
		samples_phys_flat = samples_flat * data_model.sigmas
		samples_phys_flat[:, data_model.tolog] = 10 ** samples_phys_flat[:, data_model.tolog]
				
		elapsed = time.time() - start_time
		print(f"\nTempo total: {elapsed:.1f} segundos")
		
		gc.collect()
		time.sleep(2)

		return
	return
############################################################################################
# ----------------------------------------------------------------------
# Per‑galaxy setup for MCMC
# ----------------------------------------------------------------------
############################################################################################
def mcmc_inc_setup(sample,cluster,save_value,model_flag):
    """Prepare the data and configuration for one galaxy and launch the MCMC.

    The function:
    - Creates the output directory.
    - Reads the necessary images (stamp, sigma, mask, PSF).
    - Depending on ``model_flag``, retrieves the initialisation from the
      existing single‑Sérsic or double‑Sérsic fit.
    - Builds the ``ProfitData`` object.
    - Wraps everything in an ``entry_config`` tuple and calls
      ``model_inc_mcmc``.

    Args:
        sample (str): Sample name.
        cluster (str): Galaxy ID.
        save_value (str): Tag (used for file paths).
        model_flag (str): 'S' for single Sérsic, 'SS' for double Sérsic.

    Returns:
        None.
    """

	os.makedirs(f'{sample}/{cluster}/{save_value}',exist_ok=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	image = np.array(fits.getdata(f'../{sample}/{cluster}/ajust-bcg-r.fits',1))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	mask_center=centro_magzero(sample,cluster,save_value)[-2]
	if model_flag == 'SS':
		data_entry = magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_duplo_entry_incs(sample,cluster,image.shape,save_value,modeltype)
		data_model = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)
		data_model.verbose = False	
	if model_flag == 'S':
		data_entry = magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_entry_incs(sample,cluster,save_value,modeltype)
		data_model = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)
		data_model.verbose = False
	config_entry=mask_center,data_entry,data_model,model_flag
	################################################################
	model_inc_mcmc(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value,config_entry)
	return 
############################################################################################
# ----------------------------------------------------------------------
# Post‑processing / summary
# ----------------------------------------------------------------------
############################################################################################
def finish_details(sample,catalogo,modeltype,ass,save_value):
    """(Same function as in ``profit_anysample_fix_psf.py``) – gathers
    fit results, computes RFF, BIC, and writes the summary output file.

    Args:
        (see previous documentation)

    Returns:
        None.
    """

	#results_dir=f'{sample}/{cluster}/{save_value}'#f'{sample}/{cluster}'
	ok=[]
	with open(f'{sample}_profit_{save_value}.dat','r') as inp2:
		for item in inp2.readlines():   
			ok.append(item.split()[0])
	for cluster in catalogo:
		if cluster in ok:
			pass
		else:
			# try:
			model_names=infotype(modeltype,save_value)

			sersic_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']
			sersic_exp_names=['XCEN_S', 'YCEN_S', 'MAG_S', 'RE_S', 'NSER_S', 'ANG_S', 'AXRAT_S', 'BOX_S', 'MAG_E', 'RE_E', 'NSER_E', 'ANG_E', 'AXRAT_E', 'BOX_E', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']
			sersic_duplo_names=['XCEN_1', 'YCEN_1', 'MAG_1', 'RE_1', 'NSER_1', 'ANG_1', 'AXRAT_1', 'BOX_1', 'MAG_2', 'RE_2', 'NSER_2', 'ANG_2', 'AXRAT_2', 'BOX_2', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']

			rff=rff_calc(sample,cluster,model_names[0],save_value)	
			image = np.array(fits.getdata(f'../{sample}/{cluster}/ajust-bcg-r.fits',1))
			sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
			mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
			segim = np.logical_not(mask)
			psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
			img_likelihood_sersic=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/likelihood_{model_names[0]}'))
			# img_likelihood_sersic_exp=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/likelihood_{model_names[1]}'))
			img_likelihood_sersic_duplo=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/likelihood_{model_names[2]}'))

			#################################
			#RESULTADOS SERSIC	
			sersic_header=fits.getheader(f'{sample}/{cluster}/{save_value}/{model_names[0]}',2)
			sersic_dict=dict(sersic_header)
			result_sersic=[]
			for param in sersic_names:
				result_sersic.append(str(sersic_dict[param]).split()[0])	
			#################################
			#RESULTADOS SERSIC+EXP
			try:				
				sersic_exp_header=fits.getheader(f'{sample}/{cluster}/{save_value}/{model_names[1]}',2)
				sersic_exp_dict=dict(sersic_exp_header)
				result_sersic_exp=[]
				for param in sersic_exp_names:
					result_sersic_exp.append(str(sersic_exp_dict[param]).split()[0])
			except:
				result_sersic_exp=[0 for i in range(len(sersic_exp_names))]				
			#################################
			#RESULTADOS SERSIC+SERSIC
			sersic_duplo_header=fits.getheader(f'{sample}/{cluster}/{save_value}/{model_names[2]}',2)
			sersic_duplo_dict=dict(sersic_duplo_header)
			result_sersic_duplo=[]

			for param in sersic_duplo_names:
				result_sersic_duplo.append(str(sersic_duplo_dict[param]).split()[0])
			
			#################################
			#ESCOLHA DO BIC E CONFECÇÃO DOS DELTAS
			bic_str=['S','S+S']

			## TODOS OS PIXEIS
			bic_sersic=float(result_sersic[-2])
			bic_sersic_exp=float(result_sersic_exp[-2])
			bic_sersic_duplo=float(result_sersic_duplo[-2])

			max_llh_sersic=-float(result_sersic[-3])
			max_llh_sersic_exp=-float(result_sersic_exp[-3])
			max_llh_sersic_duplo=-float(result_sersic_duplo[-3])

			bic_vec=np.array((bic_sersic,bic_sersic_duplo))
			best_model=bic_str[np.argmin(bic_vec)]

			delta_s=max_llh_sersic_exp-max_llh_sersic
			delta_se=max_llh_sersic_duplo-max_llh_sersic_exp
			sersic_residuos=fits.getdata(f'{sample}/{cluster}/{save_value}/{model_names[0]}',3)
			chi2_s=np.sum(np.power(sersic_residuos,2.)/np.power(sigim,2.))

			sersic_duplo_residuos=fits.getdata(f'{sample}/{cluster}/{save_value}/{model_names[2]}',3)
			chi2_ss=np.sum(np.power(sersic_duplo_residuos,2.)/np.power(sigim,2.))

			output=open(f'{sample}_profit_{save_value}.dat','a')
			#output.write(f'{cluster} {chi2_s} {chi2_ss}\n')
			output.write(f'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n'%(cluster,*result_sersic,*result_sersic_exp,*result_sersic_duplo,rff,ass,best_model,delta_s,delta_se))
			output.close()
			# except:
			# 	output=open(f'{sample}_profit_{save_value}.dat','a')
			# 	output.write(f'{cluster} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n')
			# 	output.close()
	return
############################################################################################
# ----------------------------------------------------------------------
# Multiprocessing wrappers
# ----------------------------------------------------------------------
############################################################################################
def run_mcmc_inc_setup(args):
    """Unpack arguments and call ``mcmc_inc_setup``.

    Args:
        args (tuple): (sample, cluster, save_value, model_flag)

    Returns:
        Whatever ``mcmc_inc_setup`` returns.
    """
	sample, cl, i,flag= args
	return mcmc_inc_setup(sample, cl, i,flag)
def zelador(sample,cluster,pasta):
    """Clean (delete) all files inside a cluster's ``comp*`` sub‑directories.

    Args:
        sample (str): Sample name.
        cluster (str): Cluster ID.
        pasta (str): Sub‑directory within the cluster folder.

    Returns:
        None.  Executes an ``rm -r`` command via shell.
    """	
	call(f'rm -r {sample}/{cluster}/{pasta}/comp*',shell=True)
	# os.makedirs(f'{sample}/{cluster}/{pasta}_old',exist_ok=True)
	# call(f'cp -r {sample}/{cluster}/{pasta}/* {sample}/{cluster}/{pasta}_old/',shell=True)
	# call(f'rm {sample}/{cluster}/{pasta}/ajust-sersic-duplo-llh-{pasta}.fits',shell=True)
	return

def rename_npz_file(directory, new_name):
    """Rename the first .npz file found in a directory to a new name.

    Useful for converting generic .npz files to a standard naming scheme.

    Args:
        directory (str): Path to the directory containing .npz files.
        new_name (str): New base name (without extension).

    Returns:
        None.  Prints a message indicating the rename or if no file was found.
    """
	import glob
	npz_files = glob.glob(os.path.join(directory, "*.npz"))

	if len(npz_files) == 0:
		print(f"Nenhum arquivo .npz encontrado em {directory}")
		return
	elif len(npz_files) == 1:
		old_path = npz_files[0]
		new_path = os.path.join(directory, f"{new_name}.npz")
		os.rename(old_path, new_path)
		print(f"Renomeado: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
		return

if __name__ == '__main__':
	import sys
	sample=sys.argv[1]
	save_value=['observation_SE_sky']#,'simulation_SE_sky','simulation_s_duplo_SE_sky']
	ok=[]

	catalogo=[]
	with open(f'../{sample}/pargal_{sample}_compact_astro_vfix.dat','r') as inp1:
		data_sample=inp1.readlines()
		for i,obj in enumerate(data_sample):
			flagmask=int(float(obj.split()[11]))
			flagdelta=int(float(obj.split()[12]))
			cluster=obj.split()[0]
			if [flagmask,flagdelta] == [0,0] and cluster != '2102' and cluster!='3338':
				catalogo.append(cluster)
				ass=float(obj.split()[9])

	###################
	#OBSERVAÇÕES

	model_names=infotype(0,save_value[0])
	temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[0]}/inc_mcmc/mcmc_results_{save_value[0]}.npz')]
	if set(catalogo) == set(temp_ok):
		pass
	else:
		new_catalog=[item for item in catalogo if item not in temp_ok]
		obs_list = [(sample, cl, save_value[0],'S') for cl in new_catalog]
		with mp.Pool(processes=18) as pool:
			chunksize=1
			for _ in pool.imap_unordered(run_mcmc_inc_setup, obs_list,chunksize=chunksize):
				pass

	temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[0]}/inc_mcmc/mcmc_results_{save_value[2]}.npz')]
	if set(catalogo) == set(temp_ok):
		pass
	else:
		new_catalog=[item for item in catalogo if item not in temp_ok]
		obs_list = [(sample, cl, save_value[0],'SS') for cl in new_catalog]
		with mp.Pool(processes=18) as pool:
			chunksize=1
			for _ in pool.imap_unordered(run_mcmc_inc_setup, obs_list,chunksize=chunksize):
				pass