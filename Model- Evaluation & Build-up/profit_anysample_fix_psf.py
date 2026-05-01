"""
profit_anysample_fix_psf.py

Automated 2D surface‑photometry fitting of BCG (Brightest Cluster Galaxy) stamps.

This module manages the full pipeline for fitting single‑Sérsic, Sérsic+Exponential,
and double‑Sérsic (Sérsic+Sérsic) models to galaxy images using the ``profit``
optimisation library.  It is designed to work with two pre‑defined samples
('WHL' and 'L07') and supports three modes:

- **Observation** (``modeltype=0``): fit real data.
- **Simulation on single‑Sérsic** (``modeltype=1``): generate a mock galaxy from
  the best single‑Sérsic model, add noise, and re‑fit.
- **Simulation on double‑Sérsic** (``modeltype=2``): generate a mock galaxy from
  the best double‑Sérsic model and re‑fit.

The script also computes the Residual Flux Fraction (RFF) and performs basic
model selection (BIC).  Parallelisation is achieved via ``multiprocessing``.

**Global dependency:** The module‑level variable ``sample`` must be set (e.g.,
via command‑line argument) before calling any function that constructs file
paths.  Many functions rely on external files with a fixed naming convention.
"""

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

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Callback functions for optimisation monitoring
# ----------------------------------------------------------------------

def make_callback(cluster, data, save_value, model):
	"""Return a callback that logs likelihood and parameters during optimisation.

	The callback is invoked by ``scipy.optimize.minimize`` at each iteration.
	It writes the current negative log‑likelihood and the parameter vector to
	text files inside the cluster's output directory.

	Args:
		cluster    (str): Cluster identifier.
		data       (ProfitData): The ``profit`` data object (not used inside
					 the callback, but required by the signature of the caller).
		save_value (str): Tag used to construct the output file names.
		model      (str): Model name (e.g., 'sersic', 'sersic_exp').

	Returns:
		function: A closure with signature ``callback(params_iter)``.

	Note:
		The returned callback depends on the global ``sample`` variable.
	"""

	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		with open(f'{sample}/{cluster}/{save_value}/hist_likelihood_{model}_{save_value}.dat','a') as hist_likelihood:
			hist_likelihood.write(f'{ll}\n')
		with open(f'{sample}/{cluster}/{save_value}/hist_params_{model}_{save_value}.dat','a') as hist_params:
			hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_s(cluster,data,save_value,model):
	"""Same as ``make_callback``, but dedicated to single‑Sérsic simulations.

	This duplicate exists for historical reasons; its behaviour is identical
	to ``make_callback``.

	Args:
		cluster, data, save_value, model: See ``make_callback``.

	Returns:
		function: Logging callback.
	"""

	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		with open(f'{sample}/{cluster}/{save_value}/hist_likelihood_{model}_{save_value}.dat','a') as hist_likelihood:
			hist_likelihood.write(f'{ll}\n')
		with open(f'{sample}/{cluster}/{save_value}/hist_params_{model}_{save_value}.dat','a') as hist_params:
			hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_ss(cluster,data,save_value,model):
	"""Same as ``make_callback``, but dedicated to double‑Sérsic simulations.

	Args:
		cluster, data, save_value, model: See ``make_callback``.

	Returns:
		function: Logging callback.
	"""


	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		with open(f'{sample}/{cluster}/{save_value}/hist_likelihood_{model}_{save_value}.dat','a') as hist_likelihood:
			hist_likelihood.write(f'{ll}\n')
		with open(f'{sample}/{cluster}/{save_value}/hist_params_{model}_{save_value}.dat','a') as hist_params:
			hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn

def make_callback_simul_se(cluster,data,save_value,model):
	"""Same as ``make_callback``, but writes to files with a ``simul_se`` tag.

	This is used specifically for the Sérsic+Exponential model during
	double‑Sérsic simulations.

	Args:
		cluster, data, save_value, model: See ``make_callback``.

	Returns:
		function: Logging callback.
	"""


	hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_simul_se_{model}_{save_value}.dat','a')
	hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_simul_se_{model}_{save_value}.dat','a')
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		hist_likelihood.write(f'{ll}\n')
		hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
#####################################################
# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def rff_calc(sample,cluster,figname,save_value):
	"""Compute the Residual Flux Fraction (RFF) from a fitted model image.

	RFF is defined as the total absolute residual flux inside the galaxy
	mask, after subtracting 0.8× the sky RMS per pixel to account for noise
	fluctuations, divided by the total galaxy flux.

	Uses:
	- HDU 1 of the FITS file ``figname`` as the galaxy image.
	- HDU 3 of the same file as the model residual image.
	- Background mask ``bcg_r_mask.fits`` and bulge mask ``bcg_r_mask_b.fits``.

	Args:
		sample     (str): Sample name ('WHL' or 'L07').
		cluster    (str): Cluster identifier.
		figname    (str): Name of the FITS file containing model and residual.
		save_value (str): Tag used as part of the path inside the cluster directory.

	Returns:
		float: Residual Flux Fraction.

	Note:
		Assumes the FITS files are located under
		``{sample}/{cluster}/{save_value}/`` and ``../{sample}/{cluster}/``.
	"""


	

	ajust1 = fits.getdata(f'{sample}/{cluster}/{save_value}/{figname}',1)
	ajust3 = fits.getdata(f'{sample}/{cluster}/{save_value}/{figname}',3)

	mask = fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')
	mask_b = fits.getdata(f'../{sample}/{cluster}/bcg_r_mask_b.fits')

	##############################################################
	#CALCULO DO RFF
	
	sbk=np.std(ajust1[np.where((mask_b == 0) & (mask==0))])
	nn2=len(ajust1[np.where((mask_b == 1) & (mask==0))])
	xy = np.sum(np.absolute(ajust3[np.where((mask_b == 1) & (mask==0))]))
	xn = np.sum(ajust1[np.where((mask_b == 1) & (mask==0))])

	rff=(xy-0.8*sbk*nn2)/xn
	return rff
def prior_func(s):
	"""Return a zero‑centred normal log‑PDF with fixed standard deviation ``s``.

	This is used to create a weak Gaussian prior for ``profit`` parameters.

	Args:
		s (float): Standard deviation of the prior.

	Returns:
		function: A callable ``norm_with_fixed_sigma(x)`` that evaluates the
		log‑probability of ``x`` under N(0, s).
	"""


	def norm_with_fixed_sigma(x):
		return stats.norm.logpdf(x, 0, s)
	return norm_with_fixed_sigma
def centro_magzero(sample,cluster,save_value,psf):
	"""Determine initial galaxy centre, magnitude, shape, and sky properties.

	This function:
	1. Converts the background‑subtracted image from nano‑maggies to flux units.
	2. Generates a SExtractor configuration file and runs SExtractor on the
	   original stamp to find the brightest object.
	3. Estimates initial values for magnitude, effective radius, position angle,
	   axial ratio, and sky background.
	4. Creates a circular mask (diameter 1.5 × PSF FWHM) around the centre to
	   exclude the central pixel region during fitting.

	Args:
		sample     (str): Sample name.
		cluster    (str): Cluster identifier.
		save_value (str): Path tag for intermediate files.
		psf        (2D ndarray): PSF image used to measure the FWHM.

	Returns:
		tuple:
			xc, yc            (float): Fitted centre coordinates (from SExtractor).
			mag               (float): Initial apparent magnitude.
			pa                (float): Position angle (deg; 0 = north, E of N).
			ax                (float): Axial ratio (b/a).
			re                (float): Estimated effective radius in arcsec.
			magzero           (float): MAGZPT + 2.5 log10(EXPTIME).
			sigma_sky         (float): Sky RMS (standard deviation).
			median_sky        (float): Median sky level.
			mask_center       (2D int array): Central mask (0 = good, 1 = masked).
			sum_vec_1         (float): Sum of pixel values inside the galaxy mask.

	Note:
		Writes a temporary SExtractor configuration and catalogue, then
		deletes the check images.  Relies on ``base_default.sex`` in the
		current working directory.
	"""


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
	# psf=fits.open(f'../{sample}/{cluster}/bcg_r_psf_b.fits',memmap=True)[0].data
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
# Model initialisation functions
# ----------------------------------------------------------------------
def sersic_unico(sample,cluster,save_value,psf):
	"""Set up the initial parameters and bounds for a single‑Sérsic fit.

	Calls ``centro_magzero`` to obtain initial guesses and then constructs
	the parameter vector (xcen, ycen, mag, re, n, ang, axrat, box, sky)
	with reasonable sigmas, lower, and upper bounds.  The Sérsic index ``n``
	is started at 4.0 and boxiness at 0.0.

	Args:
		sample, cluster, save_value, psf: See ``centro_magzero``.

	Returns:
		tuple:
			magzero (float): Magnitude zero‑point.
			names   (list of str): Parameter names in ``profit`` format.
			model0  (ndarray): Initial parameter values.
			tofit   (ndarray of bool): Which parameters are free.
			tolog   (ndarray of bool): Which parameters are fitted in log10.
			sigmas  (ndarray): Prior sigmas.
			priors  (list of callables): Prior functions.
			lowers  (ndarray): Lower bounds.
			uppers  (ndarray): Upper bounds.
	"""


	#####
	xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,_,_=centro_magzero(sample,cluster,save_value,psf)
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
def sersic_exp(sample,cluster,data_entry,img_siz):
	"""Set up initial parameters for a Sérsic + Exponential fit.

	The initial values are derived from a previously determined Sérsic
	solution ``data_entry``, with the exponential component assumed to have
	half the effective radius and a magnitude offset.

	Args:
		sample      (str): Sample name.
		cluster     (str): Cluster identifier (only used to obtain sky RMS).
		data_entry  (array_like): 9‑element vector [xc, yc, mag, re, n,
								  pa, axrat, box, sky] from a Sérsic fit.
		img_siz     (int or tuple): Image dimensions (only max extent used
								   for upper bound of effective radius).

	Returns:
		tuple: (names, model0, tofit, tolog, sigmas, priors, lowers, uppers)
			   analogous to ``sersic_unico`` but for the two‑component model.
	"""



	xc,yc,mag,re,n,pa,ax,box,sky = data_entry
	sigma_sky=centro_magzero(sample,cluster)[7]

	xc_exp=xc
	yc_exp=yc
	re_exp = re/3.
	mag_exp = mag+2.*np.log10(2.)
	n_exp = 4.
	axis_exp = ax 
	pa_exp = pa
	box_exp=box

	rs_exp=2.*re
	magd_exp =mag+3.*np.log10(2.)
	axisd_exp =ax
	pad_exp= pa
	boxd_exp=box

	
	sigmas = np.array((     5,     5,     5,     5,     1.5,    1.5,   1.,    1.,     45,    45,   0.3,   0.3,   0.3,   0.3,sigma_sky/3.))

	names  = ['%s.%s' % (profile, prop) for prop,profile in itertools.product(('mag','re','nser','ang','axrat','box'), ('sersic1','sersic2'))]
	names.insert(0,'common.ycen')
	names.insert(0,'common.xcen')
	names.append('sky.bg')

	model0 = np.array((xc_exp, yc_exp, mag_exp, magd_exp, re_exp, rs_exp, n_exp, 1.0, pa_exp, pad_exp, axis_exp,axisd_exp, box_exp,boxd_exp,sky))
	tofit  = np.array((True,  True, True,  True,  True, True, True, False, True,  True,  True, True,  True,  True, True))
	tolog  = np.array((False, False, False, False, True, True, True, False,  False, False, True,  True,  False, False,False))
	lowers = np.array((xc_exp-5,yc_exp-5,     10,    10,    0,    0,      np.log10(0.5),   np.log10(0.5),   -180,  -180,    -1,    -1,    -1,    -1, -sigma_sky))
	uppers = np.array((xc_exp+5,yc_exp+5,   30,    30,np.log10(np.max(img_siz)),np.log10(np.max(img_siz)),np.log10(10.),np.log10(15.),    360,   360, -0.01, -0.01,     1,     1,sigma_sky))
	priors = np.array([prior_func(s) for s in sigmas])

	return names, model0, tofit, tolog, sigmas, priors, lowers, uppers
def sersic_duplo(sample,cluster,data_entry,img_siz,save_value,psf):
	"""Set up initial parameters for a double‑Sérsic (Sérsic+Sérsic) fit.

	The two components are initialised from the single‑Sérsic values in
	``data_entry``, with the second component initially having a smaller
	effective radius and similar brightness.

	Args:
		sample, cluster, data_entry, img_siz, save_value, psf:
			See ``sersic_exp``.  ``save_value`` and ``psf`` are passed to
			``centro_magzero`` to recompute sky RMS.

	Returns:
		tuple: (magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers)
	"""



	xc,yc,mag,re,n,pa,ax,box,sky = data_entry

	magzero,sigma_sky=centro_magzero(sample,cluster,save_value,psf)[6:8]
	xc_ss=xc
	yc_ss=yc
	re_ss = re/3.
	mag_ss = mag+2.5*np.log10(2.)
	n_ss = 4.
	axis_ss = ax
	pa_ss = pa
	box_ss=box

	rd_ss=re/1.8
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
	"""Return standardised filenames and output file name for a given mode.

	Args:
		modeltype  (int): 0 (observation), 1 (simulation on single‑Sérsic),
						  or 2 (simulation on double‑Sérsic).
		save_value (str): A tag that is embedded in the filenames.

	Returns:
		tuple:
			name_s_ajust   (str): FITS file name for the single‑Sérsic fit.
			name_se_ajust  (str): FITS file name for the Sérsic+Exp fit.
			name_ss_ajust  (str): FITS file name for the double‑Sérsic fit.
			save_file      (str): Name of the summary output file
								 ('output_obs.dat', etc.).
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
###########################################################################################
# ----------------------------------------------------------------------
# Fitting core
# ----------------------------------------------------------------------
###########################################################################################
def build_data(cluster, image, mask, sigim, segim, psf,	magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center=None):
	"""Wrap the ``profit_setup_data`` function to build a ``ProfitData`` object.

	Args:
		cluster    (str): Cluster ID.
		image      (2D array): Galaxy stamp image.
		mask       (2D bool array): Mask of pixels to exclude (1 = masked).
		sigim      (2D array): Noise (sigma) image.
		segim      (2D bool array): Segmentation map (1 = source).
		psf        (2D array): PSF image.
		magzero    (float): Magnitude zero‑point.
		names      (list of str): Parameter names.
		model0     (array): Initial parameter values.
		tofit      (bool array): Which parameters to vary.
		tolog      (bool array): Which parameters to fit in log space.
		sigmas     (array): Prior sigmas.
		priors     (list of callables): Prior functions.
		lowers     (array): Lower bounds.
		uppers     (array): Upper bounds.
		mask_center (2D bool array, optional): Additional central mask.
					 If None, an all‑False mask of the same shape as ``image``
					 is used.

	Returns:
		profit_optim_v4.ProfitData: Data container for ``profit`` optimisation.

	Note:
		Requires the module ``profit_optim_v4`` to be importable.
	"""


	from profit_optim_v4 import profit_setup_data

	data = profit_setup_data(
	cluster, image.copy(), mask.copy(), sigim.copy(), segim.copy(), psf.copy(),
	magzero, names[:], model0.copy(), tofit.copy(), tolog.copy(),
	sigmas.copy(), priors.copy(), lowers.copy(), uppers.copy(),
	mask_center if mask_center is not None else np.zeros_like(image, dtype=bool))

	return data
def sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value):
	"""Fit a single‑Sérsic model to the galaxy image.

	If the output FITS file already exists, the function returns immediately
	(skip already‑processed object).  Otherwise it:
	1. Initialises the model via ``sersic_unico``.
	2. Builds the ``ProfitData`` object.
	3. Minimises the negative log‑likelihood using L‑BFGS‑B.
	4. Computes BIC, cleaned log‑likelihood, and parameter uncertainties
	   from the Hessian.
	5. Saves the data, model, and residual images into a three‑extension
	   FITS file, along with all relevant metadata in the headers.

	Args:
		sample     (str): Sample name.
		cluster    (str): Cluster identifier.
		image      (2D array): Galaxy stamp image.
		mask       (2D bool array): Source mask (1 = bad).
		sigim      (2D array): Noise image.
		segim      (2D bool array): Segmentation map.
		psf        (2D array): PSF image.
		modeltype  (int): Mode (0/1/2) – determines which callback and
						  output names are used.
		save_value (str): Tag for output files.

	Returns:
		None.  The results are written to disk.
	"""


	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	print(f'{cluster} SERSIC')
	name_s_ajust=infotype(modeltype,save_value)[0]
	try:#{sample}/{cluster}
		ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',2)		
		print(f'{cluster} SERSIC FINALIZADO')
		return
	except:
		magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster,save_value,psf)
		mask_center=centro_magzero(sample,cluster,save_value,psf)[-2]
		data_sersic = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic.verbose = False
		if modeltype == 0:
			hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_sersic_{save_value}.dat','w')
			hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_sersic_{save_value}.dat','w')
			callback=make_callback(cluster,data_sersic,save_value,'sersic')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic,save_value,'sersic')
		elif modeltype == 2:
			callback=make_callback_simul_ss(cluster,data_sersic,save_value,'sersic')
		result_sersic = optimize.minimize(profit_like_model, data_sersic.init, args=(data_sersic,), method='L-BFGS-B', bounds=data_sersic.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic = to_pyprofit_image_simples(data_sersic.init, data_sersic, use_mask=True)
		
		all_params_sersic, modelim_sersic = to_pyprofit_image_simples(result_sersic.x, data_sersic, use_mask=True)

		clean_max_llh_sersic,unclean_max_llh_sersic,img_likelihood_sersic=clean_model(result_sersic.x,data_sersic)
		fits.writeto(f'{sample}/{cluster}/{save_value}/likelihood_{name_s_ajust}',img_likelihood_sersic,overwrite=True)
		####
		#FAZ A IMG DE AJUST DO PROFIT
		bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
		model_img=fits.ImageHDU(modelim_sersic,name='MODEL')
		resid_img=fits.ImageHDU(image - modelim_sersic,name='RESIDUAL')
		###

		max_llh_sersic=-result_sersic.fun
		x=[(names[i],all_params_sersic[i]) for i in range(len(names))]
		
		xc,yc=model0[:2]
		region=data_sersic.region
		n_data=np.sum(region)
		bic_sersic=len(names)*np.log(n_data) - 2*(max_llh_sersic)

		temp_cov_sersic=result_sersic.hess_inv.todense()
		inc_sersic=np.sqrt(np.diag(temp_cov_sersic))
		####1
		bcg_img.header['COMP_0'] = 'SERSIC'
		model_img.header['MODELO'] = 'SERSIC'

		for i,item in enumerate(x[:8]):
			bcg_img.header[item[0].split('.')[1]] = f'{model0[:8][i]:.4f}'
			model_img.header[item[0].split('.')[1]] = f'{item[1]:.4f} +/- {inc_sersic[:8][i]:.4f}'
		bcg_img.header['SKY']=f'{model0[-1]:.4f}'
		model_img.header['SKY']=f'{x[-1][1]:.4f}'
		model_img.header['MAX_LLH'] = max_llh_sersic
		model_img.header['BIC'] = bic_sersic
		model_img.header['CL_LLH']=clean_max_llh_sersic
		model_img.header['OLD_LLH']=unclean_max_llh_sersic
		model_img.header['FLAG']=data_sersic.check_model
		model_img.header['DOF']=data_sersic.dof

		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
		hdulist.writeto(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',overwrite=True)

	print(f'{cluster} SERSIC FINALIZADO')
	return
def sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,magzero,fit_params_sersic,modeltype,save_value):
	"""Fit a Sérsic+Exponential model, using results from a previous Sérsic fit.

	If the output FITS file already exists, the parameters are read from its
	header and the function returns early.

	Args:
		sample            (str): Sample name.
		cluster           (str): Cluster identifier.
		image, mask, sigim, segim, psf: Input data.
		magzero           (float): Magnitude zero‑point.
		fit_params_sersic (array): 9 Sérsic parameters from a prior fit.
								   If the first element is -1000, a default
								   initialisation is used (fallback).
		modeltype         (int): Mode 0/1/2.
		save_value        (str): Tag for output files.

	Returns:
		None.  Results are saved to a FITS file.

	Note:
		The Exponential component's Sérsic index is fixed to 1.0 (tofit[7]
		is False), so only the surface brightness and scale length are
		free parameters for the disk.
	"""


	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	print(f'{cluster} SERSIC + EXP')
	name_se_ajust=infotype(modeltype,save_value)[1]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_se_ajust}',2)

		sersic_exp_names=['XCEN_S', 'YCEN_S', 'MAG_S', 'RE_S', 'NSER_S', 'ANG_S', 'AXRAT_S', 'BOX_S', 'MAG_E', 'RE_E', 'NSER_E', 'ANG_E', 'AXRAT_E', 'BOX_E', 'SKY']
		sersic_exp_header=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_se_ajust}',2)
		sersic_exp_dict=dict(sersic_exp_header)
		result_sersic_exp=[]
		for param in sersic_exp_names:
			result_sersic_exp.append(str(sersic_exp_dict[param]).split()[0])
		result_sersic_exp=np.asarray(result_sersic_exp)
		print(f'{cluster} SERSIC + EXP FINALIZADO')
		return
	except:
		if fit_params_sersic[0] != -1000:
			data_entry=fit_params_sersic 
		else:
			data_entry=fit_params_sersic
		names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_exp(sample,cluster,data_entry,image.shape)
		mask_center=centro_magzero(sample,cluster)[-2]
		data_sersic_exp = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic_exp.verbose = False
		if modeltype == 0:
			hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_sersic_exp_{save_value}.dat','a')
			hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_sersic_exp_{save_value}.dat','a')
			callback=make_callback(cluster,data_sersic_exp,save_value,'sersic_exp')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic_exp,save_value,'sersic_exp')
		elif modeltype == 2:
			callback=make_callback_simul_se(cluster,data_sersic_exp,save_value,'sersic_exp')

		result_sersic_exp = optimize.minimize(profit_like_model, data_sersic_exp.init, args=(data_sersic_exp,), method='L-BFGS-B', bounds=data_sersic_exp.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic_exp = to_pyprofit_image_duplo(data_sersic_exp.init, data_sersic_exp, use_mask=True)
		all_params_sersic_exp, modelim_sersic_exp  = to_pyprofit_image_duplo(result_sersic_exp.x, data_sersic_exp, use_mask=True)

		clean_max_llh_sersic_exp,unclean_max_llh_sersic_exp,img_likelihood_sersic_exp=clean_model(result_sersic_exp.x,data_sersic_exp)
		fits.writeto(f'{sample}/{cluster}/{save_value}/likelihood_{name_se_ajust}',img_likelihood_sersic_exp,overwrite=True)

		####
		
		bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
		model_img=fits.ImageHDU(modelim_sersic_exp,name='MODEL')
		resid_img=fits.ImageHDU(image - modelim_sersic_exp,name='RESIDUAL')

		########
		max_llh_sersic_exp=-result_sersic_exp.fun
		x=[(names[i],all_params_sersic_exp[i]) for i in range(len(names))]

		xc,yc=model0[:2]
		region=data_sersic_exp.region
		n_data=np.sum(region)
		bic_sersic_exp=14.*np.log(n_data) - 2*(max_llh_sersic_exp)
		
		temp_cov_sersic_exp=result_sersic_exp.hess_inv.todense()
		inc_sersic_exp=np.sqrt(np.diag(temp_cov_sersic_exp))

		inc_sersic_exp=np.insert(inc_sersic_exp,[7],[0])


		####
		bcg_img.header['COMP_1'] = 'SERSIC'
		bcg_img.header['XCEN_S'] = f'{model0[0]:.4f}'
		bcg_img.header['YCEN_S'] = f'{model0[1]:.4f}'

		model_img.header['MODELO'] = 'SERSIC + EXP'
		model_img.header['COMP_1'] = 'SERSIC'
		model_img.header['XCEN_S'] = f'{x[0][1]:.4f} +/- {inc_sersic_exp[0]:.4f}'
		model_img.header['YCEN_S'] = f'{x[1][1]:.4f} +/- {inc_sersic_exp[1]:.4f}'
		for i,item in enumerate(x[2:-1]):
			if i%2 == 0:
				bcg_img.header[item[0].split('.')[1]+'_S'] = model0[2:][i]
				model_img.header[item[0].split('.')[1]+'_S'] = f'{item[1]:.4f} +/- {inc_sersic_exp[2:-1][i]:.4f}'
		bcg_img.header['COMP_2'] = 'EXP'
		model_img.header['COMP_2'] = 'EXP'

		for i,item in enumerate(x[2:-1]):
			if i%2 != 0:
				bcg_img.header[item[0].split('.')[1]+'_E'] = model0[2:][i]
				model_img.header[item[0].split('.')[1]+'_E'] = f'{item[1]:.4f} +/- {inc_sersic_exp[2:-1][i]:.4f}'

		bcg_img.header['SKY']=f'{model0[-1]:.4f}'
		model_img.header['SKY']=f'{x[-1][1]:.4f}'
		model_img.header['MAX_LLH'] = max_llh_sersic_exp
		model_img.header['BIC'] = bic_sersic_exp
		model_img.header['CL_LLH']=clean_max_llh_sersic_exp
		model_img.header['OLD_LLH']=unclean_max_llh_sersic_exp
		model_img.header['FLAG']=data_sersic_exp.check_model
		model_img.header['DOF']=data_sersic_exp.dof
		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
		hdulist.writeto(f'{sample}/{cluster}/{save_value}/{name_se_ajust}',overwrite=True)

		print(f'{cluster} SERSIC + EXP FINALIZADO')
	return
def sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value):
	"""Fit a double‑Sérsic (Sérsic+Sérsic) model, optionally reusing a Sérsic fit.

	If the FITS result already exists, it is skipped.  Otherwise the initial
	parameters are obtained from ``sersic_duplo``, using the single‑Sérsic
	solution if available, or defaults if not.

	Args:
		sample, cluster, image, mask, sigim, segim, psf: Input data.
		modeltype  (int): Mode (0/1/2).
		save_value (str): Tag for output.

	Returns:
		None.  Results saved to FITS.
	"""


	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	print(f'{cluster} SERSIC 2 + SERSIC 1')
	name_ss_ajust=infotype(modeltype,save_value)[2]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}',2)
		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO')
		return
	except:
		name_s_ajust=infotype(modeltype,save_value)[0]
		if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
			ajuste_sersic=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',2)		
			data_entry = [float(ajuste_sersic[k].split()[0].replace('*','')) for k in ('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')]
		else:
			data_entry = [0.0 for k in ('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')]
			

		magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_duplo(sample,cluster,data_entry,image.shape,save_value,psf)
		mask_center=centro_magzero(sample,cluster,save_value,psf)[-2]
		data_sersic_duplo = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic_duplo.verbose = False
		if modeltype == 0:
			hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_sersic_duplo_{save_value}.dat','a')
			hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_sersic_duplo_{save_value}.dat','a')
			callback=make_callback(cluster,data_sersic_duplo,save_value,'sersic_duplo')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic_duplo,save_value,'sersic_duplo')
		elif modeltype == 2:
			callback=make_callback_simul_ss(cluster,data_sersic_duplo,save_value,'sersic_duplo')
		result_sersic_duplo = optimize.minimize(profit_like_model, data_sersic_duplo.init, args=(data_sersic_duplo,), method='L-BFGS-B', bounds=data_sersic_duplo.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic_duplo = to_pyprofit_image_duplo(data_sersic_duplo.init, data_sersic_duplo, use_mask=True)
		all_params_sersic_duplo, modelim_sersic_duplo  = to_pyprofit_image_duplo(result_sersic_duplo.x, data_sersic_duplo, use_mask=True)
		####
		clean_max_llh_sersic_duplo,unclean_max_llh_sersic_duplo,img_likelihood_sersic_duplo=clean_model(result_sersic_duplo.x,data_sersic_duplo)
		fits.writeto(f'{sample}/{cluster}/{save_value}/likelihood_{name_ss_ajust}',img_likelihood_sersic_duplo,overwrite=True)

		bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
		model_img=fits.ImageHDU(modelim_sersic_duplo,name='MODEL')
		resid_img=fits.ImageHDU(image - modelim_sersic_duplo,name='RESIDUAL')

		####
		max_llh_sersic_duplo=-result_sersic_duplo.fun
		x=[(names[i],all_params_sersic_duplo[i]) for i in range(len(names))]

		xc,yc=model0[:2]
		region=data_sersic_duplo.region
		n_data=np.sum(region)
		bic_sersic_duplo=(len(names))*np.log(n_data) - 2*(max_llh_sersic_duplo)

		temp_cov_sersic_duplo=result_sersic_duplo.hess_inv.todense()
		inc_sersic_duplo=np.sqrt(np.diag(temp_cov_sersic_duplo))
		####

		bcg_img.header['COMP_1'] = 'SERSIC1'
		bcg_img.header['XCEN_1'] = f'{model0[0]:.4f}'
		bcg_img.header['YCEN_1'] = f'{model0[1]:.4f}'

		model_img.header['MODELO'] = 'SERSIC1 + SERSIC2'
		model_img.header['COMP_1'] = 'SERSIC1'
		model_img.header['XCEN_1'] = f'{x[0][1]:.4f} +/- {inc_sersic_duplo[0]:.4f}'
		model_img.header['YCEN_1'] = f'{x[1][1]:.4f} +/- {inc_sersic_duplo[1]:.4f}'

		for i,item in enumerate(x[2:-1]):
			if i%2 == 0:
				bcg_img.header[item[0].split('.')[1]+'_1'] = model0[2:][i]
				model_img.header[item[0].split('.')[1]+'_1'] = f'{item[1]:.4f} +/- {inc_sersic_duplo[2:-1][i]:.4f}'

		bcg_img.header['COMP_2'] = 'SERSIC2'
		model_img.header['COMP_2'] = 'SERSIC2'
		for i,item in enumerate(x[2:-1]):
			if i%2 != 0:
				bcg_img.header[item[0].split('.')[1]+'_2'] = model0[2:][i]
				model_img.header[item[0].split('.')[1]+'_2'] = f'{item[1]:.4f} +/- {inc_sersic_duplo[2:-1][i]:.4f}'

		bcg_img.header['SKY']=f'{model0[-1]:.4f}'
		model_img.header['SKY']=f'{x[-1][1]:.4f}'
		model_img.header['MAX_LLH'] = max_llh_sersic_duplo
		model_img.header['BIC'] = bic_sersic_duplo
		model_img.header['CL_LLH']=clean_max_llh_sersic_duplo
		model_img.header['OLD_LLH']=unclean_max_llh_sersic_duplo
		model_img.header['FLAG']=data_sersic_duplo.check_model
		model_img.header['DOF']=data_sersic_duplo.dof
		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
		hdulist.writeto(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}',overwrite=True)

		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO')
	return 
############################################################################################
# ----------------------------------------------------------------------
# High‑level pipelines (called by multiprocessing wrappers)
# ----------------------------------------------------------------------
############################################################################################

def sersic_setup(sample,cluster,save_value):
	"""Run a single‑Sérsic fit.
	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	cluster_test='1030'
	call(f'mkdir -p {sample}/{cluster_test}',shell=True)
	call(f'mkdir -p {sample}/{cluster_test}/{save_value}',shell=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	image = np.array(fits.getdata(f'../{sample}/{cluster_test}/ajust-bcg-r.fits',1))
	# image = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster_test}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}/{cluster_test}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster_test,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)	
	# ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)
	return 
def sersic_duplo_setup(sample,cluster,save_value):
	"""Run single‑Sérsic and double‑Sérsic fits.

	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	##############################################################{sample}/{cluster}
	##IMAGENS
	cluster_test='1030'
	image = np.array(fits.getdata(f'../{sample}/{cluster_test}/ajust-bcg-r.fits',1))
	# image = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster_test}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}/{cluster_test}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster_test,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)	
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster_test,image, mask, sigim, segim, psf,modeltype,save_value)
	return 

def sersic_simul_s_setup(sample,cluster,save_value):
	"""Simulate a single‑Sérsic galaxy and re‑fit it.

	If the single‑Sérsic fit of the simulation was already done (its FITS
	file exists), the function uses the existing image; otherwise it creates
	a mock image by adding Gaussian noise (based on the sigma map) to the
	best‑fit single‑Sérsic model of the real observation.

	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################
	modeltype=1
	observation_file='observation_SE_sky'
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	name_s_ajust=infotype(modeltype,save_value)[0]
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
		image=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',1))	
	else:
		model_s_img=fits.getdata(f'{sample}/{cluster}/{observation_file}/ajust-sersic-llh-{observation_file}.fits',2)
		noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
		image = model_s_img+noise_img
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	# ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)

	return 
def sersic_duplo_simul_s_setup(sample,cluster,save_value):
	"""Simulate a single‑Sérsic galaxy and fit both single‑ and double‑Sérsic models.

	Same as ``sersic_simul_s_setup`` but also runs the double‑Sérsic fit on the
	mock image.

	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################
	modeltype=1
	observation_file='observation_SE_sky'
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	name_s_ajust=infotype(modeltype,save_value)[0]
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
		image=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',1))	
	else:
		model_s_img=fits.getdata(f'{sample}/{cluster}/{observation_file}/ajust-sersic-llh-{observation_file}.fits',2)
		noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
		image = model_s_img+noise_img
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)

	return 

def sersic_simul_ss_setup(sample,cluster,save_value):
	"""Simulate a double‑Sérsic galaxy and re‑fit it.

	Uses the best double‑Sérsic model from the real observation to create
	a mock image (model + noise).  Then fits a single Sérsic model.

	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################
	modeltype=2
	observation_file='observation_SE_sky'
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	name_s_ajust=infotype(modeltype,save_value)[0]
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
		image=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',1))	
	else:
		model_s_img=fits.getdata(f'{sample}/{cluster}/{observation_file}/ajust-sersic-duplo-llh-{observation_file}.fits',2)
		noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
		image = model_s_img+noise_img
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	# ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)

	return 
def sersic_duplo_simul_ss_setup(sample,cluster,save_value):
	"""Simulate a double‑Sérsic galaxy and fit both models (single + double).

	Args:
		sample, cluster, save_value: Standard identifiers.

	Returns:
		None.
	"""


	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################
	modeltype=2
	observation_file='observation_SE_sky'
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	name_s_ajust=infotype(modeltype,save_value)[0]
	if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
		image=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',1))	
	else:
		model_s_img=fits.getdata(f'{sample}/{cluster}/{observation_file}/ajust-sersic-duplo-llh-{observation_file}.fits',2)
		noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
		image = model_s_img+noise_img
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)

	return 

def desi_setup(sample,cluster,save_value,ra,dec):
	"""Run a single‑Sérsic fit on DESI‑legacy survey data.

	The input images are expected under ``../desi_{sample}/{cluster}/``.

	Args:
		sample, cluster, save_value: Standard identifiers.
		ra, dec (float): Right ascension and declination (currently not used
						 in the function body but passed for compatibility).

	Returns:
		None.
	"""


	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	image = np.array(fits.getdata(f'../desi_{sample}/{cluster}/bcg_r.fits'))
	# image = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../desi_{sample}/{cluster}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../desi_{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../desi_{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value,ra,dec)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)	
	# ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)
	return 

# ----------------------------------------------------------------------
# Post‑processing and summary
# ----------------------------------------------------------------------
def finish_details(sample,catalogo,modeltype,ass,save_value):
	"""Gather fit results, compute BIC, and write a summary file.

	For each cluster in ``catalogo`` that has not yet been processed (tracked
	by the file ``{sample}_profit_{save_value}_chi2.dat``), this function:
	- Reads the header of the three model FITS files (single‑Sérsic,
	  Sérsic+Exp, double‑Sérsic) and extracts the fitted parameters,
	  log‑likelihood, and BIC.
	- Computes the Residual Flux Fraction (RFF) via ``rff_calc``.
	- Determines the best model (single S or double S+S) based on BIC.
	- Calculates Δlog‑likelihood between the models.
	- Appends all information to the file ``{sample}_profit_{save_value}.dat``.

	Args:
		sample     (str): Sample name.
		catalogo   (list of str): List of cluster IDs to process.
		modeltype  (int): Mode identifier (0/1/2). Used to retrieve the
						  correct FITS filenames via ``infotype``.
		ass        (float): Asymmetry index to be written to the summary.
		save_value (str): Tag for file names.

	Returns:
		None.  The summary file is updated on disk.

	Note:
		This function assumes that the FITS files and the likelihood maps
		have already been created by the fitting routines.  It also writes
		an entry for every cluster in ``catalogo``, even if a fit failed
		(the earlier code had a commented try/except that filled zeros).
	"""


	#results_dir=f'{sample}/{cluster}/{save_value}'#f'{sample}/{cluster}'
	ok=[]
	with open(f'{sample}_profit_{save_value}_chi2.dat','r') as inp2:
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

			output=open(f'{sample}_profit_{save_value}.dat','a')
			output.write(f'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n'%(cluster,*result_sersic,*result_sersic_exp,*result_sersic_duplo,rff,ass,best_model,delta_s,delta_se))
			output.close()
			# except:
			# 	output=open(f'{sample}_profit_{save_value}.dat','a')
			# 	output.write(f'{cluster} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n')
			# 	output.close()
	return
##########################################
# ----------------------------------------------------------------------
# Multiprocessing wrappers
# ----------------------------------------------------------------------
##########################################

def run_sersic_setup(args):
	"""Unpack arguments and call ``sersic_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		None
	"""


	sample, cl, i= args
	return sersic_setup(sample, cl, i)
def run_sersic_duplo_setup(args):
	"""Unpack arguments and call ``sersic_duplo_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		None.
	"""


	sample, cl, i= args
	return sersic_duplo_setup(sample, cl, i)
def run_sersic_simul_s_setup(args):
	"""Unpack arguments and call ``sersic_simul_s_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		None
	"""


	sample, cl, i= args
	return sersic_simul_s_setup(sample, cl, i)
def run_sersic_duplo_simul_s_setup(args):
	"""Unpack arguments and call ``sersic_duplo_simul_s_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		None
	"""


	sample, cl, i= args
	return sersic_duplo_simul_s_setup(sample, cl, i)

def run_sersic_simul_ss_setup(args):
	"""Unpack arguments and call ``sersic_simul_ss_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		None
	"""


	sample, cl, i= args
	return sersic_simul_ss_setup(sample, cl, i)
def run_sersic_duplo_simul_ss_setup(args):
	"""Unpack arguments and call ``sersic_duplo_simul_ss_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		Whatever ``sersic_duplo_simul_ss_setup`` returns.
	"""


	sample, cl, i= args
	return sersic_duplo_simul_ss_setup(sample, cl, i)


def run_simul_setup(args):
	"""(Deprecated? Unused in current code.) Unpack and call ``simul_s_setup``.

	Args:
		args (tuple): (sample, cluster, save_value)

	Returns:
		Whatever ``simul_s_setup`` returns.
	"""


	sample, cl, i= args
	return simul_s_setup(sample, cl, i)
def run_desi_setup(args):
	"""Unpack arguments and call ``desi_setup``.

	Args:
		args (tuple): (sample, cluster, save_value, ra, dec)

	Returns:
		Whatever ``desi_setup`` returns.
	"""


	sample, cl, i,ra,dec= args
	return desi_setup(sample, cl, i,ra,dec)

def zelador(sample,cluster,pasta):
	"""Clean (delete) all files inside a cluster's temporary directory.

	Args:
		sample  (str): Sample name.
		cluster (str): Cluster identifier.
		pasta   (str): Sub‑directory to be emptied.

	Returns:
		None.  Executes an ``rm -r`` command.
	"""
	
	call(f'rm -r {sample}/{cluster}/{pasta}/*',shell=True)
	return


if __name__ == '__main__':
	import sys
	sample=sys.argv[1]
	save_value=[]#'observation_SE_sky','simulation_SE_sky','simulation_s_duplo_SE_sky']
	for i in range(10):
		save_value.append(f'psf_test_{str(i)}')
	ok=[]
	inp2=open(f'ass_{sample}.dat','a')
	# inp3=open(f'data_desi_{sample}_entry.dat','a')
	with open(f'ass_{sample}.dat','r') as inp2:
		for item in inp2.readlines():   
			ok.append(item.split()[0])
	# catalogo=[]
	# # data_indiv=np.loadtxt('data_indiv_ecd.dat',usecols=[5,6])
	# with open(f'../{sample}/pargal_{sample}_compact_astro_vfix.dat','r') as inp1:
	# 	data_sample=inp1.readlines()
	# 	for i,obj in enum['1206' '2039' '3035' '2211' '2173' '1313' '2214' '1415' '3092' '1169']erate(data_sample):
	# 		flagmask=int(float(obj.split()[11]))
	# 		flagdelta=int(float(obj.split()[12]))
	# 		cluster=obj.split()[0]
	# 		if [flagmask,flagdelta] == [0,0] and cluster != '2102' and cluster!='3338':
	# 			catalogo.append(cluster)
	# 			ass=float(obj.split()[9])
	# print(np.random.choice(catalogo,10))
	###################
	catalogo_psf=['1206','2039','3035','2211','2173','1313','2214','1415','3092','1169']
	# #OBSERVAÇÕES
	obs_list=[]
	for i in range(10):
		obs_list.append((sample, catalogo_psf[i], save_value[i]))
	with mp.Pool(processes=19) as pool:
		chunksize=1
		for _ in pool.imap_unordered(run_sersic_setup, obs_list,chunksize=chunksize):
			pass
	with mp.Pool(processes=19) as pool:
		chunksize=1
		for _ in pool.imap_unordered(run_sersic_duplo_setup, obs_list,chunksize=chunksize):
			pass


	# temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[0]}/{model_names[2]}')]
	# if set(catalogo) == set(temp_ok):
	# 	pass
	# else:
	# 	new_catalog=[item for item in catalogo if item not in temp_ok]
	# 	obs_list = [(sample, cl, save_value[0]) for cl in new_catalog]
	# 	with mp.Pool(processes=19) as pool:
	# 		chunksize=1
	# 		for _ in pool.imap_unordered(run_sersic_duplo_setup, obs_list,chunksize=chunksize):
	# 			pass
	# output=open(f'{sample}_profit_{save_value[0]}.dat','a')
	# finish_details(sample,catalogo,0,ass,save_value[0])
	# # #####################
	# ##SIMULAÇÕES
	# ## Sersic simples
	# model_names=infotype(1,save_value[1])
	# temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[1]}/{model_names[0]}')]
	# if set(catalogo) == set(temp_ok):
	# 	pass
	# else:
	# 	new_catalog=[item for item in catalogo if item not in temp_ok]
	# 	sim_s_list = [(sample, cl, save_value[1]) for cl in new_catalog]
	# 	with mp.Pool(processes=19) as pool:
	# 		chunksize=1
	# 		for _ in pool.imap_unordered(run_sersic_simul_s_setup, sim_s_list,chunksize=chunksize):
	# 			pass

	# model_names=infotype(1,save_value[1])
	# temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[1]}/{model_names[2]}')]
	# if set(catalogo) == set(temp_ok):
	# 	pass
	# else:
	# 	new_catalog=[item for item in catalogo if item not in temp_ok]
	# 	sim_s_list = [(sample, cl, save_value[1]) for cl in new_catalog]
	# 	with mp.Pool(processes=19) as pool:
	# 		chunksize=1
	# 		for _ in pool.imap_unordered(run_sersic_duplo_simul_s_setup, sim_s_list,chunksize=chunksize):
	# 			pass

	# output=open(f'{sample}_profit_{save_value[1]}.dat','a')
	# finish_details(sample,catalogo,1,ass,save_value[1])

	# ## Sersic duplo
	# model_names=infotype(2,save_value[2])
	# temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[2]}/{model_names[0]}')]
	# if set(catalogo) == set(temp_ok):
	# 	pass
	# else:
	# 	new_catalog=[item for item in catalogo if item not in temp_ok]
	# 	sim_ss_list = [(sample, cl, save_value[2]) for cl in new_catalog]
	# 	with mp.Pool(processes=19) as pool:
	# 		chunksize=1
	# 		for _ in pool.imap_unordered(run_sersic_simul_ss_setup, sim_ss_list,chunksize=chunksize):
	# 			pass
	# model_names=infotype(2,save_value[2])
	# temp_ok=[cl for cl in catalogo if os.path.exists(f'{sample}/{cl}/{save_value[2]}/{model_names[2]}')]
	# if set(catalogo) == set(temp_ok):
	# 	pass
	# else:
	# 	new_catalog=[item for item in catalogo if item not in temp_ok]
	# 	sim_ss_list = [(sample, cl, save_value[2]) for cl in new_catalog]
	# 	with mp.Pool(processes=19) as pool:
	# 		chunksize=1
	# 		for _ in pool.imap_unordered(run_sersic_duplo_simul_ss_setup, sim_ss_list,chunksize=chunksize):
	# 			pass
	# output=open(f'{sample}_profit_{save_value[2]}.dat','a')
	# finish_details(sample,catalogo,2,ass,save_value[2])
