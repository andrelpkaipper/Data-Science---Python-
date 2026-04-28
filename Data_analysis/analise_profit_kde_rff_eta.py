import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Ou 'TkAgg'
import matplotlib.pyplot as plt
from subprocess import call
from scipy.stats import pearsonr,iqr,f,chi2,ks_2samp,sem
from scipy.stats import gaussian_kde as kde
import seaborn as sns
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore",message='invalid value encountered in log10',category=RuntimeWarning)

"""
Comprehensive statistical analysis and visualisation of structural parameter obtained from galaxy surface-
photometry. This code processes single-Sérsic and Sérsic+Sérsic decompositions photometric parameters for large samples 
of early-type galaxies and provides:

- KDE fitting and two-sample KS tests to compare morphological subsamples.
- Diagnostic plots (histograms, KDE curves, scatter plots, joint plots,
  colour‑coded scatter).
- Investigation of the Kormendy relation (log Rₑ vs. <μₑ>) for each
  component and morphological subgroup.
- Analysis of the RFF–η (Residual Flux Fraction vs. asymmetry) plane,
  including sub‑classification based on BIC differences.
- Parameter vs. M200 (cluster mass) scaling relations and redshift trends.
- Automated SVM classification tests to separate cD‑like from E‑like 
  galaxies.

The script is written for two fixed samples: 'WHL' and 'L07'.  Data files
are expected to follow a strict naming convention and are assumed to
reside in appropriate sub‑directories.  Many functions depend on global
variables that are defined in the main block; this is a legacy design
and is clearly documented below.

These code was made as part of my PhD on Astrophysics with the goal to 
clarify and analyze the relevant parameter and found associations and trendings 
between different classes of objects.
"""

# ---------------------------------------------------------------------------
# Utility / mathematical functions
# ---------------------------------------------------------------------------


def dist_pc(re, z):
	"""Convert effective radius from arcseconds to log₁₀ kiloparsec.

	The routine uses a simplified Hubble law with a deceleration parameter
	q₀ = -0.55 (Ωₘ = 0.3, ΩΛ = 0.7) and H₀ = 70 km/s/Mpc.  The pixel
	scale is hard‑coded as 0.396″/pixel.

	Steps:
	1. Scale the radius from pixels to arcseconds (re × 0.396).
	2. Convert to radians.
	3. Compute distance from the Hubble law, corrected by the
	Taylor expansion term (1 + (1‑q₀)z/2).
	4. Multiply by the angular size to obtain physical size in kpc.
	5. Return log₁₀ of the physical size, ensuring that zero radii
	are replaced by 0 to avoid -∞.

	Args:
	re (ndarray or float): Effective radius in pixels.
	z  (ndarray or float): Redshift.

	Returns:
	ndarray: log₁₀(re / kpc) with zeros handling.
	"""

	h0 = 70          # km/s/Mpc
	q0 = -0.55       # omega_m = 0.3 e omega_v = 0.7
	c = 299792.458   # km/s

	re = re * 0.396
	re_rad = re * np.pi / 648000.  # converte para radianos

	hubble_law = c * z / h0
	taylor_q0z = 1 + ((1 - q0) * z / 2)
	dist = hubble_law * taylor_q0z

	re_kpc = dist * re_rad * 1000.  # converte para pc

	# evita log10(0)
	x = np.where(re_kpc != 0, np.log10(re_kpc), 0)

	return x
def linfunc(x,a,b):
    """Evaluate a linear function y = a*x + b.

    This tiny routine is used throughout the code as a compact way to
    generate model lines (e.g., for regression fits).

    Args:
        x (array_like): Independent variable.
        a (float): Slope.
        b (float): Intercept.

    Returns:
        array_like: a*x + b.
    """
	return a*x+b
def exp_func(x,a,b):
	"""Exponential function y = a * exp(b * x).
	Used in some exploratory modelling.
	Args:
	x (array_like): Independent variable.
	a (float): Amplitude.
	b (float): Growth rate (can be negative).

	Returns:
	array_like: a * exp(b * x).
	"""

	return a*np.exp(x*(b))
def diff(x,param):
	"""Difference between a Gaussian and a log‑normal distribution.

	This function is used specifically to find the zero‑crossing of the
	combined RFF distribution model (Gaussian + log‑normal) in order to
	determine an automatic split point for the RFF histogram.

	Args:
	x(array_like): Independent variable (RFF value).
	param (list of 6 floats):
		Model parameters:[μ_Gauss, σ_Gauss, amp_Gauss, μ_log, σ_log, amp_log]

	Returns:
	array_like: gauss(x, *param[:3]) - lognormal(x, *param[3:6]).
	"""

	px=gauss(x,*param[:3]) - lognormal(x,*param[3:6])
	return px
def gauss(x,mu,sigma,amp):
	"""Evaluate a Gaussian function.

	y = amp * exp(-(x - mu)² / (2 * sigma²))

	Args:
		x     (array_like): Independent variable.
		mu    (float): Mean.
		sigma (float): Standard deviation.
		amp   (float): Amplitude (peak height).

	Returns:
	    array_like: Gaussian values.
	"""

	return amp*np.exp(-((x-mu)**2)/(2*(sigma)**2))
def lognormal(x,mu,sigma,amp):
	"""Log‑normal distribution.

	y = amp * exp(-(ln(x) - mu)² / (2 * sigma²))
	The input x must be positive.

	Args:
		x     (array_like): Independent variable (x > 0).
		mu    (float): Mean of ln(x).
		sigma (float): Standard deviation of ln(x).
		amp   (float): Amplitude.

	Returns:
	array_like: Log‑normal values.
	"""

	return (amp)*np.exp(-((np.log(x)-mu)**2)/(2*(sigma)**2))
def calc_1_sigma(vec):
	"""Compute the 16th and 84th percentiles (1‑σ limits).

	For a distribution close to normal, these percentiles approximate the
	±1σ range around the median/mean.

	Args:
		vec (array_like): Data array.

	Returns:
		ndarray: [16th percentile, 84th percentile].
	"""

	x=np.percentile(vec,[16,84])
	return x
def dlognorm(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
	"""Sum of a Gaussian and a log‑normal distribution.

	This is the full model that is fitted to the RFF histogram:
	a Gaussian component for the "symmetric" RFF population and a log‑normal
	component for the asymmetric tail.

	Args:
		x     (array_like): Independent variable.
		mu1, sigma1, amp1: Parameters for the Gaussian component.
		mu2, sigma2, amp2: Parameters for the log‑normal component.

	Returns:
		array_like: Sum of the two components.
	"""

    px=amp1*np.exp(-((x-mu1)**2)/(2*(sigma1)**2))+amp2*np.exp(-(np.log(x)-mu2)**2/(2*sigma2**2))
    return px
def log_model(x, a, b):
	"""Logarithmic model y = a * ln(x) + b.

	Used to fit the boundary in the ΔBIC–redshift plane to split
	the data in 2 parts, where the predicted envelope show a strong evidence
	against an another part which the predicted envelope are more scarce and shows 
	a dubiuos existence.

	Args:
		x (array_like): Independent variable (redshift, x > 0).
		a (float): Scaling factor.
		b (float): Constant offset.

	Returns:
		array_like: a * ln(x) + b.
	"""
	return a * np.log(x) + b  
def bt_ratio(mag_bojo,mag_env):
	"""Compute the bulge‑to‑total flux ratio from component magnitudes.

	Flux_bulge = 10^(-0.4 * mag_bojo)
	Flux_env   = 10^(-0.4 * mag_env)
	B/T = Flux_bulge / (Flux_bulge + Flux_env)

	Args:
		mag_bojo (array_like): Magnitude of the inner component (bulge).
		mag_env  (array_like): Magnitude of the envelope.

	Returns:
		ndarray: B/T ratio (floats between 0 and 1).
	"""

	fluxo_bojo=	np.power(10,-0.4*mag_bojo)
	fluxo_envelope=np.power(10,-0.4*mag_env)
	ii = np.divide(fluxo_bojo,fluxo_bojo+fluxo_envelope)
	return ii
def bt_ratio_corr(mag_bojo,mag_env,corr_int,corr_ext):
	"""Corrected B/T ratio using multiplicative flux corrections.

	The corrections (corr_int, corr_ext) account for missing flux that
	was not attributed to either component during the fit (e.g., due to
	the limited radial range of the model).  Only galaxies where the
	corrected inner flux is non‑zero are returned.

	Args:
		mag_bojo, mag_env: Magnitudes of inner and outer components.
		corr_int, corr_ext: Flux correction factors (>0).

	Returns:
		ndarray: Corrected B/T for objects where inner flux ≠ 0.
	"""

	fluxo_bojo=	np.power(10,-0.4*mag_bojo)*corr_int
	fluxo_envelope=np.power(10,-0.4*mag_env)*corr_ext
	ii = np.divide(fluxo_bojo[fluxo_bojo!=0],(fluxo_bojo+fluxo_envelope)[fluxo_bojo!=0])
	return ii
def f_test(chi2_s,chi2_ss,ndof_s,ndof_ss):
	"""Perform an F‑test comparing two nested models.

	The simpler model (e.g., single Sérsic) is compared to the more
	complex one (e.g., Sérsic+Sérsic).  The F‑statistic is computed as
	(χ²_s / ndof_s) / (χ²_ss / ndof_ss).

	Args:
		chi2_s  (float): χ² of the simpler model.
		chi2_ss (float): χ² of the complex model.
		ndof_s  (int):   Degrees of freedom of the simpler model.
		ndof_ss (int):   Degrees of freedom of the complex model.

	Returns:
		float: p‑value from the F‑distribution.
	"""


	chi2nu_s = chi2_s / ndof_s
	chi2nu_ss = chi2_ss / ndof_ss
	F = chi2nu_s / chi2nu_ss
	dfn = ndof_s
	dfd = ndof_ss
	p_value = f.sf(F, dfn, dfd)
	return p_value
# ---------------------------------------------------------------------------
# Piecewise linear fitting and uncertainty estimation
# ---------------------------------------------------------------------------
def two_line_model(params, x, y):
	"""Evaluate a piecewise linear model with a break point at x_div. This function was 
	primarily used o find the break point on the Kormendy relation from the inner core of the double sérsic profiles.

	y = slope1 * x + intercept1   for x ≤ x_div
	y = slope2 * x + intercept2   for x > x_div

	The `y` argument is not used; it is present only for compatibility with
	minimisation routines that expect a function of (params, x, y).

	Args:
		params (list of 5 floats): [slope1, intercept1, slope2, intercept2, x_div].
		x      (array_like): Independent variable.
		y      (array_like): Ignored.

	Returns:
		ndarray: Predicted y values.
	"""

	slope1, intercept1, slope2, intercept2, x_div = params

	y_pred = np.zeros_like(x)
	mask1 = x <= x_div
	mask2 = x > x_div

	y_pred[mask1] = slope1 * x[mask1] + intercept1
	y_pred[mask2] = slope2 * x[mask2] + intercept2

	return y_pred
def chi_square(params, x, y, y_err=None):
	"""Reduced χ² for the two‑line model.

	If `y_err` is None, unit weights are used (i.e., unweighted χ²).

	Args:
		params (list): Parameters for the two‑line model.
		x, y   (array_like): Data points.
		y_err  (array_like, optional): Measurement errors on y.

	Returns:
		float: χ² = Σ ((y - y_pred) / y_err)².
	"""
	if y_err is None:
		y_err = np.ones_like(y)  # Assume equal errors if not provided

	y_pred = two_line_model(params, x, y)
	chi2_val = np.sum(((y - y_pred) / y_err) ** 2)
	return chi2_val
def find_optimal_division(x, y, y_err=None, initial_guess=None):
    from scipy.optimize import minimize
	"""Automatically fit a piecewise linear model with an optimal break point.

	The initial guess is normally obtained by splitting the data at the
	median x and fitting separate linear regressions.  The break point is
	constrained to lie between (min(x)+0.1, max(x)-0.1) to avoid edge effects.
	Minimisation uses the Nelder‑Mead algorithm.

	Args:
		x, y           (array_like): Data.
		y_err          (array_like, optional): Errors on y.
		initial_guess  (list, optional): [slope1, inter1, slope2, inter2, x_div].
	                 If None, a heuristic guess is computed.

	Returns:
		scipy.optimize.OptimizeResult: Contains the optimal parameters
		    (result.x) and the minimum χ² (result.fun).
	"""


    if initial_guess is None:
        # x_min, x_max = -0.283,2.716#np.min(x), np.max(x)
        # x_mid = 0.7#0.5*(x_min+x_max)
        x_min, x_max =-0.3,2.5# np.min(x), np.max(x)
        x_mid = 0.65#0.5*(x_min+x_max)
        
        # Fit initial lines to left and right halves
        mask_left = x <= x_mid
        mask_right = x > x_mid
        
        if np.sum(mask_left) > 1:
            coeffs_left = np.polyfit(x[mask_left], y[mask_left], 1)
        else:
            coeffs_left = [-3, 0.0]
            
        if np.sum(mask_right) > 1:
            coeffs_right = np.polyfit(x[mask_right], y[mask_right], 1)
        else:
            coeffs_right = [-3, 0.0]
        
        initial_guess = [coeffs_left[0], coeffs_left[1], 
                        coeffs_right[0], coeffs_right[1], x_mid]
    
    # Constraints: x_div should be within data range
    bounds = [(None, None), (None, None), (None, None), (None, None), 
              (np.min(x) + 0.1, np.max(x) - 0.1)]
    
    # Minimize chi-square
    result = minimize(chi_square, initial_guess, args=(x, y, y_err),
                     method='Nelder-Mead', bounds=bounds)
    
    return result
def bootstrap_uncertainty(x, y, params, y_err=None, n_bootstrap=100):
	"""Estimate parameter uncertainties for the piecewise linear fit using
	bootstrap resampling.

	For each bootstrap iteration, the data are resampled with replacement and
	the optimal division is refitted.  The standard deviation of the bootstrap
	parameter values is returned as the uncertainty.

	Args:
		x, y         (array_like): Data.
		params       (list): Initial fit parameters (only used for shape reference).
		y_err        (array_like, optional): Measurement errors.
		n_bootstrap  (int): Number of bootstrap resamples.

	Returns:
		ndarray: Standard deviation of each parameter (5 elements).
	"""
    bootstrap_params = []
    n_points = len(x)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_points, n_points, replace=True)
        x_bs = x[indices]
        y_bs = y[indices]
        
        if y_err is not None:
            y_err_bs = y_err[indices]
        else:
            y_err_bs = None
        
        # Fit to bootstrap sample
        try:
            result = find_optimal_division(x_bs, y_bs, y_err_bs)
            if result.success:
                bootstrap_params.append(result.x)
        except:
            continue
    
    bootstrap_params = np.array(bootstrap_params)
    
    if len(bootstrap_params) > 0:
        uncertainties = np.std(bootstrap_params, axis=0)
        return uncertainties
    else:
        return np.zeros_like(params)
# ---------------------------------------------------------------------------
# Statistical tests and SVM classifiers
# ---------------------------------------------------------------------------
def ks_calc(vecs):
	"""Compute the two‑sample Kolmogorov‑Smirnov test for all pairs
	of arrays in a list.

	Args:
		vecs (list of array_like): List of 1D data arrays.

	Returns:
		tuple: (p_values, D_statistics) – both as lists of lists.
	       results_p[i][j] is the p‑value between vecs[i] and vecs[j].
	"""

	n = len(vecs)
	results_p = [[] for i in range(n)]
	results_d = [[] for i in range(n)]

	for i in range(n):
		for j in range(n):
			D, p = ks_2samp(vecs[i], vecs[j])
			results_p[i].append(p)
			results_d[i].append(D)
	return results_p,results_d
def svc_calc_trio(vecs_x,vecs_y,name,outfile):
	from sklearn.svm import SVC
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	"""Train a linear SVM classifier on 2D features and save the result.

	This variant is used for triples of parameter combinations;

	Args:
		vecs_x  (ndarray, shape (n, 2)): Feature data (two parameters).
		vecs_y  (ndarray, shape (n,)): Class labels (0/1).
		name    (list of str): Ignored (place holder).
		outfile (str): Path to output file (opened in append mode).

	Returns:
		tuple: (coefs, inter, acc_check) – SVM coefficients, intercept,
	       and training accuracy.
	"""

	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	coefs  = res.named_steps['svc'].coef_.ravel()
	inter  = res.named_steps['svc'].intercept_.ravel()

	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	name=name1,name2,name3
	save_vec = np.concatenate([[name1],[name2],[name3],[acc_check]])

	with open(outfile, "ab") as f:
	    np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_calc_dupla(vecs_x,vecs_y,name,outfile):
	from sklearn.svm import SVC
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	"""Linear SVM classifier for two‑parameter combinations (pair mode).

	Similar to `svc_calc_trio` but for pairs.  It writes the two parameter
	names (global variables `name1`, `name2`) and accuracy to `outfile`.

	Args:
		vecs_x, vecs_y: Data and labels.
		name: Placeholder.
		outfile: Output file path.

	Returns:
		tuple: (coefs, inter, acc_check).
	"""
	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	coefs  = res.named_steps['svc'].coef_.ravel()
	inter  = res.named_steps['svc'].intercept_.ravel()

	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	name=name1,name2
	save_vec = np.concatenate([[name1],[name2],[acc_check]])

	with open(outfile, "ab") as f:
	    np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_calc(vecs_x,vecs_y):
	from sklearn.svm import SVC
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	"""Train a linear SVM (with StandardScaler) and return coefficients
	transformed back to the original feature space.

	No file output is generated.  This function is used in the interactive
	plotting of the n₁ vs nₛ decision boundary.

	Args:
		vecs_x (ndarray, shape (n, 2)): Features.
		vecs_y (ndarray, shape (n,)): Labels.

	Returns:
		tuple: (coefs, inter, acc_check) – coefficients in original units.
	"""
	test_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
	res = test_svc.fit(vecs_x, vecs_y)

	acc_check = test_svc.score(vecs_x, vecs_y)

	scaler = res.named_steps['standardscaler']
	svc    = res.named_steps['svc']

	coefs_pad = svc.coef_[0]     # coeficientes no espaço padronizado
	inter_pad = svc.intercept_[0]

	# Desfaz a padronização
	coefs = coefs_pad / scaler.scale_
	inter = inter_pad - np.sum(coefs_pad * scaler.mean_ / scaler.scale_)
	# save_vec = np.concatenate([[name],coefs, inter, [acc_check]])
	# name=name1,name2
	# save_vec = np.concatenate([[name1],[name2],[acc_check]])

	# with open(outfile, "ab") as f:
	#     np.savetxt(f, save_vec[None, :], fmt="%s")

	return coefs, inter, acc_check
def svc_line_plot(x,a,b,c):
	"""Compute y = -(a*x + b) / c, the line of the SVM decision boundary
	in a 2D scatter plot.

	Args:
		x (array_like): x‑coordinates.
		a, b, c (float): Coefficients of the plane a*x + b*y + c = 0.

	Returns:
		ndarray: y‑coordinates of the line.
	"""

	y_plot = -(a*x+b)/c
	return y_plot
# ---------------------------------------------------------------------------
# Plotting and animation utilities
# ---------------------------------------------------------------------------
def make_gif(vec_data,vec_label,zlim,xlim,ylim,save_place):
	from matplotlib.animation import PillowWriter, FFMpegWriter
	"""Create two rotating 3D scatter‑plot GIFs showing different
	morphological classes.

	This function relies heavily on global variables like lim_cd_small,
	lim_cd_big and elip_lim. This function was used only for observation of the 
	formats that was found on the possibles combinations from parameters to see
	that exist or not some kind of dependence on the particularly plane at the time.
	Args:
		vec_data   (list of 3 arrays): [x, y, z] coordinates.
		vec_label  (list of str): Axis labels.
		zlim, xlim, ylim (tuple): Axis limits.
		save_place (list of two str): Output file paths for the two GIFs.

	Returns:
		None.  The GIFs are written to disk.
	"""

	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')

	writer_2c = PillowWriter(fps=2)

	with writer_2c.saving(fig, save_place[0], dpi=100):
		for angle in np.arange(0,360,60):
			for angle2 in np.arange(0,180,30):
				ax.clear()
				ax.scatter(vec_data[0][lim_cd_small], vec_data[1][lim_cd_small],vec_data[2][lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
				ax.scatter(vec_data[0][lim_cd_big], vec_data[1][lim_cd_big], vec_data[2][lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='cD')
				ax.set_zlim(zlim)
				if xlim != None:
					ax.set_xlim(xlim)
				if ylim != None:
					ax.set_xlim(ylim)
				ax.set_xlabel(vec_label[0])
				ax.set_ylabel(vec_label[1])
				ax.set_zlabel(vec_label[2])
				ax.legend()
				ax.view_init(elev=angle, azim=angle2)
				writer_2c.grab_frame()
		plt.close()

	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')

	writer_sample = PillowWriter(fps=2)

	with writer_sample.saving(fig, save_place[1], dpi=100):

		for angle in np.arange(0,360,60):
			for angle2 in np.arange(0,180,30):
				ax.clear()
				ax.scatter(vec_data[0][elip_lim], vec_data[1][elip_lim],vec_data[2][elip_lim],alpha=0.4, c='green', edgecolor='black', label='E')
				ax.scatter(vec_data[0][lim_cd_small], vec_data[1][lim_cd_small],vec_data[2][lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
				ax.scatter(vec_data[0][lim_cd_big], vec_data[1][lim_cd_big], vec_data[2][lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='cD')
				ax.set_zlim(zlim)
				ax.set_xlabel(vec_label[0])
				ax.set_ylabel(vec_label[1])
				ax.set_zlabel(vec_label[2])
				ax.legend()
				ax.view_init(elev=angle, azim=angle2)
				writer_sample.grab_frame()
		plt.close()
	return
def bic_clean(bic,max_llh,llh_cl):
    """Recompute BIC after replacing the log‑likelihood with a “cleaned”
    version.
	
	This cleaned version was obtained masking the central region from the galaxy, 
	on attempt to understand the liability of the central region that, as we know, are heavily 
	dependent from the PSF image that was convoluted with the model to explain how the flux was 
	scatter through the pixels on the astronomic image.

    Cleaned BIC = (original_BIC + 2*max_llh) - 2*llh_cl.

    Args:
        bic     (float): Original BIC.
        max_llh (float): Maximum log‑likelihood.
        llh_cl  (float): Cleaned log‑likelihood.

    Returns:
        float: Cleaned BIC.
    """

	cl_llh=-llh_cl
	x_data=bic+2*max_llh
	cl_bic=x_data-2*cl_llh
	return cl_bic
# ---------------------------------------------------------------------------
# High‑level analysis functions
# ---------------------------------------------------------------------------
def mass_calc(info_need):
	"""Calculate stellar masses for the two components using B/T and
	total stellar mass.

	Uses the global mask `lim_casjobs` to select objects with available
	CASJOBS data.

	Args:
		info_need (tuple): (bt_vals, starmass_vals) – both 1D arrays.
		bt_vals: It's the B/T ratio from the data 
		starmass_vals: It's the star mass on the same object
	Returns:
		tuple: (mass_c1, mass_c2) – log₁₀ stellar masses for
	       component 1 and 2.
	"""
	bt_vals,starmass_vals=info_need
	bt_mass=bt_vals[lim_casjobs]

	mass_c1=np.log10(np.multiply(bt_mass,np.power(10,starmass)))
	mass_c2=np.log10(np.multiply((1-bt_mass),np.power(10,starmass)))

	return mass_c1,mass_c2
def photo_counter(param,lim_region,l07_regions,split_value,entry_label,save_labels):
	"""Count objects above/below a threshold and produce heatmaps.

	This function creates summary maps showing the absolute number and
	fraction of galaxies that satisfy `param ≤ split_value` for each of
	the three morphological regions.  If `l07_regions` is not None
	(i.e., L07 sample), additional maps are made for the cD, E, and
	mixed classes.

	Args:
		param       (array_like): Parameter values for the full sample.
		lim_region  (list of masks): Three boolean masks for the main morphological sub‑samples.
		l07_regions (tuple of masks or None): cD, E, E/cD masks (L07).
		split_value (float): Threshold.
		entry_label (tuple): (split_label, title) for plot title.
		save_labels (tuple): (directory, filename_prefix).

	Returns:
		None.  Several PNG heatmaps are saved.
	"""
	split_label,title=entry_label
	labelsx = [r'$R_{sim}$',r'$R_{low}$',r'$R_{asy}$']

	#GERAL
	qt_less=[]
	qt_great=[]

	frac_less=[]
	frac_great=[]

	for i,region in enumerate(lim_region):
		val_low_region=param[region] <= split_value
		val_high_region=param[region] > split_value
		
		qt_less.append(np.sum(val_low_region))
		qt_great.append(np.sum(val_high_region))
		frac_less.append(np.mean(val_low_region))
		frac_great.append(np.mean(val_high_region))
		if l07_regions != None:

			cd_cut,e_cut,misc_cut=l07_regions
			
			val_low_region_cD=param[region & cd_cut] <= split_value
			val_high_region_cD=param[region & cd_cut] > split_value

			val_low_region_E=param[region & e_cut] <= split_value
			val_high_region_E=param[region & e_cut] > split_value

			val_low_region_misc=param[region & misc_cut] <= split_value
			val_high_region_misc=param[region & misc_cut] > split_value
			
			qt_less_l07=[np.sum(val_low_region_cD),np.sum(val_low_region_E),np.sum(val_low_region_misc)]
			qt_great_l07=[np.sum(val_high_region_cD),np.sum(val_high_region_E),np.sum(val_high_region_misc)]
			frac_less_l07=[np.mean(val_low_region_cD),np.mean(val_low_region_E),np.mean(val_low_region_misc)]
			frac_great_l07=[np.mean(val_high_region_cD),np.mean(val_high_region_E),np.mean(val_high_region_misc)]

			labelsx_l07 = [f'{labelsx[i]}[cD]',f'{labelsx[i]}[E]',f'{labelsx[i]}[E/cD]']
			vec=np.asarray([qt_less_l07,qt_great_l07])
			plt.figure(figsize=(6, 4))
			plt.suptitle(title)
			sns.heatmap(vec, annot=True, fmt='d', cmap='Reds',xticklabels=labelsx_l07,yticklabels=split_label)
			plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/counter_{save_labels[1]}_{labelsx[i]}.png')
			plt.close()

			vec=np.asarray([frac_less_l07,frac_great_l07])
			plt.figure(figsize=(6, 4))
			plt.suptitle(f'{title} (frac)')
			sns.heatmap(vec,annot=True,fmt='.3f',cmap='Reds',xticklabels=labelsx_l07,yticklabels=split_label)
			plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/counter_frac_{save_labels[1]}_{labelsx[i]}.png')
			plt.close()
	vec=np.asarray([qt_less,qt_great])
	plt.figure(figsize=(6, 4))
	plt.suptitle(f'{title}')
	sns.heatmap(vec, annot=True, fmt='d', cmap='Reds',xticklabels=labelsx,yticklabels=split_label)
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/counter_{save_labels[1]}.png')
	plt.close()

	vec=np.asarray([frac_less,frac_great])
	plt.figure(figsize=(6, 4))
	plt.suptitle(f'{title} (frac)')
	sns.heatmap(vec,annot=True,fmt='.3f',cmap='Reds',xticklabels=labelsx,yticklabels=split_label)
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/counter_frac_{save_labels[1]}.png')
	plt.close()

	return 
###################################################################################
#FUNÇÕES AUXILIARES
def par_multi_kde(par_entry):
	"""KDE comparison for three parameters (Sérsic, comp1, comp2) across
	morphological subsamples.

	For each component type (Sérsic, internal, external), a plot is created
	showing the KDE curves for the three morphological classes, along with
	vertical lines at the mean and a text box with the two‑sample KS
	p‑values.  If the sample is 'L07', an extra set of plots restricted
	to cD galaxies (Zhao classification) is also produced.

	Args:
		par_entry (tuple):
		param         : tuple of three arrays (parameter values for the three morphological splits).
		lim_region    : list of three boolean masks.
		l07_regions   : tuple of masks or None.
		param_linspace: ndarray – x‑axis grid for KDE evaluation.
		par_labels    : tuple (suptitle, (xlabel1, xlabel2, xlabel3)).
		save_labels   : tuple (subdir, filename_prefix).

	Returns:
		None.  Multiple PNG figures are saved.

	Note:
		Global styling variables are used (cores, line_width, alpha_vec,
		names_simples, save_path).
	"""
	param,lim_region,l07_regions,param_linspace,par_labels,save_labels=par_entry
	param_s,param1,param2=param
	##SÉRSIC 
	param_sersic_high,param_sersic_low_left,param_sersic_low_right=param_s[lim_region[0]],param_s[lim_region[1]],param_s[lim_region[2]]
	##COMPONENTES -- INTERNO/EXTERNO
	param_interno_high,param_externo_high,param_interno_low_left,param_externo_low_left,param_interno_low_right,param_externo_low_right=param1[lim_region[0]],param2[lim_region[0]],param1[lim_region[1]],param2[lim_region[1]],param1[lim_region[2]],param2[lim_region[2]]

	ks_param_sersic=ks_calc([param_sersic_high,param_sersic_low_left,param_sersic_low_right])
	ks_param_intern=ks_calc([param_interno_high,param_interno_low_left,param_interno_low_right])
	ks_param_extern=ks_calc([param_externo_high,param_externo_low_left,param_externo_low_right])

	param_kde_entry=np.hstack((param_s,param1,param2))
	kde_param=kde(param_kde_entry)
	param_factor = kde_param.factor

	vec_ks_param_sersic=ks_param_sersic[0][0][2],ks_param_sersic[0][1][2],ks_param_sersic[0][0][1]
	vec_ks_param_intern=ks_param_intern[0][0][2],ks_param_intern[0][1][2],ks_param_intern[0][0][1]
	vec_ks_param_extern=ks_param_extern[0][0][2],ks_param_extern[0][1][2],ks_param_extern[0][0][1]
	#
	vec_med_param_sersic=med_param_sersic_high,med_param_sersic_low_left,med_param_sersic_low_right=np.average(param_sersic_high),np.average(param_sersic_low_left),np.average(param_sersic_low_right)
	vec_med_param_interno=med_param_interno_high,med_param_interno_low_left,med_param_interno_low_right=np.average(param_interno_high),np.average(param_interno_low_left),np.average(param_interno_low_right)
	vec_med_param_externo=med_param_interno_high,med_param_externo_low_left,med_param_externo_low_right=np.average(param_externo_high),np.average(param_externo_low_left),np.average(param_externo_low_right)

	##SÉRSIC
	vec_kde_param_simples=param_sersic_kde_high,param_sersic_kde_low_left,param_sersic_kde_low_right=kde(param_sersic_high,bw_method=param_factor),kde(param_sersic_low_left,bw_method=param_factor),kde(param_sersic_low_right,bw_method=param_factor)
	##COMPONENTES -- INTERNO/EXTERNO
	vec_kde_param_intern=param_interno_kde_high,param_interno_kde_low_left,param_interno_kde_low_right=kde(param_interno_high,bw_method=param_factor),kde(param_interno_low_left,bw_method=param_factor),kde(param_interno_low_right,bw_method=param_factor)
	vec_kde_param_extern=param_externo_kde_high,param_externo_kde_low_left,param_externo_kde_low_right=kde(param_externo_high,bw_method=param_factor),kde(param_externo_low_left,bw_method=param_factor),kde(param_externo_low_right,bw_method=param_factor)
	########################
	#MODELO SIMPLES
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - Modelos Simples')
	for i,dist in enumerate(vec_kde_param_simples):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_sersic[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][0])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_sersic[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_sersic[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_sersic[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	# plt.show()
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_simples_kde.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO

	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - Componente interno')
	for i,dist in enumerate(vec_kde_param_intern):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_interno[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][1])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_intern[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_intern[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_intern[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_interno_kde.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - Componente externo')
	for i,dist in enumerate(vec_kde_param_extern):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_externo[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][2])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_extern[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_extern[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_extern[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_externo_kde.png')
	plt.close()
	if sample == 'L07':
		##SÉRSIC -- cD,E(EL),cD(std),E
		#CLASSIFICAÇÃO ZHAO PURA
		#lim_region,l07_regions
		param_sersic_cd_zhao,param_sersic_e_zhao=param_s[l07_regions[0]],param_s[l07_regions[1]]	
		##COMPONENTES -- INTERNO/EXTERNO cD(any),INTERNO/EXTERNO E DAS CLASSIFICAÇÕES ZHAO
		param_interno_cd_zhao,param_externo_cd_zhao,param_interno_e_zhao,param_externo_e_zhao=param1[l07_regions[0]],param2[l07_regions[0]],param1[l07_regions[1]],param2[l07_regions[1]]
		##

		#CLASSIFICAÇÃO NOSSA / SUBGRUPOS MORFOLOGICOS DO ZHAO
		param_sersic_high_E,param_sersic_high_cD,param_sersic_high_misc=param_s[lim_region[0] & l07_regions[1]],param_s[lim_region[0] & l07_regions[0]],param_s[lim_region[0] & l07_regions[2]]
		param_sersic_low_left_E,param_sersic_low_left_cD,param_sersic_low_left_misc=param_s[lim_region[1] & l07_regions[1]],param_s[lim_region[1] & l07_regions[0]],param_s[lim_region[1] & l07_regions[2]]
		param_sersic_low_right_E,param_sersic_low_right_cD,param_sersic_low_right_misc=param_s[lim_region[2] & l07_regions[1]],param_s[lim_region[2] & l07_regions[0]],param_s[lim_region[2] & l07_regions[2]]
		#
		#COMPONENTES -- INTERNO/EXTERNO NOSSO COM SUBDIVISÃO POR ZHAO 
		param_interno_high_E,param_interno_high_cD,param_interno_high_misc=param1[lim_region[0] & l07_regions[1]],param1[lim_region[0] & l07_regions[0]],param1[lim_region[0] & l07_regions[2]]
		param_externo_high_E,param_externo_high_cD,param_externo_high_misc=param2[lim_region[0] & l07_regions[1]],param2[lim_region[0] & l07_regions[0]],param2[lim_region[0] & l07_regions[2]]
		#
		param_interno_low_left_E,param_interno_low_left_cD,param_interno_low_left_misc=param1[lim_region[1] & l07_regions[1]],param1[lim_region[1] & l07_regions[0]],param1[lim_region[1] & l07_regions[2]]
		param_externo_low_left_E,param_externo_low_left_cD,param_externo_low_left_misc=param2[lim_region[1] & l07_regions[1]],param2[lim_region[1] & l07_regions[0]],param2[lim_region[1] & l07_regions[2]]
		#
		param_interno_low_right_E,param_interno_low_right_cD,param_interno_low_right_misc=param1[lim_region[2] & l07_regions[1]],param1[lim_region[2] & l07_regions[0]],param1[lim_region[2] & l07_regions[2]]
		param_externo_low_right_E,param_externo_low_right_cD,param_externo_low_right_misc=param2[lim_region[2] & l07_regions[1]],param2[lim_region[2] & l07_regions[0]],param2[lim_region[2] & l07_regions[2]]
		###
		ks_param_cD_zhao=ks_2samp(param_sersic_high_cD,param_sersic_low_right_cD)[1],ks_2samp(param_sersic_low_left_cD,param_sersic_low_right_cD)[1],ks_2samp(param_sersic_high_cD,param_sersic_low_left_cD)[1]
		ks_param_intern_cD_zhao=ks_2samp(param_interno_high_cD,param_interno_low_right_cD)[1],ks_2samp(param_interno_low_left_cD,param_interno_low_right_cD)[1],ks_2samp(param_interno_high_cD,param_interno_low_left_cD)[1]
		ks_param_extern_cD_zhao=ks_2samp(param_externo_high_cD,param_externo_low_right_cD)[1],ks_2samp(param_externo_low_left_cD,param_externo_low_right_cD)[1],ks_2samp(param_externo_high_cD,param_externo_low_left_cD)[1]
		###
		vec_med_param_sersic_cD=med_param_sersic_high_cD,med_param_sersic_low_left_cD,med_param_sersic_low_right_cD=np.average(param_sersic_high_cD),np.average(param_sersic_low_left_cD),np.average(param_sersic_low_right_cD)
		vec_med_param_interno_cD=med_param_interno_high_cD,med_param_interno_low_left_cD,med_param_interno_low_right_cD=np.average(param_interno_high_cD),np.average(param_interno_low_left_cD),np.average(param_interno_low_right_cD)
		vec_med_param_externo_cD=med_param_interno_high_cD,med_param_externo_low_left_cD,med_param_externo_low_right_cD=np.average(param_externo_high_cD),np.average(param_externo_low_left_cD),np.average(param_externo_low_right_cD)

		##SÉRSIC -- KDE
		vec_kde_param_simples_zhao=param_sersic_kde_high_cD,param_sersic_kde_low_left_cD,param_sersic_kde_low_right_cD=kde(param_sersic_high_cD,bw_method=param_factor),kde(param_sersic_low_left_cD,bw_method=param_factor),kde(param_sersic_low_right_cD,bw_method=param_factor)
		##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
		vec_kde_param_intern_cD=param_interno_kde_high_cD,param_interno_kde_low_left_cD,param_interno_kde_low_right_cD=kde(param_interno_high_cD,bw_method=param_factor),kde(param_interno_low_left_cD,bw_method=param_factor),kde(param_interno_low_right_cD,bw_method=param_factor)
		vec_kde_param_extern_cD=param_externo_kde_high_cD,param_externo_kde_low_left_cD,param_externo_kde_low_right_cD=kde(param_externo_high_cD,bw_method=param_factor),kde(param_externo_low_left_cD,bw_method=param_factor),kde(param_externo_low_right_cD,bw_method=param_factor)
		##########################
		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - Modelos Simples - cD Zhao')
		for i,dist in enumerate(vec_kde_param_simples_zhao):
			axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}[cD]')
			axs.axvline(vec_med_param_sersic_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_sersic_cD[i]:.3e}$')
		axs.legend()
		axs.set_xlabel(par_labels[1][0])
		info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={ks_param_cD_zhao[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={ks_param_cD_zhao[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={ks_param_cD_zhao[2]:.3e}')
		fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_simples_kde_cD_zhao.png')
		plt.close()

		#MODELO COMPOSTO - COMPONENTE INTERNO

		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - Componente interno - cD Zhao')
		for i,dist in enumerate(vec_kde_param_intern_cD):
			axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}[cD]')
			axs.axvline(vec_med_param_interno_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_interno_cD[i]:.3e}$')
		axs.legend()
		axs.set_xlabel(par_labels[1][1])
		info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={ks_param_intern_cD_zhao[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={ks_param_intern_cD_zhao[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={ks_param_intern_cD_zhao[2]:.3e}')
		fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_interno_kde_cD_zhao.png')
		plt.close()

		#MODELO COMPOSTO - COMPONENTE EXTERNO
		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - Componente externo - cD Zhao')
		for i,dist in enumerate(vec_kde_param_extern_cD):
			axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}[cD]')
			axs.axvline(vec_med_param_externo_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_externo_cD[i]:.3f}$')
		axs.legend()
		axs.set_xlabel(par_labels[1][2])
		info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={ks_param_extern_cD_zhao[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={ks_param_extern_cD_zhao[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={ks_param_extern_cD_zhao[2]:.3e}')
		fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_externo_kde_cD_zhao.png')
		plt.close()
	return
def par_multi_hist(par_entry):
	"""Histogram version of `par_multi_kde`.

	Generates histograms (normalised) for the same splits, with optional
	x‑axis limits (`limx`).  L07 cD subsets are added when applicable.

	Args:
	par_entry (tuple): Similar structure to `par_multi_kde`, plus `region_names` and, optionally, `limx`.

	Returns:
	None.

	param=parametro de entrada
	lim_region=regiãoi de interesse e o corte que da a subamostra
	param_linspace=região que da para melhor olhar a amostra
	par_labels= labels das figuras de interesse
	save_labels= local de salvamento
	"""
	param,lim_region,l07_regions,par_labels,save_labels,region_names,limx=par_entry
	param_s,param1,param2=param
	########################
	#MODELO SIMPLES
	#MODELO COMPOSTO - COMPONENTE INTERNO
	#MODELO COMPOSTO - COMPONENTE EXTERNO
	model_type=[('Sérsic','simples'),('Componente interno','interno'),('Componente externo','externo')]
	bins_vec=[]
	for i in range(3):
		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - {model_type[i][0]}')
		if limx != None:
			count,bins,_=axs.hist(param[i][(param[i] >= limx[0]) & (param[i] <= limx[1])],bins='auto',color='black',density=True,histtype='step',label=f'{sample}')
			bins_vec.append(bins)
		else:
			count,bins,_=axs.hist(param[i],bins='auto',color='black',density=True,histtype='step',label=f'{sample}')
			bins_vec.append(bins)
		axs.axvline(np.average(param[i]),color='black',ls='--',label=fr'$\mu = {np.average(param[i]):.3f}$')
		axs.legend()
		axs.set_xlabel(par_labels[1][0])
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{sample}_{model_type[i][1]}_hist.png')
		plt.close()

		for j in range(len(lim_region)):
			fig,axs=plt.subplots(1,1,figsize=(10,5))
			plt.title(f'{par_labels[0]} - {model_type[i][0]}')
			axs.hist(param[i][lim_region[j]],bins=bins,histtype='step',density=True,color=cores[j],label=f'{names_simples[j]}')
			axs.axvline(np.average(param[i][lim_region[j]]),color=cores[j],ls='--',label=fr'$\mu = {np.average(param[i][lim_region[j]]):.3f}$')
			axs.legend()
			axs.set_xlabel(par_labels[1][0])
			plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{region_names[j]}_{model_type[i][1]}_hist.png')
			plt.close()
	if sample == 'L07':
		for i in range(3):
			bins=bins_vec[i]
			for j in range(len(lim_region)):
				morf_region=lim_region[j] & l07_regions[0]
				fig,axs=plt.subplots(1,1,figsize=(10,5))
				plt.title(f'{par_labels[0]} - {model_type[i][0]} - cD Zhao')
				axs.hist(param[i][morf_region],bins=bins,histtype='step',density=True,color=cores[j],label=f'{names_simples[j]}[cD]')
				axs.axvline(np.average(param[i][morf_region]),color=cores[j],ls='--',label=fr'$\mu = {np.average(param[i][morf_region]):.3f}$')
				axs.legend()
				axs.set_xlabel(par_labels[1][0])
				plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{region_names[j]}_{model_type[i][1]}_hist_cD.png')
				plt.close()
	return
def par_multi_kde_stripes(stripe,stripe_med,lim_region_z,to_do_test,pasta_analise,pasta_save,par_stripe_entry):
	"""KDE for a single bin of a global variable (e.g., redshift slice or
	M200 bin).

	If `to_do_test == 'feito'`, only the mean and standard error of the
	mean (SEM) are returned without generating any plot – this is used to
	collect statistics for the tracer plots in `analise_m200`.

	Otherwise, three KDE comparison figures (Sérsic, internal, external)
	are saved.

	Args:
		stripe           (int): Bin index.
		stripe_med       (float): Median of the bin variable (e.g., mean z).
		lim_region_z     (list of masks): Morphological masks for this bin.
		to_do_test       (str): 'feito' → return stats only.
		pasta_analise    (str): Parent analysis directory.
		pasta_save       (str): Sub‑directory for this bin.
		par_stripe_entry (tuple): (param, param_linspace, par_labels, save_labels) where `param` is a tuple of three arrays.

	Returns:
		tuple: (med_cond, sem_cond) – each a (3,3) array:
		Row: morphological class (high, low_left, low_right).
		Column: parameter type (Sérsic, internal, external).
		Values are mean and SEM.
	"""


	"""
	param=parametro de entrada
	param_linspace=região que da para melhor olhar a amostra
	par_labels= labels das figuras de interesse
	save_labels= local de salvamento
	"""
	param,param_linspace,par_labels,save_labels=par_stripe_entry
	param_s,param1,param2=param
	##SÉRSIC
	param_sersic_high,param_sersic_low_left,param_sersic_low_right=param_s[lim_region_z[0]],param_s[lim_region_z[1]],param_s[lim_region_z[2]]
	##COMPONENTES -- INTERNO/EXTERNO
	param_interno_high,param_externo_high,param_interno_low_left,param_externo_low_left,param_interno_low_right,param_externo_low_right=param1[lim_region_z[0]],param2[lim_region_z[0]],param1[lim_region_z[1]],param2[lim_region_z[1]],param1[lim_region_z[2]],param2[lim_region_z[2]]

	ks_param_sersic=ks_calc([param_sersic_high,param_sersic_low_left,param_sersic_low_right])
	ks_param_intern=ks_calc([param_interno_high,param_interno_low_left,param_interno_low_right])
	ks_param_extern=ks_calc([param_externo_high,param_externo_low_left,param_externo_low_right])

	param_kde_entry=np.hstack((param_s,param1,param2))
	kde_param=kde(param_kde_entry)
	param_factor = kde_param.factor

	vec_ks_param_sersic=ks_param_sersic[0][0][2],ks_param_sersic[0][1][2],ks_param_sersic[0][0][2]
	vec_ks_param_intern=ks_param_intern[0][0][2],ks_param_intern[0][1][2],ks_param_intern[0][0][2]
	vec_ks_param_extern=ks_param_extern[0][0][2],ks_param_extern[0][1][2],ks_param_extern[0][0][2]
	#
	vec_med_param_sersic=med_param_sersic_high,med_param_sersic_low_left,med_param_sersic_low_right=np.average(param_sersic_high),np.average(param_sersic_low_left),np.average(param_sersic_low_right)
	vec_med_param_interno=med_param_interno_high,med_param_interno_low_left,med_param_interno_low_right=np.average(param_interno_high),np.average(param_interno_low_left),np.average(param_interno_low_right)
	vec_med_param_externo=med_param_interno_high,med_param_externo_low_left,med_param_externo_low_right=np.average(param_externo_high),np.average(param_externo_low_left),np.average(param_externo_low_right)

	vec_sem_param_sersic=sem_param_sersic_high,sem_param_sersic_low_left,sem_param_sersic_low_right=sem(param_sersic_high),sem(param_sersic_low_left),sem(param_sersic_low_right)
	vec_sem_param_interno=sem_param_interno_high,sem_param_interno_low_left,sem_param_interno_low_right=sem(param_interno_high),sem(param_interno_low_left),sem(param_interno_low_right)
	vec_sem_param_externo=sem_param_interno_high,sem_param_externo_low_left,sem_param_externo_low_right=sem(param_externo_high),sem(param_externo_low_left),sem(param_externo_low_right)
	#

	##SÉRSIC -- cD,E(EL),cD(std),E
	vec_kde_param_simples=param_sersic_kde_high,param_sersic_kde_low_left,param_sersic_kde_low_right=kde(param_sersic_high,bw_method=param_factor),kde(param_sersic_low_left,bw_method=param_factor),kde(param_sersic_low_right,bw_method=param_factor)
	##COMPONENTES -- INTERNO/EXTERNO
	vec_kde_param_intern=param_interno_kde_high,param_interno_kde_low_left,param_interno_kde_low_right=kde(param_interno_high,bw_method=param_factor),kde(param_interno_low_left,bw_method=param_factor),kde(param_interno_low_right,bw_method=param_factor)
	vec_kde_param_extern=param_externo_kde_high,param_externo_kde_low_left,param_externo_kde_low_right=kde(param_externo_high,bw_method=param_factor),kde(param_externo_low_left,bw_method=param_factor),kde(param_externo_low_right,bw_method=param_factor)

	med_cond=vec_med_param_sersic,vec_med_param_interno,vec_med_param_externo
	sem_cond=vec_sem_param_sersic,vec_sem_param_interno,vec_sem_param_externo

	if to_do_test == 'feito':
		return med_cond,sem_cond
	############################################
	#MODELO SIMPLES
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - {stripe_med} - Modelos Simples')
	for i,dist in enumerate(vec_kde_param_simples):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_sersic[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_sersic[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][0])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_sersic[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_sersic[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_sersic[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/{pasta_analise}/{save_labels[0]}/{pasta_save}/{save_labels[1]}_simples_kde_{stripe}.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE INTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - {stripe_med} - Componente interno')
	for i,dist in enumerate(vec_kde_param_intern):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_interno[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_interno[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][1])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_intern[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_intern[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_intern[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/{pasta_analise}/{save_labels[0]}/{pasta_save}/{save_labels[1]}_interno_kde_{stripe}.png')
	plt.close()

	#MODELO COMPOSTO - COMPONENTE EXTERNO
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} - {stripe_med} - Componente externo')
	for i,dist in enumerate(vec_kde_param_extern):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param_externo[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_externo[i]:.3f}$')
	axs.legend()
	axs.set_xlabel(par_labels[1][2])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param_extern[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param_extern[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param_extern[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/{pasta_analise}/{save_labels[0]}/{pasta_save}/{save_labels[1]}_externo_kde_{stripe}.png')
	plt.close()
	return med_cond,sem_cond
################################
def par_only_kde(par_entry):
	"""KDE plot for a single parameter (not split by component type).

	The logic mirrors `par_multi_kde` but for a single vector instead of
	three.  It also adds a sub‑plot for cD galaxies when sample='L07'.

	Args:
	    par_entry: (param, lim_region, l07_regions, param_linspace, par_labels, save_labels)

	Returns:
	    None.

	param=parametro de entrada
	lim_region=regiãoi de interesse e o corte que da a subamostra
	param_linspace=região que da para melhor olhar a amostra
	par_labels= labels das figuras de interesse
	save_labels= local de salvamento
	"""
	param,lim_region,l07_regions,param_linspace,par_labels,save_labels=par_entry
	param_high,param_low_left,param_low_right=param[lim_region[0]],param[lim_region[1]],param[lim_region[2]]
	ks_param=ks_calc([param_high,param_low_left,param_low_right])

	vec_ks_param=ks_param[0][0][2],ks_param[0][1][2],ks_param[0][0][1]

	##média
	kde_param=kde(param)
	param_factor = kde_param.factor

	vec_kde_param=param_kde_high,param_kde_low_left,param_kde_low_right=kde(param_high,bw_method=param_factor),kde(param_low_left,bw_method=param_factor),kde(param_low_right,bw_method=param_factor)

	vec_med_param=med_param_high,med_param_low_left,med_param_low_right=np.average(param_high),np.average(param_low_left),np.average(param_low_right)
	
	###########################
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]}') 
	for i,dist in enumerate(vec_kde_param):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(par_labels[1])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[2]})={vec_ks_param[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=8, va='center', ha='left')
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_kde.png')
	plt.close()

	if sample=='L07' and save_labels[1] != 'hline':
		#param - E,cD,E/cD DO ZHAO
		param_cd_zhao,param_e_zhao,param_misc_zhao=param[l07_regions[0]],param[l07_regions[1]],param[l07_regions[2]]
		#param - SUBGRUPO cD DO ZHAO
		param_high_cD,param_low_left_cD,param_low_right_cD=param[lim_region[0] & l07_regions[0]],param[lim_region[1] & l07_regions[0]],param[lim_region[2] & l07_regions[0]]
		
		##KDE
		param_kde_cd_zhao,param_kde_e_zhao=kde(param_cd_zhao,bw_method=param_factor),kde(param_e_zhao,bw_method=param_factor)
		vec_kde_param_cD=param_kde_high_cD,param_kde_low_left_cD,param_kde_low_right_cD=kde(param_high_cD,bw_method=param_factor),kde(param_low_left_cD,bw_method=param_factor),kde(param_low_right_cD,bw_method=param_factor)

		#MÉDIAS
		vec_med_param_cD=med_param_high_cD,med_param_low_left_cD,med_param_low_right_cD=np.average(param_high_cD),np.average(param_low_left_cD),np.average(param_low_right_cD)
		#
		ks_param_cD_zhao=ks_2samp(param_high_cD,param_low_right_cD)[1],ks_2samp(param_low_left_cD,param_low_right_cD)[1],ks_2samp(param_high_cD,param_low_left_cD)[1]
		##
		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - cDs do ZHAO')
		for i,dist in enumerate(vec_kde_param_cD):
			axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
			axs.axvline(vec_med_param_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param_cD[i]:.3e}$')
		axs.legend()
		axs.set_xlabel(par_labels[1])
		info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={ks_param_cD_zhao[0]:.3e} ' f'K-S({names_simples[1]},{names_simples[2]})={ks_param_cD_zhao[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={ks_param_cD_zhao[2]:.3e}')
		fig.text(0.65, 0.95, info_labels, fontsize=9, va='center', ha='left')
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_kde_cD_zhao.png')
		plt.close()
	return
def par_only_hist(par_entry):
	"""Histogram equivalent of `par_only_kde`.

	Args:
		par_entry: Similar tuple plus `region_names` and optional `limx`.

	Returns:
		None.
	param=parametro de entrada
	lim_region=regiãoi de interesse e o corte que da a subamostra
	param_linspace=região que da para melhor olhar a amostra
	par_labels= labels das figuras de interesse
	save_labels= local de salvamento
	"""
	param,lim_region,l07_regions,par_labels,save_labels,region_names,limx=par_entry
	########################
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]}')
	if limx != None:
		count,bins,_=axs.hist(param[(param >= limx[0]) & (param <= limx[1])],bins='auto',color='black',density=True,histtype='step',label=f'{sample}')
	else:
		count,bins,_=axs.hist(param,bins='auto',color='black',density=True,histtype='step',label=f'{sample}')
	axs.legend()
	axs.axvline(np.average(param),color='black',ls='--',label=fr'$\mu = {np.average(param):.3f}$')
	axs.set_xlabel(par_labels[1][0])
	plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{sample}_hist.png')
	plt.close()

	for j in range(len(lim_region)):
		fig,axs=plt.subplots(1,1,figsize=(10,5))
		plt.title(f'{par_labels[0]} - {region_names[j]}')
		axs.hist(param[lim_region[j]],bins=bins,histtype='step',density=True,color=cores[j],label=f'{names_simples[j]}')
		axs.axvline(np.average(param[lim_region[j]]),color=cores[j],ls='--',label=fr'$\mu = {np.average(param[lim_region[j]]):.3f}$')
		axs.legend()
		axs.set_xlabel(par_labels[1][0])
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{region_names[j]}_hist.png')
		plt.close()
		if sample == 'L07':
			morf_region=lim_region[j] & l07_regions[0]
			fig,axs=plt.subplots(1,1,figsize=(10,5))
			plt.title(f'{par_labels[0]} - {region_names[j]} - cD Zhao')
			axs.hist(param[morf_region],bins=bins,histtype='step',density=True,color=cores[j],label=f'{names_simples[j]}[cD]')
			axs.axvline(np.average(param[morf_region]),color=cores[j],ls='--',label=fr'$\mu = {np.average(param[morf_region]):.3f}$')
			axs.legend()
			axs.set_xlabel(par_labels[1][0])
			plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_{region_names[j]}_hist_cD.png')
			plt.close()
	return

def par_only_kde_stripes(stripe,stripe_med,lim_region_z,to_do_test,pasta_analise,pasta_save,par_stripe_entry):
	"""Single‑parameter KDE inside a bin (for redshift/M200 analysis).

	Returns mean and SEM if `to_do_test == 'feito'`.

	Args:
		stripe, stripe_med, lim_region_z, to_do_test, pasta_analise,
		pasta_save: as in `par_multi_kde_stripes`.
		par_stripe_entry: (param, param_linspace, par_labels, save_labels) where `param` is a single array.

	Returns:
		tuple: (vec_med_param, vec_sem_param) – arrays of length 3
		(mean and SEM for the three morphological classes).
	"""
	param,param_linspace,par_labels,save_labels=par_stripe_entry
	param_high,param_low_left,param_low_right=param[lim_region_z[0]],param[lim_region_z[1]],param[lim_region_z[2]]
	
	ks_param=ks_calc([param_high,param_low_left,param_low_right])

	vec_ks_param=ks_param[0][0][2],ks_param[0][1][2],ks_param[0][0][2]

	##média
	kde_param=kde(param)
	param_factor = kde_param.factor

	vec_kde_param=param_kde_high,param_kde_low_left,param_kde_low_right=kde(param_high,bw_method=param_factor),kde(param_low_left,bw_method=param_factor),kde(param_low_right,bw_method=param_factor)

	vec_med_param=med_param_high,med_param_low_left,med_param_low_right=np.average(param_high),np.average(param_low_left),np.average(param_low_right)
	vec_sem_param=sem_param_high,sem_param_low_left,sem_param_low_right=sem(param_high),sem(param_low_left),sem(param_low_right)

	if to_do_test == 'feito':
		return vec_med_param,vec_sem_param 
	########################
	fig,axs=plt.subplots(1,1,figsize=(10,5))
	plt.title(f'{par_labels[0]} {stripe_med}')
	for i,dist in enumerate(vec_kde_param):
		axs.plot(param_linspace,dist(param_linspace),color=cores[i],lw=line_width[i],alpha=alpha_vec[i],label=f'{names_simples[i]}')
		axs.axvline(vec_med_param[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_param[i]:.3e}$')
	axs.legend()
	axs.set_xlabel(par_labels[1])
	info_labels = (f'K-S({names_simples[0]},{names_simples[2]}) ={vec_ks_param[0]:.3e}\n' f'K-S({names_simples[1]},{names_simples[2]})={vec_ks_param[1]:.3e}\n' f'K-S({names_simples[0]},{names_simples[1]})={vec_ks_param[2]:.3e}')
	fig.text(0.7, 0.95, info_labels, fontsize=11, va='center', ha='left')
	plt.savefig(f'{save_path}/{pasta_analise}/{save_labels[0]}/{pasta_save}/{save_labels[1]}_kde_{stripe}.png')
	plt.close()

	return vec_med_param,vec_sem_param
####################################################
def casjobs_jointplots(info_need,lim_region,l07_regions):
	"""Create a large set of joint scatter+KDE marginal plots for CASJOBS
	parameters (stellar mass, age, M200, component‑wise stellar masses).

	Each plot includes:
		- Scatter with linear fits and bootstrapped errors for the three
		morphological sub‑samples.
		- Top and right marginal KDE curves.
		- Special versions for the L07 sample showing cD subsets.

	**This function uses a huge number of global arrays** (starmass, age,
	m200_temp, bt_vec_corr, etc.).  It is designed to be called directly
	after they are defined.

	Args:
		info_need  (tuple): (bt_vec_corr, starmass)
		lim_region (list of masks): For the CASJOBS‑available sample.
		l07_regions (tuple of masks or None): L07 morphological subsets.

	Returns:
		None.  Many PNG files are saved.
	"""


	#######################################################
	starmass_high,starmass_low_left,starmass_low_right=starmass[lim_region[0]],starmass[lim_region[1]],starmass[lim_region[2]]

	kde_starmass=kde(starmass)
	starmass_factor = kde_starmass.factor
	starmass_linspace=np.linspace(10.8,max(starmass),3000)
	vec_kde_starmass=starmass_kde_high,starmass_kde_low_left,starmass_kde_low_right=kde(starmass_high,bw_method=starmass_factor),kde(starmass_low_left,bw_method=starmass_factor),kde(starmass_low_right,bw_method=starmass_factor)

	vec_med_starmass=med_starmass_high,med_starmass_low_left,med_starmass_low_right=np.average(starmass_high),np.average(starmass_low_left),np.average(starmass_low_right)
	###############################################
	age_high,age_low_left,age_low_right=age[lim_region[0]],age[lim_region[1]],age[lim_region[2]]

	kde_age=kde(age)
	age_factor = kde_age.factor
	age_linspace=np.linspace(min(age),max(age),3000)
	vec_kde_age=age_kde_high,age_kde_low_left,age_kde_low_right=kde(age_high,bw_method=age_factor),kde(age_low_left,bw_method=age_factor),kde(age_low_right,bw_method=age_factor)
	vec_med_age=med_age_high,med_age_low_left,med_age_low_right=np.average(age_high),np.average(age_low_left),np.average(age_low_right)

	##############################################
	m200=m200_temp[lim_casjobs]
	m200_high,m200_low_left,m200_low_right=m200[lim_region[0]],m200[lim_region[1]],m200[lim_region[2]]

	kde_m200=kde(m200)
	m200_factor = kde_m200.factor
	m200_linspace=np.linspace(min(m200),max(m200),3000)
	vec_kde_m200=m200_kde_high,m200_kde_low_left,m200_kde_low_right=kde(m200_high,bw_method=m200_factor),kde(m200_low_left,bw_method=m200_factor),kde(m200_low_right,bw_method=m200_factor)
	vec_med_m200=med_m200_high,med_m200_low_left,med_m200_low_right=np.average(m200_high),np.average(m200_low_left),np.average(m200_low_right)
	
	if sample=='L07':
		##slope - E,cD,E/cD DO ZHAO
		starmass_cD_zhao,starmass_e_zhao,starmass_misc_zhao=starmass[l07_regions[0]],starmass[l07_regions[1]],starmass[l07_regions[2]]
		##slope - SUBGRUPO cD DO ZHAO
		starmass_high_cD,starmass_low_left_cD,starmass_low_right_cD=starmass[lim_region[0] & l07_regions[0]],starmass[lim_region[1] & l07_regions[0]],starmass[lim_region[2] & l07_regions[0]]

		##slope - KDE
		vec_kde_starmass_zhao=starmass_kde_cD_zhao,starmass_kde_e_zhao=kde(starmass_cD_zhao,bw_method=starmass_factor),kde(starmass_e_zhao,bw_method=starmass_factor)
		vec_kde_starmass_cD=starmass_kde_high_cD,starmass_kde_low_left_cD,starmass_kde_low_right_cD=kde(starmass_high_cD,bw_method=starmass_factor),kde(starmass_low_left_cD,bw_method=starmass_factor),kde(starmass_low_right_cD,bw_method=starmass_factor)

		#MÉDIAS
		vec_med_starmass_zhao=med_starmass_cD_zhao,med_starmass_e_zhao=np.average(starmass_cD_zhao),np.average(starmass_e_zhao)
		vec_med_starmass_cD=med_starmass_high,med_starmass_low_left,med_starmass_low_right=np.average(starmass_high_cD),np.average(starmass_low_left_cD),np.average(starmass_low_right_cD)
		###########################
		###########################
		##slope - E,cD,E/cD DO ZHAO
		age_cD_zhao,age_e_zhao,age_misc_zhao=age[l07_regions[0]],age[l07_regions[1]],age[l07_regions[2]]
		##slope - SUBGRUPO cD DO ZHAO
		age_high_cD,age_low_left_cD,age_low_right_cD=age[lim_region[0] & l07_regions[0]],age[lim_region[1] & l07_regions[0]],age[lim_region[2] & l07_regions[0]]
		
		##slope - KDE
		vec_kde_age_zhao=age_kde_cD_zhao,age_kde_e_zhao=kde(age_cD_zhao,bw_method=age_factor),kde(age_e_zhao,bw_method=age_factor)
		vec_kde_age_cD=age_kde_high_cD,age_kde_low_left_cD,age_kde_low_right_cD=kde(age_high_cD,bw_method=age_factor),kde(age_low_left_cD,bw_method=age_factor),kde(age_low_right_cD,bw_method=age_factor)
		#MÉDIAS
		vec_med_age_zhao=med_age_cD_zhao,med_age_e_zhao=np.average(age_cD_zhao),np.average(age_e_zhao)
		vec_med_age_cD=med_age_high_cD,med_age_low_left_cD,med_age_low_right_cD=np.average(age_high_cD),np.average(age_low_left_cD),np.average(age_low_right_cD)
		###########################
		###########################
		##slope - E,cD,E/cD DO ZHAO
		m200_cD_zhao,m200_e_zhao,m200_misc_zhao=m200[l07_regions[0]],m200[l07_regions[1]],m200[l07_regions[2]]
		##slope - SUBGRUPO cD DO ZHAO
		m200_high_cD,m200_low_left_cD,m200_low_right_cD=m200[lim_region[0] & l07_regions[0]],m200[lim_region[1] & l07_regions[0]],m200[lim_region[2] & l07_regions[0]]
		
		##slope - KDE
		vec_kde_m200_zhao=m200_kde_cD_zhao,m200_kde_e_zhao=kde(m200_cD_zhao,bw_method=m200_factor),kde(m200_e_zhao,bw_method=m200_factor)
		vec_kde_m200_cD=m200_kde_high_cD,m200_kde_low_left_cD,m200_kde_low_right_cD=kde(m200_high_cD,bw_method=m200_factor),kde(m200_low_left_cD,bw_method=m200_factor),kde(m200_low_right_cD,bw_method=m200_factor)

		#MÉDIAS
		vec_med_m200_zhao=med_m200_cD_zhao,med_m200_e_zhao=np.average(m200_cD_zhao),np.average(m200_e_zhao)
		vec_med_m200_cD=med_m200_cd,med_m200_low_left,med_m200_low_right=np.average(m200_high_cD),np.average(m200_low_left_cD),np.average(m200_low_right_cD)
	#######################################

	#JOINTPLOT DA MASSA x M200

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

	ajust_high,cov_high=np.polyfit(m200_high,starmass_high,1,cov=True)
	ajust_low_left,cov_low_left=np.polyfit(m200_low_left,starmass_low_left,1,cov=True)
	ajust_low_right,cov_low_right=np.polyfit(m200_low_right,starmass_low_right,1,cov=True)

	ax_center.scatter(m200_high,starmass_high,marker='o',edgecolor='black',label=names_simples[0],color=cores[0])
	ax_center.scatter(m200_low_left,starmass_low_left,marker='o',edgecolor='black',label=names_simples[1],color=cores[1])
	ax_center.scatter(m200_low_right,starmass_low_right,marker='o',edgecolor='black',label=names_simples[2],color=cores[2])
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_high),color=cores[0],label=fr'$\alpha$={ajust_high[0]:.3f}$\pm${np.sqrt(cov_high[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_high[1]:.3f}$\pm${np.sqrt(cov_high[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_low_left),color=cores[1],label=fr'$\alpha$={ajust_low_left[0]:.3f}$\pm${np.sqrt(cov_low_left[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_left[1]:.3f}$\pm${np.sqrt(cov_low_left[1,1]):.3f}')
	ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_low_right),color=cores[2],label=fr'$\alpha$={ajust_low_right[0]:.3f}$\pm${np.sqrt(cov_low_right[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_right[1]:.3f}$\pm${np.sqrt(cov_low_right[1,1]):.3f}')
	ax_center.legend(fontsize='x-small')
	ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	ax_center.set_ylabel(r'$\log M_{\bigstar}$')

	for i,dist in enumerate(vec_kde_m200):
		ax_topx.plot(m200_linspace,dist(m200_linspace),color=cores[i],label=f'{names_simples[i]}')
		ax_topx.axvline(vec_med_m200[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_m200[i]:.3f}$')
	ax_topx.legend(fontsize='x-small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_starmass):
		ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cores[i],label=f'{names_simples[i]}')
		ax_righty.axhline(vec_med_starmass[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass[i]:.3f}$')
	ax_righty.legend(fontsize='x-small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{save_path}/test_ks/starmass_m200.png')
	plt.close()

	#JOINTPLOT DA MASSA x IDADE ESTELAR

	fig = plt.figure(figsize=(10, 10))
	gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
	ax_center = fig.add_subplot(gs[1,0])
	ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
	ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)
	ajust_high,cov_high=np.polyfit(starmass_high,age_high,1,cov=True)
	print(ajust_high)
	ajust_low_left,cov_low_left=np.polyfit(starmass_low_left,age_low_left,1,cov=True)
	ajust_low_right,cov_low_right=np.polyfit(starmass_low_right,age_low_right,1,cov=True)

	ax_center.scatter(starmass_high,age_high,marker='o',edgecolor='black',label=names_simples[0],color=cores[0])
	ax_center.scatter(starmass_low_left,age_low_left,marker='o',edgecolor='black',label=names_simples[1],color=cores[1])
	ax_center.scatter(starmass_low_right,age_low_right,marker='o',edgecolor='black',label=names_simples[2],color=cores[2])
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_high),color=cores[0],label=fr'$\alpha$={ajust_high[0]:.3f}$\pm${np.sqrt(cov_high[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_high[1]:.3f}$\pm${np.sqrt(cov_high[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_low_left),color=cores[1],label=fr'$\alpha$={ajust_low_left[0]:.3f}$\pm${np.sqrt(cov_low_left[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_left[1]:.3f}$\pm${np.sqrt(cov_low_left[1,1]):.3f}')
	ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_low_right),color=cores[2],label=fr'$\alpha$={ajust_low_right[0]:.3f}$\pm${np.sqrt(cov_low_right[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_right[1]:.3f}$\pm${np.sqrt(cov_low_right[1,1]):.3f}')
	ax_center.legend()
	ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
	ax_center.set_ylabel(r'$\tau$ (Gyr)')
	ax_center.set_xlabel(r'$\log M_{\bigstar}$')

	for i,dist in enumerate(vec_kde_starmass):
		ax_topx.plot(starmass_linspace,dist(starmass_linspace),color=cores[i],label=f'{names_simples[i]}')
		ax_topx.axvline(vec_med_starmass[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass[i]:.3f}$')
	ax_topx.legend(fontsize='small')
	ax_topx.tick_params(labelbottom=False)

	for i,dist in enumerate(vec_kde_age):
		ax_righty.plot(dist(age_linspace),age_linspace,color=cores[i],label=f'{names_simples[i]}')
		ax_righty.axhline(vec_med_age[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_age[i]:.3f}$')
	ax_righty.legend(fontsize='small')
	ax_righty.tick_params(labelleft=False)
	plt.savefig(f'{save_path}/test_ks/starmass_age.png')
	plt.close()
	##############################################################
	info_need=bt_vec_corr,starmass
	mass_c1,mass_c2=mass_calc(info_need)

	mass_c1_high,mass_c1_low_left,mass_c1_low_right=mass_c1[lim_region[0]],mass_c1[lim_region[1]],mass_c1[lim_region[2]]
	mass_c2_high,mass_c2_low_left,mass_c2_low_right=mass_c2[lim_region[0]],mass_c2[lim_region[1]],mass_c2[lim_region[2]]
	##
	vec_med_mass_c1=med_mass_c1_high,med_mass_c1_low_left,med_mass_c1_low_right=np.average(mass_c1_high),np.average(mass_c1_low_left),np.average(mass_c1_low_right)
	vec_med_mass_c2=med_mass_c2_high,med_mass_c2_low_left,med_mass_c2_low_right=np.average(mass_c2_high),np.average(mass_c2_low_left),np.average(mass_c2_low_right)
	##
	mass_c1_kde_entry=mass_c1
	kde_mass_c1=kde(mass_c1_kde_entry)
	mass_c1_factor = kde_mass_c1.factor
	mass_c1_linspace=np.linspace(min(mass_c1_kde_entry),max(mass_c1_kde_entry),3000)
	##
	mass_c2_kde_entry=mass_c2
	kde_mass_c2=kde(mass_c2_kde_entry)
	mass_c2_factor = kde_mass_c2.factor
	mass_c2_linspace=np.linspace(min(mass_c2_kde_entry),max(mass_c2_kde_entry),3000)
	##
	##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
	vec_kde_mass_c1=mass_c1_kde_high,mass_c1_kde_low_left,mass_c1_kde_low_right=kde(mass_c1_high,bw_method=mass_c1_factor),kde(mass_c1_low_left,bw_method=mass_c1_factor),kde(mass_c1_low_right,bw_method=mass_c1_factor)
	vec_kde_mass_c2=mass_c2_kde_high,mass_c2_kde_low_left,mass_c2_kde_low_right=kde(mass_c2_high,bw_method=mass_c2_factor),kde(mass_c2_low_left,bw_method=mass_c2_factor),kde(mass_c2_low_right,bw_method=mass_c2_factor)
	###########################

	fig,axs=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE INTERNO')

	ajust_c1_high,cov_c1_high=np.polyfit(m200_high,mass_c1_high,1,cov=True)
	ajust_c1_low_left,cov_c1_low_left=np.polyfit(m200_low_left,mass_c1_low_left,1,cov=True)
	ajust_c1_low_right,cov_c1_low_right=np.polyfit(m200_low_right,mass_c1_low_right,1,cov=True)

	ajust_c2_high,cov_c2_high=np.polyfit(m200_high,mass_c2_high,1,cov=True)
	ajust_c2_low_left,cov_c2_low_left=np.polyfit(m200_low_left,mass_c2_low_left,1,cov=True)
	ajust_c2_low_right,cov_c2_low_right=np.polyfit(m200_low_right,mass_c2_low_right,1,cov=True)

	axs[0].scatter(m200_high,mass_c1_high,marker='o',edgecolor='black',alpha=0.6,label=names_simples[0],color=cores[0])
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_high),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_high[0]:.3f}$\pm${np.sqrt(cov_c1_high[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_high[1]:.3f}$\pm${np.sqrt(cov_c1_high[1,1]):.3f}')

	axs[1].scatter(m200_low_left,mass_c1_low_left,marker='o',edgecolor='black',alpha=0.6,label=names_simples[1],color=cores[1])
	axs[1].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_low_left),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_low_left[0]:.3f}$\pm${np.sqrt(cov_c1_low_left[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_low_left[1]:.3f}$\pm${np.sqrt(cov_c1_low_left[1,1]):.3f}')

	axs[2].scatter(m200_low_right,mass_c1_low_right,marker='o',edgecolor='black',alpha=0.6,label=names_simples[2],color=cores[2])
	axs[2].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[2].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_low_right),color='black',ls='-',label=fr'$\alpha$={ajust_c1_low_right[0]:.3f}$\pm${np.sqrt(cov_c1_low_right[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_low_right[1]:.3f}$\pm${np.sqrt(cov_c1_low_right[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{save_path}/test_ks/m200_mass_c1.png')
	plt.close()

	fig,axs=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
	plt.suptitle('COMPONENTE EXTERNO')

	axs[0].scatter(m200_high,mass_c2_high,marker='o',edgecolor='black',alpha=0.6,label=names_simples[0],color=cores[0])
	axs[0].set_ylim(9.2,12.2)
	axs[0].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[0].set_ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')
	axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_high),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_high[0]:.3f}$\pm${np.sqrt(cov_c2_high[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_high[1]:.3f}$\pm${np.sqrt(cov_c2_high[1,1]):.3f}')

	axs[1].scatter(m200_low_left,mass_c2_low_left,marker='o',edgecolor='black',alpha=0.6,label=names_simples[1],color=cores[1])
	axs[1].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_low_left),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_low_left[0]:.3f}$\pm${np.sqrt(cov_c2_low_left[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_low_left[1]:.3f}$\pm${np.sqrt(cov_c2_low_left[1,1]):.3f}')

	axs[2].scatter(m200_low_right,mass_c2_low_right,marker='o',edgecolor='black',alpha=0.6,label=names_simples[2],color=cores[2])
	axs[2].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
	axs[2].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_low_right),color='black',ls='-',label=fr'$\alpha$={ajust_c2_low_right[0]:.3f}$\pm${np.sqrt(cov_c2_low_right[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_low_right[1]:.3f}$\pm${np.sqrt(cov_c2_low_right[1,1]):.3f}')

	fig.legend()
	plt.savefig(f'{save_path}/test_ks/m200_mass_c2.png')
	plt.close()
	if sample == 'L07':
		mass_c1_cD_zhao,mass_c1_e_zhao=mass_c1[l07_regions[0]],mass_c1[l07_regions[1]]
		mass_c2_cD_zhao,mass_c2_e_zhao=mass_c2[l07_regions[0]],mass_c2[l07_regions[1]]

		mass_c1_high_cD,mass_c1_low_left_cD,mass_c1_low_right_cD=mass_c1[lim_region[0] & l07_regions[0]],mass_c1[lim_region[1] & l07_regions[0]],mass_c1[lim_region[2] & l07_regions[0]]
		mass_c2_high_cD,mass_c2_low_left_cD,mass_c2_low_right_cD=mass_c2[lim_region[0] & l07_regions[0]],mass_c2[lim_region[1] & l07_regions[0]],mass_c2[lim_region[2] & l07_regions[0]]
		###
		vec_med_mass_c1_cD=med_mass_c1_high_cD,med_mass_c1_small_cD,med_mass_c1_low_right_cD=np.average(mass_c1_high_cD),np.average(mass_c1_low_left_cD),np.average(mass_c1_low_right_cD)
		vec_med_mass_c2_cD=med_mass_c2_high_cD,med_mass_c2_small_cD,med_mass_c2_low_right_cD=np.average(mass_c2_high_cD),np.average(mass_c2_low_left_cD),np.average(mass_c2_low_right_cD)
		##COMPONENTES -- INTERNO cD(any),EXTERNO cD(any),INTERNO cD (std),EXTERNO cD(std),INTERNO E(EL),EXTERNO E(EL) - KDE
		vec_kde_mass_c1_cD=mass_c1_kde_high_cD,mass_c1_kde_small_cD,mass_c1_kde_low_right_cD=kde(mass_c1_high_cD,bw_method=mass_c1_factor),kde(mass_c1_low_left_cD,bw_method=mass_c1_factor),kde(mass_c1_low_right_cD,bw_method=mass_c1_factor)
		vec_kde_mass_c2_cD=mass_c2_kde_high_cD,mass_c2_kde_small_cD,mass_c2_kde_low_right_cD=kde(mass_c2_high_cD,bw_method=mass_c2_factor),kde(mass_c2_low_left_cD,bw_method=mass_c2_factor),kde(mass_c2_low_right_cD,bw_method=mass_c2_factor)
	
		#JOINTPLOT DA MASSA x M200 -- cDS DO ZHAO PELA NOSSA CLASSIFICAÇÃO

		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ajust_high_cD,cov_high_cD=np.polyfit(m200_high_cD,starmass_high_cD,1,cov=True)
		ajust_low_left_cD,cov_low_left_cD=np.polyfit(m200_low_left_cD,starmass_low_left_cD,1,cov=True)
		ajust_low_right_cD,cov_low_right_cD=np.polyfit(m200_low_right_cD,starmass_low_right_cD,1,cov=True)

		ax_center.scatter(m200_high_cD,starmass_high_cD,marker='o',edgecolor='black',label=f'{names_simples[0]}[cD]',color=cores[0])
		ax_center.scatter(m200_low_left_cD,starmass_low_left_cD,marker='o',edgecolor='black',label=f'{names_simples[1]}[cD]',color=cores[1])
		ax_center.scatter(m200_low_right_cD,starmass_low_right_cD,marker='o',edgecolor='black',label=f'{names_simples[2]}[cD]',color=cores[2])
		ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_high_cD),color=cores[0],label=fr'$\alpha$={ajust_high_cD[0]:.3f}$\pm${np.sqrt(cov_high_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_high_cD[1]:.3f}$\pm${np.sqrt(cov_high_cD[1,1]):.3f}')
		ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_low_left_cD),color=cores[1],label=fr'$\alpha$={ajust_low_left_cD[0]:.3f}$\pm${np.sqrt(cov_low_left_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_left_cD[1]:.3f}$\pm${np.sqrt(cov_low_left_cD[1,1]):.3f}')
		ax_center.plot(m200_linspace,linfunc(m200_linspace,*ajust_low_right_cD),color=cores[2],label=fr'$\alpha$={ajust_low_right_cD[0]:.3f}$\pm${np.sqrt(cov_low_right_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_right_cD[1]:.3f}$\pm${np.sqrt(cov_low_right_cD[1,1]):.3f}')
		ax_center.legend(fontsize='x-small')
		ax_center.set_ylim(min(starmass_linspace),max(starmass_linspace))
		ax_center.set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		ax_center.set_ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')

		for i,dist in enumerate(vec_kde_m200_cD):
			ax_topx.plot(m200_linspace,dist(m200_linspace),color=cores[i],label=f'{names_simples[i]}[cD]')
			ax_topx.axvline(vec_med_m200_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_m200_cD[i]:.3f}$')
		ax_topx.legend(fontsize='x-small')
		ax_topx.tick_params(labelbottom=False)

		for i,dist in enumerate(vec_kde_starmass_cD):
			ax_righty.plot(dist(starmass_linspace),starmass_linspace,color=cores[i],label=f'{names_simples[i]}[cD]')
			ax_righty.axhline(vec_med_starmass_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[i]:.3f}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)
		plt.savefig(f'{save_path}/test_ks/starmass_m200_highs_zhao.png')
		plt.close()

		#####################################		
		#JOINTPLOT DA MASSA x AGE -- cDS DO ZHAO

		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		ajust_high_cD,cov_high_cD=np.polyfit(starmass_high_cD,age_high_cD,1,cov=True)
		ajust_low_left_cD,cov_low_left_cD=np.polyfit(starmass_low_left_cD,age_low_left_cD,1,cov=True)
		ajust_low_right_cD,cov_low_right_cD=np.polyfit(starmass_low_right_cD,age_low_right_cD,1,cov=True)

		ax_center.scatter(starmass_high_cD,age_high_cD,marker='o',edgecolor='black',label='E[cD]',color='green')
		ax_center.scatter(starmass_low_left_cD,age_low_left_cD,marker='o',edgecolor='black',label=f'E(EL)[cD]',color='blue')
		ax_center.scatter(starmass_low_right_cD,age_low_right_cD,marker='o',edgecolor='black',label='cD[cD]',color='red')
		ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_high_cD),color='green',label=fr'$\alpha$={ajust_high_cD[0]:.3f}$\pm${np.sqrt(cov_high_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_high_cD[1]:.3f}$\pm${np.sqrt(cov_high_cD[1,1]):.3f}')
		ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_low_left_cD),color='blue',label=fr'$\alpha$={ajust_low_left_cD[0]:.3f}$\pm${np.sqrt(cov_low_left_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_left_cD[1]:.3f}$\pm${np.sqrt(cov_low_left_cD[1,1]):.3f}')
		ax_center.plot(starmass_linspace,linfunc(starmass_linspace,*ajust_low_right_cD),color='red',label=fr'$\alpha$={ajust_low_right_cD[0]:.3f}$\pm${np.sqrt(cov_low_right_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_low_right_cD[1]:.3f}$\pm${np.sqrt(cov_low_right_cD[1,1]):.3f}')
		ax_center.legend(fontsize='x-small')
		ax_center.set_xlim(min(starmass_linspace),max(starmass_linspace))
		ax_center.set_xlabel(r'$\log M_{\bigstar} \ (M_\odot)$')
		ax_center.set_ylabel(r'$\tau \ (Gyr)$')

		for i,dist in enumerate(vec_kde_starmass_cD):
			ax_topx.plot(dist(starmass_linspace),starmass_linspace,color=cores[i],label=f'{names_simples[i]}[cD]')
			ax_topx.axhline(vec_med_starmass_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_starmass_cD[i]:.3f}$')
		ax_topx.legend(fontsize='x-small')
		ax_topx.tick_params(labelbottom=False)

		for i,dist in enumerate(vec_kde_age_cD):
			ax_righty.plot(age_linspace,dist(age_linspace),color=cores[i],label=f'{names_simples[i]}[cD]')
			ax_righty.axvline(vec_med_age_cD[i],color=cores[i],ls='--',label=fr'$\mu = {vec_med_age_cD[i]:.3f}$')
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)
		plt.savefig(f'{save_path}/test_ks/starmass_age_highs_zhao.png')
		plt.close()
		####################################################
		fig,axs=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
		plt.suptitle(f'COMPONENTE INTERNO - cD Zhao')

		ajust_c1_high_cD,cov_c1_high_cD=np.polyfit(m200_high_cD,mass_c1_high_cD,1,cov=True)
		ajust_c1_low_left_cD,cov_c1_low_left_cD=np.polyfit(m200_low_left_cD,mass_c1_low_left_cD,1,cov=True)
		ajust_c1_low_right_cD,cov_c1_low_right_cD=np.polyfit(m200_low_right_cD,mass_c1_low_right_cD,1,cov=True)

		ajust_c2_high_cD,cov_c2_high_cD=np.polyfit(m200_high_cD,mass_c2_high_cD,1,cov=True)
		ajust_c2_low_left_cD,cov_c2_low_left_cD=np.polyfit(m200_low_left_cD,mass_c2_low_left_cD,1,cov=True)
		ajust_c2_low_right_cD,cov_c2_low_right_cD=np.polyfit(m200_low_right_cD,mass_c2_low_right_cD,1,cov=True)

		axs[0].scatter(m200_high_cD,mass_c1_high_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[0]}[cD]',color=cores[0])
		axs[0].set_ylim(9.2,12.2)
		axs[0].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[0].set_ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')
		axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_high_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_high_cD[0]:.3f}$\pm${np.sqrt(cov_c1_high_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_high_cD[1]:.3f}$\pm${np.sqrt(cov_c1_high_cD[1,1]):.3f}')

		axs[1].scatter(m200_low_left_cD,mass_c1_low_left_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[1]}[cD]',color=cores[1])
		axs[1].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_low_left_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c1_low_left_cD[0]:.3f}$\pm${np.sqrt(cov_c1_low_left_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_low_left_cD[1]:.3f}$\pm${np.sqrt(cov_c1_low_left_cD[1,1]):.3f}')

		axs[2].scatter(m200_low_right_cD,mass_c1_low_right_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[2]}[cD]',color=cores[2])
		axs[2].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[2].plot(m200_linspace,linfunc(m200_linspace,*ajust_c1_low_right_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c1_low_right_cD[0]:.3f}$\pm${np.sqrt(cov_c1_low_right_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c1_low_right_cD[1]:.3f}$\pm${np.sqrt(cov_c1_low_right_cD[1,1]):.3f}')

		fig.legend()
		plt.savefig(f'{save_path}/test_ks/m200_mass_c1_high.png')
		plt.close()

		fig,axs=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
		plt.suptitle(f'COMPONENTE EXTERNO - cD Zhao')

		axs[0].scatter(m200_high_cD,mass_c2_high_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[0]}[cD]',color=cores[0])
		axs[0].set_ylim(9.2,12.2)
		axs[0].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[0].set_ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')
		axs[0].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_high_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_high_cD[0]:.3f}$\pm${np.sqrt(cov_c2_high_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_high_cD[1]:.3f}$\pm${np.sqrt(cov_c2_high_cD[1,1]):.3f}')

		axs[1].scatter(m200_low_left_cD,mass_c2_low_left_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[1]}[cD]',color=cores[1])
		axs[1].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[1].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_low_left_cD),color='black',ls='-.',label=fr'$\alpha$={ajust_c2_low_left_cD[0]:.3f}$\pm${np.sqrt(cov_c2_low_left_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_low_left_cD[1]:.3f}$\pm${np.sqrt(cov_c2_low_left_cD[1,1]):.3f}')

		axs[2].scatter(m200_low_right_cD,mass_c2_low_right_cD,marker='o',edgecolor='black',alpha=0.6,label=f'{names_simples[2]}[cD]',color=cores[2])
		axs[2].set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs[2].plot(m200_linspace,linfunc(m200_linspace,*ajust_c2_low_right_cD),color='black',ls='-',label=fr'$\alpha$={ajust_c2_low_right_cD[0]:.3f}$\pm${np.sqrt(cov_c2_low_right_cD[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_c2_low_right_cD[1]:.3f}$\pm${np.sqrt(cov_c2_low_right_cD[1,1]):.3f}')

		fig.legend()
		plt.savefig(f'{save_path}/test_ks/m200_mass_c2_high.png')
		plt.close()
	return
###################################################################################
# FUNÇÕES COM VÁRIAVEIS GLOBAIS - QUE ENVOLVE CHAMAR PRIMEIRO OS RESULTS
def plots_kde(cut_data):
	import pandas as pd
	"""Master orchestrator that generates KDE, histogram, KS‑test, and
	colour‑coded plots for all relevant parameters.

	The sample can optionally be split into bins defined by an external
	variable (redshift or M200) by setting the `flag` parameter.  When a
	split is requested, the `redshift_cut_loop` inner function creates
	tracer plots showing how the mean of each morphological class evolves
	across bins.

	Args:
	    cut_data (tuple):
			param_cut    : 1D array used to define bins (e.g., redshift, M200).
			n_faixas     : Number of bins (only if flag is not None).
			faixas_label : Label string for the binning variable.
			config       : Dictionary containing region masks, colour,settings, and the `save_names` list.
			flag         : 'redshift' or 'm200' → perform binned analysis; None → only full‑sample plots.

	Returns:
	    None.  Saves a large number of figures in an organised directory
	    tree.
	"""

	param_cut,n_faixas,faixas_label,config,flag=cut_data
	def par_only_tracer_plots(medias,sems,par_labels,save_labels):
		""" Simple function that serve as a cosmological tracer to redshift or the virial mass (M200) given the
		designed entry.

		Args.:
			medias: the mean values array, from a deisred number of bins.
			sems: the SEM - Standard Error of the mean - values array, from a deisred number of bins.
			par_labels: The tuple of labels used on the image.
			save_labels: the designed save name of the image.	

		"""
		fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,6))
		plt.suptitle(f'{par_labels[0]}')
		for j in range(3):
			axs.errorbar(stripes_med,medias[:,j],yerr=sems[:,j],fmt='o',color=cores[j],alpha=0.4)
			axs.plot(stripes_med,medias[:,j],color=cores[j],alpha=0.4,label=names_simples[j])
		axs.invert_xaxis()
		axs.set_xlabel(r'$z$')
		axs.set_ylabel(par_labels[1])
		axs.legend()
		plt.tight_layout()
		plt.savefig(f'{save_path}/test_ks/{save_labels[0]}/{save_labels[1]}_tracer_{pasta_save}.png')
		plt.close()
		return

	def redshift_cut_loop(cut_entry,par_stripe_entry):
		lim_region_z,to_do_test,pasta_analise,pasta_save=cut_entry
		_,_,par_labels,save_labels=par_stripe_entry
		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			if type(par_labels[1]) == tuple:
				bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_region_z[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			else:
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_region_z[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		if len(medias.shape) == 3:
			for i in range(len(titles)):
				if save_labels[0] == 're_kpc':
					par_labels,save_labels=(f'Raio efetivo - {titles[i]}',r'$\log_{10}(R_e)\ (kpc)$'+f' {sub_labels[i]}'),('re_kpc',f're_kpc_{titles[i]}')
				if save_labels[0] == 'mue_med':
					par_labels,save_labels=(f'Brilho efetivo médio - {titles[i]}',r'$<\mu_e>$'+f' {sub_labels[i]}'),('mue_med',f'brilho_eff_{titles[i]}')
				if save_labels[0] == 'indice_sersic':
					par_labels,save_labels=(f'n - {titles[i]}',fr'$n_{sub_labels[i]}$'),('indice_sersic',f'n_{titles[i]}')
				if save_labels[0] == 'ax_ratio':
					par_labels,save_labels=(f'q - {titles[i]}',fr'$q_{sub_labels[i]}$'),('ax_ratio',f'q_{titles[i]}')
				if save_labels[0] == 'box':
					par_labels,save_labels=(f'Box - {titles[i]}',r'$(a_4/a)$'+f' {sub_labels[i]}'),('box',f'box_{titles[i]}')	
				par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)
		else:
			par_only_tracer_plots(medias,sems,par_labels,save_labels)
		return

	if sample == 'WHL':
		l07_regions=None
		l07_regions_photutils=None
		l07_regions_casjobs=None
		l07_regions_cor_gr=None
	if sample == 'L07':
		l07_regions=(cd_cut,e_cut,(ecd_cut | cde_cut))
		l07_regions_photutils=(cd_cut_photutils,e_cut_photutils,(ecd_cut_photutils | cde_cut_photutils))
		l07_regions_casjobs=(cd_cut_casjobs,e_cut_casjobs,(ecd_cut_casjobs | cde_cut_casjobs))
		l07_regions_cor_gr=(cd_cut_cor_gr,e_cut_cor_gr,(ecd_cut_cor_gr | cde_cut_cor_gr))

	titles=['Sérsic','Comp_1','Comp_2']
	sub_labels=['s','1','2']
	to_do_test='refaz'
	pasta_analise='test_ks'
	lim_region=config['lim_region']
	lim_region_photutils=config['lim_region_photutils']
	lim_region_casjobs=config['lim_region_casjobs']
	lim_region_cor_gr=config['lim_region_cor_gr']
	lim_region_halpha=config['lim_region_halpha']
	names_simples=config['names_simples']
	alpha_vec=config['alpha_vec']
	cores=config['cores']
	line_width=config['line_width']
	region_names=config['save_names']
	if flag == None:
		pass
	else:
		idx_faixas=pd.qcut(param_cut,n_faixas,labels=False)
		idx_faixas_photutils=idx_faixas[lim_photutils]
		idx_faixas_casjobs=idx_faixas[lim_casjobs]
		idx_faixas_cor_gr=idx_faixas[lim_cor_gr]
		stripes=np.unique(idx_faixas)
		stripes=np.flip(stripes)
		stripes_med=np.asarray([np.round(np.average(param_cut[idx_faixas==idx]),4) for idx in stripes],dtype=float)

		lim_vec=[],[],[]
		lim_vec_photutils=[],[],[]
		lim_vec_casjobs=[],[],[]
		lim_vec_cor_gr=[],[],[]

		pasta_save=faixas_label
		for i,n in enumerate(stripes):
			z_sub_sample_only=idx_faixas==n
			redshift_z=param_cut[z_sub_sample_only]
			lim_vec[0].append(lim_region[0] & z_sub_sample_only)
			lim_vec[1].append(lim_region[1] & z_sub_sample_only)
			lim_vec[2].append(lim_region[2] & z_sub_sample_only)
			#
			z_sub_sample_only_photutils=idx_faixas_photutils==n
			redshift_photutils_z=param_cut[lim_photutils][z_sub_sample_only_photutils]
			lim_vec_photutils[0].append(lim_region_photutils[0] & z_sub_sample_only_photutils)
			lim_vec_photutils[1].append(lim_region_photutils[1] & z_sub_sample_only_photutils)
			lim_vec_photutils[2].append(lim_region_photutils[2] & z_sub_sample_only_photutils)
			#
			z_sub_sample_only_casjobs=idx_faixas_casjobs==n
			redshift_casjobs_z=param_cut[lim_casjobs][z_sub_sample_only_casjobs]
			lim_vec_casjobs[0].append(lim_region_casjobs[0] & z_sub_sample_only_casjobs)
			lim_vec_casjobs[1].append(lim_region_casjobs[1] & z_sub_sample_only_casjobs)
			lim_vec_casjobs[2].append(lim_region_casjobs[2] & z_sub_sample_only_casjobs)
			#
			z_sub_sample_only_cor_gr=idx_faixas_cor_gr==n
			redshift_cor_gr_z=param_cut[lim_cor_gr][z_sub_sample_only_cor_gr]
			lim_vec_cor_gr[0].append(lim_region_cor_gr[0] & z_sub_sample_only_cor_gr)
			lim_vec_cor_gr[1].append(lim_region_cor_gr[1] & z_sub_sample_only_cor_gr)
			lim_vec_cor_gr[2].append(lim_region_cor_gr[2] & z_sub_sample_only_cor_gr)

		lim_vec=np.asarray(lim_vec)
		lim_vec_photutils=np.asarray(lim_vec_photutils)
		lim_vec_casjobs=np.asarray(lim_vec_casjobs)
		lim_vec_cor_gr=np.asarray(lim_vec_cor_gr)

		lim_vec=lim_vec.transpose(1,0,2)
		lim_vec_photutils=lim_vec_photutils.transpose(1,0,2)
		lim_vec_casjobs=lim_vec_casjobs.transpose(1,0,2)
		lim_vec_cor_gr=lim_vec_cor_gr.transpose(1,0,2)
	##################################################
	#RAIO EFETIVO (KPC)
	os.makedirs(f'{save_path}/test_ks/re_kpc',exist_ok=True)

	re_kpc_values=np.hstack((re_s_kpc,re_1_kpc,re_2_kpc))
	re_kpc_linspace=np.linspace(min(re_kpc_values),max(re_kpc_values),3000)
	par_entry_multi=(re_s_kpc,re_1_kpc,re_2_kpc),lim_region,l07_regions,re_kpc_linspace,('Raio Efetivo (kpc)',(r'$\log_{10}(Re_S)\ (kpc)$',r'$\log_{10}(Re_1)\ (kpc)$',r'$\log_{10}(Re_2)\ (kpc)$',r'$\log_{10}(Re)\ (kpc)$')),('re_kpc','re_kpc')
	par_multi_kde(par_entry_multi)
	par_entry_hist=(re_s_kpc,re_1_kpc,re_2_kpc),lim_region,l07_regions,('Raio Efetivo (kpc)',(r'$\log_{10}(Re_S)\ (kpc)$',r'$\log_{10}(Re_1)\ (kpc)$',r'$\log_{10}(Re_2)\ (kpc)$',r'$\log_{10}(Re)\ (kpc)$')),('re_kpc','re_kpc'),region_names,None
	par_multi_hist(par_entry_hist)
	par_stripe_entry=(re_s_kpc,re_1_kpc,re_2_kpc),re_kpc_linspace,(f'Raio Efetivo (kpc)',(r'$\log_{10}(Re_S)\ (kpc)$',r'$\log_{10}(Re_1)\ (kpc)$',r'$\log_{10}(Re_2)\ (kpc)$',r'$\log_{10}(Re)\ (kpc)$')),('re_kpc','re_kpc')
	if flag == None:
		pass
	else:
		cut_entry=lim_vec,to_do_test,pasta_analise,pasta_save
		os.makedirs(f'{save_path}/test_ks/re_kpc/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#RE1/RE2 RATIO
	os.makedirs(f'{save_path}/test_ks/re_ratio_12',exist_ok=True)

	re_ratio_12_linspace=np.linspace(-0.1,1.1,3000)
	par_entry=re_ratio_12,lim_region,l07_regions,re_ratio_12_linspace,('re_ratio_12 (Comp 1/Comp 2)',r'$Re_1/Re_2$'),('re_ratio_12','re_ratio_12')
	par_only_kde(par_entry)
	par_entry_hist=re_ratio_12,lim_region,l07_regions,('re_ratio_12 (Comp 1/Comp 2)',r'$Re_1/Re_2$'),('re_ratio_12','re_ratio_12'),region_names,[-0.1,1.1]
	par_only_hist(par_entry_hist)
	#
	photo_counter(re_ratio_12,lim_region,l07_regions,0.5,(['re_ratio_12 <= 0.5','re_ratio_12 > 0.5'],'Razão de Re_1/Re_2 - 0.5'),('re_ratio_12','re_ratio_12_05'))
	photo_counter(re_ratio_12,lim_region,l07_regions,1,(['re_ratio_12 <= 1','re_ratio_12 > 1'],'Razão de Re_1/Re_2 - 1'),('re_ratio_12','re_ratio_12_1'))
	#
	par_stripe_entry=re_ratio_12,re_ratio_12_linspace,('re_ratio_12 (Comp 1/Comp 2)',r'$Re_1/Re_2$'),('re_ratio_12','re_ratio_12')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/re_ratio_12/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#RE1/RE_S RATIO
	os.makedirs(f'{save_path}/test_ks/re_ratio_1s',exist_ok=True)

	re_ratio_1s_linspace=np.linspace(-0.1,1.1,3000)
	par_entry=re_ratio_1s,lim_region,l07_regions,re_ratio_1s_linspace,('re_ratio_1s (Comp 1/Sérsic)',r'$Re_1/Re_S$'),('re_ratio_1s','re_ratio_1s')
	par_only_kde(par_entry)
	par_entry_hist=re_ratio_1s,lim_region,l07_regions,('re_ratio_1s (Comp 1/Sérsic)',r'$Re_1/Re_S$'),('re_ratio_1s','re_ratio_1s'),region_names,[-0.1,1.1]
	par_only_hist(par_entry_hist)
	#
	photo_counter(re_ratio_1s,lim_region,l07_regions,0.5,(['re_ratio_1s <= 0.5','re_ratio_1s > 0.5'],'Razão de Re_1/Re_S - 0.5'),('re_ratio_1s','re_ratio_1s_05'))
	photo_counter(re_ratio_1s,lim_region,l07_regions,1,(['re_ratio_1s <= 1','re_ratio_1s > 1'],'Razão de Re_1/Re_S - 1'),('re_ratio_1s','re_ratio_1s_1'))
	#
	par_stripe_entry=re_ratio_1s,re_ratio_1s_linspace,('re_ratio_1s (Comp 1/Sérsic)',r'$Re_1/Re_S$'),('re_ratio_1s','re_ratio_1s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/re_ratio_1s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#RE2/RE_S RATIO
	os.makedirs(f'{save_path}/test_ks/re_ratio_2s',exist_ok=True)

	re_ratio_2s_linspace=np.linspace(-0.5,2.5,3000)
	par_entry=re_ratio_2s,lim_region,l07_regions,re_ratio_2s_linspace,('re_ratio_2s (Comp 2/Sérsic)',r'$Re_2/Re_S$'),('re_ratio_2s','re_ratio_2s')
	par_only_kde(par_entry)
	par_entry_hist=re_ratio_2s,lim_region,l07_regions,('re_ratio_2s (Comp 2/Sérsic)',r'$Re_2/Re_S$'),('re_ratio_2s','re_ratio_2s'),region_names,[-0.5,2.5]
	par_only_hist(par_entry_hist)
	#
	photo_counter(re_ratio_2s,lim_region,l07_regions,0.5,(['re_ratio_2s <= 0.5','re_ratio_2s > 0.5'],'Razão de Re_2/Re_S - 0.5'),('re_ratio_2s','re_ratio_2s_05'))
	photo_counter(re_ratio_2s,lim_region,l07_regions,1,(['re_ratio_2s <= 1','re_ratio_2s > 1'],'Razão de Re_2/Re_S - 1'),('re_ratio_2s','re_ratio_2s_1'))
	#
	par_stripe_entry=re_ratio_2s,re_ratio_2s_linspace,('re_ratio_2s (Comp 2/Sérsic)',r'$Re_2/Re_S$'),('re_ratio_2s','re_ratio_2s')

	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/re_ratio_2s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#MAGNITUDE MÉDIA EFETIVA -MUE
	os.makedirs(f'{save_path}/test_ks/mue_med',exist_ok=True)

	mue_med_values=np.hstack((mue_med_s,mue_med_comp_1,mue_med_comp_2))
	mue_med_linspace=np.linspace(min(mue_med_values),27,3000)
	par_entry_multi=(mue_med_s,mue_med_comp_1,mue_med_comp_2),lim_region,l07_regions,mue_med_linspace,('Brilho Efetivo Médio',(r'$<\mu_e>_s$',r'$<\mu_e>_1$',r'$<\mu_e>_2$',r'$<\mu_e>$')),('mue_med','mue_med')
	par_multi_kde(par_entry_multi)
	par_entry_hist=(mue_med_s,mue_med_comp_1,mue_med_comp_2),lim_region,l07_regions,('Brilho Efetivo Médio',(r'$<\mu_e>_s$',r'$<\mu_e>_1$',r'$<\mu_e>_2$',r'$<\mu_e>$')),('mue_med','mue_med'),region_names,[min(mue_med_values),27]
	par_multi_hist(par_entry_hist)

	par_stripe_entry=(mue_med_s,mue_med_comp_1,mue_med_comp_2),mue_med_linspace,('Brilho Efetivo Médio',(r'$<\mu_e>_s$',r'$<\mu_e>_1$',r'$<\mu_e>_2$',r'$<\mu_e>$')),('mue_med','mue_med')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mue_med/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#MUE1/MUE2 RATIO
	os.makedirs(f'{save_path}/test_ks/mue_ratio_12',exist_ok=True)

	mue_ratio_12_linspace=np.linspace(min(mue_med_ratio_12),1.2,3000)
	par_entry=mue_med_ratio_12,lim_region,l07_regions,mue_ratio_12_linspace,('mue_ratio_12 (Comp 1/Comp 2)',r'$<\mu_{e,1}>/<\mu_{e,2}>$'),('mue_ratio_12','mue_ratio_12')
	par_only_kde(par_entry)
	par_entry_hist=mue_med_ratio_12,lim_region,l07_regions,('mue_ratio_12 (Comp 1/Comp 2)',r'$<\mu_{e,1}>/<\mu_{e,2}>$'),('mue_ratio_12','mue_ratio_12'),region_names,[min(mue_med_ratio_12),1.2]
	par_only_hist(par_entry_hist)
	#
	photo_counter(mue_med_ratio_12,lim_region,l07_regions,0.5,(['mue_ratio_12 <= 0.5','mue_ratio_12 > 0.5'],'Razão de mue_1/mue_2 - 0.5'),('mue_ratio_12','mue_ratio_12_05'))
	photo_counter(mue_med_ratio_12,lim_region,l07_regions,1,(['mue_ratio_12 <= 1','mue_ratio_12 > 1'],'Razão de mue_1/mue_2 - 1'),('mue_ratio_12','mue_ratio_12_1'))
	#
	par_stripe_entry=mue_med_ratio_12,mue_ratio_12_linspace,('mue_ratio_12 (Comp 1/Comp 2)',r'$<\mu_{e,1}>/<\mu_{e,2}>$'),('mue_ratio_12','mue_ratio_12')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mue_ratio_12/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#MUE1/MUE_S RATIO
	os.makedirs(f'{save_path}/test_ks/mue_ratio_1s',exist_ok=True)

	mue_ratio_1s_linspace=np.linspace(0.7,1.25,3000)
	par_entry=mue_med_ratio_1s,lim_region,l07_regions,mue_ratio_1s_linspace,('mue_ratio_1s (Comp 1/Sérsic)',r'$<\mu_{e,1}>/<\mu_{e,S}>$'),('mue_ratio_1s','mue_ratio_1s')
	par_only_kde(par_entry)
	par_entry_hist=mue_med_ratio_1s,lim_region,l07_regions,('mue_ratio_1s (Comp 1/Sérsic)',r'$<\mu_{e,1}>/<\mu_{e,S}>$'),('mue_ratio_1s','mue_ratio_1s'),region_names,[0.7,1.25]
	par_only_hist(par_entry_hist)
	#
	photo_counter(mue_med_ratio_1s,lim_region,l07_regions,0.5,(['mue_ratio_1s <= 0.5','mue_ratio_1s > 0.5'],'Razão de mue_1/mue_S - 0.5'),('mue_ratio_1s','mue_ratio_1s_05'))
	photo_counter(mue_med_ratio_1s,lim_region,l07_regions,1,(['mue_ratio_1s <= 1','mue_ratio_1s > 1'],'Razão de mue_1/mue_S - 1'),('mue_ratio_1s','mue_ratio_1s_1'))
	#
	par_stripe_entry=mue_med_ratio_1s,mue_ratio_1s_linspace,('mue_ratio_1s (Comp 1/Sérsic)',r'$<\mu_{e,1}>/<\mu_{e,S}>$'),('mue_ratio_1s','mue_ratio_1s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mue_ratio_1s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#MUE2/MUE_S RATIO
	os.makedirs(f'{save_path}/test_ks/mue_ratio_2s',exist_ok=True)

	mue_ratio_2s_linspace=np.linspace(0.7,1.25,3000)
	par_entry=mue_med_ratio_2s,lim_region,l07_regions,mue_ratio_2s_linspace,('mue_ratio_2s (Comp 2/Sérsic)',r'$<\mu_{e,2}>/<\mu_{e,S}>$'),('mue_ratio_2s','mue_ratio_2s')
	par_only_kde(par_entry)
	par_entry_hist=mue_med_ratio_2s,lim_region,l07_regions,('mue_ratio_2s (Comp 2/Sérsic)',r'$<\mu_{e,2}>/<\mu_{e,S}>$'),('mue_ratio_2s','mue_ratio_2s'),region_names,[0.7,1.25]
	par_only_hist(par_entry_hist)
	#
	photo_counter(mue_med_ratio_2s,lim_region,l07_regions,0.5,(['mue_ratio_2s <= 0.5','mue_ratio_2s > 0.5'],'Razão de mue_2/mue_S - 0.5'),('mue_ratio_2s','mue_ratio_2s_05'))
	photo_counter(mue_med_ratio_2s,lim_region,l07_regions,1,(['mue_ratio_2s <= 1','mue_ratio_2s > 1'],'Razão de mue_2/mue_S - 1'),('mue_ratio_2s','mue_ratio_2s_1'))
	#
	par_stripe_entry=mue_med_ratio_2s,mue_ratio_2s_linspace,('mue_ratio_2s (Comp 2/Sérsic)',r'$<\mu_{e,2}>/<\mu_{e,S}>$'),('mue_ratio_2s','mue_ratio_2s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mue_ratio_2s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#ETA
	os.makedirs(f'{save_path}/test_ks/eta',exist_ok=True)

	eta_linspace=np.linspace(-0.05,0.1,3000)
	par_entry=eta,lim_region,l07_regions,eta_linspace,('eta (RFF - A1)',r'$\eta$'),('eta','eta')
	par_only_kde(par_entry)
	par_entry_hist=eta,lim_region,l07_regions,('eta (RFF - A1)',r'$\eta$'),('eta','eta'),region_names,[-0.05,0.1]
	par_only_hist(par_entry_hist)

	par_stripe_entry=eta,eta_linspace,('eta (RFF - A1)',r'$\eta$'),('eta','eta')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/eta/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#CHI2 RATIO
	os.makedirs(f'{save_path}/test_ks/chi2_ratio',exist_ok=True)

	chi2_ratio_linspace=np.linspace(min(chi2_ratio),5,3000)
	par_entry=chi2_ratio,lim_region,l07_regions,chi2_ratio_linspace,('chi2_ratio (S/SS)',r'$\chi^2_S/\chi^2_{SS}$'),('chi2_ratio','chi2_ratio')
	par_only_kde(par_entry)
	par_entry_hist=chi2_ratio,lim_region,l07_regions,('chi2_ratio (S/SS)',r'$\chi^2_S/\chi^2_{SS}$'),('chi2_ratio','chi2_ratio'),region_names,[min(chi2_ratio),5]
	par_only_hist(par_entry_hist)
	#
	photo_counter(chi2_ratio,lim_region,l07_regions,0.5,(['chi2_ratio <= 0.5','chi2_ratio > 0.5'],'Razão de chi2 - 0.5'),('chi2_ratio','chi2_ratio_05'))
	photo_counter(chi2_ratio,lim_region,l07_regions,1,(['chi2_ratio <= 1','chi2_ratio > 1'],'Razão de chi2 - 1'),('chi2_ratio','chi2_ratio_1'))
	#
	par_stripe_entry=chi2_ratio,chi2_ratio_linspace,('chi2_ratio (S/SS)',r'$\chi^2_S/\chi^2_{SS}$'),('chi2_ratio','chi2_ratio')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/chi2_ratio/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#RFFss/RFFs
	os.makedirs(f'{save_path}/test_ks/rff_ratio',exist_ok=True)

	rff_ratio_linspace=np.linspace(-0.5,3,3000)
	par_entry=rff_ratio,lim_region,l07_regions,rff_ratio_linspace,('rff_ratio (SS/S)',r'$RFF_{SS}/RFF_{S}$'),('rff_ratio','rff_ratio')
	par_only_kde(par_entry)
	par_entry_hist=rff_ratio,lim_region,l07_regions,('rff_ratio (SS/S)',r'$RFF_{SS}/RFF_{S}$'),('rff_ratio','rff_ratio'),region_names,[-0.5,3]
	par_only_hist(par_entry_hist)
	#
	photo_counter(rff_ratio,lim_region,l07_regions,0.5,(['rff_ratio <= 0.5','rff_ratio > 0.5'],'Razão de rff - 0.5'),('rff_ratio','rff_ratio_05'))
	photo_counter(rff_ratio,lim_region,l07_regions,1,(['rff_ratio <= 1','rff_ratio > 1'],'Razão de rff - 1'),('rff_ratio','rff_ratio_1'))
	#
	par_stripe_entry=rff_ratio,rff_ratio_linspace,('rff_ratio (SS/S)',r'$RFF_{SS}/RFF_{S}$'),('rff_ratio','rff_ratio')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/rff_ratio/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	############################################################
	#RAZÃO BT
	os.makedirs(f'{save_path}/test_ks/bt',exist_ok=True)

	bt_linspace=np.linspace(min(bt_vec_corr),max(bt_vec_corr),3000)
	par_entry=bt_vec_corr,lim_region,l07_regions,bt_linspace,('B/T',r'$B/T$'),('bt','bt')
	par_only_kde(par_entry)
	par_entry_hist=bt_vec_corr,lim_region,l07_regions,('B/T',r'$B/T$'),('bt','bt'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(bt_vec_corr,lim_region,l07_regions,0.5,(['B/T <= 0.5','B/T > 0.5'],'Razão BT - 0.5'),('bt','bt_05'))
	photo_counter(bt_vec_corr,lim_region,l07_regions,0.8,(['B/T <= 0.8','B/T > 0.8'],'Razão BT - 0.8'),('bt','bt_08'))
	#
	par_stripe_entry=bt_vec_corr,bt_linspace,('B/T',r'$B/T$'),('bt','bt')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/bt/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	############################################################
	#RFF_SS
	os.makedirs(f'{save_path}/test_ks/rff_ss',exist_ok=True)

	rff_ss_linspace=np.linspace(0.0,0.1,3000)
	par_entry=rff_ss,lim_region,l07_regions,rff_ss_linspace,('RFF (SS)',r'$RFF$'),('rff_ss','rff_ss')
	par_only_kde(par_entry)
	par_entry_hist=rff_ss,lim_region,l07_regions,('RFF (SS)',r'$RFF$'),('rff_ss','rff_ss'),region_names,[0.0,0.1]
	par_only_hist(par_entry_hist)

	par_stripe_entry=rff_ss,rff_ss_linspace,('RFF (SS)',r'$RFF$'),('rff_ss','rff_ss')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/rff_ss/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	############################################################
	#INDICE DE SÉRSIC 
	os.makedirs(f'{save_path}/test_ks/indice_sersic',exist_ok=True)

	n_values=np.hstack((n_s,n1,n2))
	n_linspace=np.linspace(min(n_values),max(n_values),3000)
	par_entry_multi=(n_s,n1,n2),lim_region,l07_regions,n_linspace,('Índice de Sérsic',(r'$n_s$',r'$n_1$',r'$n_2$',r'$n$')),('indice_sersic','n')
	par_multi_kde(par_entry_multi)
	par_entry_hist=(n_s,n1,n2),lim_region,l07_regions,('Índice de Sérsic',(r'$n_s$',r'$n_1$',r'$n_2$',r'$n$')),('indice_sersic','n'),region_names,None
	par_multi_hist(par_entry_hist)

	par_stripe_entry=(n_s,n1,n2),n_linspace,('Índice de Sérsic',(r'$n_s$',r'$n_1$',r'$n_2$',r'$n$')),('indice_sersic','n')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/indice_sersic/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#n1/n2 RATIO
	os.makedirs(f'{save_path}/test_ks/n_ratio_12',exist_ok=True)

	n_ratio_12_linspace=np.linspace(-0.5,10,3000)
	par_entry=n_ratio_12,lim_region,l07_regions,n_ratio_12_linspace,('n_ratio_12 (Comp 1/Comp 2)',r'$n_1/n_2$'),('n_ratio_12','n_ratio_12')
	par_only_kde(par_entry)
	par_entry_hist=n_ratio_12,lim_region,l07_regions,('n_ratio_12 (Comp 1/Comp 2)',r'$n_1/n_2$'),('n_ratio_12','n_ratio_12'),region_names,[-0.5,10]
	par_only_hist(par_entry_hist)
	#
	photo_counter(n_ratio_12,lim_region,l07_regions,0.5,(['n_ratio_12 <= 0.5','n_ratio_12 > 0.5'],'Razão de n_1/n_2 - 0.5'),('n_ratio_12','n_ratio_12_05'))
	photo_counter(n_ratio_12,lim_region,l07_regions,1,(['n_ratio_12 <= 1','n_ratio_12 > 1'],'Razão de n_1/n_2 - 1'),('n_ratio_12','n_ratio_12_1'))
	#
	par_stripe_entry=n_ratio_12,n_ratio_12_linspace,('n_ratio_12 (Comp 1/Comp 2)',r'$n_1/n_2$'),('n_ratio_12','n_ratio_12')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/n_ratio_12/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#n1/n_S RATIO
	os.makedirs(f'{save_path}/test_ks/n_ratio_1s',exist_ok=True)

	n_ratio_1s_linspace=np.linspace(-0.5,4,3000)
	par_entry=n_ratio_1s,lim_region,l07_regions,n_ratio_1s_linspace,('n_ratio_1s (Comp 1/Sérsic)',r'$n_1/n_S$'),('n_ratio_1s','n_ratio_1s')
	par_only_kde(par_entry)
	par_entry_hist=n_ratio_1s,lim_region,l07_regions,('n_ratio_1s (Comp 1/Sérsic)',r'$n_1/n_S$'),('n_ratio_1s','n_ratio_1s'),region_names,[-0.5,4]
	par_only_hist(par_entry_hist)
	#
	photo_counter(n_ratio_1s,lim_region,l07_regions,0.5,(['n_ratio_1s <= 0.5','n_ratio_1s > 0.5'],'Razão de n_1/n_S - 0.5'),('n_ratio_1s','n_ratio_1s_05'))
	photo_counter(n_ratio_1s,lim_region,l07_regions,1,(['n_ratio_1s <= 1','n_ratio_1s > 1'],'Razão de n_1/n_S - 1'),('n_ratio_1s','n_ratio_1s_1'))
	#
	par_stripe_entry=n_ratio_1s,n_ratio_1s_linspace,('n_ratio_1s (Comp 1/Sérsic)',r'$n_1/n_S$'),('n_ratio_1s','n_ratio_1s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/n_ratio_1s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#n2/n_S RATIO
	os.makedirs(f'{save_path}/test_ks/n_ratio_2s',exist_ok=True)

	n_ratio_2s_linspace=np.linspace(-0.5,2,3000)
	par_entry=n_ratio_2s,lim_region,l07_regions,n_ratio_2s_linspace,('n_ratio_2s (Comp 2/Sérsic)',r'$n_2/n_S$'),('n_ratio_2s','n_ratio_2s')
	par_only_kde(par_entry)
	par_entry_hist=n_ratio_2s,lim_region,l07_regions,('n_ratio_2s (Comp 2/Sérsic)',r'$n_2/n_S$'),('n_ratio_2s','n_ratio_2s'),region_names,[-0.5,2]
	par_only_hist(par_entry_hist)
	#
	photo_counter(n_ratio_2s,lim_region,l07_regions,0.5,(['n_ratio_2s <= 0.5','n_ratio_2s > 0.5'],'Razão de n_2/n_S - 0.5'),('n_ratio_2s','n_ratio_2s_05'))
	photo_counter(n_ratio_2s,lim_region,l07_regions,1,(['n_ratio_2s <= 1','n_ratio_2s > 1'],'Razão de n_2/n_S - 1'),('n_ratio_2s','n_ratio_2s_1'))
	#
	par_stripe_entry=n_ratio_2s,n_ratio_2s_linspace,('n_ratio_2s (Comp 2/Sérsic)',r'$n_2/n_S$'),('n_ratio_2s','n_ratio_2s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/n_ratio_2s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#RAZÃO AXIAL
	os.makedirs(f'{save_path}/test_ks/ax_ratio',exist_ok=True)

	q_values=np.hstack((e_s,e1,e2))
	q_linspace=np.linspace(min(q_values),max(q_values),3000)
	par_entry_multi=(e_s,e1,e2),lim_region,l07_regions,q_linspace,('Razão Axial',(r'$q_s$',r'$q_1$',r'$q_2$',r'$q$')),('ax_ratio','axrat')
	par_multi_kde(par_entry_multi)
	par_entry_hist=(e_s,e1,e2),lim_region,l07_regions,('Razão Axial',(r'$q_s$',r'$q_1$',r'$q_2$',r'$q$')),('ax_ratio','axrat'),region_names,None
	par_multi_hist(par_entry_hist)
	#
	photo_counter(e_s,lim_region,l07_regions,0.5,(['q <= 0.5','q > 0.5'],'Razão Axial Sérsic - 0.5'),('ax_ratio','axrat_s_05'))
	photo_counter(e_s,lim_region,l07_regions,0.8,(['q <= 0.8','q > 0.8'],'Razão Axial Sérsic - 0.8'),('ax_ratio','axrat_s_08'))
	#
	photo_counter(e1,lim_region,l07_regions,0.5,(['q1 <= 0.5','q1 > 0.5'],'Razão Axial Comp. 1 - 0.5'),('ax_ratio','axrat_1_05'))
	photo_counter(e1,lim_region,l07_regions,0.8,(['q1 <= 0.8','q1 > 0.8'],'Razão Axial Comp. 1 - 0.8'),('ax_ratio','axrat_1_08'))
	#
	photo_counter(e2,lim_region,l07_regions,0.5,(['q2 <= 0.5','q2 > 0.5'],'Razão Axial Comp. 2 - 0.5'),('ax_ratio','axrat_2_05'))
	photo_counter(e2,lim_region,l07_regions,0.8,(['q2 <= 0.8','q2 > 0.8'],'Razão Axial Comp. 2 - 0.8'),('ax_ratio','axrat_2_08'))
	#
	par_stripe_entry=(e_s,e1,e2),q_linspace,('Razão Axial',(r'$q_s$',r'$q_1$',r'$q_2$',r'$q$')),('ax_ratio','axrat')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/ax_ratio/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#q1/q2 RATIO
	os.makedirs(f'{save_path}/test_ks/axrat_ratio_12',exist_ok=True)

	axrat_ratio_12_linspace=np.linspace(min(axrat_ratio_12),3,3000)
	par_entry=axrat_ratio_12,lim_region,l07_regions,axrat_ratio_12_linspace,('axrat_ratio_12 (Comp 1/Comp 2)',r'$q_1/q_2$'),('axrat_ratio_12','axrat_ratio_12')
	par_only_kde(par_entry)
	par_entry_hist=axrat_ratio_12,lim_region,l07_regions,('axrat_ratio_12 (Comp 1/Comp 2)',r'$q_1/q_2$'),('axrat_ratio_12','axrat_ratio_12'),region_names,[min(axrat_ratio_12),3]
	par_only_hist(par_entry_hist)
	#
	photo_counter(axrat_ratio_12,lim_region,l07_regions,0.5,(['axrat_ratio_12 <= 0.5','axrat_ratio_12 > 0.5'],'Razão de q_1/q_2 - 0.5'),('axrat_ratio_12','axrat_ratio_12_05'))
	photo_counter(axrat_ratio_12,lim_region,l07_regions,1,(['axrat_ratio_12 <= 1','axrat_ratio_12 > 1'],'Razão de q_1/q_2 - 1'),('axrat_ratio_12','axrat_ratio_12_1'))
	#
	par_stripe_entry=axrat_ratio_12,axrat_ratio_12_linspace,('axrat_ratio_12 (Comp 1/Comp 2)',r'$q_1/q_2$'),('axrat_ratio_12','axrat_ratio_12')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/axrat_ratio_12/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#q1/q_S RATIO
	os.makedirs(f'{save_path}/test_ks/axrat_ratio_1s',exist_ok=True)

	axrat_ratio_1s_linspace=np.linspace(min(axrat_ratio_1s),max(axrat_ratio_1s),3000)
	par_entry=axrat_ratio_1s,lim_region,l07_regions,axrat_ratio_1s_linspace,('axrat_ratio_1s (Comp 1/Sérsic)',r'$q_1/q_S$'),('axrat_ratio_1s','axrat_ratio_1s')
	par_only_kde(par_entry)
	par_entry_hist=axrat_ratio_1s,lim_region,l07_regions,('axrat_ratio_1s (Comp 1/Sérsic)',r'$q_1/q_S$'),('axrat_ratio_1s','axrat_ratio_1s'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(axrat_ratio_1s,lim_region,l07_regions,0.5,(['axrat_ratio_1s <= 0.5','axrat_ratio_1s > 0.5'],'Razão de q_1/q_S - 0.5'),('axrat_ratio_1s','axrat_ratio_1s_05'))
	photo_counter(axrat_ratio_1s,lim_region,l07_regions,1,(['axrat_ratio_1s <= 1','axrat_ratio_1s > 1'],'Razão de q_1/q_S - 1'),('axrat_ratio_1s','axrat_ratio_1s_1'))
	#
	par_stripe_entry=axrat_ratio_1s,axrat_ratio_1s_linspace,('axrat_ratio_1s (Comp 1/Sérsic)',r'$q_1/q_S$'),('axrat_ratio_1s','axrat_ratio_1s')

	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/axrat_ratio_1s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#q2/q_S RATIO
	os.makedirs(f'{save_path}/test_ks/axrat_ratio_2s',exist_ok=True)

	axrat_ratio_2s_linspace=np.linspace(min(axrat_ratio_2s),max(axrat_ratio_2s),3000)
	par_entry=axrat_ratio_2s,lim_region,l07_regions,axrat_ratio_2s_linspace,('axrat_ratio_2s (Comp 1/Sérsic)',r'$q_2/q_S$'),('axrat_ratio_2s','axrat_ratio_2s')
	par_only_kde(par_entry)
	par_entry_hist=axrat_ratio_2s,lim_region,l07_regions,('axrat_ratio_2s (Comp 1/Sérsic)',r'$q_2/q_S$'),('axrat_ratio_2s','axrat_ratio_2s'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(axrat_ratio_2s,lim_region,l07_regions,0.5,(['axrat_ratio_2s <= 0.5','axrat_ratio_2s > 0.5'],'Razão de q_2/q_S - 0.5'),('axrat_ratio_2s','axrat_ratio_2s_05'))
	photo_counter(axrat_ratio_2s,lim_region,l07_regions,1,(['axrat_ratio_2s <= 1','axrat_ratio_2s > 1'],'Razão de q_2/q_S - 1'),('axrat_ratio_2s','axrat_ratio_2s_1'))
	#
	par_stripe_entry=axrat_ratio_2s,axrat_ratio_2s_linspace,('axrat_ratio_2s (Comp 1/Sérsic)',r'$q_2/q_S$'),('axrat_ratio_2s','axrat_ratio_2s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/axrat_ratio_2s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	###################################################################################
	#BOX
	os.makedirs(f'{save_path}/test_ks/box',exist_ok=True)

	box_values=np.hstack((box_s,box1,box2))
	box_linspace=np.linspace(min(box_values),max(box_values),3000)
	par_entry_multi=(box_s,box1,box2),lim_region,l07_regions,box_linspace,('Boxiness',(r'$(a_4/a)_s$',r'$(a_4/a)_1$',r'$(a_4/a)_2$',r'$a_4/a$')),('box','box')
	par_multi_kde(par_entry_multi)
	par_entry_hist=(box_s,box1,box2),lim_region,l07_regions,('Boxiness',(r'$(a_4/a)_s$',r'$(a_4/a)_1$',r'$(a_4/a)_2$',r'$a_4/a$')),('box','box'),region_names,None
	par_multi_hist(par_entry_hist)
	#
	photo_counter(box_s,lim_region,l07_regions,0,(['boxy','disky'],'Boxiness Sérsic'),('box','box_s_0'))
	#
	photo_counter(box1,lim_region,l07_regions,0,(['boxy','disky'],'Boxiness Comp. 1'),('box','box_1_0'))
	#
	photo_counter(box2,lim_region,l07_regions,0,(['boxy','disky'],'Boxiness Comp. 2'),('box','box_2_0'))
	#
	par_stripe_entry=(box_s,box1,box2),box_linspace,('Boxiness',(r'$(a_4/a)_s$',r'$(a_4/a)_1$',r'$(a_4/a)_2$',r'$a_4/a$')),('box','box')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/box/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#box1/box2 RATIO
	os.makedirs(f'{save_path}/test_ks/box_ratio_12',exist_ok=True)

	box_ratio_12_linspace=np.linspace(min(box_ratio_12),max(box_ratio_12),3000)
	par_entry=box_ratio_12,lim_region,l07_regions,box_ratio_12_linspace,('box_ratio_12 (Comp 1/Comp 2)',r'$(a4/a)_1/(a4/a)_2$'),('box_ratio_12','box_ratio_12')
	par_only_kde(par_entry)
	par_entry_hist=box_ratio_12,lim_region,l07_regions,('box_ratio_12 (Comp 1/Comp 2)',r'$(a4/a)_1/(a4/a)_2$'),('box_ratio_12','box_ratio_12'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(box_ratio_12,lim_region,l07_regions,0.5,(['box_ratio_12 <= 0.5','box_ratio_12 > 0.5'],'Razão de (a4/a)_1/(a4/a)_2 - 0.5'),('box_ratio_12','box_ratio_12_05'))
	photo_counter(box_ratio_12,lim_region,l07_regions,1,(['box_ratio_12 <= 1','box_ratio_12 > 1'],'Razão de (a4/a)_1/(a4/a)_2 - 1'),('box_ratio_12','box_ratio_12_1'))
	#
	par_stripe_entry=box_ratio_12,box_ratio_12_linspace,('box_ratio_12 (Comp 1/Comp 2)',r'$(a4/a)_1/(a4/a)_2$'),('box_ratio_12','box_ratio_12')

	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/box_ratio_12/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#box1/box_S RATIO
	os.makedirs(f'{save_path}/test_ks/box_ratio_1s',exist_ok=True)

	box_ratio_1s_linspace=np.linspace(min(box_ratio_1s),max(box_ratio_1s),3000)
	par_entry=box_ratio_1s,lim_region,l07_regions,box_ratio_1s_linspace,('box_ratio_1s (Comp 1/Sérsic)',r'$(a4/a)_1/(a4/a)_S$'),('box_ratio_1s','box_ratio_1s')
	par_only_kde(par_entry)
	par_entry_hist=box_ratio_1s,lim_region,l07_regions,('box_ratio_1s (Comp 1/Sérsic)',r'$(a4/a)_1/(a4/a)_S$'),('box_ratio_1s','box_ratio_1s'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(box_ratio_1s,lim_region,l07_regions,0.5,(['box_ratio_1s <= 0.5','box_ratio_1s > 0.5'],'Razão de (a4/a)_1/(a4/a)_S - 0.5'),('box_ratio_1s','box_ratio_1s_05'))
	photo_counter(box_ratio_1s,lim_region,l07_regions,1,(['box_ratio_1s <= 1','box_ratio_1s > 1'],'Razão de (a4/a)_1/(a4/a)_S - 1'),('box_ratio_1s','box_ratio_1s_1'))
	#
	par_stripe_entry=box_ratio_1s,box_ratio_1s_linspace,('box_ratio_1s (Comp 1/Sérsic)',r'$(a4/a)_1/(a4/a)_S$'),('box_ratio_1s','box_ratio_1s')

	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/box_ratio_1s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##################################
	#box2/box_S RATIO
	os.makedirs(f'{save_path}/test_ks/box_ratio_2s',exist_ok=True)

	box_ratio_2s_linspace=np.linspace(min(box_ratio_2s),max(box_ratio_2s),3000)
	par_entry=box_ratio_2s,lim_region,l07_regions,box_ratio_2s_linspace,('box_ratio_2s (Comp 2/Sérsic)',r'$(a4/a)_2/(a4/a)_S$'),('box_ratio_2s','box_ratio_2s')
	par_only_kde(par_entry)
	par_entry_hist=box_ratio_2s,lim_region,l07_regions,('box_ratio_2s (Comp 2/Sérsic)',r'$(a4/a)_2/(a4/a)_S$'),('box_ratio_2s','box_ratio_2s'),region_names,None
	par_only_hist(par_entry_hist)
	#
	photo_counter(box_ratio_2s,lim_region,l07_regions,0.5,(['box_ratio_2s <= 0.5','box_ratio_2s > 0.5'],'Razão de (a4/a)_2/(a4/a)_S - 0.5'),('box_ratio_2s','box_ratio_2s_05'))
	photo_counter(box_ratio_2s,lim_region,l07_regions,1,(['box_ratio_2s <= 1','box_ratio_2s > 1'],'Razão de (a4/a)_2/(a4/a)_S - 1'),('box_ratio_2s','box_ratio_2s_1'))
	#
	par_stripe_entry=box_ratio_2s,box_ratio_2s_linspace,('box_ratio_2s (Comp 2/Sérsic)',r'$(a4/a)_2/(a4/a)_S$'),('box_ratio_2s','box_ratio_2s')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/box_ratio_2s/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	###############################################
	#M200
	os.makedirs(f'{save_path}/test_ks/m200',exist_ok=True)
	m200_linspace=np.linspace(min(m200_temp),max(m200_temp),3000)

	par_entry=m200_temp,lim_region,l07_regions,m200_linspace,('Massa do Cluster',r'$\log M_{200} \ (M_\odot)$'),('m200','m200')
	par_only_kde(par_entry)
	par_entry_hist=m200_temp,lim_region,l07_regions,('Massa do Cluster',r'$\log M_{200} \ (M_\odot)$'),('m200','m200'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=m200_temp,m200_linspace,('Massa do Cluster',r'$\log M_{200} \ (M_\odot)$'),('m200','m200')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/m200/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	###############################################
	#RAIO EFETIVO EM ARCSEC
	os.makedirs(f'{save_path}/test_ks/re1',exist_ok=True)

	re1_linspace=np.linspace(min(re1*0.396),20,3000)
	par_entry=re1*0.396,lim_region,l07_regions,re1_linspace,('Raio efetivo (arcsec)',r'$Re_{1} (arcsec)$'),('re1','re1')
	par_only_kde(par_entry)
	par_entry_hist=re1*0.396,lim_region,l07_regions,('Raio efetivo (arcsec)',r'$Re_{1} (arcsec)$'),('re1','re1'),region_names,[min(re1*0.396),20]
	par_only_hist(par_entry_hist)

	par_stripe_entry=re1*0.396,re1_linspace,('Raio efetivo (arcsec)',r'$Re_{1} (arcsec)$'),('re1','re1')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/re1/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#########################################################
	#PSF FWHM EM ARCSEC
	os.makedirs(f'{save_path}/test_ks/psf_fwhm',exist_ok=True)

	psf_fwhm_linspace=np.linspace(min(psf_fwhm*0.396),max(psf_fwhm*0.396),3000)
	par_entry=psf_fwhm*0.396,lim_region,l07_regions,psf_fwhm_linspace,('PSF FWHM (arcsec)',r'$PSF_{FWHM} (arcsec)$'),('psf_fwhm','psf_fwhm')
	par_only_kde(par_entry)
	par_entry_hist=psf_fwhm*0.396,lim_region,l07_regions,('PSF FWHM (arcsec)',r'$PSF_{FWHM} (arcsec)$'),('psf_fwhm','psf_fwhm'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=psf_fwhm*0.396,psf_fwhm_linspace,('PSF FWHM (arcsec)',r'$PSF_{FWHM} (arcsec)$'),('psf_fwhm','psf_fwhm')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/psf_fwhm/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##############################################################
	#COEFICIENTES DE FOURIER
	coef_titles=['médio','slope']
	#A3
	os.makedirs(f'{save_path}/test_ks/a3',exist_ok=True)
	a3_vecs=[med_a3,slope_a3]
	a3_labels=[r'\bar{a_3}',r'$\alpha \ a_3$']
	for i,vet in enumerate(a3_vecs):
		a3_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
		par_entry=vet,lim_region_photutils,l07_regions_photutils,a3_linspace,(f'a3 - {coef_titles[i]}',a3_labels[i]),('a3',f'a3_{coef_titles[i]}')
		par_only_kde(par_entry)
		par_entry_hist=vet,lim_region_photutils,l07_regions_photutils,(f'a3 - {coef_titles[i]}',a3_labels[i]),('a3',f'a3_{coef_titles[i]}'),region_names,[np.percentile(vet,5),np.percentile(vet,95)]
		par_only_hist(par_entry_hist)
		photo_counter(vet,lim_region_photutils,l07_regions,0,(['a3<0','a3>0'],f'a3 {coef_titles[i]}'),('a3',f'a3_{coef_titles[i]}_0'))

		par_labels,save_labels=(f'a3 - {coef_titles[i]}',a3_labels[i]),('a3',f'a3_{coef_titles[i]}')
		par_stripe_entry=vet,a3_linspace,par_labels,save_labels
		if flag == None:
			pass
		else:
			cut_entry=lim_vec_photutils,to_do_test,pasta_analise,pasta_save
			os.makedirs(f'{save_path}/test_ks/a3/{pasta_save}',exist_ok=True)
			redshift_cut_loop(cut_entry,par_stripe_entry)

	######################################################
	#A4
	os.makedirs(f'{save_path}/test_ks/a4',exist_ok=True)
	a4_vecs=[med_a4,slope_a4]
	a4_labels=[r'\bar{a_4}',r'$\alpha \ a_4$']
	for i,vet in enumerate(a4_vecs):
		a4_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
		par_entry=vet,lim_region_photutils,l07_regions_photutils,a4_linspace,(f'a4 - {coef_titles[i]}',a4_labels[i]),('a4',f'a4_{coef_titles[i]}')
		par_only_kde(par_entry)
		par_entry_hist=vet,lim_region_photutils,l07_regions_photutils,(f'a4 - {coef_titles[i]}',a4_labels[i]),('a4',f'a4_{coef_titles[i]}'),region_names,[np.percentile(vet,5),np.percentile(vet,95)]
		par_only_hist(par_entry_hist)
		photo_counter(vet,lim_region_photutils,l07_regions,0,(['a4<0','a4>0'],f'a4 {coef_titles[i]}'),('a4',f'a4_{coef_titles[i]}_0'))

		par_labels,save_labels=(f'a4 - {coef_titles[i]}',a4_labels[i]),('a4',f'a4_{coef_titles[i]}')
		par_stripe_entry=vet,a4_linspace,par_labels,save_labels
		if flag == None:
			pass
		else:
			os.makedirs(f'{save_path}/test_ks/a4/{pasta_save}',exist_ok=True)
			redshift_cut_loop(cut_entry,par_stripe_entry)

	######################################################
	#B3
	os.makedirs(f'{save_path}/test_ks/b3',exist_ok=True)

	b3_vecs=[med_b3,slope_b3]
	b3_labels=[r'\bar{b_3}',r'$\alpha \ b_3$']
	for i,vet in enumerate(b3_vecs):
		b3_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
		par_entry=vet,lim_region_photutils,l07_regions_photutils,b3_linspace,(f'b3 - {coef_titles[i]}',b3_labels[i]),('b3',f'b3_{coef_titles[i]}')
		par_only_kde(par_entry)
		par_entry_hist=vet,lim_region_photutils,l07_regions_photutils,(f'b3 - {coef_titles[i]}',b3_labels[i]),('b3',f'b3_{coef_titles[i]}'),region_names,[np.percentile(vet,5),np.percentile(vet,95)]
		par_only_hist(par_entry_hist)
		photo_counter(vet,lim_region_photutils,l07_regions,0,(['b3<0','b3>0'],f'b3 {coef_titles[i]}'),('b3',f'b3_{coef_titles[i]}_0'))

		par_labels,save_labels=(f'b3 - {coef_titles[i]}',b3_labels[i]),('b3',f'b3_{coef_titles[i]}')
		par_stripe_entry=vet,b3_linspace,par_labels,save_labels
		if flag == None:
			pass
		else:
			os.makedirs(f'{save_path}/test_ks/b3/{pasta_save}',exist_ok=True)
			redshift_cut_loop(cut_entry,par_stripe_entry)
	######################################################
	#B4
	os.makedirs(f'{save_path}/test_ks/b4',exist_ok=True)

	b4_vecs=[med_b4,slope_b4]
	b4_labels=[r'\bar{b_4}',r'$\alpha \ b_4$']
	for i,vet in enumerate(b4_vecs):
		b4_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
		par_entry=vet,lim_region_photutils,l07_regions_photutils,b4_linspace,(f'b4 - {coef_titles[i]}',b4_labels[i]),('b4',f'b4_{coef_titles[i]}')
		par_only_kde(par_entry)
		par_entry_hist=vet,lim_region_photutils,l07_regions_photutils,(f'b4 - {coef_titles[i]}',b4_labels[i]),('b4',f'b4_{coef_titles[i]}'),region_names,[np.percentile(vet,5),np.percentile(vet,95)]
		par_only_hist(par_entry_hist)
		photo_counter(vet,lim_region_photutils,l07_regions,0,(['b4<0','b4>0'],f'b4 {coef_titles[i]}'),('b4',f'b4_{coef_titles[i]}_0'))

		par_labels,save_labels=(f'b4 - {coef_titles[i]}',b4_labels[i]),('b4',f'b4_{coef_titles[i]}')
		par_stripe_entry=vet,b4_linspace,par_labels,save_labels
		if flag == None:
			pass
		else:
			os.makedirs(f'{save_path}/test_ks/b4/{pasta_save}',exist_ok=True)
			redshift_cut_loop(cut_entry,par_stripe_entry)
	######################################################################
	#G-R
	os.makedirs(f'{save_path}/test_ks/test_gr',exist_ok=True)

	slope_gr_linspace=np.linspace(np.percentile(slope_gr,5),np.percentile(slope_gr,95),3000)
	par_entry=slope_gr,lim_region_photutils,l07_regions_photutils,slope_gr_linspace,('slope g-r',r'$\alpha g-r$'),('test_gr','slope_gr')
	par_only_kde(par_entry)
	par_entry_hist=slope_gr,lim_region_photutils,l07_regions_photutils,('slope g-r',r'$\alpha g-r$'),('test_gr','slope_gr'),region_names,[np.percentile(slope_gr,5),np.percentile(slope_gr,95)]
	par_only_hist(par_entry_hist)

	par_stripe_entry=slope_gr,slope_gr_linspace,('slope g-r',r'$\alpha g-r$'),('test_gr','slope_gr')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/test_gr/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	######################################################################
	#GRADIENTE E
	os.makedirs(f'{save_path}/test_ks/grad_e',exist_ok=True)

	grad_e_linspace=np.linspace(min(grad_e),max(grad_e),3000)
	par_entry=grad_e,lim_region_photutils,l07_regions_photutils,grad_e_linspace,('Gradiente de Elipticidade',r'$\nabla_e$'),('grad_e','grad_e')
	par_only_kde(par_entry)
	par_entry_hist=grad_e,lim_region_photutils,l07_regions_photutils,('Gradiente de Elipticidade',r'$\nabla_e$'),('grad_e','grad_e'),region_names,None
	par_only_hist(par_entry_hist)
	photo_counter(grad_e,lim_region_photutils,l07_regions,0,(['flat to round','round to flat'],f'Gradiente de Elipticidade'),('grad_e','grad_e_0'))

	par_stripe_entry=grad_e,grad_e_linspace,('Gradiente de Elipticidade',r'$\nabla_e$'),('grad_e','grad_e')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/grad_e/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################################
	#GRADIENTE PA
	os.makedirs(f'{save_path}/test_ks/grad_pa',exist_ok=True)

	grad_pa_linspace=np.linspace(min(grad_pa),5,3000)
	par_entry=grad_pa,lim_region_photutils,l07_regions_photutils,grad_pa_linspace,('Gradiente de Posição Angular',r'$\nabla_PA$'),('grad_pa','grad_pa')
	par_only_kde(par_entry)
	par_entry_hist=grad_pa,lim_region_photutils,l07_regions_photutils,('Gradiente de Posição angular',r'$\nabla_PA$'),('grad_pa','grad_pa'),region_names,[min(grad_pa),5]
	par_only_hist(par_entry_hist)

	par_stripe_entry=grad_pa,grad_pa_linspace,('Gradiente de Posição Angular',r'$\nabla_PA$'),('grad_pa','grad_pa')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/grad_pa/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	########################################################################
	#REDSHIFT
	os.makedirs(f'{save_path}/test_ks/redshift',exist_ok=True)

	redshift_linspace=np.linspace(min(redshift),max(redshift),3000)
	par_entry=redshift,lim_region,l07_regions,redshift_linspace,('z',r'$z$'),('redshift','redshift')
	par_only_kde(par_entry)
	par_entry_hist=redshift,lim_region,l07_regions,('z',r'$z$'),('redshift','redshift'),region_names,None
	par_only_hist(par_entry_hist)

	#LINHA DO H ALPHA
	if sample == 'L07' and mode != 'delta_bic':
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/h_line',exist_ok=True)
		h_line_linspace=np.linspace(min(h_line),4.5,3000)
		par_entry=h_line,lim_region_halpha,l07_regions,h_line_linspace,('Linha do h_alpha',r'$H_\alpha$'),('h_line','hline')
		par_only_kde(par_entry)
		par_entry_hist=h_line,lim_region_halpha,l07_regions,('Linha do h_alpha',r'$H_\alpha$'),('h_line','hline'),region_names,None
		par_only_hist(par_entry_hist)

	#######################
	#CASJOBS
	os.makedirs(f'{save_path}/test_ks/starmass',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/magabs',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/age',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/conc',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/mass_c1',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/mass_c2',exist_ok=True)
	os.makedirs(f'{save_path}/test_ks/m_sersic',exist_ok=True)

	info_need=bt_vec_corr,starmass#,(lim_casjobs,(cd_lim_casjobs,cd_lim_casjobs_small,cd_lim_casjobs_big,elip_lim_casjobs)),l07_regions_casjobs
	casjobs_jointplots(info_need,lim_region_casjobs,l07_regions_casjobs)	
		
	#MAG ABS
	magabs_linspace=np.linspace(min(magabs),max(magabs),3000)
	
	par_entry=magabs,lim_region_casjobs,l07_regions_casjobs,magabs_linspace,('Magnitude Absoluta',r'$Mag_{r,bol}$'),('magabs','magabs')
	par_only_kde(par_entry)
	par_entry_hist=magabs,lim_region_casjobs,l07_regions_casjobs,('Magnitude Absoluta',r'$Mag_{r,bol}$'),('magabs','magabs'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=magabs,magabs_linspace,('Magnitude Absoluta',r'$Mag_{r,bol}$'),('magabs','magabs')
	if flag == None:
		pass
	else:
		cut_entry=lim_vec_casjobs,to_do_test,pasta_analise,pasta_save
		os.makedirs(f'{save_path}/test_ks/magabs/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	############################################################
	#MASS ESTELAR
	starmass_linspace=np.linspace(10.8,max(starmass),3000)
	par_entry=starmass,lim_region_casjobs,l07_regions_casjobs,starmass_linspace,('Massa estelar',r'$\log M_{\bigstar} \ (M_\odot)$'),('starmass','starmass')
	par_only_kde(par_entry)
	par_entry_hist=starmass,lim_region_casjobs,l07_regions_casjobs,('Massa estelar',r'$\log M_{\bigstar} \ (M_\odot)$'),('starmass','starmass'),region_names,[10.8,max(starmass)]
	par_only_hist(par_entry_hist)

	par_stripe_entry=starmass,starmass_linspace,('Massa estelar',r'$\log M_{\bigstar} \ (M_\odot)$'),('starmass','starmass')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/starmass/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#########################################################
	#IDADE

	age_linspace=np.linspace(min(age),max(age),3000)
	par_entry=age,lim_region_casjobs,l07_regions_casjobs,age_linspace,('Idade estelar',r'$\tau \ (Gyr)$'),('age','age')
	par_only_kde(par_entry)
	par_entry_hist=age,lim_region_casjobs,l07_regions_casjobs,('Idade estelar',r'$\tau \ (Gyr)$'),('age','age'),region_names,None
	par_only_kde(par_entry)
	par_stripe_entry=age,age_linspace,('Idade estelar',r'$\tau \ (Gyr)$'),('age','age')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/age/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#########################################################
	#CONCENTRAÇAO
	conc_linspace=np.linspace(min(conc),4,3000)

	par_entry=conc,lim_region_casjobs,l07_regions_casjobs,conc_linspace,('Concentração R90/R50',r'$C_{90,50}$'),('conc','conc')
	par_only_kde(par_entry)
	par_entry_hist=conc,lim_region_casjobs,l07_regions_casjobs,('Concentração R90/R50',r'$C_{90,50}$'),('conc','conc'),region_names,[min(conc),4]
	par_only_hist(par_entry_hist)

	par_stripe_entry=conc,conc_linspace,('Concentração R90/R50',r'$C_{90,50}$'),('conc','conc')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/conc/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	###############################################
	#MASSAS DO COMPONENTE INTERNO/EXTERNO
	info_need=bt_vec_corr,starmass
	mass_c1,mass_c2=mass_calc(info_need)

	#MASSA C1

	mass_c1_linspace=np.linspace(min(mass_c1),max(mass_c1),3000)

	par_entry=mass_c1,lim_region_casjobs,l07_regions_casjobs,mass_c1_linspace,('Massa Estelar Componente 1',r'$\log M_{1,\bigstar} \ (M_\odot)$'),('mass_c1','mass_c1')
	par_only_kde(par_entry)
	par_entry_hist=mass_c1,lim_region_casjobs,l07_regions_casjobs,('Massa Estelar Componente 1',r'$\log M_{1,\bigstar} \ (M_\odot)$'),('mass_c1','mass_c1'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=mass_c1,mass_c1_linspace,('Massa Estelar Componente 1',r'$\log M_{1,\bigstar} \ (M_\odot)$'),('mass_c1','mass_c1')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mass_c1/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	##############
	#MASSA C2
	mass_c2_linspace=np.linspace(min(mass_c2),max(mass_c2),3000)

	par_entry=mass_c2,lim_region_casjobs,l07_regions_casjobs,mass_c2_linspace,('Massa Estelar Componente 2',r'$\log M_{2,\bigstar} \ (M_\odot)$'),('mass_c2','mass_c2')
	par_only_kde(par_entry)
	par_entry_hist=mass_c2,lim_region_casjobs,l07_regions_casjobs,('Massa Estelar Componente 2',r'$\log M_{2,\bigstar} \ (M_\odot)$'),('mass_c2','mass_c2'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=mass_c2,mass_c2_linspace,('Massa Estelar Componente 2',r'$\log M_{2,\bigstar} \ (M_\odot)$'),('mass_c2','mass_c2')
	if flag == None:
		pass
	else:
		os.makedirs(f'{save_path}/test_ks/mass_c2/{pasta_save}',exist_ok=True)
		redshift_cut_loop(cut_entry,par_stripe_entry)
	#####################################################
	#G-R RESTFRAME
	os.makedirs(f'{save_path}/test_ks/gr_rest',exist_ok=True)
	
	gr_rest_linspace=np.linspace(min(gr_rest),max(gr_rest),3000)
	par_entry=gr_rest,lim_region_cor_gr,l07_regions_cor_gr,gr_rest_linspace,('g-r (z=0)',r'$g-r (z=0)$'),('gr_rest','gr_rest')
	par_only_kde(par_entry)
	par_entry_hist=gr_rest,lim_region_cor_gr,l07_regions_cor_gr,('g-r (z=0)',r'$g-r (z=0)$'),('gr_rest','gr_rest'),region_names,None
	par_only_hist(par_entry_hist)

	par_stripe_entry=gr_rest,gr_rest_linspace,('g-r (z=0)',r'$g-r (z=0)$'),('gr_rest','gr_rest')
	if sample == 'WHL':
		if flag == None:
			pass
		else:
			cut_entry=lim_vec_cor_gr,to_do_test,pasta_analise,pasta_save
			os.makedirs(f'{save_path}/test_ks/gr_rest/{pasta_save}',exist_ok=True)
			redshift_cut_loop(cut_entry,par_stripe_entry)
	return
def split_point_2C():
    """Fit a piecewise linear model to the internal component of the
    Kormendy relation for the whole 2C sample.

    The function calls `find_optimal_division` on (log Rₑ, ⟨μₑ⟩) for
    the inner component of all 2C galaxies, then displays the data with
    the two fitted lines, the break point, and residuals.

    Returns:
        None.  A figure is saved.
    """

	x,y=re_1_kpc[cd_lim], mue_med_comp_1[cd_lim]
	y_err=None
	result = find_optimal_division(x,y, y_err)
	slope1, intercept1, slope2, intercept2, x_div = params_opt= result.x
	chi2_min = result.fun
	dof = len(x) - 5  # degrees of freedom
	inc_slope1,inc_intercept1,inc_slope2,inc_intercept2,inc_divx=uncertainties = bootstrap_uncertainty(x, y, params_opt, y_err)

	fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 6))
	ax1.scatter(x, y, alpha=0.7, label=label_bojo_2c)
	x_fine = np.linspace(np.min(x), np.max(x), 1000)
	y_pred_fine = two_line_model(params_opt, x_fine, np.zeros_like(x_fine))
	mask1 = x_fine <= x_div
	mask2 = x_fine > x_div
	ax1.plot(x_fine[mask1], y_pred_fine[mask1], 'r-',label=f'Line 1: y = ({slope1:.3f}±{inc_slope1:.3f})x + {intercept1:.3f}±{inc_intercept1:.3f}', linewidth=2)
	ax1.plot(x_fine[mask2], y_pred_fine[mask2], 'g-',label=f'Line 2: y = ({slope2:.3f}±{inc_slope2:.3f})x + {intercept2:.3f}±{inc_intercept2:.3f}', linewidth=2)
	ax1.axvline(x=x_div, color='k', linestyle='--',label=f'Division at x = {x_div:.3f}±{inc_divx:.3f}')
	ax1.axvspan(x_div - inc_divx, x_div + inc_divx,color='black',alpha=0.15,label='Division Uncertanty Region')

	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax1.yaxis.set_inverted(True)
	ax1.legend(fontsize='small')
	y_pred = two_line_model(params_opt, x, y)
	residuals = y - y_pred
	ax2.scatter(x, residuals, alpha=0.7)
	ax2.axhline(y=0, color='r', linestyle='--')
	ax2.set_xlabel(xlabel)
	ax2.set_ylabel('Residuals')
	plt.tight_layout()
	# plt.show()
	plt.savefig(f'{save_path}/2c_intern_split_line.png')
	plt.close()
	return
def kormendy_plots(config):
	"""Produce an exhaustive set of joint plots for the Kormendy relation.

	For each morphological class and each model type (single Sérsic,
	component 1, component 2) the code creates:
		- Single‑class joint plots (scatter + marginal KDE).
		- Overlaid two‑class and three‑class comparisons.
		- Colour‑coded scatter plots where the third variable is a chosen
		auxiliary parameter (e.g., B/T, n, q, *re₁/re₂*, RFF ratio).
		The colour bar is placed on the left side of the figure.
		- For the L07 sample, extra subsets restricted to cD galaxies (Zhao).

	The Kormendy relation is displayed with the effective radius on a
	logarithmic scale (log₁₀ Rₑ/kpc) and the mean effective surface
	brightness ⟨μₑ⟩ in mag/arcsec², with brighter values at the top.

	Args:
		config (dict): Must contain at least:
			'lim_region'    : masks for the three morphological classes.
			'names_simples' : list of three class labels.
			'cores'         : list of three main colours.
			'alpha_vec', 'line_width', 'save_names'.

	Returns:
		None.  Numerous PNG figures are saved.
	"""


	from mpl_toolkits.axes_grid1.inset_locator import inset_axes
	os.makedirs(f'{save_path}/kormendy_rel_split',exist_ok=True)
	os.makedirs(f'{save_path}/kormendy_rel_split/intern',exist_ok=True)
	os.makedirs(f'{save_path}/kormendy_rel_split/extern',exist_ok=True)
	os.makedirs(f'{save_path}/kormendy_rel_split/sersic',exist_ok=True)
	def label_format(valor):
		return format(valor,'.3f')

	def jointplot_builder(stats_entry):
		"""
		params,ajuste,kde_vec,cores,par_labels,save_labels=stats_entry
		
		params=vetores dos parametros
		ajuste=valores em tupla (alpha,beta) dos ajustes de linha
		cov_ajuste=valores (matriz) das incertezas
		kde_vec=vetores das distribuições de KDE
		alpha_values= valores de alpha (tupla)
		cores= cores em tupla (tupla)
		par_labels= lapelas (tupla) -> (scatter,linha do ajuste)
		line_styles=linhas de cada tupla
		save_names=tupla--> pasta,nome da figura
		"""
		params,ajuste,cov_ajuste,kde_vec,cores,alpha_values,par_labels,line_styles,save_names=stats_entry
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)
		for i,data in enumerate(params):
			line_label=fr'$\alpha$={label_format(ajuste[i][0])}$\pm${label_format(np.sqrt(cov_ajuste[i][0,0]))}'+'\n'+fr'$\beta$={label_format(ajuste[i][1])}$\pm${label_format(np.sqrt(cov_ajuste[i][1,1]))}'
			ax_center.scatter(data[0],data[1],marker='s',edgecolor='black',alpha=alpha_values[i],label=par_labels[i],color=cores[i][0])
			ax_center.plot(re_linspace, linfunc(re_linspace,*ajuste[i]), color=cores[i][1], linestyle=line_styles[i], label=line_label)#fr'$\alpha={label_format(ajuste[i][0])}$'+'\n'+fr'$\beta={label_format(ajuste[i][1])}$')
		for j,dist in enumerate(kde_vec):
			ax_topx.plot(re_linspace,dist[0](re_linspace),ls=line_styles[j],color=cores[j][1])
			ax_topx.axvline(np.average(params[j][0]),ls=line_styles[j],color=cores[j][1],label=fr'$\mu={label_format(np.average(params[j][0]))}$')
			ax_righty.plot(dist[1](mue_linspace),mue_linspace,ls=line_styles[j],color=cores[j][1])
			ax_righty.axhline(np.average(params[j][1]),ls=line_styles[j],color=cores[j][1],label=fr'$\mu={label_format(np.average(params[j][1]))}$')
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)
		ax_righty.legend(fontsize='small')
		ax_righty.tick_params(labelleft=False)
		if save_names[0] == None:
			plt.savefig(f'{save_path}/kormendy_rel_split/rel_kormendy_{save_names[1]}.png')
		else:
			plt.savefig(f'{save_path}/kormendy_rel_split/{save_names[0]}/rel_kormendy_{save_names[1]}.png')
		plt.close()
		return
	def jointplot_builder_only(stats_entry):
		"""
		params,ajuste,kde_vec,cores,par_labels,save_labels=stats_entry
		
		params=vetores dos parametros
		ajuste=valores em tupla (alpha,beta) dos ajustes de linha
		cov_ajuste=valores (matriz) das incertezas
		kde_vec=vetores das distribuições de KDE
		cores= cores em tupla (tupla)
		par_labels= lapelas (tupla) -> (scatter,linha do ajuste)
		save_names=tupla--> pasta, nome da figura
		"""
		params,ajuste,cov_ajuste,kde_vec,cores,par_labels,save_names=stats_entry
		line_styles='-'
		fig = plt.figure(figsize=(8, 8))
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)
		line_label=fr'$\alpha$={label_format(ajuste[0])}$\pm${label_format(np.sqrt(cov_ajuste[0,0]))}'+'\n'+fr'$\beta$={label_format(ajuste[1])}$\pm${label_format(np.sqrt(cov_ajuste[1,1]))}'
		ax_center.scatter(params[0],params[1],marker='s',edgecolor='black',alpha=0.3,label=par_labels[0],color=cores[0])
		ax_center.plot(re_linspace, linfunc(re_linspace,*ajuste), color=cores[1], linestyle=line_styles, label=line_label)
		
		ax_topx.plot(re_linspace,kde_vec[0](re_linspace),ls=line_styles,color=cores[1])
		ax_topx.axvline(np.average(params[0]),ls=line_styles,color=cores[1],label=fr'$\mu={label_format(np.average(params[0]))}$')
		ax_righty.plot(kde_vec[1](mue_linspace),mue_linspace,ls=line_styles,color=cores[1])
		ax_righty.axhline(np.average(params[1]),ls=line_styles,color=cores[1],label=fr'$\mu={label_format(np.average(params[1]))}$')
		
		ax_center.legend(fontsize='small')
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)
		ax_righty.legend(fontsize='small')
		ax_righty.tick_params(labelleft=False)

		if save_names[0] == None:
			plt.savefig(f'{save_path}/kormendy_rel_split/rel_kormendy_{save_names[1]}.png')
		else:
			plt.savefig(f'{save_path}/kormendy_rel_split/{save_names[0]}/rel_kormendy_{save_names[1]}.png')
		plt.close()
		return

	def jointplot_cmap_builder(stats_entry):
		"""
		params,ajuste,kde_vec,cores,par_labels,save_labels=stats_entry
		
		params=vetores dos parametros
		ajuste=valores em tupla (alpha,beta) dos ajustes de linha
		cov_ajuste=valores (matriz) das incertezas
		kde_vec=vetores das distribuições de KDE
		alpha_values= valores de alpha (tupla)
		cores= cores em tupla (tupla) --> aqui o primeiro valor é o do mapa de cores
		par_labels= lapelas (tupla) -> (scatter,linha do ajuste) --> sendo o ultimo valor o label da cor
		line_styles=linhas de cada tupla
		save_names = tupla -> parametros xy, parametro da cor
		"""
		params,ajuste,cov_ajuste,kde_vec,cores,par_labels,line_styles,cor_lim,save_names=stats_entry

		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)

		for i,data in enumerate(params):
			line_label=fr'$\alpha$={label_format(ajuste[i][0])}$\pm${label_format(np.sqrt(cov_ajuste[i][0,0]))}'+'\n'+fr'$\beta$={label_format(ajuste[i][1])}$\pm${label_format(np.sqrt(cov_ajuste[i][1,1]))}'
			ax_center.scatter(data[0],data[1],marker='o',edgecolor='black',alpha=0.9,vmin=cor_lim[0],vmax=cor_lim[1],label=par_labels[i],c=cores[i][0],cmap=cmap)
			ax_center.plot(re_linspace,linfunc(re_linspace,*ajuste[i]), color=cores[i][1], linestyle=line_styles[i], label=line_label)

		for j,dist in enumerate(kde_vec):
			ax_topx.plot(re_linspace,dist[0](re_linspace),ls=line_styles[j],color=cores[j][1])
			ax_topx.axvline(np.average(params[j][0]),ls=line_styles[j],color=cores[j][1],label=fr'$\mu={label_format(np.average(params[j][0]))}$')
			ax_righty.plot(dist[1](mue_linspace),mue_linspace,ls=line_styles[j],color=cores[j][1])
			ax_righty.axhline(np.average(params[j][1]),ls=line_styles[j],color=cores[j][1],label=fr'$\mu={label_format(np.average(params[j][1]))}$')

		ax_center.legend(fontsize='small')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(ax_center.collections[0], cax=cax)
		cbar.set_label(par_labels[-1])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')
		if save_names[0] == None:
			plt.savefig(f'{save_path}/kormendy_rel_split/kormendy_rel_{save_names[1]}_{save_names[2]}.png')
		else:
			plt.savefig(f'{save_path}/kormendy_rel_split/{save_names[0]}/kormendy_rel_{save_names[1]}_{save_names[2]}.png')
		plt.close()

		return
	def jointplot_cmap_builder_only(stats_entry):
		"""
		params,ajuste,kde_vec,cores,par_labels,save_labels=stats_entry
		
		params=vetores dos parametros
		ajuste=valores em tupla (alpha,beta) dos ajustes de linha
		cov_ajuste=valores (matriz) das incertezas
		kde_vec=vetores das distribuições de KDE
		alpha_values= valores de alpha (tupla)
		cores= cores em tupla (tupla) --> aqui o primeiro valor é o do mapa de cores
		par_labels= lapelas (tupla) -> (scatter,linha do ajuste) --> sendo o ultimo valor o label da cor
		line_styles=linhas de cada tupla
		save_names = tupla -> parametros xy, parametro da cor
		"""
		params,ajuste,cov_ajuste,kde_vec,cores,par_labels,cor_lim,save_names=stats_entry

		fig = plt.figure(figsize=(8,8))
		fig.subplots_adjust(left=0.18)
		gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
		ax_center = fig.add_subplot(gs[1,0])
		ax_topx = fig.add_subplot(gs[0,0],sharex=ax_center)
		ax_righty = fig.add_subplot(gs[1,1],sharey=ax_center)
		line_styles='-'
		line_label=fr'$\alpha$={label_format(ajuste[0])}$\pm${label_format(np.sqrt(cov_ajuste[0,0]))}'+'\n'+fr'$\beta$={label_format(ajuste[1])}$\pm${label_format(np.sqrt(cov_ajuste[1,1]))}'
		ax_center.scatter(params[0],params[1],marker='o',edgecolor='black',alpha=0.9,vmin=cor_lim[0],vmax=cor_lim[1],label=par_labels[0],c=cores[0],cmap=cmap)
		ax_center.plot(re_linspace,linfunc(re_linspace,*ajuste), color=cores[1], linestyle=line_styles, label=line_label)

		ax_topx.plot(re_linspace,kde_vec[0](re_linspace),ls=line_styles,color=cores[1])
		ax_topx.axvline(np.average(params[0]),ls=line_styles,color=cores[1],label=fr'$\mu={label_format(np.average(params[0]))}$')
		ax_righty.plot(kde_vec[1](mue_linspace),mue_linspace,ls=line_styles,color=cores[1])
		ax_righty.axhline(np.average(params[1]),ls=line_styles,color=cores[1],label=fr'$\mu={label_format(np.average(params[1]))}$')

		ax_center.legend(fontsize='small')
		ax_topx.legend(fontsize='small')
		ax_topx.tick_params(labelbottom=False)
		ax_righty.legend(fontsize='x-small')
		ax_righty.tick_params(labelleft=False)
		ax_center.set_ylim(y2ss,y1ss)
		ax_center.set_xlim(x1ss,x2ss)
		ax_center.set_xlabel(xlabel)
		ax_center.set_ylabel(ylabel)

		cax = inset_axes(ax_center,width="5%",height="100%",loc='lower left',bbox_to_anchor=(-0.2, 0., 1, 1),bbox_transform=ax_center.transAxes,borderpad=0)
		cbar = plt.colorbar(ax_center.collections[0], cax=cax)
		cbar.set_label(par_labels[-1])
		cbar.ax.yaxis.set_ticks_position('left')
		cbar.ax.yaxis.set_label_position('left')

		plt.savefig(f'{save_path}/kormendy_rel_split/{save_names[0]}/kormendy_rel_{save_names[1]}_{save_names[2]}.png')
		plt.close()

		return
	x1ss,x2ss=(-0.28322921653227484,2.716867381942037)
	y1ss,y2ss=(16.824997655983005,26.727550943781317)
	#################################################
	lim_region=config['lim_region']
	names_simples=config['names_simples']
	alpha_vec=config['alpha_vec']
	cores=config['cores']
	line_width=config['line_width']
	#################################################
	##TRADICIONAL - MUE MÉDIO EM RELAÇÃO AOS MODELOS DE CADA UM DOS COMPONENTES
	#################################################
	#PARÂMETROS DA RELAÇÃO DE KORMENDY
	###RAIO EFETIVO
	#GRUPOS HIGH -- LOW_LEFT -- LOW_RIGHT
	##SOMENTE SÉRSIC
	re_sersic_high,re_sersic_low_left,re_sersic_low_right=re_s_kpc[lim_region[0]],re_s_kpc[lim_region[1]],re_s_kpc[lim_region[2]]
	#COMPONENTES INTERNO-EXTERNO
	##HIGH
	re_intern_high,re_extern_high=re_1_kpc[lim_region[0]],re_2_kpc[lim_region[0]]
	##LOW_LEFT
	re_intern_low_left,re_extern_low_left=re_1_kpc[lim_region[1]],re_2_kpc[lim_region[1]]
	##LOW_RIGHT
	re_intern_low_right,re_extern_low_right=re_1_kpc[lim_region[2]],re_2_kpc[lim_region[2]]
	###
	re_kde_entry=np.hstack((re_s_kpc,re_1_kpc,re_2_kpc))
	kde_re=kde(re_kde_entry)
	re_factor = kde_re.factor
	re_linspace=np.linspace(x2ss,x1ss,3000)

	#SERSIC SIMPLES
	re_kde_sersic_high,re_kde_sersic_low_left,re_kde_sersic_low_right=kde(re_sersic_high,bw_method=re_factor),kde(re_sersic_low_left,bw_method=re_factor),kde(re_sersic_low_right,bw_method=re_factor)
	#COMPONENTES -- INTERNO,EXTERNO
	re_kde_intern_high,re_kde_extern_high=kde(re_intern_high,bw_method=re_factor),kde(re_extern_high,bw_method=re_factor)
	re_kde_intern_low_left,re_kde_extern_low_left=kde(re_intern_low_left,bw_method=re_factor),kde(re_extern_low_left,bw_method=re_factor)
	re_kde_intern_low_right,re_kde_extern_low_right=kde(re_intern_low_right,bw_method=re_factor),kde(re_extern_low_right,bw_method=re_factor)

	#######################
	#BRILHO EFETIVO MÉDIO

	#GRUPOS HIGH -- LOW_LEFT -- LOW_RIGHT
	##SOMENTE SÉRSIC
	mue_sersic_high,mue_sersic_low_left,mue_sersic_low_right=mue_med_s[lim_region[0]],mue_med_s[lim_region[1]],mue_med_s[lim_region[2]]
	#COMPONENTES INTERNO-EXTERNO
	##HIGH
	mue_intern_high,mue_extern_high=mue_med_comp_1[lim_region[0]],mue_med_comp_2[lim_region[0]]
	##LOW_LEFT
	mue_intern_low_left,mue_extern_low_left=mue_med_comp_1[lim_region[1]],mue_med_comp_2[lim_region[1]]
	##LOW_RIGHT
	mue_intern_low_right,mue_extern_low_right=mue_med_comp_1[lim_region[2]],mue_med_comp_2[lim_region[2]]

	#KDE
	mue_kde_entry=np.hstack((mue_med_s,mue_med_comp_1,mue_med_comp_2))
	kde_mue=kde(mue_kde_entry)
	mue_factor = kde_mue.factor
	mue_linspace=np.linspace(y2ss,y1ss,3000)

	#SERSIC SIMPLES
	mue_kde_sersic_high,mue_kde_sersic_low_left,mue_kde_sersic_low_right=kde(mue_sersic_high,bw_method=mue_factor),kde(mue_sersic_low_left,bw_method=mue_factor),kde(mue_sersic_low_right,bw_method=mue_factor)
	#COMPONENTES -- INTERNO,EXTERNO
	mue_kde_intern_high,mue_kde_extern_high=kde(mue_intern_high,bw_method=mue_factor),kde(mue_extern_high,bw_method=mue_factor)
	mue_kde_intern_low_left,mue_kde_extern_low_left=kde(mue_intern_low_left,bw_method=mue_factor),kde(mue_extern_low_left,bw_method=mue_factor)
	mue_kde_intern_low_right,mue_kde_extern_low_right=kde(mue_intern_low_right,bw_method=mue_factor),kde(mue_extern_low_right,bw_method=mue_factor)
	#########################################
	#POLYFITS

	ajuste_s_high,cov_s_high = np.polyfit(re_sersic_high,mue_sersic_high,1,cov=True)
	ajuste_1_high,cov_1_high = np.polyfit(re_intern_high,mue_intern_high,1,cov=True)
	ajuste_2_high,cov_2_high = np.polyfit(re_extern_high,mue_extern_high,1,cov=True)

	ajuste_s_low_left,cov_s_low_left = np.polyfit(re_sersic_low_left,mue_sersic_low_left,1,cov=True)
	ajuste_1_low_left,cov_1_low_left = np.polyfit(re_intern_low_left,mue_intern_low_left,1,cov=True)
	ajuste_2_low_left,cov_2_low_left = np.polyfit(re_extern_low_left,mue_extern_low_left,1,cov=True)
	
	ajuste_s_low_right,cov_s_low_right = np.polyfit(re_sersic_low_right,mue_sersic_low_right,1,cov=True)
	ajuste_1_low_right,cov_1_low_right = np.polyfit(re_intern_low_right,mue_intern_low_right,1,cov=True)
	ajuste_2_low_right,cov_2_low_right = np.polyfit(re_extern_low_right,mue_extern_low_right,1,cov=True)
	######################################
	#GRÁFICO JOINTPLOT DA RELAÇÃO DE KORMENDY
	################################################
	##SÉRSIC -- 2C (SMALL E BIG) vs ELIPTICAS
	xlabel=r'$\log_{10} R_e (Kpc)$'
	ylabel=r'$<\mu_e>$'

	names_morf=['cD','E/cD & cD/E','E']
	#############################################
	#SÉRSIC SIMPLES -- POR GRUPO
	# HIGH
	
	params=([re_sersic_high,mue_sersic_high])
	ajuste=(ajuste_s_high)
	cov_ajuste=(cov_s_high)
	kde_vec=([re_kde_sersic_high,mue_kde_sersic_high])
	cores_entry=(cores[0],'black')
	par_labels=(names_simples[0])
	save_labels=('sersic','sersic_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_LEFT 
	params=([re_sersic_low_left,mue_sersic_low_left])
	ajuste=(ajuste_s_low_left)
	cov_ajuste=(cov_s_low_left)
	kde_vec=([re_kde_sersic_low_left,mue_kde_sersic_low_left])
	cores_entry=(cores[1],'black')
	par_labels=(names_simples[1])
	save_labels=('sersic','sersic_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_RIGHT
	params=([re_sersic_low_right,mue_sersic_low_right])
	ajuste=(ajuste_s_low_right)
	cov_ajuste=(cov_s_low_right)
	kde_vec=([re_kde_sersic_low_right,mue_kde_sersic_low_right])
	cores_entry=(cores[2],'black')
	par_labels=(names_simples[2])
	save_labels=('sersic','sersic_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	##########################
	#COMPONENTE INTERNO -- POR GRUPO
	# HIGH
	params=([re_intern_high,mue_intern_high])
	ajuste=(ajuste_1_high)
	cov_ajuste=(cov_1_high)
	kde_vec=([re_kde_intern_high,mue_kde_intern_high])
	cores_entry=(cores[0],'black')
	par_labels=(names_simples[0])
	save_labels=('intern','intern_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_LEFT 
	params=([re_intern_low_left,mue_intern_low_left])
	ajuste=(ajuste_1_low_left)
	cov_ajuste=(cov_1_low_left)
	kde_vec=([re_kde_intern_low_left,mue_kde_intern_low_left])
	cores_entry=(cores[1],'black')
	par_labels=(names_simples[1])
	save_labels=('intern','intern_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_RIGHT
	params=([re_intern_low_right,mue_intern_low_right])
	ajuste=(ajuste_1_low_right)
	cov_ajuste=(cov_1_low_right)
	kde_vec=([re_kde_intern_low_right,mue_kde_intern_low_right])
	cores_entry=(cores[2],'black')
	par_labels=(names_simples[2])
	save_labels=('intern','intern_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)
	#############################################
	#COMPONENTE EXTERNO -- POR GRUPO
	# HIGH
	params=([re_extern_high,mue_extern_high])
	ajuste=(ajuste_2_high)
	cov_ajuste=(cov_2_high)
	kde_vec=([re_kde_extern_high,mue_kde_extern_high])
	cores_entry=(cores[0],'black')
	par_labels=(names_simples[0])
	save_labels=('extern','extern_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_LEFT 
	params=([re_extern_low_left,mue_extern_low_left])
	ajuste=(ajuste_2_low_left)
	cov_ajuste=(cov_2_low_left)
	kde_vec=([re_kde_extern_low_left,mue_kde_extern_low_left])
	cores_entry=(cores[1],'black')
	par_labels=(names_simples[1])
	save_labels=('extern','extern_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)

	# LOW_RIGHT
	params=([re_extern_low_right,mue_extern_low_right])
	ajuste=(ajuste_2_low_right)
	cov_ajuste=(cov_2_low_right)
	kde_vec=([re_kde_extern_low_right,mue_kde_extern_low_right])
	cores_entry=(cores[2],'black')
	par_labels=(names_simples[2])
	save_labels=('extern','extern_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,save_labels
	jointplot_builder_only(stats_entry)
	
	#############################################
	"""
	SÉRSIC
	high x low_left
	high x low_right
	low_left x low_right
	high x low_left x low_right
	COMPONENTE 1 
	high x low_left
	high x low_right
	low_left x low_right
	high x low_left x low_right
	COMPONENTE 2 
	high x low_left
	high x low_right
	low_left x low_right
	high x low_left x low_right
	"""
	# SÉRSIC
	
	## HIGH X LOW_LEFT
	params=([re_sersic_high,mue_sersic_high],[re_sersic_low_left,mue_sersic_low_left])
	ajuste=(ajuste_s_high,ajuste_s_low_left)
	cov_ajuste=(cov_s_high,cov_s_low_left)
	kde_vec=([re_kde_sersic_high,mue_kde_sersic_high],[re_kde_sersic_low_left,mue_kde_sersic_low_left])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]))
	alpha_values=(0.3,0.3)
	par_labels=(names_simples[0],names_simples[1])
	line_styles=('-','-')
	save_labels=('sersic','sersic_high_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_RIGHT
	params=([re_sersic_high,mue_sersic_high],[re_sersic_low_right,mue_sersic_low_right])
	ajuste=(ajuste_s_high,ajuste_s_low_right)
	cov_ajuste=(cov_s_high,cov_s_low_right)
	kde_vec=([re_kde_sersic_high,mue_kde_sersic_high],[re_kde_sersic_low_right,mue_kde_sersic_low_right])
	cores_entry=((cores[0],cores[0]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(names_simples[0],names_simples[2])
	line_styles=('-','-')
	save_labels=('sersic','sersic_high_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## LOW_LEFT X LOW_RIGHT
	params=([re_sersic_low_left,mue_sersic_low_left],[re_sersic_low_right,mue_sersic_low_right])
	ajuste=(ajuste_s_low_left,ajuste_s_low_right)
	cov_ajuste=(cov_s_low_left,cov_s_low_right)
	kde_vec=([re_kde_sersic_low_left,mue_kde_sersic_low_left],[re_kde_sersic_low_right,mue_kde_sersic_low_right])
	cores_entry=((cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(names_simples[1],names_simples[2])
	line_styles=('-','-')
	save_labels=('sersic','sersic_low_left_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_LEFT X LOW_RIGHT
	params=([re_sersic_high,mue_sersic_high],[re_sersic_low_left,mue_sersic_low_left],[re_sersic_low_right,mue_sersic_low_right])
	ajuste=(ajuste_s_high,ajuste_s_low_left,ajuste_s_low_right)
	cov_ajuste=(cov_s_high,cov_s_low_left,cov_s_low_right)
	kde_vec=([re_kde_sersic_high,mue_kde_sersic_high],[re_kde_sersic_low_left,mue_kde_sersic_low_left],[re_kde_sersic_low_right,mue_kde_sersic_low_right])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3,0.3)
	par_labels=(names_simples[0],names_simples[1],names_simples[2])
	line_styles=('-','-','-')
	save_labels=('sersic','sersic_low_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)
	#######################################
	#COMPONENTE 1 

	## HIGH X LOW_LEFT
	params=([re_intern_high,mue_intern_high],[re_intern_low_left,mue_intern_low_left])
	ajuste=(ajuste_1_high,ajuste_1_low_left)
	cov_ajuste=(cov_1_high,cov_1_low_left)
	kde_vec=([re_kde_intern_high,mue_kde_intern_high],[re_kde_intern_low_left,mue_kde_intern_low_left])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 1. {names_simples[0]}',f'Comp 1. {names_simples[1]}')
	line_styles=('-','-')
	save_labels=('intern','comp_intern_high_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_RIGHT
	params=([re_intern_high,mue_intern_high],[re_intern_low_right,mue_intern_low_right])
	ajuste=(ajuste_1_high,ajuste_1_low_right)
	cov_ajuste=(cov_1_high,cov_1_low_right)
	kde_vec=([re_kde_intern_high,mue_kde_intern_high],[re_kde_intern_low_right,mue_kde_intern_low_right])
	cores_entry=((cores[0],cores[0]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 1. {names_simples[0]}',f'Comp 1. {names_simples[2]}')
	line_styles=('-','-')
	save_labels=('intern','comp_intern_high_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## LOW_LEFT X LOW_RIGHT
	params=([re_intern_low_left,mue_intern_low_left],[re_intern_low_right,mue_intern_low_right])
	ajuste=(ajuste_1_low_left,ajuste_1_low_right)
	cov_ajuste=(cov_1_low_left,cov_1_low_right)
	kde_vec=([re_kde_intern_low_left,mue_kde_intern_low_left],[re_kde_intern_low_right,mue_kde_intern_low_right])
	cores_entry=((cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 1. {names_simples[1]}',f'Comp 1. {names_simples[2]}')
	line_styles=('-','-')
	save_labels=('intern','comp_intern_low_left_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_LEFT X LOW_RIGHT
	params=([re_intern_high,mue_intern_high],[re_intern_low_left,mue_intern_low_left],[re_intern_low_right,mue_intern_low_right])
	ajuste=(ajuste_1_high,ajuste_1_low_left,ajuste_1_low_right)
	cov_ajuste=(cov_1_high,cov_1_low_left,cov_1_low_right)
	kde_vec=([re_kde_intern_high,mue_kde_intern_high],[re_kde_intern_low_left,mue_kde_intern_low_left],[re_kde_intern_low_right,mue_kde_intern_low_right])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3,0.3)
	par_labels=(f'Comp 1. {names_simples[0]}',f'Comp 1. {names_simples[1]}',f'Comp 1. {names_simples[2]}')
	line_styles=('-','-','-')
	save_labels=('intern','comp_intern_low_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	#######################################
	#COMPONENTE 2 

	## HIGH X LOW_LEFT
	params=([re_extern_high,mue_extern_high],[re_extern_low_left,mue_extern_low_left])
	ajuste=(ajuste_2_high,ajuste_2_low_left)
	cov_ajuste=(cov_2_high,cov_2_low_left)
	kde_vec=([re_kde_extern_high,mue_kde_extern_high],[re_kde_extern_low_left,mue_kde_extern_low_left])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 2. {names_simples[0]}',f'Comp 2. {names_simples[1]}')
	line_styles=('-','-')
	save_labels=('extern','comp_extern_high_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_RIGHT
	params=([re_extern_high,mue_extern_high],[re_extern_low_right,mue_extern_low_right])
	ajuste=(ajuste_2_high,ajuste_2_low_right)
	cov_ajuste=(cov_2_high,cov_2_low_right)
	kde_vec=([re_kde_extern_high,mue_kde_extern_high],[re_kde_extern_low_right,mue_kde_extern_low_right])
	cores_entry=((cores[0],cores[0]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 2. {names_simples[0]}',f'Comp 2. {names_simples[2]}')
	line_styles=('-','-')
	save_labels=('extern','comp_extern_high_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## LOW_LEFT X LOW_RIGHT
	params=([re_extern_low_left,mue_extern_low_left],[re_extern_low_right,mue_extern_low_right])
	ajuste=(ajuste_2_low_left,ajuste_2_low_right)
	cov_ajuste=(cov_2_low_left,cov_2_low_right)
	kde_vec=([re_kde_extern_low_left,mue_kde_extern_low_left],[re_kde_extern_low_right,mue_kde_extern_low_right])
	cores_entry=((cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3)
	par_labels=(f'Comp 2. {names_simples[1]}',f'Comp 2. {names_simples[2]}')
	line_styles=('-','-')
	save_labels=('extern','comp_extern_low_left_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	## HIGH X LOW_LEFT X LOW_RIGHT
	params=([re_extern_high,mue_extern_high],[re_extern_low_left,mue_extern_low_left],[re_extern_low_right,mue_extern_low_right])
	ajuste=(ajuste_2_high,ajuste_2_low_left,ajuste_2_low_right)
	cov_ajuste=(cov_2_high,cov_2_low_left,cov_2_low_right)
	kde_vec=([re_kde_extern_high,mue_kde_extern_high],[re_kde_extern_low_left,mue_kde_extern_low_left],[re_kde_extern_low_right,mue_kde_extern_low_right])
	cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
	alpha_values=(0.3,0.3,0.3)
	par_labels=(f'Comp 2. {names_simples[0]}',f'Comp 2. {names_simples[1]}',f'Comp 2. {names_simples[2]}')
	line_styles=('-','-','-')
	save_labels=('intern','comp_intern_low_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)
	
	########################################
	# HIGH
	params=([re_intern_high,mue_intern_high],[re_extern_high,mue_extern_high])
	ajuste=(ajuste_1_high,ajuste_2_high)
	cov_ajuste=(cov_1_high,cov_2_high)
	kde_vec=([re_kde_intern_high,mue_kde_intern_high],[re_kde_extern_high,mue_kde_extern_high])
	cores_entry=(('white',cores[0]),('black',cores[0]))
	alpha_values=(0.5,0.5)
	par_labels=(f'Comp 1. {names_simples[0]}',f'Comp 2. {names_simples[0]}')
	line_styles=('-','-.')
	save_labels=(None,'intern_extern_high')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	# LOW_LEFT
	pov=[650.88187589,1177.47577485]

	print('OBJETOS EM QUE RE1/PSF_FWHM<=1 NOS GRUPOS DE RFF-ETA')
	print(f'{names_simples[0]}',len(re_intern_high[np.divide(re1,psf_fwhm)[lim_region[0]]<=1.]))
	print(f'{names_simples[1]}',len(re_intern_low_left[np.divide(re1,psf_fwhm)[lim_region[1]]<=1.]))
	print(f'{names_simples[2]}',len(re_intern_low_right[np.divide(re1,psf_fwhm)[lim_region[2]]<=1.]))
	print('OBJETOS EXTRALUZ NOS GRUPOS RFF-ETA')
	print(f'{names_simples[0]}',len(re_intern_high[(delta_bic_obs < log_model(redshift,*pov))[lim_region[0]] & (np.divide(re1,psf_fwhm)[lim_region[0]]<=1.)]))
	print(f'{names_simples[1]}',len(re_intern_low_left[(delta_bic_obs < log_model(redshift,*pov))[lim_region[1]] & (np.divide(re1,psf_fwhm)[lim_region[1]]<=1.)]))
	print(f'{names_simples[2]}',len(re_intern_low_right[(delta_bic_obs < log_model(redshift,*pov))[lim_region[2]] & (np.divide(re1,psf_fwhm)[lim_region[2]]<=1.)]))
	params=([re_intern_low_left,mue_intern_low_left],[re_extern_low_left,mue_extern_low_left])
	ajuste=(ajuste_1_low_left,ajuste_2_low_left)
	cov_ajuste=(cov_1_low_left,cov_2_low_left)
	kde_vec=([re_kde_intern_low_left,mue_kde_intern_low_left],[re_kde_extern_low_left,mue_kde_extern_low_left])
	cores_entry=(('white',cores[1]),('black',cores[1]))
	alpha_values=(0.5,0.5)
	par_labels=(f'Comp 1. {names_simples[1]}',f'Comp 2. {names_simples[1]}')
	line_styles=('-','-.')
	save_labels=(None,'intern_extern_low_left')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)

	# LOW_RIGHT
	params=([re_intern_low_right,mue_intern_low_right],[re_extern_low_right,mue_extern_low_right])
	ajuste=(ajuste_1_low_right,ajuste_2_low_right)
	cov_ajuste=(cov_1_low_right,cov_2_low_right)
	kde_vec=([re_kde_intern_low_right,mue_kde_intern_low_right],[re_kde_extern_low_right,mue_kde_extern_low_right])
	cores_entry=(('white',cores[2]),('black',cores[2]))
	alpha_values=(0.5,0.5)
	par_labels=(f'Comp 1. {names_simples[2]}',f'Comp 2. {names_simples[2]}')
	line_styles=('-','-.')
	save_labels=(None,'intern_extern_low_right')
	stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
	jointplot_builder(stats_entry)
	# PARTE DO L07
	if sample == 'L07':
		#RAIO EFETIVO
		re_sersic_high_cd,re_sersic_low_left_cd,re_sersic_low_right_cd=re_s_kpc[lim_region[0] & cd_cut],re_s_kpc[lim_region[1] & cd_cut],re_s_kpc[lim_region[2] & cd_cut]
		
		re_intern_high_cd,re_extern_high_cd=re_1_kpc[lim_region[0] & cd_cut],re_2_kpc[lim_region[0] & cd_cut]
		re_intern_low_left_cd,re_extern_low_left_cd=re_1_kpc[lim_region[1] & cd_cut],re_2_kpc[lim_region[1] & cd_cut]
		re_intern_low_right_cd,re_extern_low_right_cd=re_1_kpc[lim_region[2] & cd_cut],re_2_kpc[lim_region[2] & cd_cut]

		re_kde_sersic_high_cd,re_kde_sersic_low_left_cd,re_kde_sersic_low_right_cd=kde(re_sersic_high_cd,bw_method=re_factor),kde(re_sersic_low_left_cd,bw_method=re_factor),kde(re_sersic_low_right_cd,bw_method=re_factor)

		re_kde_intern_high_cd,re_kde_extern_high_cd=kde(re_intern_high_cd,bw_method=re_factor),kde(re_extern_high_cd,bw_method=re_factor)
		re_kde_intern_low_left_cd,re_kde_extern_low_left_cd=kde(re_intern_low_left_cd,bw_method=re_factor),kde(re_extern_low_left_cd,bw_method=re_factor)
		re_kde_intern_low_right_cd,re_kde_extern_low_right_cd=kde(re_intern_low_right_cd,bw_method=re_factor),kde(re_extern_low_right_cd,bw_method=re_factor)

		#MAGNITUDE EFETIVA

		mue_sersic_high_cd,mue_sersic_low_left_cd,mue_sersic_low_right_cd=mue_med_s[lim_region[0] & cd_cut],mue_med_s[lim_region[1] & cd_cut],mue_med_s[lim_region[2] & cd_cut]

		mue_intern_high_cd,mue_extern_high_cd=mue_med_comp_1[lim_region[0] & cd_cut],mue_med_comp_2[lim_region[0] & cd_cut]
		mue_intern_low_left_cd,mue_extern_low_left_cd=mue_med_comp_1[lim_region[1] & cd_cut],mue_med_comp_2[lim_region[1] & cd_cut]
		mue_intern_low_right_cd,mue_extern_low_right_cd=mue_med_comp_1[lim_region[2] & cd_cut],mue_med_comp_2[lim_region[2] & cd_cut]

		mue_kde_sersic_high_cd,mue_kde_sersic_low_left_cd,mue_kde_sersic_low_right_cd=kde(mue_sersic_high_cd,bw_method=mue_factor),kde(mue_sersic_low_left_cd,bw_method=mue_factor),kde(mue_sersic_low_right_cd,bw_method=mue_factor)

		mue_kde_intern_high_cd,mue_kde_extern_high_cd=kde(mue_intern_high_cd,bw_method=mue_factor),kde(mue_extern_high_cd,bw_method=mue_factor)
		mue_kde_intern_low_left_cd,mue_kde_extern_low_left_cd=kde(mue_intern_low_left_cd,bw_method=mue_factor),kde(mue_extern_low_left_cd,bw_method=mue_factor)
		mue_kde_intern_low_right_cd,mue_kde_extern_low_right_cd=kde(mue_intern_low_right_cd,bw_method=mue_factor),kde(mue_extern_low_right_cd,bw_method=mue_factor)
		#####################################################
		#LINHAS - SUBGRUPO CDS DO ZHAO
		##SÉRSIC 
		ajuste_s_high_cd,cov_s_high_cd = np.polyfit(re_sersic_high_cd,mue_sersic_high_cd,1,cov=True)
		ajuste_s_low_left_cd,cov_s_low_left_cd = np.polyfit(re_sersic_low_left_cd,mue_sersic_low_left_cd,1,cov=True)
		ajuste_s_low_right_cd,cov_s_low_right_cd = np.polyfit(re_sersic_low_right_cd,mue_sersic_low_right_cd,1,cov=True)
		## COMPONENTES HIGH
		ajuste_1_high_cd,cov_1_high_cd = np.polyfit(re_intern_high_cd,mue_intern_high_cd,1,cov=True)
		ajuste_2_high_cd,cov_2_high_cd = np.polyfit(re_extern_high_cd,mue_extern_high_cd,1,cov=True)
		## COMPONENTES LOW_LEFT
		ajuste_1_low_left_cd,cov_1_low_left_cd = np.polyfit(re_intern_low_left_cd,mue_intern_low_left_cd,1,cov=True)
		ajuste_2_low_left_cd,cov_2_low_left_cd = np.polyfit(re_extern_low_left_cd,mue_extern_low_left_cd,1,cov=True)
		## COMPONENTES LOW_RIGHT
		ajuste_1_low_right_cd,cov_1_low_right_cd = np.polyfit(re_intern_low_right_cd,mue_intern_low_right_cd,1,cov=True)
		ajuste_2_low_right_cd,cov_2_low_right_cd = np.polyfit(re_extern_low_right_cd,mue_extern_low_right_cd,1,cov=True)

		#SÉRSIC - HIGH x LOW_LEFT x LOW_RIGHT - SUBGRUPO CDS DO ZHAO

		params=([re_sersic_high_cd,mue_sersic_high_cd],[re_sersic_low_left_cd,mue_sersic_low_left_cd],[re_sersic_low_right_cd,mue_sersic_low_right_cd])
		ajuste=(ajuste_s_high_cd,ajuste_s_low_left_cd,ajuste_s_low_right_cd)
		cov_ajuste=(cov_s_high_cd,cov_s_low_left_cd,cov_s_low_right_cd)
		kde_vec=([re_kde_sersic_high_cd,mue_kde_sersic_high_cd],[re_kde_sersic_low_left_cd,mue_kde_sersic_low_left_cd],[re_kde_sersic_low_right_cd,mue_kde_sersic_low_right_cd])
		cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
		alpha_values=(0.3,0.3,0.3)
		par_labels=(f'{names_simples[0]}[cD]',f'{names_simples[1]}[cD]',f'{names_simples[2]}[cD]')
		line_styles=('-','-','-')
		save_labels=('sersic','sersic_low_high_cD_zhao')
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
		jointplot_builder(stats_entry)

		##COMPONENTE - INTERNO -- SUBGRUPO cDS ZHAO

		params=([re_intern_high_cd,mue_intern_high_cd],[re_intern_low_left_cd,mue_intern_low_left_cd],[re_intern_low_right_cd,mue_intern_low_right_cd])
		ajuste=(ajuste_1_high_cd,ajuste_1_low_left_cd,ajuste_1_low_right_cd)
		cov_ajuste=(cov_1_high_cd,cov_1_low_left_cd,cov_1_low_right_cd)
		kde_vec=([re_kde_intern_high_cd,mue_kde_intern_high_cd],[re_kde_intern_low_left_cd,mue_kde_intern_low_left_cd],[re_kde_intern_low_right_cd,mue_kde_intern_low_right_cd])
		cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
		alpha_values=(0.3,0.3,0.3)
		par_labels=(f'Comp 1.{names_simples[0]}[cD]',f'Comp 1.{names_simples[1]}[cD]',f'Comp 1.{names_simples[2]}[cD]')
		line_styles=('-','-','-')
		save_labels=('intern','intern_low_high_cD_zhao')
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
		jointplot_builder(stats_entry)

		##COMPONENTE - EXTERNO -- SUBGRUPO cDS ZHAO

		params=([re_extern_high_cd,mue_extern_high_cd],[re_extern_low_left_cd,mue_extern_low_left_cd],[re_extern_low_right_cd,mue_extern_low_right_cd])
		ajuste=(ajuste_2_high_cd,ajuste_2_low_left_cd,ajuste_2_low_right_cd)
		cov_ajuste=(cov_2_high_cd,cov_2_low_left_cd,cov_2_low_right_cd)
		kde_vec=([re_kde_extern_high_cd,mue_kde_extern_high_cd],[re_kde_extern_low_left_cd,mue_kde_extern_low_left_cd],[re_kde_extern_low_right_cd,mue_kde_extern_low_right_cd])
		cores_entry=((cores[0],cores[0]),(cores[1],cores[1]),(cores[2],cores[2]))
		alpha_values=(0.3,0.3,0.3)
		par_labels=(f'Comp 2.{names_simples[0]}[cD]',f'Comp 2.{names_simples[1]}[cD]',f'Comp 2.{names_simples[2]}[cD]')
		line_styles=('-','-','-')
		save_labels=('extern','extern_low_high_cD_zhao')
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,alpha_values,par_labels,line_styles,save_labels
		jointplot_builder(stats_entry)
	
	##RAZÕES ENTRE PARAMETROS
	data_to_look=[box2,n2,e2,n_s,e_s,box_s,box1,n1,e1,n_s,e_s,box_s,bt_vec_corr,n_ratio_12,re_ratio_12,rff_ratio,chi2_ratio,np.log10(rff_s),np.log10(rff_ss)]
	cor_to_look=[(0,0.6),(0.5,8),(0,0.85),(0.5,12),(0,0.9),(0,0.6),(0,0.6),(0.5,8),(0,0.85),(0.5,12),(0,0.9),(0,0.6),(0,1),(1,1.5),(1,1.5),(0,0.9),(0.9,1.5),(-2.5,-0.7),(-2.5,-0.7)]
	labels_to_look=[r'$a_4/a \ 2$',r'$n_2$',r'$q_2$',r'$n_s$',r'$q_s$',r'$a_4/a \ s$',r'$a_4/a \ 1$',r'$n_1$',r'$q_1$',r'$n_s$',r'$q_s$',r'$a_4/a \ s$',r'$B/T$',r'$n_1/n_2$',r'$Re_1/Re_2 (Kpc)$',r'$RFF_{S+S}/RFF_S$',r'$\chi^2_{S+S}/\chi^2_S$',r'$\log_{10} \ RFF_{S}$',r'$\log_{10} \ RFF_{SS}$']
	save_names=['box2','n2','e2','n_s','e_s','box_s','box1','n1','e1','n_s','e_s','box_s','bt','n_ratio_12','re_ratio_12','rff_ratio','chi2_ratio','rff_s','rff_ss']
	for i,item in enumerate(data_to_look):
		#SÉRSIC
		##HIGH
		params=([re_sersic_high,mue_sersic_high])
		ajuste=(ajuste_s_high)
		cov_ajuste=(cov_s_high)
		kde_vec=([re_kde_sersic_high,mue_kde_sersic_high])
		cores_entry=(item[lim_region[0]],cores[0])
		par_labels=(names_simples[0],labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('sersic','sersic_high',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)

		##LOW_LEFT
		params=([re_sersic_low_left,mue_sersic_low_left])
		ajuste=(ajuste_s_low_left)
		cov_ajuste=(cov_s_low_left)
		kde_vec=([re_kde_sersic_low_left,mue_kde_sersic_low_left])
		cores_entry=(item[lim_region[1]],cores[1])
		par_labels=(names_simples[1],labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('sersic','sersic_low_left',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
		##LOW_RIGHT		
		params=([re_sersic_low_right,mue_sersic_low_right])
		ajuste=(ajuste_s_low_right)
		cov_ajuste=(cov_s_low_right)
		kde_vec=([re_kde_sersic_low_right,mue_kde_sersic_low_right])
		cores_entry=(item[lim_region[2]],cores[2])
		par_labels=(names_simples[2],labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('sersic','sersic_low_right',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
		#########
		#COMPONENTE INTERNO
		##HIGH
		params=([re_intern_high,mue_intern_high])
		ajuste=(ajuste_1_high)
		cov_ajuste=(cov_1_high)
		kde_vec=([re_kde_intern_high,mue_kde_intern_high])
		cores_entry=(item[lim_region[0]],cores[0])
		par_labels=(f'Comp. 1 {names_simples[0]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('intern','sersic_comp1_high',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)

		##LOW_LEFT
		params=([re_intern_low_left,mue_intern_low_left])
		ajuste=(ajuste_1_low_left)
		cov_ajuste=(cov_1_low_left)
		kde_vec=([re_kde_intern_low_left,mue_kde_intern_low_left])
		cores_entry=(item[lim_region[1]],cores[1])
		par_labels=(f'Comp. 1 {names_simples[1]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('intern','sersic_comp1_low_left',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
		##LOW_RIGHT
		params=([re_intern_low_right,mue_intern_low_right])
		ajuste=(ajuste_1_low_right)
		cov_ajuste=(cov_1_low_right)
		kde_vec=([re_kde_intern_low_right,mue_kde_intern_low_right])
		cores_entry=(item[lim_region[2]],cores[2])
		par_labels=(f'Comp. 1 {names_simples[2]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('intern','sersic_comp1_low_right',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
		#########
		#COMPONENTE EXTERNO
		##HIGH
		params=([re_extern_high,mue_extern_high])
		ajuste=(ajuste_2_high)
		cov_ajuste=(cov_2_high)
		kde_vec=([re_kde_extern_high,mue_kde_extern_high])
		cores_entry=(item[lim_region[0]],cores[0])
		par_labels=(f'Comp. 2 {names_simples[0]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('extern','sersic_comp1_high',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)

		##LOW_LEFT
		params=([re_extern_low_left,mue_extern_low_left])
		ajuste=(ajuste_2_low_left)
		cov_ajuste=(cov_2_low_left)
		kde_vec=([re_kde_extern_low_left,mue_kde_extern_low_left])
		cores_entry=(item[lim_region[1]],cores[1])
		par_labels=(f'Comp. 2 {names_simples[1]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('extern','sersic_comp1_low_left',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
		##LOW_RIGHT
		params=([re_extern_low_right,mue_extern_low_right])
		ajuste=(ajuste_2_low_right)
		cov_ajuste=(cov_2_low_right)
		kde_vec=([re_kde_extern_low_right,mue_kde_extern_low_right])
		cores_entry=(item[lim_region[2]],cores[2])
		par_labels=(f'Comp. 2 {names_simples[2]}',labels_to_look[i])
		cor_lim=cor_to_look[i]
		save_labels=('extern','sersic_comp1_low_right',save_names[i])
		stats_entry=params,ajuste,cov_ajuste,kde_vec,cores_entry,par_labels,cor_lim,save_labels
		jointplot_cmap_builder_only(stats_entry)
	
	####
	return
def svc_tester():
	from itertools import combinations	
	"""Perform an exhaustive pairwise and triplet SVM classification to
	separate E(EL) from cD galaxies.

	The routine loops over all combinations of 2 or 3 structural parameters
	from a predefined list (ellipticities, boxiness, B/T, n, RFF ratios,
	etc.) and trains a linear SVM.  The training accuracy for each
	combination is saved to a text file.

	**Global dependencies:** it relies on many global arrays (e_s, e1, e2,
	box_s, bt_vec_corr, ...) and on the masks `cd_lim` and `lim_cd_small`.

	Returns:
		None.  Output files are written as `svc_score_el_cd_{...}.dat`.
	"""

	var_name_vec=['ax_ratio','ax_ratio_1','ax_ratio_2','box','bt','bt_corr','chi2_ratio','n1','n_sersic','q1q2','n1n2','rff_ratio','re1re2','n2','box1','box2','ass','rff','eta']
	par_vec=[e_s,e1,e2,box_s,bt_vec_12,bt_vec_corr,chi2_ratio,n1,n_s,axrat_ratio_12,n_ratio_12,rff_ratio,np.log10(re_ratio_12),n2,box1,box2,ass,np.log10(rff_s),eta]

	vec_espec_photutils=['slope_a3','slope_a4','slope_b3','slope_b4','test_gr','grad_e','grad_pa']
	par_vec_photutils=[slope_a3,slope_a4,slope_b3,slope_b4,slope_gr,grad_e,grad_pa]

	vec_espec_casjobs=['conc','magabs','starmass','age']
	par_vec_casjobs=[conc_temp,magabs_temp,starmass_temp,age_temp]

	# var_name_vec.extend(vec_espec_photutils)
	# par_vec.extend(par_vec_photutils)
	# svc_lim=cd_lim & lim_photutils & np.isfinite(np.log10(rff_s))
	# save_file=f'{save_path}/svc_score_el_cd_dupla_{sample}_photutils.dat'

	svc_lim=cd_lim & np.isfinite(np.log10(rff_s))
	save_file=f'{save_path}/svc_score_el_cd_dupla_{sample}.dat'

	# var_name_vec.extend(vec_espec_casjobs)
	# par_vec.extend(par_vec_casjobs)
	# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
	# save_file=f'{save_path}/svc_score_el_cd_dupla_{sample}_casjobs.dat'

	out=open(save_file,'w')
	vec_comb=combinations(var_name_vec,2)
	for i,dupla in enumerate(vec_comb):
		name1,name2=dupla
		ind1,ind2=var_name_vec.index(name1),var_name_vec.index(name2)
		par_temp=[par_vec[ind1][svc_lim],par_vec[ind2][svc_lim]]

		vec_x_el_cd=np.column_stack(par_temp)
		split_cd_el=svc_calc_dupla(vec_x_el_cd,lim_cd_small[svc_lim].astype(int).T,dupla,save_file)
	# var_name_vec.extend(vec_espec_photutils)
	# par_vec.extend(par_vec_photutils)
	# svc_lim=cd_lim & lim_photutils & np.isfinite(np.log10(rff_s))
	# save_file=f'{save_path}/svc_score_el_cd_{sample}_photutils.dat'

	# var_name_vec.extend(vec_espec_casjobs)
	# par_vec.extend(par_vec_casjobs)
	# svc_lim=cd_lim & lim_casjobs & np.isfinite(np.log10(rff_s))
	# save_file=f'{save_path}/svc_score_el_cd_{sample}_casjobs.dat'
	#bt_corr re1re2 magabs 0.8805970149253731

	# out=open(save_file,'w')
	# vec_comb=combinations(var_name_vec,3)
	# for i,trio in enumerate(vec_comb):
	# 	name1,name2,name3=trio
	# 	ind1,ind2,ind3=var_name_vec.index(name1),var_name_vec.index(name2),var_name_vec.index(name3)
	# 	par_temp=[par_vec[ind1][svc_lim],par_vec[ind2][svc_lim],par_vec[ind3][svc_lim]]

	# 	vec_x_el_cd=np.column_stack(par_temp)
	# 	split_cd_el=svc_calc_trio(vec_x_el_cd,lim_cd_small[svc_lim].astype(int).T,trio,save_file)
	return
def rff_eta_plots(config):
	"""Complete visualisation of the RFF–η (Residual Flux Fraction vs.
	asymmetry) plane.

	This includes:
		- A histogram of RFF with the best‑fitting Gaussian+log‑normal model
		and the automatic split point computed from the model's zero crossing.
		- Scatter plots in the log₁₀ RFF vs. η plane, with lines of constant
		residual fraction (0.1 and 0.5 of the total).
		- KDE contour overlays for each morphological class.
		- Colour‑coded versions where the colour represents a third physical
		parameter (B/T, χ² ratio, ellipticities, Sérsic index, etc.).
		- L07‑specific subdivisions when the sample is 'L07'.

	Args:
		config (dict): Contains the usual region and style definitions.

	Returns:
		None.  Many figures are saved.
	"""


	def color_plots_plane(plots_entry,rff_split=False):
		"""
		param
		par_labels=titulo,color_label
		l07_regions
		cor_lim
		save_labels=pasta,figname
		rff_split
		"""
		param,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names=plots_entry

		x=np.linspace(0.001,0.3,1000)
		vec=[0.1,0.5]

		#SAMPLE
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {sample}')
		for item in vec:
			plt.plot(np.log10(x),x-item*x,label=str(item))
		plt.scatter(np.log10(rff_s),eta,c=param,edgecolors='black',cmap=cmap)
		if rff_split:
			plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(limx)
		plt.ylim(limy)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}.png')
		plt.close()
		
		for i in range(len(lim_region)):
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[i]}')
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			plt.scatter(np.log10(rff_s[lim_region[i]]),eta[lim_region[i]],c=param[lim_region[i]],edgecolors='black',cmap=cmap)
			if rff_split:
				plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.xlim(limx)
			plt.ylim(limy)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}_{region_names[i]}.png')
			plt.close()
		if sample=='L07':
			cd_cut,e_cut,misc_cut=l07_regions
			
			#ELIPTICAS ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[Zhao]')
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			plt.scatter(np.log10(rff_s[e_cut]),eta[e_cut],c=param[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
			if rff_split:
				plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(limx)
			plt.ylim(limy)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}_e_zhao.png')
			plt.close()

			#cDs ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - cD[Zhao]')
			plt.scatter(np.log10(rff_s[cd_cut]),eta[cd_cut],c=param[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			if rff_split:
				plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(limx)
			plt.ylim(limy)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}_cd_zhao.png')
			plt.close()

			#LOW_LEFT[E & E/cD]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - [E/cD]')
			plt.scatter(np.log10(re_ratio_12[misc_cut]),n2[misc_cut],c=param[misc_cut],edgecolors='black',label='E/cD',cmap=cmap)
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			if rff_split:
				plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(limx)
			plt.ylim(limy)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}_Ecd.png')
			plt.close()

			for i in range(len(lim_region)):
				fig1=plt.figure(figsize=(9,7))
				plt.suptitle(f'{par_labels[0]} - {names_simples[i]}[cD]')
				plt.scatter(np.log10(re_ratio_12[lim_region[i] & cd_cut]),n2[lim_region[i] & cd_cut],c=param[lim_region[i] & cd_cut],edgecolors='black',label=f'{names_simples[i]}[Zhao]',cmap=cmap)
				for item in vec:
					plt.plot(np.log10(x),x-item*x,label=str(item))
				if rff_split:
					plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
				cbar=plt.colorbar()
				cbar.set_label(par_labels[1], rotation=90)
				plt.clim(cor_lim)
				plt.legend()
				plt.xlim(limx)
				plt.ylim(limy)
				plt.xlabel(label_x)
				plt.ylabel(label_y)
				plt.savefig(f'{save_path}/plano_rff_eta/{save_labels[0]}/rffxeta_color_{save_labels[1]}_{region_names[i]}_cd.png')
				plt.close()
		return
	os.makedirs(f'{save_path}/plano_rff_eta',exist_ok=True)

	lim_region=config['lim_region']
	names_simples=config['names_simples']
	alpha_vec=config['alpha_vec']
	cores=config['cores']
	line_width=config['line_width']
	region_names=config['save_names']
	rff_split=False
	x0=np.linspace(0,0.1,10000)
	if sample == 'WHL':
		bins = np.arange(0,0.1,0.0005)
		dlpov=[1.54622724e-02,4.61969720e-03,1.23642039e+02,-3.93661732e+00,-4.81476825e-01,7.57926648e+01]
		converter=(7057*0.0005)
	elif sample == 'L07':
		bins = np.arange(0,0.1,0.0017)
		dlpov=[0.017835,0.005812,42.214771,-3.522754,0.328877,14.599428]
		converter=1
	ynorm=[]
	idx_split_rff = np.argwhere(np.diff(np.sign(diff(x0,dlpov)))).flatten()

	fig0=plt.figure()
	y,x,_ = plt.hist(rff_s,bins=bins,density=True,histtype='step',color='white',edgecolor='black')
	x = (x[1:]+x[:-1])/2
	ll=y!=0.0
	y=y[ll]
	x=x[ll]
	sigma_y = np.sqrt(y/converter)
	ynorm=np.power(y-(dlognorm(x,*dlpov)/converter),2)/(sigma_y**2)
	chi2_norm=np.sum(ynorm)/(len(y)-7)
	dlog_norm_conv=dlognorm(x0,*dlpov)/converter
	gauss_conv=gauss(x0,*dlpov[:3])/converter
	log_norm_conv=lognormal(x0,*dlpov[3:6])/converter
	plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
	plt.plot(x0,gauss_conv,linewidth=1,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]/converter),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
	plt.plot(x0,log_norm_conv,linewidth=1,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]/converter),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
	plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	plt.savefig(f'{save_path}/plano_rff_eta/histogram_rff_lines.png')
	plt.close(fig0)

	#HISTOGRAMA DO RFF PARA ELIPTICAS
	for i in range(len(lim_region)):
		print(names_simples[i],np.sum(lim_region[i]))
		fig0=plt.figure(figsize=(9,7))
		plt.hist(rff_s[lim_region[i]],bins=bins,color=cores[i],edgecolor='black',alpha=0.4,label=f'{names_simples[i]}')
		plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
		plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.4')))
		plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
		plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
		plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.legend()
		plt.xlabel('RFF')
		plt.ylabel('Objetos')
		plt.savefig(f'{save_path}/plano_rff_eta/histogram_rff_lines_{region_names[i]}.png')
		plt.close(fig0)

	if sample == 'L07':

		#HISTOGRAMA DO RFF PARA cDs do Zhao

		fig0=plt.figure(figsize=(9,7))
		plt.hist(rff_s[cd_cut],bins=bins,color='red',edgecolor='black',alpha=0.4,label='cD[Zhao]')
		plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
		plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
		plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
		if rff_split:
			plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.legend()
		plt.xlabel('RFF')
		plt.ylabel('Objetos')
		plt.savefig(f'{save_path}/plano_rff_eta/histogram_rff_lines_cDs_zhao.png')
		plt.close(fig0)

		#HISTOGRAMA DO RFF PARA ELIPTICAS do Zhao

		fig0=plt.figure(figsize=(9,7))
		plt.hist(rff_s[e_cut],bins=bins,color='green',edgecolor='black',alpha=0.4,label='E[Zhao]')
		plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
		plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
		plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
		if rff_split:
			plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.legend()
		plt.xlabel('RFF')
		plt.ylabel('Objetos')
		plt.savefig(f'{save_path}/plano_rff_eta/histogram_rff_lines_e_zhao.png')
		plt.close(fig0)

		#HISTOGRAMA DO RFF PARA ELIPTICAS -- subgrupo cD do zhao
		for i in range(len(lim_region)):
			fig0=plt.figure(figsize=(9,7))
			plt.hist(rff_s[lim_region[i] & cd_cut],bins=bins,color=cores[i],edgecolor='black',alpha=0.4,label=f'{names_simples[i]}[cD]')
			plt.plot(x0,dlog_norm_conv,linewidth=2, color='b',label='G+log')
			plt.plot(x0,gauss_conv,linewidth=1.5,ls=':',color='black',label='G'+'\n'+'A '+ str(format(abs(dlpov[:3][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[:3][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[:3][1]),'.3f')))
			plt.plot(x0,log_norm_conv,linewidth=1.5,ls='--',color='g',label='log'+'\n'+'A '+ str(format(abs(dlpov[3:6][2]),'.3f'))+'\n'+r'$\mu$ '+str(format(abs(dlpov[3:6][0]),'.3f'))+ '\n'+ r'$\sigma$ '+str(format(abs(dlpov[3:6][1]),'.3f')))
			if rff_split:
				plt.axvline(x0[idx_split_rff][-1],label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			plt.legend()
			plt.xlabel('RFF')
			plt.ylabel('Objetos')
			plt.savefig(f'{save_path}/plano_rff_eta/histogram_rff_lines_{names_simples[i]}_cd.png')
			plt.close(fig0)
	#########################################
	#PLANO DE RFF - ETA

	x=np.linspace(0.001,0.3,1000)
	y=np.linspace(0.001,0.3,1000)
	limx=[-2.5,-0.5]
	limy=[-0.02,0.1]
	label_x=r'$\log\,RFF$'
	label_y=r'$\eta$'

	#RFF - ETA / PONTOS COLORIDOS
	fig1=plt.figure(figsize=(9,7))
	vec=[0.1,0.5]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	if rff_split:
		plt.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
	for i in range(len(lim_region)):
		plt.scatter(np.log10(rff_s)[lim_region[i]],eta[lim_region[i]],c=cores[i],edgecolors='black',label=names_simples[i])
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_color_coded.png')
	plt.close()

	#UNITÁRIOS
	for i in range(len(lim_region)):
		fig,axs=plt.subplots(1,1,figsize=(9,7))
		axs.scatter(np.log10(rff_s)[lim_region[i]],eta[lim_region[i]],c=cores[i],edgecolors='black',alpha=0.2,label=names_simples[i])
		for item in vec:
			plt.plot(np.log10(x),x-item*x,label=str(item))
		sns.kdeplot(x=np.log10(rff_s)[lim_region[i]],y=eta[lim_region[i]],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(limx)
		axs.set_ylim(limy)
		axs.set_xlabel(label_x)
		axs.set_ylabel(label_y)
		axs.legend()
		if rff_split:
			axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_{region_names[i]}.png')
		plt.close()


	if sample == 'L07':
		#ELIPTICAS ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7))
		axs.scatter(np.log10(rff_s)[e_cut],eta[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
		for item in vec:
			plt.plot(np.log10(x),x-item*x,label=str(item))
		sns.kdeplot(x=np.log10(rff_s)[e_cut],y=eta[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(limx)
		axs.set_ylim(limy)
		axs.set_xlabel(label_x)
		axs.set_ylabel(label_y)
		axs.legend()
		if rff_split:
			axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_elipticas_zhao.png')
		plt.close()

		#cDs ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7))
		axs.scatter(np.log10(rff_s)[cd_cut],eta[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
		for item in vec:
			plt.plot(np.log10(x),x-item*x,label=str(item))
		sns.kdeplot(x=np.log10(rff_s)[cd_cut],y=eta[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(limx)
		axs.set_ylim(limy)
		axs.set_xlabel(label_x)
		axs.set_ylabel(label_y)
		axs.legend()
		if rff_split:
			axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
		plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_cds_zhao.png')
		plt.close()

		for i in range(len(lim_region)):
			fig,axs=plt.subplots(1,1,figsize=(9,7))
			axs.scatter(np.log10(rff_s)[lim_region[i] & e_cut],eta[lim_region[i] & e_cut],c=cores[i],edgecolors='black',alpha=0.2,label=f'{names_simples[i]}[E]')
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			sns.kdeplot(x=np.log10(rff_s)[lim_region[i] & e_cut],y=eta[lim_region[i] & e_cut],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
			axs.set_xlim(limx)
			axs.set_ylim(limy)
			axs.set_xlabel(label_x)
			axs.set_ylabel(label_y)
			axs.legend()
			if rff_split:
				axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_{region_names[i]}_e.png')
			plt.close()

			#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
			fig,axs=plt.subplots(1,1,figsize=(9,7))
			axs.scatter(np.log10(rff_s)[lim_region[i] & cd_cut],eta[lim_region[i] & cd_cut],c=cores[i],edgecolors='red',alpha=0.2,label=f'{names_simples[i]}[cD]')
			for item in vec:
				plt.plot(np.log10(x),x-item*x,label=str(item))
			if rff_split:
				axs.axvline(np.log10(x0[idx_split_rff][-1]),label='RFF='+str(format(x0[idx_split_rff][-1],'.3f')),color='black')
			sns.kdeplot(x=np.log10(rff_s)[lim_region[i] & cd_cut],y=eta[lim_region[i] & cd_cut],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
			axs.set_xlim(limx)
			axs.set_ylim(limy)
			axs.set_xlabel(label_x)
			axs.set_ylabel(label_y)
			axs.legend()
			plt.savefig(f'{save_path}/plano_rff_eta/rffxeta_{region_names[i]}_cd.png')
			plt.close()

	###############################################
	#MAPA DE COR - RAZÃO BT - SEM CORREÇÃO
	if sample == 'L07':
		l07_regions=cd_cut,e_cut,(ecd_cut | cde_cut)
	else:
		l07_regions=None
	##################################
	#MAPA DE COR - RAZÃO BT - CORRIGIDO
	os.makedirs(f'{save_path}/plano_rff_eta/bt',exist_ok=True)
	
	par_labels=('BT',r'$B/T$')
	save_labels=('bt','bt')
	cor_lim=(0,1)
	plots_entry=bt_vec_corr,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	###################################
	#MAPA DE RAZÃO DE RFF
	os.makedirs(f'{save_path}/plano_rff_eta/rff_ratio',exist_ok=True)

	par_labels=('Razão de RFF',r'$RFF_{SS}/RFF_S$')
	save_labels=('rff_ratio','rff_ratio')
	cor_lim=(0,1)
	plots_entry=rff_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	#################################
	#MAPA DE COR RAZÃO DE CHI2
	os.makedirs(f'{save_path}/plano_rff_eta/chi2_ratio',exist_ok=True)

	par_labels=(r'Razão de $\chi^2$',r'$\chi_{S}^2/\chi_{S+S}^2$')
	save_labels=('chi2_ratio','chi2_ratio')
	cor_lim=(0.9,1.1)
	plots_entry=chi2_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	#################################
	#MAPA DE COR DELTA BIC
	os.makedirs(f'{save_path}/plano_rff_eta/delta_bic',exist_ok=True)

	par_labels=('Delta BIC',r'$\Delta \ BIC$')
	save_labels=('delta_bic','delta_bic')
	cor_lim=(1000,-1000)
	plots_entry=delta_bic_obs,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR RAZÃO AXIAL
	os.makedirs(f'{save_path}/plano_rff_eta/ax_ratio',exist_ok=True)

	par_labels=('Razão Axial Sérsic',r'$q$')
	save_labels=('ax_ratio','ax_ratio')
	cor_lim=(0.1,1.)
	plots_entry=e_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR RAZÃO - Q1/Q2
	os.makedirs(f'{save_path}/plano_rff_eta/q1q2',exist_ok=True)

	par_labels=('Razão de q1/q2',r'$q_1/q_2$')
	save_labels=('q1q2','q_ratio')
	cor_lim=(0.7,2.)
	plots_entry=axrat_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR BOXINESS
	os.makedirs(f'{save_path}/plano_rff_eta/box',exist_ok=True)

	par_labels=('Boxiness',r'$a_4/a$')
	save_labels=('box','box')
	cor_lim=(-0.8,0.8)
	plots_entry=box_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR BOXINESS 1
	os.makedirs(f'{save_path}/plano_rff_eta/box1',exist_ok=True)

	par_labels=('Boxiness Comp. 1',r'$a_4/a \ 1$')
	save_labels=('box1','box1')
	cor_lim=(-0.8,0.8)
	plots_entry=box1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	##############################
	#MAPA DE COR BOXINESS 2
	os.makedirs(f'{save_path}/plano_rff_eta/box2',exist_ok=True)

	par_labels=('Boxiness Comp. 2',r'$a_4/a \ 2$')
	save_labels=('box2','box2')
	cor_lim=(-0.8,0.8)
	plots_entry=box2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR RAZÃO - n1/n2
	os.makedirs(f'{save_path}/plano_rff_eta/n1n2',exist_ok=True)

	par_labels=('Razão de indices de Sérsic',r'$n_1/n_2$')
	save_labels=('n1n2','n_ratio')
	cor_lim=(0,2)
	plots_entry=n_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR - INDICE DE SÉRSIC
	os.makedirs(f'{save_path}/plano_rff_eta/indice_sersic',exist_ok=True)

	par_labels=('Indice de Sérsic',r'$n$')
	save_labels=('indice_sersic','n')
	cor_lim=(2,8)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
	os.makedirs(f'{save_path}/plano_rff_eta/n1',exist_ok=True)

	par_labels=('Indice de Sérsic Comp. 1',r'$n_1$')
	save_labels=('n1','n1')
	cor_lim=(2,6)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 1 - SS
	os.makedirs(f'{save_path}/plano_rff_eta/ax_ratio_1',exist_ok=True)

	par_labels=('Razão Axial Comp. 1',r'$q_1$')
	save_labels=('ax_ratio_1','q1')
	cor_lim=(0.2,1)
	plots_entry=e1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 2 - SS
	os.makedirs(f'{save_path}/plano_rff_eta/ax_ratio_2',exist_ok=True)

	par_labels=('Razão Axial Comp. 2',r'$q_2$')
	save_labels=('ax_ratio_2','q2')
	cor_lim=(0.2,1)
	plots_entry=e2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	return
def re_ratio_n2_plots(config):
	"""Scatter and colour‑coded plots in the plane of
	log₁₀(re₁/re₂) vs. n₂ (Sérsic index of the outer component).

	For each morphological class, plain scatter, KDE contours, and
	colour maps (using auxiliary parameters) are generated.

	Args:
		config (dict): Region and style configuration.

	Returns:
		None.
	"""

	def color_plots_plane(plots_entry):
		"""
		param
		par_labels=titulo,color_label
		l07_regions
		cor_lim
		save_labels=pasta,figname
		"""
		param,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names=plots_entry

		#SAMPLE
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {sample}')
		plt.scatter(np.log10(re_ratio_12),n2,c=param,edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(re_rat_lim)
		plt.ylim(n2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}.png')
		plt.close()
		
		#HIGH
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[0]}')
		plt.scatter(np.log10(re_ratio_12[lim_region[0]]),n2[lim_region[0]],c=param[lim_region[0]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(re_rat_lim)
		plt.ylim(n2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[0]}.png')
		plt.close()
		
		#LOW_LEFT
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[1]}')
		plt.scatter(np.log10(re_ratio_12[lim_region[1]]),n2[lim_region[1]],c=param[lim_region[1]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(re_rat_lim)
		plt.ylim(n2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}.png')
		plt.close()

		#LOW_RIGHT
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[2]}')
		plt.scatter(np.log10(re_ratio_12[lim_region[2]]),n2[lim_region[2]],c=param[lim_region[2]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(re_rat_lim)
		plt.ylim(n2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[2]}.png')
		plt.close()
		if sample=='L07':
			cd_cut,e_cut,misc_cut=l07_regions
			
			#ELIPTICAS ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[Zhao]')
			plt.scatter(np.log10(re_ratio_12[e_cut]),n2[e_cut],c=param[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_e_zhao.png')
			plt.close()

			#cDs ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - cD[Zhao]')
			plt.scatter(np.log10(re_ratio_12[cd_cut]),n2[cd_cut],c=param[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_cd_zhao.png')
			plt.close()

			#HIGH[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[0]}[cD]')
			plt.scatter(np.log10(re_ratio_12[lim_region[0] & cd_cut]),n2[lim_region[0] & cd_cut],c=param[lim_region[0] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[0]}_cd.png')
			plt.close()

			#LOW_LEFT[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[1]}[cD]')
			plt.scatter(np.log10(re_ratio_12[lim_region[1] & cd_cut]),n2[lim_region[1] & cd_cut],c=param[lim_region[1] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}_cd.png')
			plt.close()

			#LOW_LEFT[E & E/cD]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[1]}[E & E/cD]')
			plt.scatter(np.log10(re_ratio_12[lim_region[1] & (e_cut | misc_cut)]),n2[lim_region[1] & (e_cut | misc_cut)],c=param[lim_region[1] & (e_cut | misc_cut)],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}_E_Ecd.png')
			plt.close()

			#LOW_RIGHT[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[2]}[cD]')
			plt.scatter(np.log10(re_ratio_12[lim_region[2] & cd_cut]),n2[lim_region[2] & cd_cut],c=param[lim_region[2] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(re_rat_lim)
			plt.ylim(n2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_re_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[2]}_cd.png')
			plt.close()
		return

	os.makedirs(f'{save_path}/plano_re_n2',exist_ok=True)

	label_x=r'$R_{1}/R_{2}$'
	label_y=r'$n_{2}$'
	re_rat_lim=[-2,1.5]
	n2_ylim=[0,15]

	names_simples=config['names_simples']
	lim_region=config['lim_region']
	region_names=config['save_names']
	cores=config['cores']

	fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
	axs[0].scatter(np.log10(re_ratio_12)[lim_region[0]],n2[lim_region[0]],c=cores[0],edgecolor='black',label=names_simples[0])
	axs[0].set_xlabel(label_x)
	axs[0].set_ylabel(label_y)
	axs[0].set_xlim(re_rat_lim)
	axs[0].set_ylim(n2_ylim)
	axs[0].legend()
	axs[1].scatter(np.log10(re_ratio_12)[lim_region[1]],n2[lim_region[1]],c=cores[1],edgecolor='black',label=names_simples[1])
	axs[1].set_xlabel(label_x)
	axs[1].legend()
	axs[2].scatter(np.log10(re_ratio_12)[lim_region[2]],n2[lim_region[2]],c=cores[2],edgecolor='black',label=names_simples[2])
	axs[2].set_xlabel(label_x)
	axs[2].legend()
	plt.savefig(f'{save_path}/plano_re_n2/scatter_re_ratio_n2_geral_subplots.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
	axs.scatter(np.log10(re_ratio_12)[lim_region[0]],n2[lim_region[0]],c=cores[0],edgecolor='black',label=names_simples[0])
	axs.scatter(np.log10(re_ratio_12)[lim_region[1]],n2[lim_region[1]],c=cores[1],edgecolor='black',label=names_simples[1])
	axs.scatter(np.log10(re_ratio_12)[lim_region[2]],n2[lim_region[2]],c=cores[2],edgecolor='black',label=names_simples[2])
	axs.set_xlim(re_rat_lim)
	axs.set_ylim(n2_ylim)
	axs.set_xlabel(label_x)
	axs.set_ylabel(label_y)
	axs.legend()
	plt.savefig(f'{save_path}/plano_re_n2/scatter_re_ratio_n2_geral.png')
	plt.close()

	#UNITÁRIOS
	for i in range(3):
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(np.log10(re_ratio_12)[lim_region[i]],n2[lim_region[i]],c=cores[i],edgecolors='black',alpha=0.2,label=names_simples[i])
		sns.kdeplot(x=np.log10(re_ratio_12)[lim_region[i]],y=n2[lim_region[i]],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(re_rat_lim)
		axs.set_ylim(n2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_re_n2/rexn2_{region_names[i]}.png')
		plt.close()
	if sample == 'L07':
		#ELIPTICAS ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(np.log10(re_ratio_12)[e_cut],n2[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
		sns.kdeplot(x=np.log10(re_ratio_12)[e_cut],y=n2[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(re_rat_lim)
		axs.set_ylim(n2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_re_n2/rexn2_elipticas_zhao.png')
		plt.close()

		#cDs ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(np.log10(re_ratio_12)[cd_cut],n2[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
		sns.kdeplot(x=np.log10(re_ratio_12)[cd_cut],y=n2[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(re_rat_lim)
		axs.set_ylim(n2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_re_n2/rexn2_cds_zhao.png')
		plt.close()

		#HIGH[cDs]
		#LOW_LEFT[cDs]
		#LOW_RIGHT[cDs]
		for i in range(3):
			fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
			axs.scatter(np.log10(re_ratio_12)[lim_region[i] & cd_cut],n2[lim_region[i] & cd_cut],c=cores[i],edgecolors='black',alpha=0.2,label=f'{names_simples[i]}[Zhao]')
			sns.kdeplot(x=np.log10(re_ratio_12)[lim_region[i] & cd_cut],y=n2[lim_region[i] & cd_cut],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
			axs.set_xlim(re_rat_lim)
			axs.set_ylim(n2_ylim)
			axs.set_ylabel(label_y)
			axs.set_xlabel(label_x)
			axs.legend()
			plt.savefig(f'{save_path}/plano_re_n2/rexn2_{region_names[i]}_cds_zhao.png')
			plt.close()

		#LOW_LEFT[E & E/cD]

		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(np.log10(re_ratio_12)[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],n2[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],c=cores[1],edgecolors='black',alpha=0.2,label=f'{names_simples[1]}[Zhao]')
		sns.kdeplot(x=np.log10(re_ratio_12)[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],y=n2[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],fill=False,levels=30,color=cores[1],linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(re_rat_lim)
		axs.set_ylim(n2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_re_n2/rexn2_{region_names[1]}_cds_zhao.png')
		plt.close()

	##################################
	if sample == 'L07':
		l07_regions=cd_cut,e_cut,(ecd_cut | cde_cut)
	else:
		l07_regions=None
	##################################
	#MAPA DE COR - RAZÃO BT - CORRIGIDO
	os.makedirs(f'{save_path}/plano_re_n2/bt',exist_ok=True)
	
	par_labels=('BT',r'$B/T$')
	save_labels=('bt','bt')
	cor_lim=(0,1)
	plots_entry=bt_vec_corr,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	#######################################################
	###################################
	#MAPA DE RAZÃO DE RFF
	os.makedirs(f'{save_path}/plano_re_n2/rff_ratio',exist_ok=True)

	par_labels=('Razão de RFF',r'$RFF_{SS}/RFF_S$')
	save_labels=('rff_ratio','rff_ratio')
	cor_lim=(0,1)
	plots_entry=rff_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	#################################
	#MAPA DE COR RAZÃO DE CHI2
	os.makedirs(f'{save_path}/plano_re_n2/chi2_ratio',exist_ok=True)

	par_labels=(r'Razão de $\chi^2$',r'$\chi_{S}^2/\chi_{S+S}^2$')
	save_labels=('chi2_ratio','chi2_ratio')
	cor_lim=(0.9,1.1)
	plots_entry=chi2_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	#################################
	#MAPA DE COR DELTA BIC
	os.makedirs(f'{save_path}/plano_re_n2/delta_bic',exist_ok=True)

	par_labels=('Delta BIC',r'$\Delta \ BIC$')
	save_labels=('delta_bic','delta_bic')
	cor_lim=(1000,-1000)
	plots_entry=delta_bic_obs,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR RAZÃO AXIAL
	os.makedirs(f'{save_path}/plano_re_n2/ax_ratio',exist_ok=True)

	par_labels=('Razão Axial Sérsic',r'$q$')
	save_labels=('ax_ratio','ax_ratio')
	cor_lim=(0.1,1.)
	plots_entry=e_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR RAZÃO - Q1/Q2
	os.makedirs(f'{save_path}/plano_re_n2/q1q2',exist_ok=True)

	par_labels=('Razão de q1/q2',r'$q_1/q_2$')
	save_labels=('q1q2','q_ratio')
	cor_lim=(0.7,2.)
	plots_entry=axrat_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR BOXINESS
	os.makedirs(f'{save_path}/plano_re_n2/box',exist_ok=True)

	par_labels=('Boxiness',r'$a_4/a$')
	save_labels=('box','box')
	cor_lim=(-0.8,0.8)
	plots_entry=box_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##############################
	#MAPA DE COR BOXINESS 1
	os.makedirs(f'{save_path}/plano_re_n2/box1',exist_ok=True)

	par_labels=('Boxiness Comp. 1',r'$a_4/a \ 1$')
	save_labels=('box1','box1')
	cor_lim=(-0.8,0.8)
	plots_entry=box1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)
	##############################
	#MAPA DE COR BOXINESS 2
	os.makedirs(f'{save_path}/plano_re_n2/box2',exist_ok=True)

	par_labels=('Boxiness Comp. 2',r'$a_4/a \ 2$')
	save_labels=('box2','box2')
	cor_lim=(-0.8,0.8)
	plots_entry=box2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR RAZÃO - n1/n2
	os.makedirs(f'{save_path}/plano_re_n2/n1n2',exist_ok=True)

	par_labels=('Razão de indices de Sérsic',r'$n_1/n_2$')
	save_labels=('n1n2','n_ratio')
	cor_lim=(0,2)
	plots_entry=n_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR - INDICE DE SÉRSIC
	os.makedirs(f'{save_path}/plano_re_n2/indice_sersic',exist_ok=True)

	par_labels=('Indice de Sérsic',r'$n$')
	save_labels=('indice_sersic','n')
	cor_lim=(2,8)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
	os.makedirs(f'{save_path}/plano_re_n2/n1',exist_ok=True)

	par_labels=('Indice de Sérsic Comp. 1',r'$n_1$')
	save_labels=('n1','n1')
	cor_lim=(2,6)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 1 - SS
	os.makedirs(f'{save_path}/plano_re_n2/ax_ratio_1',exist_ok=True)

	par_labels=('Razão Axial Comp. 1',r'$q_1$')
	save_labels=('ax_ratio_1','q1')
	cor_lim=(0.2,1)
	plots_entry=e1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 2 - SS
	os.makedirs(f'{save_path}/plano_re_n2/ax_ratio_2',exist_ok=True)

	par_labels=('Razão Axial Comp. 2',r'$q_2$')
	save_labels=('ax_ratio_2','q2')
	cor_lim=(0.2,1)
	plots_entry=e2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	return
def plano_n1_ns():
	"""Scatter and decision boundary analysis for n₁ vs. nₛ.

	The function:
		1. Trains a linear SVM to separate E(EL) from cD galaxies.
		2. Plots the data with the decision boundary and accuracy.
		3. Shows KDE contours for the two classes.
		4. Generates colour‑coded versions for several auxiliary parameters
		(B/T, RFF ratio, boxiness, etc.).

	All data used are global variables defined in the main block.

	Returns:
		None.
	"""

	def color_plots_plane(plots_entry):
		"""
		param
		lim_region
		par_labels=titulo,color_label
		l07_regions
		cor_lim
		save_labels=pasta,figname
		"""
		param,lim_region,l07_regions,cor_lim,par_labels,save_labels=plots_entry
		cd_lim,lim_cd_small,lim_cd_big,elip_lim=lim_region
		#SAMPLE
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {sample} - {titulo}')
		plt.scatter(n1,n_s,c=param,edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n1_xlim)
		plt.ylim(ns_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}.png')
		plt.close()
		#2C
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - 2C - {titulo}')
		plt.scatter(n1[cd_lim],n_s[cd_lim],c=param[cd_lim],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n1_xlim)
		plt.ylim(ns_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_2c.png')
		plt.close()
		#cD
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - cD - {titulo}')
		plt.scatter(n1[lim_cd_big],n_s[lim_cd_big],c=param[lim_cd_big],edgecolors='black',label='cD',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.legend()
		plt.xlim(n1_xlim)
		plt.ylim(ns_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_cd.png')
		plt.close()
		#E(EL)
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - E(EL) - {titulo}')
		plt.scatter(n1[lim_cd_small],n_s[lim_cd_small],c=param[lim_cd_small],edgecolors='black',label='E(EL)',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.legend()
		plt.xlim(n1_xlim)
		plt.ylim(ns_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_eel.png')
		plt.close()
		#E
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - E - {titulo}')
		plt.scatter(n1[elip_lim],n_s[elip_lim],c=param[elip_lim],edgecolors='black',label='E',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.legend()
		plt.xlim(n1_xlim)
		plt.ylim(ns_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_elip.png')
		plt.close()
		if sample=='L07' and mode == 'delta_bic':
			cd_cut,e_cut,misc_cut=l07_regions
			#ELIPTICAS ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[Zhao] - {titulo}')
			plt.scatter(n1[e_cut],n_s[e_cut],c=param[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_e_zhao.png')
			plt.close()

			#cDs ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - cD[Zhao] - {titulo}')
			plt.scatter(n1[cd_cut],n_s[cd_cut],c=param[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_cd_zhao.png')
			plt.close()

			#NOSSAS ELIPTICAS -- SUBGRUPO ELIPTICAS ZHAO

			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[E] - {titulo}')
			plt.scatter((n1[elip_lim & e_cut]),n2[elip_lim & e_cut],c=param[elip_lim & e_cut],edgecolors='black',label='E[E]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_elip_E.png')
			plt.close()

			#NOSSAS ELIPTICAS -- SUBGRUPO cDs ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[cD] - {titulo}')
			plt.scatter(n1[elip_lim & cd_cut],n_s[elip_lim & cd_cut],c=param[elip_lim & cd_cut],edgecolors='black',label='E[cD]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_elip_cD.png')
			plt.close()

			#NOSSAS ELIPTICAS -- SUBGRUPO cD/E & E/cD ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[E/cD] - {titulo}')
			plt.scatter(n1[elip_lim & misc_cut],n_s[elip_lim & misc_cut],c=param[elip_lim & misc_cut],edgecolors='black',label='E[E/cD]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_elip_misc.png')
			plt.close()
			#2C
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - 2C[cD] - {titulo}')
			plt.scatter(n1[cd_lim & cd_cut],n_s[cd_lim & cd_cut],c=param[cd_lim & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)		
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_cd_cD.png')
			plt.close()

			#cDs -- SUBGRUPO cD ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - cD[cD] - {titulo}')
			plt.scatter(n1[lim_cd_big & cd_cut],n_s[lim_cd_big & cd_cut],c=param[lim_cd_big & cd_cut],edgecolors='black',label='cD[cD]',cmap=cmap)		
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_cd_cD.png')
			plt.close()

			#EXTRA LIGHT -- SUBGRUPO cD ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E(EL)[cD] - {titulo}')
			plt.scatter(n1[lim_cd_small & cd_cut],n_s[lim_cd_small & cd_cut],c=param[lim_cd_small & cd_cut],edgecolors='black',label='E(EL)[cD]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n1_xlim)
			plt.ylim(ns_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n1_ns/{save_labels[0]}/n1xns_color_{save_labels[1]}_eel_cD.png')
			plt.close()
		return

	os.makedirs(f'{save_path}/plano_n1_ns',exist_ok=True)

	n1_xlim=[-0.5,10.5]
	ns_ylim=[-0.5,12.5]

	label_x=r'$n_1$'
	label_y=r'$n_s$'


	par_temp = np.column_stack([n1[cd_lim], n_s[cd_lim]])
	coefs, inter, acc = svc_calc(par_temp, lim_cd_small[cd_lim].astype(int))

	a, c = coefs
	b = inter
	# Plot
	plt.figure()
	plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.5, c='blue', edgecolor='black', label='E(EL)')
	plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.6, c='red', edgecolor='black', label='cD')
	xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
	yplot = -(a*xplot + b) / c
	lim_y=yplot<np.max(n_s[cd_lim])
	m = -(a/c)
	b_plot = -(b/c)
	label_line = rf'$F_s={acc:.3f}$'+'\n'+rf'$\alpha={m:.3f}$'+'\n'+rf'$\beta={b_plot:.3f}$'
	plt.plot(xplot[lim_y], yplot[lim_y], color='black', lw=3, label=label_line)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.tight_layout()
	plt.savefig(f'{save_path}/plano_n1_ns/n1_ns_split.png')
	plt.close()

	plt.figure(figsize=(8, 7))
	plt.scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.1, c='blue', edgecolor='black', label='E(EL)')
	plt.scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='cD')
	sns.kdeplot(x=n1[cd_lim],y=n_s[cd_lim],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,bw_adjust=0.7,thresh=0.05)#thresh=0)
	plt.plot([],[],color='black',label=r'$\rho$')

	# GRADE DE X MAIS SUAVE
	xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
	yplot = -(a*xplot + b) / c
	lim_y=yplot<np.max(n_s[cd_lim])
	m = -(a/c)
	b_plot = -(b/c)
	label_line = rf'$F_s={acc:.3f}$'+'\n'+rf'$\alpha={m:.3f}$'+'\n'+rf'$\beta={b_plot:.3f}$'
	plt.plot(xplot[lim_y], yplot[lim_y], color='black', lw=3, label=label_line)
	plt.legend()
	plt.xlim(n1_xlim)
	plt.ylim(ns_ylim)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.tight_layout()
	plt.savefig(f'{save_path}/plano_n1_ns/n1_ns_contorno.png')
	plt.close()

	xplot = np.linspace(np.min(n1[cd_lim]), np.max(n1[cd_lim]), 300)
	yplot = -(a*xplot + b) / c
	lim_y=yplot<np.max(n_s[cd_lim])
	fig,axs=plt.subplots(1,2)
	axs[0].scatter(n1[lim_cd_small], n_s[lim_cd_small],alpha=0.1, c='blue', edgecolor='black', label='E(EL)')
	sns.kdeplot(x=n1[lim_cd_small],y=n_s[lim_cd_small],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,ax=axs[0])#thresh=0)
	axs[0].plot([],[],color='black',label=r'$\rho$')
	axs[0].plot(xplot[lim_y], yplot[lim_y], color='black', lw=3)
	axs[0].legend()
	axs[0].set_xlim(n1_xlim)
	axs[0].set_ylim(ns_ylim)
	axs[0].set_xlabel(label_x)
	axs[0].set_ylabel(label_y)

	axs[1].scatter(n1[lim_cd_big], n_s[lim_cd_big],alpha=0.1, c='red', edgecolor='black', label='cD')
	sns.kdeplot(x=n1[lim_cd_big],y=n_s[lim_cd_big],fill=False,levels=50,color="black",linewidths=1,alpha=0.7,ax=axs[1])#thresh=0)
	axs[1].plot([],[],color='black',label=r'$\rho$')
	axs[1].plot(xplot[lim_y], yplot[lim_y], color='black', lw=3)
	axs[1].legend()
	axs[1].set_xlim(n1_xlim)
	axs[1].set_ylim(ns_ylim)
	axs[1].set_xlabel(r'$n_1$')
	plt.tight_layout()
	plt.savefig(f'{save_path}/plano_n1_ns/n1_ns_contorno.png')
	plt.close()
	##################################################
	if sample == 'L07':
		l07_regions=cd_cut,e_cut,(ecd_cut | cde_cut)
	else:
		l07_regions=None
	lim_region=cd_lim,lim_cd_small,lim_cd_big,elip_lim
	#########################################################
	#MAPA DE COR - RAZÃO BT - COM CORREÇÃO
	os.makedirs(f'{save_path}/plano_n1_ns/bt',exist_ok=True)

	par_labels=('BT',r'$B/T$')
	save_labels=('bt','bt')
	cor_lim=(0,1)
	plots_entry=bt_vec_corr,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	###################################################################################
	#MAPA DE COR - RFF RATIO
	os.makedirs(f'{save_path}/plano_n1_ns/rff_ratio',exist_ok=True)

	par_labels=('Razão de RFF',r'$RFF_{S+S}/RFF_S$')
	save_labels=('rff_ratio','rff_ratio')
	cor_lim=(-0.5,3)
	plots_entry=rff_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	############
	#MAPA DE COR - N2
	os.makedirs(f'{save_path}/plano_n1_ns/n2',exist_ok=True)

	par_labels=('Índice de Sérsic Comp. 2',r'$n_2$')
	save_labels=('n2','n2')
	cor_lim=(0.5,8)
	plots_entry=n2,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)

	#############################
	#MAPA DE COR - CHI2 RATIO
	os.makedirs(f'{save_path}/plano_n1_ns/chi2_ratio',exist_ok=True)

	par_labels=('Razão de chi2',r'$\chi^{2}_{S}/\chi^{2}_{S+S}$')
	save_labels=('chi2_ratio','chi2_ratio')
	cor_lim=(0.9,1.1)
	plots_entry=chi2_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	#############################
	#MAPA DE COR - BOX 1
	os.makedirs(f'{save_path}/plano_n1_ns/box1',exist_ok=True)
	
	par_labels=('Boxiness Comp. 1',r'$(a_4/a)_1$')
	save_labels=('box1','box1')
	cor_lim=(0.,0.6)
	plots_entry=box1,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	
	#############################
	#MAPA DE COR - BOX 2
	os.makedirs(f'{save_path}/plano_n1_ns/box2',exist_ok=True)

	par_labels=('Boxiness Comp. 2',r'$a_4/a$')
	save_labels=('box2','box2')
	cor_lim=(0,0.6)
	plots_entry=box2,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	#############################
	#MAPA DE COR - ETA
	os.makedirs(f'{save_path}/plano_n1_ns/eta',exist_ok=True)

	par_labels=('Eta (RFF - A1)',r'$\eta$')
	save_labels=('eta','eta')
	cor_lim=(0.,0.1)
	plots_entry=eta,lim_region,l07_regions,cor_lim,par_labels,save_labels
	color_plots_plane(plots_entry)
	
	return
def n_ratio_n2_plots(config):
	"""Plane plots for n₁/n₂ vs. log₁₀(n₂).

	The same logic as `re_ratio_n2_plots`, applied to the Sérsic index ratio.

	Args:
		config (dict): Configuration dictionary.

	Returns:
		None.
	"""

	def color_plots_plane(plots_entry):
		"""
		param
		par_labels=titulo,color_label
		l07_regions
		cor_lim
		save_labels=pasta,figname
		"""
		param,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names=plots_entry

		#SAMPLE
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {sample}')
		plt.scatter(n_ratio_12,np.log10(n2),c=param,edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n_rat_lim)
		plt.ylim(logn2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}.png')
		plt.close()
		
		#HIGH
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[0]}')
		plt.scatter(n_ratio_12[lim_region[0]],np.log10(n2[lim_region[0]]),c=param[lim_region[0]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n_rat_lim)
		plt.ylim(logn2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[0]}.png')
		plt.close()
		
		#LOW_LEFT
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[1]}')
		plt.scatter(n_ratio_12[lim_region[1]],np.log10(n2[lim_region[1]]),c=param[lim_region[1]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n_rat_lim)
		plt.ylim(logn2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}.png')
		plt.close()

		#LOW_RIGHT
		fig1=plt.figure(figsize=(9,7))
		plt.suptitle(f'{par_labels[0]} - {names_simples[2]}')
		plt.scatter(n_ratio_12[lim_region[2]],np.log10(n2[lim_region[2]]),c=param[lim_region[2]],edgecolors='black',cmap=cmap)
		cbar=plt.colorbar()
		cbar.set_label(par_labels[1], rotation=90)
		plt.clim(cor_lim)
		plt.xlim(n_rat_lim)
		plt.ylim(logn2_ylim)
		plt.xlabel(label_x)
		plt.ylabel(label_y)
		plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[2]}.png')
		plt.close()
		if sample=='L07':
			cd_cut,e_cut,misc_cut=l07_regions
			
			#ELIPTICAS ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - E[Zhao]')
			plt.scatter(n_ratio_12[e_cut],np.log10(n2[e_cut]),c=param[e_cut],edgecolors='black',label='E[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_e_zhao.png')
			plt.close()

			#cDs ZHAO
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - cD[Zhao]')
			plt.scatter(n_ratio_12[cd_cut],np.log10(n2[cd_cut]),c=param[cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_cd_zhao.png')
			plt.close()

			#HIGH[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[0]}[cD]')
			plt.scatter(n_ratio_12[lim_region[0] & cd_cut],np.log10(n2[lim_region[0] & cd_cut]),c=param[lim_region[0] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[0]}_cd.png')
			plt.close()

			#LOW_LEFT[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[1]}[cD]')
			plt.scatter(n_ratio_12[lim_region[1] & cd_cut],np.log10(n2[lim_region[1] & cd_cut]),c=param[lim_region[1] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}_cd.png')
			plt.close()

			#LOW_LEFT[E & E/cD]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[1]}[E & E/cD]')
			plt.scatter(n_ratio_12[lim_region[1] & (e_cut | misc_cut)],np.log10(n2[lim_region[1] & (e_cut | misc_cut)]),c=param[lim_region[1] & (e_cut | misc_cut)],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[1]}_E_Ecd.png')			
			plt.close()

			#LOW_RIGHT[cDs]
			fig1=plt.figure(figsize=(9,7))
			plt.suptitle(f'{par_labels[0]} - {names_simples[2]}[cD]')
			plt.scatter(n_ratio_12[lim_region[2] & cd_cut],np.log10(n2[lim_region[2] & cd_cut]),c=param[lim_region[2] & cd_cut],edgecolors='black',label='cD[Zhao]',cmap=cmap)
			cbar=plt.colorbar()
			cbar.set_label(par_labels[1], rotation=90)
			plt.clim(cor_lim)
			plt.legend()
			plt.xlim(n_rat_lim)
			plt.ylim(logn2_ylim)
			plt.xlabel(label_x)
			plt.ylabel(label_y)
			plt.savefig(f'{save_path}/plano_n_ratio_n2/{save_labels[0]}/rexn2_color_{save_labels[1]}_{region_names[2]}_cd.png')
			plt.close()
		return

	os.makedirs(f'{save_path}/plano_n_ratio_n2',exist_ok=True)

	label_x=r'$n_{1}/n_{2}$'
	label_y=r'$log_{10} n_{2}$'
	n_rat_lim=[-0.5,18]
	logn2_ylim=[-0.5,1.3]

	names_simples=config['names_simples']
	lim_region=config['lim_region']
	region_names=config['save_names']
	cores=config['cores']

	fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True,constrained_layout=True)
	for i in range(len(lim_region)):
		axs[i].scatter(n_ratio_12[lim_region[i]],np.log10(n2)[lim_region[i]],c=cores[i],edgecolor='black',label=names_simples[i])
		axs[i].legend()
	axs[0].set_xlim(n_rat_lim)
	axs[0].set_ylim(logn2_ylim)
	axs[0].set_xlabel(label_x)
	axs[0].set_ylabel(label_y)
	axs[1].set_xlabel(label_x)
	axs[1].legend()
	axs[2].set_xlabel(label_x)
	axs[2].legend()
	plt.savefig(f'{save_path}/plano_n_ratio_n2/scatter_n_ratio_n2_geral_subplots.png')
	plt.close()

	fig,axs=plt.subplots(1,1,figsize=(10,8),sharey=True,sharex=True,constrained_layout=True)
	for i in range(len(lim_region)):
		axs.scatter(n_ratio_12[lim_region[i]],np.log10(n2)[lim_region[i]],c=cores[i],edgecolor='black',label=names_simples[i])
	axs.set_xlim(n_rat_lim)
	axs.set_ylim(logn2_ylim)
	axs.set_xlabel(label_x)
	axs.set_ylabel(label_y)
	axs.legend()
	plt.savefig(f'{save_path}/plano_n_ratio_n2/scatter_n_ratio_n2_geral.png')
	plt.close()

	#UNITÁRIOS

	for i in range(len(lim_region)):
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(n_ratio_12[lim_region[i]],np.log10(n2)[lim_region[i]],c=cores[i],edgecolors='black',alpha=0.2,label=names_simples[i])
		sns.kdeplot(x=n_ratio_12[lim_region[i]],y=np.log10(n2)[lim_region[i]],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(n_rat_lim)
		axs.set_ylim(logn2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_n_ratio_n2/nxn2_{region_names[i]}.png')
		plt.close()

	if sample == 'L07':
		#ELIPTICAS ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(n_ratio_12[e_cut],np.log10(n2)[e_cut],c='green',edgecolors='black',alpha=0.2,label='E[Zhao]')
		sns.kdeplot(x=n_ratio_12[e_cut],y=np.log10(n2)[e_cut],fill=False,levels=30,color="green",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(n_rat_lim)
		axs.set_ylim(logn2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_n_ratio_n2/nxn2_elipticas_zhao.png')
		plt.close()

		#cDs ZHAO
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(n_ratio_12[cd_cut],np.log10(n2)[cd_cut],c='red',edgecolors='black',alpha=0.2,label='cD[Zhao]')
		sns.kdeplot(x=n_ratio_12[cd_cut],y=np.log10(n2)[cd_cut],fill=False,levels=30,color="red",linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(n_rat_lim)
		axs.set_ylim(logn2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_n_ratio_n2/nxn2_cds_zhao.png')
		plt.close()

		#HIGH[cDs]
		#LOW_LEFT[cDs]
		#LOW_RIGHT[cDs]
		for i in range(3):
			fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
			axs.scatter(n_ratio_12[lim_region[i] & cd_cut],np.log10(n2)[lim_region[i] & cd_cut],c=cores[i],edgecolors='black',alpha=0.2,label=f'{names_simples[i]}[cD]')
			sns.kdeplot(x=n_ratio_12[lim_region[i] & cd_cut],y=np.log10(n2)[lim_region[i] & cd_cut],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
			axs.set_xlim(n_rat_lim)
			axs.set_ylim(logn2_ylim)
			axs.set_ylabel(label_y)
			axs.set_xlabel(label_x)
			axs.legend()
			plt.savefig(f'{save_path}/plano_n_ratio_n2/nxn2_{region_names[i]}_cd.png')
			plt.close()

		#LOW_LEFT[E & E/cD]
		fig,axs=plt.subplots(1,1,figsize=(9,7),constrained_layout=True)
		axs.scatter(n_ratio_12[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],np.log10(n2)[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],c=cores[i],edgecolors='black',alpha=0.2,label=f'{names_simples[1]}[cD & E/cD]')
		sns.kdeplot(x=n_ratio_12[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],y=np.log10(n2)[lim_region[1] & (cd_cut | (ecd_cut | cde_cut))],fill=False,levels=30,color=cores[i],linewidths=1,alpha=0.8,thresh=0,ax=axs)
		axs.set_xlim(n_rat_lim)
		axs.set_ylim(logn2_ylim)
		axs.set_ylabel(label_y)
		axs.set_xlabel(label_x)
		axs.legend()
		plt.savefig(f'{save_path}/plano_n_ratio_n2/nxn2_{region_names[1]}_misc_cD.png')
		plt.close()
	##################################
		l07_regions=cd_cut,e_cut,(ecd_cut | cde_cut)
	else:
		l07_regions=None
	################################################
	#MAPA DE COR - RAZÃO BT
	os.makedirs(f'{save_path}/plano_n_ratio_n2/bt',exist_ok=True)
	
	par_labels=('BT',r'$B/T$')
	save_labels=('bt','bt')
	cor_lim=(0,1)
	plots_entry=bt_vec_corr,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###################################
	#MAPA DE RAZÃO DE RFF
	os.makedirs(f'{save_path}/plano_n_ratio_n2/rff_ratio',exist_ok=True)

	par_labels=('Razão de RFF',r'$RFF_{SS}/RFF_S$')
	save_labels=('rff_ratio','rff_ratio')
	cor_lim=(0,1)
	plots_entry=rff_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	#################################
	#MAPA DE COR RAZÃO DE CHI2
	os.makedirs(f'{save_path}/plano_n_ratio_n2/chi2_ratio',exist_ok=True)

	par_labels=(r'Razão de $\chi^2$',r'$\chi_{S}^2/\chi_{S+S}^2$')
	save_labels=('chi2_ratio','chi2_ratio')
	cor_lim=(0.9,1.1)
	plots_entry=chi2_ratio,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	#################################
	#MAPA DE COR DELTA BIC
	os.makedirs(f'{save_path}/plano_n_ratio_n2/delta_bic',exist_ok=True)

	par_labels=('Delta BIC',r'$\Delta \ BIC$')
	save_labels=('delta_bic','delta_bic')
	cor_lim=(1000,-1000)
	plots_entry=delta_bic_obs,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR RAZÃO AXIAL
	os.makedirs(f'{save_path}/plano_n_ratio_n2/ax_ratio',exist_ok=True)

	par_labels=('Razão Axial Sérsic',r'$q$')
	save_labels=('ax_ratio','ax_ratio')
	cor_lim=(0.1,1.)
	plots_entry=e_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR RAZÃO - Q1/Q2
	os.makedirs(f'{save_path}/plano_n_ratio_n2/q1q2',exist_ok=True)

	par_labels=('Razão de q1/q2',r'$q_1/q_2$')
	save_labels=('q1q2','q_ratio')
	cor_lim=(0.7,2.)
	plots_entry=axrat_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR BOXINESS
	os.makedirs(f'{save_path}/plano_n_ratio_n2/box',exist_ok=True)

	par_labels=('Boxiness',r'$a_4/a$')
	save_labels=('box','box')
	cor_lim=(-0.8,0.8)
	plots_entry=box_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR BOXINESS 1
	os.makedirs(f'{save_path}/plano_n_ratio_n2/box1',exist_ok=True)

	par_labels=('Boxiness Comp. 1',r'$a_4/a \ 1$')
	save_labels=('box1','box1')
	cor_lim=(-0.8,0.8)
	plots_entry=box1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	###############################
	#MAPA DE COR BOXINESS 2
	os.makedirs(f'{save_path}/plano_n_ratio_n2/box2',exist_ok=True)

	par_labels=('Boxiness Comp. 2',r'$a_4/a \ 2$')
	save_labels=('box2','box2')
	cor_lim=(-0.8,0.8)
	plots_entry=box2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR RAZÃO - re1/re2
	os.makedirs(f'{save_path}/plano_n_ratio_n2/re1re2',exist_ok=True)

	par_labels=('Razão de Raios Efetivos',r'$Re_1/Re_2$')
	save_labels=('re1re2','re_ratio')
	cor_lim=(0,2)
	plots_entry=re_ratio_12,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	################################
	#MAPA DE COR - INDICE DE SÉRSIC
	os.makedirs(f'{save_path}/plano_n_ratio_n2/indice_sersic',exist_ok=True)

	par_labels=('Indice de Sérsic',r'$n$')
	save_labels=('indice_sersic','n')
	cor_lim=(2,8)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - INDICE DE SÉRSIC 1 - SS
	os.makedirs(f'{save_path}/plano_n_ratio_n2/n1',exist_ok=True)

	par_labels=('Indice de Sérsic Comp. 1',r'$n_1$')
	save_labels=('n1','n1')
	cor_lim=(2,6)
	plots_entry=n_s,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 1 - SS
	os.makedirs(f'{save_path}/plano_n_ratio_n2/ax_ratio_1',exist_ok=True)

	par_labels=('Razão Axial Comp. 1',r'$q_1$')
	save_labels=('ax_ratio_1','q1')
	cor_lim=(0.2,1)
	plots_entry=e1,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	##########################
	#MAPA DE COR - RAZÃO AXIAL 2 - SS
	os.makedirs(f'{save_path}/plano_n_ratio_n2/ax_ratio_2',exist_ok=True)

	par_labels=('Razão Axial Comp. 2',r'$q_2$')
	save_labels=('ax_ratio_2','q2')
	cor_lim=(0.2,1)
	plots_entry=e2,lim_region,l07_regions,cor_lim,par_labels,save_labels,names_simples,region_names
	color_plots_plane(plots_entry)

	return

def analise_m200(cut_data):
	import pandas as pd
	"""Global analysis of scaling relations with M200 (cluster mass) and
	their evolution with redshift.

	Depending on the `flag`:
		- 'm200': the sample is binned by M200 and the mean parameter values
		for each morphological class are traced as a function of M200.
		- 'redshift': the sample is binned by redshift, and for each bin a
		linear regression (param vs M200) is performed.  The evolution of
		the slope and intercept with redshift is plotted.

	Parameters handled include: n, νₑ, ellipticities, Rₑ, boxiness,
	RFF, B/T, RFF ratio, photutils coefficients (Fourier slopes),
	stellar population properties (age, concentration, stellar mass,
	component masses), Hα line strength, and rest‑frame colour (g‑r).

	The function is very complex and calls many of the previously
	defined stripe‑wise helpers.

	Args:
		cut_data (tuple):
			param_cut    : Array used to define bins (redshift or M200).
			n_faixas     : Number of bins.
			faixas_label : Description string for the binning.
			config       : Dictionary with region masks and styles.
			flag         : 'redshift' or 'm200'.

	Returns:
		None.  A vast number of PNG figures is produced.
	"""

	def redshift_cut_loop(coeff_entry):
		####
		xgrid = np.linspace(xlim_m200[0], xlim_m200[1], 200)
		ygrid = np.linspace(y_lim[0], y_lim[1], 200)

		xy_mesh = np.meshgrid(xgrid, ygrid)

		cmap = plt.get_cmap('jet')
		cor_points = cmap(np.linspace(0,1,n_faixas))
		####
		param_vec,lim_region,lim_vec,config,param_label,pasta,param_name=coeff_entry
		coeff_entry_stripe=param_vec,config,param_label,pasta,param_name
		os.makedirs(f'{save_path}/analise_m200/{pasta[0]}/{pasta[1]}',exist_ok=True)
		x_peak_vec,y_peak_vec,alpha_values_vec,alpha_incs_vec,beta_values_vec,beta_incs_vec=[],[],[],[],[],[]
		for i,stripe in enumerate(stripes):
			stripe_entry=stripe,lim_vec[i]
			x_peak,y_peak,alpha_values,alpha_incs,beta_values,beta_incs=m200_coeff_stripes(xy_mesh,stripe_entry,coeff_entry_stripe,m200_vet)
			x_peak_vec.append(x_peak)
			y_peak_vec.append(y_peak)
			alpha_values_vec.append(alpha_values)
			alpha_incs_vec.append(alpha_incs)
			beta_values_vec.append(beta_values)
			beta_incs_vec.append(beta_incs)
		x_peak_vec,y_peak_vec,alpha_values_vec,alpha_incs_vec,beta_values_vec,beta_incs_vec=np.asarray(x_peak_vec),np.asarray(y_peak_vec),np.asarray(alpha_values_vec),np.asarray(alpha_incs_vec),np.asarray(beta_values_vec),np.asarray(beta_incs_vec)
		for i,vetor in enumerate(x_peak_vec[0,:]):
			plt.figure()
			plt.scatter(m200_vet[lim_region[i]],param_vec[lim_region[i]],c='white',edgecolors='black',alpha=0.2,label=f'{names_simples[i]}')
			for j,stripe in enumerate(stripes):
				plt.scatter(x_peak_vec[j,i],y_peak_vec[j,i],edgecolors='black',c=[cor_points[j]],label=f'z={stripes_med[j]}')
			plt.legend()
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(param_label)
			plt.savefig(f'{save_path}/analise_m200/{pasta[0]}/{pasta[1]}/central_peak_{param_name}_{pasta[1]}_{region_names[i]}.png')
			plt.close()

		plt.figure()
		plt.suptitle(f'{param_label}')
		for i,vetor in enumerate(x_peak_vec[0,:]):
			plt.errorbar(stripes_med,alpha_values_vec[:,i],yerr=alpha_incs_vec[:,i],c=cores[i],label=f'{names_simples[i]}')
		plt.xlabel(r'$z$')
		plt.ylabel(r'$\alpha$')
		plt.legend()
		plt.savefig(f'{save_path}/analise_m200/{pasta[0]}/{pasta[1]}/alphas_{param_name}_{pasta[1]}.png')
		plt.close()

		plt.figure()
		plt.suptitle(f'{param_label}')
		for i,vetor in enumerate(x_peak_vec[0,:]):
			plt.errorbar(stripes_med,beta_values_vec[:,i],yerr=beta_incs_vec[:,i],c=cores[i],label=f'{names_simples[i]}')
		plt.xlabel(r'$z$')
		plt.ylabel(fr'$\beta$')
		plt.legend()
		plt.savefig(f'{save_path}/analise_m200/{pasta[0]}/{pasta[1]}/beta_{param_name}_{pasta[1]}.png')
		plt.close()
		return		

	def par_only_tracer_plots(medias,sems,par_labels,save_labels):
		fig,axs=plt.subplots(1,1,sharey=True,figsize=(8,6))
		plt.suptitle(f'{par_labels[0]}')
		for j in range(3):
			axs.errorbar(stripes_med,medias[:,j],yerr=sems[:,j],fmt='o',color=cores[j],alpha=0.4)
			axs.plot(stripes_med,medias[:,j],color=cores[j],alpha=0.4,label=names_simples[j])
		axs.set_xlabel(r'$\log M_{200} \ (M_\odot)$')
		axs.set_ylabel(par_labels[1])
		axs.legend()
		plt.tight_layout()
		plt.savefig(f'{save_path}/analise_m200/{save_labels[0]}/{save_labels[1]}_tracer_{pasta_save}.png')
		plt.close()
		return

	def param_plots(plots_entry):
		param,lim_region,cores,names_simples,region_names,l07_regions,param_str=plots_entry

		ajust_param_m200_high,cov_param_m200_high=np.polyfit(m200_vet[lim_region[0]],param[lim_region[0]],1,cov=True)
		ajust_param_m200_low_left,cov_param_m200_low_left=np.polyfit(m200_vet[lim_region[1]],param[lim_region[1]],1,cov=True)
		ajust_param_m200_low_right,cov_param_m200_low_right=np.polyfit(m200_vet[lim_region[2]],param[lim_region[2]],1,cov=True)

		fit_lines=(ajust_param_m200_high,cov_param_m200_high),(ajust_param_m200_low_left,cov_param_m200_low_left),(ajust_param_m200_low_right,cov_param_m200_low_right)
		for i in range(3):
			ajust_line=fit_lines[i]
			fig1=plt.figure(figsize=(9,7))
			plt.scatter(m200_vet[lim_region[i]],param[lim_region[i]],c=cores[i],edgecolors='black',alpha=0.5,label=f'{names_simples[i]}')
			plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_line[0]),color=cores[i],lw=2,label=fr'$\alpha$={ajust_line[0][0]:.3f}$\pm${np.sqrt(ajust_line[1][0,0]):.3f}'+'\n'+fr'$\beta$={ajust_line[0][1]:.3f}$\pm${np.sqrt(ajust_line[1][1,1]):.3f}')
			plt.xlim(xlim_m200)
			plt.ylim(y_lim)
			plt.legend()
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(param_str[0])
			plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_{region_names[i]}.png')
			plt.close()
		############
		#TODAS
		fig1=plt.figure(figsize=(9,7))
		for i in range(3):
			ajust_line=fit_lines[i]
			plt.scatter(m200_vet[lim_region[i]],param[lim_region[i]],c=cores[i],edgecolors='black',alpha=0.4,label=f'{names_simples[i]}')
			plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_line[0]),color=cores[i],lw=2,label=fr'$\alpha$={ajust_line[0][0]:.3f}$\pm${np.sqrt(ajust_line[1][0,0]):.3f}'+'\n'+fr'$\beta$={ajust_line[0][1]:.3f}$\pm${np.sqrt(ajust_line[1][1,1]):.3f}')
		plt.xlim(xlim_m200)
		plt.ylim(y_lim)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(param_str[0])
		plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_3class.png')
		plt.close()

		if sample == 'L07':
			cd_cut,e_cut,misc_cut=l07_regions
			ajust_param_m200_cd_zhao,cov_param_m200_cd_zhao=np.polyfit(m200_vet[cd_cut],param[cd_cut],1,cov=True)
			ajust_param_m200_e_zhao,cov_param_m200_e_zhao=np.polyfit(m200_vet[e_cut],param[e_cut],1,cov=True)

			ajust_param_m200_high_cD,cov_param_m200_high_cD=np.polyfit(m200_vet[lim_region[0] & cd_cut],param[lim_region[0] & cd_cut],1,cov=True)
			ajust_param_m200_low_left_cD,cov_param_m200_low_left_cD=np.polyfit(m200_vet[lim_region[1] & cd_cut],param[lim_region[1] & cd_cut],1,cov=True)
			ajust_param_m200_low_right_cD,cov_param_m200_low_right_cD=np.polyfit(m200_vet[lim_region[2] & cd_cut],param[lim_region[2] & cd_cut],1,cov=True)

			fit_lines_cD=(ajust_param_m200_high_cD,cov_param_m200_high_cD),(ajust_param_m200_low_left_cD,cov_param_m200_low_left_cD),(ajust_param_m200_low_right_cD,cov_param_m200_low_right_cD)
			#classificação ZHAO
			##ELIPTICAS
			fig1=plt.figure(figsize=(9,7))
			plt.scatter(m200_vet[e_cut],param[e_cut],c='green',edgecolors='black',alpha=0.5,label=f'E[Zhao]')
			plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_param_m200_e_zhao),color='black',lw=2,label=fr'$\alpha$={ajust_param_m200_e_zhao[0]:.3f}$\pm${np.sqrt(cov_param_m200_e_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_param_m200_e_zhao[1]:.3f}$\pm${np.sqrt(cov_param_m200_e_zhao[1,1]):.3f}')
			plt.xlim(xlim_m200)
			plt.ylim(y_lim)
			plt.legend()
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(param_str[0])
			plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_e_zhao.png')
			plt.close()
			##cD
			fig1=plt.figure(figsize=(9,7))
			plt.scatter(m200_vet[cd_cut],param[cd_cut],c='blue',edgecolors='black',alpha=0.5,label=f'cD[Zhao]')
			plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_param_m200_cd_zhao),color='black',lw=2,label=fr'$\alpha$={ajust_param_m200_cd_zhao[0]:.3f}$\pm${np.sqrt(cov_param_m200_cd_zhao[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_param_m200_cd_zhao[1]:.3f}$\pm${np.sqrt(cov_param_m200_cd_zhao[1,1]):.3f}')
			plt.xlim(xlim_m200)
			plt.ylim(y_lim)
			plt.legend()
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(param_str[0])
			plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_cd_zhao.png')
			plt.close()
			################
			for i in range(3):
				ajust_line_cD=fit_lines_cD[i]
				fig1=plt.figure(figsize=(9,7))
				plt.scatter(m200_vet[lim_region[i] & cd_cut],param[lim_region[i] & cd_cut],c=cores[i],edgecolors='black',alpha=0.5,label=f'{names_simples[i]}[cD]')
				plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_line_cD[0]),color=cores[i],lw=2,label=fr'$\alpha$={ajust_line_cD[0][0]:.3f}$\pm${np.sqrt(ajust_line_cD[1][0,0]):.3f}'+'\n'+fr'$\beta$={ajust_line_cD[0][1]:.3f}$\pm${np.sqrt(ajust_line_cD[1][1,1]):.3f}')
				plt.xlim(xlim_m200)
				plt.ylim(y_lim)
				plt.legend()
				plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
				plt.ylabel(param_str[0])
				plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_{region_names[i]}_cD.png')
				plt.close()

			fig1=plt.figure(figsize=(9,7))
			plt.scatter(m200_vet[lim_region[1] & (cd_cut | misc_cut)],param[lim_region[1] & (cd_cut | misc_cut)],c=cores[i],edgecolors='black',alpha=0.5,label=f'{names_simples[i]}[cD]')
			plt.plot(m200_linspace,linfunc(m200_linspace,*fit_lines_cD[1][0]),color='red',lw=2,label=fr'$\alpha$={fit_lines_cD[1][0][0]:.3f}$\pm${np.sqrt(fit_lines_cD[1][1][0,0]):.3f}'+'\n'+fr'$\beta$={fit_lines_cD[1][0][1]:.3f}$\pm${np.sqrt(fit_lines_cD[1][1][1,1]):.3f}')
			plt.xlim(xlim_m200)
			plt.ylim(y_lim)
			plt.legend()
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(param_str[0])
			plt.savefig(f'{save_path}/analise_m200/{param_str[1]}/m200_{param_str[2]}_{region_names[1]}_cD_misc.png')
			plt.close()
		return

	param_cut,n_faixas,faixas_label,config,flag=cut_data
	os.makedirs(f'{save_path}/analise_m200',exist_ok=True)

	if sample == 'WHL':
		l07_regions=None
		l07_regions_photutils=None
		l07_regions_casjobs=None
		l07_regions_cor_gr=None
	if sample == 'L07':
		l07_regions=(cd_cut,e_cut,(ecd_cut | cde_cut))
		l07_regions_photutils=(cd_cut_photutils,e_cut_photutils,(ecd_cut_photutils | cde_cut_photutils))
		l07_regions_casjobs=(cd_cut_casjobs,e_cut_casjobs,(ecd_cut_casjobs | cde_cut_casjobs))
		l07_regions_cor_gr=(cd_cut_cor_gr,e_cut_cor_gr,(ecd_cut_cor_gr | cde_cut_cor_gr))

	m200_linspace=np.linspace(min(m200_temp),max(m200_temp),3000)

	lim_region=config['lim_region']
	lim_region_photutils=config['lim_region_photutils']
	lim_region_casjobs=config['lim_region_casjobs']
	lim_region_cor_gr=config['lim_region_cor_gr']
	lim_region_halpha=config['lim_region_halpha']
	names_simples=config['names_simples']
	alpha_vec=config['alpha_vec']
	cores=config['cores']
	line_width=config['line_width']
	region_names=config['save_names']
	sub_labels=['s','1','2']
	to_do_test='refaz'
	pasta_analise='analise_m200'
	pasta_save=faixas_label
	titles=['Sérsic','Comp_1','Comp_2']
	
	idx_faixas=pd.qcut(param_cut,n_faixas,labels=False)
	idx_faixas_photutils=idx_faixas[lim_photutils]
	idx_faixas_casjobs=idx_faixas[lim_casjobs]
	idx_faixas_halpha=idx_faixas[lim_halpha]
	idx_faixas_cor_gr=idx_faixas[lim_cor_gr]
	m200_vet=m200_temp
	stripes=np.unique(idx_faixas)
	if flag == 'redshift':
		stripes=np.flip(stripes)
	stripes_med=np.asarray([np.round(np.average(param_cut[idx_faixas==idx]),4) for idx in stripes],dtype=float)
	lim_vec=[],[],[]
	lim_vec_photutils=[],[],[]
	lim_vec_casjobs=[],[],[]
	lim_vec_cor_gr=[],[],[]
	lim_vec_halpha=[],[],[]
	for i,n in enumerate(stripes):
		z_sub_sample_only=idx_faixas==n
		redshift_z=param_cut[z_sub_sample_only]
		lim_vec[0].append(lim_region[0] & z_sub_sample_only)
		lim_vec[1].append(lim_region[1] & z_sub_sample_only)
		lim_vec[2].append(lim_region[2] & z_sub_sample_only)
		#
		z_sub_sample_only_photutils=idx_faixas_photutils==n
		redshift_photutils_z=param_cut[lim_photutils][z_sub_sample_only_photutils]
		lim_vec_photutils[0].append(lim_region_photutils[0] & z_sub_sample_only_photutils)
		lim_vec_photutils[1].append(lim_region_photutils[1] & z_sub_sample_only_photutils)
		lim_vec_photutils[2].append(lim_region_photutils[2] & z_sub_sample_only_photutils)
		#
		z_sub_sample_only_casjobs=idx_faixas_casjobs==n
		redshift_casjobs_z=param_cut[lim_casjobs][z_sub_sample_only_casjobs]
		lim_vec_casjobs[0].append(lim_region_casjobs[0] & z_sub_sample_only_casjobs)
		lim_vec_casjobs[1].append(lim_region_casjobs[1] & z_sub_sample_only_casjobs)
		lim_vec_casjobs[2].append(lim_region_casjobs[2] & z_sub_sample_only_casjobs)
		#
		z_sub_sample_only_cor_gr=idx_faixas_cor_gr==n
		redshift_cor_gr_z=param_cut[lim_cor_gr][z_sub_sample_only_cor_gr]
		lim_vec_cor_gr[0].append(lim_region_cor_gr[0] & z_sub_sample_only_cor_gr)
		lim_vec_cor_gr[1].append(lim_region_cor_gr[1] & z_sub_sample_only_cor_gr)
		lim_vec_cor_gr[2].append(lim_region_cor_gr[2] & z_sub_sample_only_cor_gr)
		#
		z_sub_sample_only_halpha=idx_faixas_halpha==n
		redshift_halpha_z=param_cut[lim_halpha][z_sub_sample_only_halpha]
		lim_vec_halpha[0].append(lim_region_halpha[0] & z_sub_sample_only_halpha)
		lim_vec_halpha[1].append(lim_region_halpha[1] & z_sub_sample_only_halpha)
		lim_vec_halpha[2].append(lim_region_halpha[2] & z_sub_sample_only_halpha)

	lim_vec=np.asarray(lim_vec)
	lim_vec_photutils=np.asarray(lim_vec_photutils)
	lim_vec_casjobs=np.asarray(lim_vec_casjobs)
	lim_vec_cor_gr=np.asarray(lim_vec_cor_gr)
	lim_vec_halpha=np.asarray(lim_vec_halpha)

	lim_vec=lim_vec.transpose(1,0,2)
	lim_vec_photutils=lim_vec_photutils.transpose(1,0,2)
	lim_vec_casjobs=lim_vec_casjobs.transpose(1,0,2)
	lim_vec_cor_gr=lim_vec_cor_gr.transpose(1,0,2)
	lim_vec_halpha=lim_vec_halpha.transpose(1,0,2)
	#############################################
	#############################################
	# N
	os.makedirs(f'{save_path}/analise_m200/indice_sersic',exist_ok=True)
	ajust_n_m200_geral,cov_n_m200_geral=np.polyfit(m200_temp,n_s,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,n_s,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_n_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_n_m200_geral[0]:.3f}$\pm${np.sqrt(cov_n_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_n_m200_geral[1]:.3f}$\pm${np.sqrt(cov_n_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$n$')
	plt.savefig(f'{save_path}/analise_m200/indice_sersic/m200_n.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=n_s,lim_region,lim_vec,config,r'$n$',('indice_sersic',f'{faixas_label}'),'n'
		redshift_cut_loop(coeff_entry)
	####
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/indice_sersic/{pasta_save}',exist_ok=True)
		n_values=np.hstack((n_s,n1,n2))
		n_linspace=np.linspace(min(n_values),max(n_values),3000)
		par_stripe_entry=(n_s,n1,n2),n_linspace,('Índice de Sérsic',(r'$n_s$',r'$n_1$',r'$n_2$',r'$n$')),('indice_sersic','n')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		for i in range(len(titles)):
			par_labels,save_labels=(f'n - {titles[i]}',fr'$n_{sub_labels[i]}$'),('indice_sersic',f'n_{titles[i]}')
			par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)
	####
	plots_entry=n_s,lim_region,cores,names_simples,region_names,l07_regions,(r'$n$','indice_sersic','n')
	param_plots(plots_entry)

	#############################################
	# N1
	ajust_n1_m200_geral,cov_n1_m200_geral=np.polyfit(m200_temp,n1,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,n1,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_n1_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_n1_m200_geral[0]:.3f}$\pm${np.sqrt(cov_n1_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_n1_m200_geral[1]:.3f}$\pm${np.sqrt(cov_n1_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$n_1$')
	plt.savefig(f'{save_path}/analise_m200/indice_sersic/m200_n1.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=n1,lim_region,lim_vec,config,r'$n_1$',('indice_sersic',f'{faixas_label}'),'n1'
		redshift_cut_loop(coeff_entry)

	plots_entry=n1,lim_region,cores,names_simples,region_names,l07_regions,(r'$n_1$','indice_sersic','n1')
	param_plots(plots_entry)
	#####################
	# N2
	ajust_n2_m200_geral,cov_n2_m200_geral=np.polyfit(m200_temp,n2,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,n2,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_n2_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_n2_m200_geral[0]:.3f}$\pm${np.sqrt(cov_n2_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_n2_m200_geral[1]:.3f}$\pm${np.sqrt(cov_n2_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$n_2$')
	plt.savefig(f'{save_path}/analise_m200/indice_sersic/m200_n2.png')
	plt.close()
	####

	if flag == 'redshift':
		coeff_entry=n2,lim_region,lim_vec,config,r'$n_2$',('indice_sersic',f'{faixas_label}'),'n2'
		redshift_cut_loop(coeff_entry)

	plots_entry=n2,lim_region,cores,names_simples,region_names,l07_regions,(r'$n_2$','indice_sersic','n2')
	param_plots(plots_entry)

	##########################
	# N_RATIO
	ajust_n_ratio_12_m200_geral,cov_n_ratio_12_m200_geral=np.polyfit(m200_temp,n_ratio_12,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,n_ratio_12,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_n_ratio_12_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_n_ratio_12_m200_geral[0]:.3f}$\pm${np.sqrt(cov_n_ratio_12_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_n_ratio_12_m200_geral[1]:.3f}$\pm${np.sqrt(cov_n_ratio_12_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$n_1/n_2$')
	plt.savefig(f'{save_path}/analise_m200/indice_sersic/m200_n_ratio_12.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=n_ratio_12,lim_region,lim_vec,config,r'$n_1/n_2$',('indice_sersic',f'{faixas_label}'),'n_ratio_12'
		redshift_cut_loop(coeff_entry)

	plots_entry=n_ratio_12,lim_region,cores,names_simples,region_names,l07_regions,(r'$n_1/n_2$','indice_sersic','n_ratio_12')
	param_plots(plots_entry)

	#############################################
	#############################################
	# MUE MED EFETIVO
	os.makedirs(f'{save_path}/analise_m200/mue_med',exist_ok=True)

	ajust_mue_med_s_m200_geral,cov_mue_med_s_m200_geral=np.polyfit(m200_temp,mue_med_s,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,mue_med_s,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_mue_med_s_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_mue_med_s_m200_geral[0]:.3f}$\pm${np.sqrt(cov_mue_med_s_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_mue_med_s_m200_geral[1]:.3f}$\pm${np.sqrt(cov_mue_med_s_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$<\mu_e>$')
	plt.savefig(f'{save_path}/analise_m200/mue_med/m200_mue_med_s.png')
	plt.close()
	
	####
	if flag == 'redshift':
		coeff_entry=mue_med_s,lim_region,lim_vec,config,r'$<\mu>$',('mue_med',f'{faixas_label}'),'mue_med_s'
		redshift_cut_loop(coeff_entry)
	####
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/mue_med/{pasta_save}',exist_ok=True)
		mue_med_values=np.hstack((mue_med_s,mue_med_comp_1,mue_med_comp_2))
		mue_med_linspace=np.linspace(min(mue_med_values),max(mue_med_values),3000)
		par_stripe_entry=(mue_med_s,mue_med_comp_1,mue_med_comp_2),mue_med_linspace,('Brilho Efetivo Médio',(r'$<\mu_e>_s$',r'$<\mu_e>_1$',r'$<\mu_e>_2$',r'$<\mu_e>$')),('mue_med','mue_med')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		for i in range(len(titles)):
			par_labels,save_labels=(f'Brilho efetivo médio - {titles[i]}',r'$<\mu_e>$'+f' {sub_labels[i]}'),('mue_med',f'brilho_eff_{titles[i]}')
			par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)

	plots_entry=mue_med_s,lim_region,cores,names_simples,region_names,l07_regions,(r'$<\mu_e>$','mue_med','mue_med_s')
	param_plots(plots_entry)
	####
	#############################################
	# mue_med_comp_1
	ajust_mue_med_comp_1_m200_geral,cov_mue_med_comp_1_m200_geral=np.polyfit(m200_temp,mue_med_comp_1,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,mue_med_comp_1,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_mue_med_comp_1_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_mue_med_comp_1_m200_geral[0]:.3f}$\pm${np.sqrt(cov_mue_med_comp_1_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_mue_med_comp_1_m200_geral[1]:.3f}$\pm${np.sqrt(cov_mue_med_comp_1_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$<\mu_e> \ 1$')
	plt.savefig(f'{save_path}/analise_m200/mue_med/m200_mue_med_comp_1.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=mue_med_comp_1,lim_region,lim_vec,config,r'$<\mu_1>$',('mue_med',f'{faixas_label}'),'mue_med_comp_1'
		redshift_cut_loop(coeff_entry)

	plots_entry=mue_med_comp_1,lim_region,cores,names_simples,region_names,l07_regions,(r'$<\mu_e> \ 1$','mue_med','mue_med_comp_1')
	param_plots(plots_entry)
	#####################
	# mue_med_comp_2
	ajust_mue_med_comp_2_m200_geral,cov_mue_med_comp_2_m200_geral=np.polyfit(m200_temp,mue_med_comp_2,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,mue_med_comp_2,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_mue_med_comp_2_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_mue_med_comp_2_m200_geral[0]:.3f}$\pm${np.sqrt(cov_mue_med_comp_2_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_mue_med_comp_2_m200_geral[1]:.3f}$\pm${np.sqrt(cov_mue_med_comp_2_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$<\mu_e> \ 2$')
	plt.savefig(f'{save_path}/analise_m200/mue_med/m200_mue_med_comp_2.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=mue_med_comp_2,lim_region,lim_vec,config,r'$<\mu_e> \ 2$',('mue_med',f'{faixas_label}'),'mue_med_comp_2'
		redshift_cut_loop(coeff_entry)
	plots_entry=mue_med_comp_2,lim_region,cores,names_simples,region_names,l07_regions,(r'$<\mu_e> \ 2$','mue_med','mue_med_comp_2')
	param_plots(plots_entry)

	#############################################
	#############################################
	# RAZÃO AXIAL
	os.makedirs(f'{save_path}/analise_m200/ax_ratio',exist_ok=True)

	ajust_ax_ratio_m200_geral,cov_ax_ratio_m200_geral=np.polyfit(m200_temp,e_s,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,e_s,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_ax_ratio_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_ax_ratio_m200_geral[0]:.3f}$\pm${np.sqrt(cov_ax_ratio_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_ax_ratio_m200_geral[1]:.3f}$\pm${np.sqrt(cov_ax_ratio_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$q$')
	plt.savefig(f'{save_path}/analise_m200/ax_ratio/m200_q.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=e_s,lim_region,lim_vec,config,r'$q$',('ax_ratio',f'{faixas_label}'),'q'
		redshift_cut_loop(coeff_entry)
	####
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/ax_ratio/{pasta_save}',exist_ok=True)

		q_values=np.hstack((e_s,e1,e2))
		q_linspace=np.linspace(min(q_values),max(q_values),3000)
		par_stripe_entry=(e_s,e1,e2),q_linspace,('Razão Axial',(r'$q_s$',r'$q_1$',r'$q_2$',r'$q$')),('ax_ratio','axrat')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		for i in range(len(titles)):
			par_labels,save_labels=(f'Razão axial - {titles[i]}',fr'$q_{sub_labels[i]}$'),('ax_ratio',f'q_{titles[i]}')
			par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)

	plots_entry=e_s,lim_region,cores,names_simples,region_names,l07_regions,(r'$q$','ax_ratio','ax_ratio')
	param_plots(plots_entry)

	#############################################
	# e1
	ajust_e1_m200_geral,cov_e1_m200_geral=np.polyfit(m200_temp,e1,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,e1,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_e1_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_e1_m200_geral[0]:.3f}$\pm${np.sqrt(cov_e1_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e1_m200_geral[1]:.3f}$\pm${np.sqrt(cov_e1_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$q_1$')
	plt.savefig(f'{save_path}/analise_m200/ax_ratio/m200_e1.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=e1,lim_region,lim_vec,config,r'$q_1$',('ax_ratio',f'{faixas_label}'),'e1'
		redshift_cut_loop(coeff_entry)

	plots_entry=e1,lim_region,cores,names_simples,region_names,l07_regions,(r'$q_1$','ax_ratio','e1')
	param_plots(plots_entry)

	#####################
	# e2
	ajust_e2_m200_geral,cov_e2_m200_geral=np.polyfit(m200_temp,e2,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,e2,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_e2_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_e2_m200_geral[0]:.3f}$\pm${np.sqrt(cov_e2_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_e2_m200_geral[1]:.3f}$\pm${np.sqrt(cov_e2_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$q_2$')
	plt.savefig(f'{save_path}/analise_m200/ax_ratio/m200_e2.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=e2,lim_region,lim_vec,config,r'$q_2$',('ax_ratio',f'{faixas_label}'),'e2'
		redshift_cut_loop(coeff_entry)
	plots_entry=e2,lim_region,cores,names_simples,region_names,l07_regions,(r'$q_2$','ax_ratio','e2')
	param_plots(plots_entry)
	#############################################
	#############################################
	#############################################
	# Re
	os.makedirs(f'{save_path}/analise_m200/raio_efetivo_kpc',exist_ok=True)

	ajust_re_kpc_m200_geral,cov_re_kpc_m200_geral=np.polyfit(m200_temp,re_s_kpc,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,re_s_kpc,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_re_kpc_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_re_kpc_m200_geral[0]:.3f}$\pm${np.sqrt(cov_re_kpc_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_re_kpc_m200_geral[1]:.3f}$\pm${np.sqrt(cov_re_kpc_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$Re_s \ (Kpc)$')
	plt.savefig(f'{save_path}/analise_m200/raio_efetivo_kpc/m200_re_s_kpc.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=re_s_kpc,lim_region,lim_vec,config,r'$Re_s \ (Kpc)$',('raio_efetivo_kpc',f'{faixas_label}'),'re_s_kpc'
		redshift_cut_loop(coeff_entry)

	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/raio_efetivo_kpc/{pasta_save}',exist_ok=True)
		re_kpc_values=np.hstack((re_s_kpc,re_1_kpc,re_2_kpc))
		re_kpc_linspace=np.linspace(min(re_kpc_values),max(re_kpc_values),3000)
		par_stripe_entry=(re_s_kpc,re_1_kpc,re_2_kpc),re_kpc_linspace,('Raio Efetivo (kpc)',(r'$\log_{10}(Re_S)\ (kpc)$',r'$\log_{10}(Re_1)\ (kpc)$',r'$\log_{10}(Re_2)\ (kpc)$',r'$\log_{10}(Re)\ (kpc)$')),('raio_efetivo_kpc','re_kpc')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		for i in range(len(titles)):
			par_labels,save_labels=(f'Raio Efetivo - {titles[i]}',r'$Re \ (Kpc)$'+f' {sub_labels[i]}$'),('raio_efetivo_kpc',f're_kpc_{titles[i]}')
			par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)

	plots_entry=re_s_kpc,lim_region,cores,names_simples,region_names,l07_regions,(r'$Re_s \ (Kpc)$','raio_efetivo_kpc','re_kpc')
	param_plots(plots_entry)
	#############################################
	# RE 1
	ajust_re_1_kpc_m200_geral,cov_re_1_kpc_m200_geral=np.polyfit(m200_temp,re_1_kpc,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,re_1_kpc,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_re_1_kpc_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_re_1_kpc_m200_geral[0]:.3f}$\pm${np.sqrt(cov_re_1_kpc_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_re_1_kpc_m200_geral[1]:.3f}$\pm${np.sqrt(cov_re_1_kpc_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$Re_1 \ (Kpc)$')
	plt.savefig(f'{save_path}/analise_m200/raio_efetivo_kpc/m200_re_1_kpc.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=re_1_kpc,lim_region,lim_vec,config,r'$Re_1 \ (Kpc)$',('raio_efetivo_kpc',f'{faixas_label}'),'re_1_kpc'
		redshift_cut_loop(coeff_entry)
	plots_entry=re_1_kpc,lim_region,cores,names_simples,region_names,l07_regions,(r'$Re_1 \ (Kpc)$','raio_efetivo_kpc','re_1_kpc')
	param_plots(plots_entry)

	#####################
	#RE 2
	ajust_re_2_kpc_m200_geral,cov_re_2_kpc_m200_geral=np.polyfit(m200_temp,re_2_kpc,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,re_2_kpc,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_re_2_kpc_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_re_2_kpc_m200_geral[0]:.3f}$\pm${np.sqrt(cov_re_2_kpc_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_re_2_kpc_m200_geral[1]:.3f}$\pm${np.sqrt(cov_re_2_kpc_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$Re_2 \ (Kpc)$')
	plt.savefig(f'{save_path}/analise_m200/raio_efetivo_kpc/m200_re_2_kpc.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=re_2_kpc,lim_region,lim_vec,config,r'$Re_2 \ (Kpc)$',('raio_efetivo_kpc',f'{faixas_label}'),'re_2_kpc'
		redshift_cut_loop(coeff_entry)
	y_lim=(y_lim[0],2)
	plots_entry=re_2_kpc,lim_region,cores,names_simples,region_names,l07_regions,(r'$Re_2 \ (Kpc)$','raio_efetivo_kpc','re_2_kpc')
	param_plots(plots_entry)
	##########################
	# RE_RATIO
	ajust_re_ratio_12_m200_geral,cov_re_ratio_12_m200_geral=np.polyfit(m200_temp,re_ratio_12,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,re_ratio_12,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_re_ratio_12_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_re_ratio_12_m200_geral[0]:.3f}$\pm${np.sqrt(cov_re_ratio_12_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_re_ratio_12_m200_geral[1]:.3f}$\pm${np.sqrt(cov_re_ratio_12_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$Re_1/Re_2$')
	plt.savefig(f'{save_path}/analise_m200/raio_efetivo_kpc/m200_re_ratio_12.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=re_ratio_12,lim_region,lim_vec,config,r'$Re_1/Re_2$',('raio_efetivo_kpc',f'{faixas_label}'),'re_ratio_12'
		redshift_cut_loop(coeff_entry)

	plots_entry=re_ratio_12,lim_region,cores,names_simples,region_names,l07_regions,(r'$Re_1/Re_2$','raio_efetivo_kpc','re_ratio_12')
	param_plots(plots_entry)

	######################################################
	######################################################
	######################################################	
	# BOX
	os.makedirs(f'{save_path}/analise_m200/box',exist_ok=True)

	ajust_box_m200_geral,cov_box_m200_geral=np.polyfit(m200_temp,box_s,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,box_s,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_box_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_box_m200_geral[0]:.3f}$\pm${np.sqrt(cov_box_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_box_m200_geral[1]:.3f}$\pm${np.sqrt(cov_box_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$a_4/a$')
	plt.savefig(f'{save_path}/analise_m200/box/m200_box.png')
	plt.close()
	####
	if flag == 'redshift':
		coeff_entry=box_s,lim_region,lim_vec,config,r'$a_4/a$',('box',f'{faixas_label}'),'box'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/box/{pasta_save}',exist_ok=True)
		box_values=np.hstack((box_s,box1,box2))
		box_linspace=np.linspace(min(box_values),max(box_values),3000)
		par_stripe_entry=(box_s,box1,box2),box_linspace,('Boxiness',(r'$(a_4/a)_s$',r'$(a_4/a)_1$',r'$(a_4/a)_2$',r'$a_4/a$')),('box','box')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_multi_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		for i in range(len(titles)):
			par_labels,save_labels=(f'Boxiness - {titles[i]}',r'$a_4/a$'+f' {sub_labels[i]}$'),('box',f'box_{titles[i]}')
			par_only_tracer_plots(medias[:,i],sems[:,i],par_labels,save_labels)

	plots_entry=box_s,lim_region,cores,names_simples,region_names,l07_regions,(r'$a_4/a$','box','box')
	param_plots(plots_entry)

	#############################################
	# BOX 1
	ajust_box1_m200_geral,cov_box1_m200_geral=np.polyfit(m200_temp,box1,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,box1,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_box1_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_box1_m200_geral[0]:.3f}$\pm${np.sqrt(cov_box1_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_box1_m200_geral[1]:.3f}$\pm${np.sqrt(cov_box1_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$(a_4/a)_1$')
	plt.savefig(f'{save_path}/analise_m200/box/m200_box1.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=box1,lim_region,lim_vec,config,r'$(a_4/a)_1$',('box',f'{faixas_label}'),'box1'
		redshift_cut_loop(coeff_entry)

	plots_entry=box1,lim_region,cores,names_simples,region_names,l07_regions,(r'$(a_4/a)_1$','box','box1')
	param_plots(plots_entry)

	#####################
	# BOX 2
	ajust_box2_m200_geral,cov_box2_m200_geral=np.polyfit(m200_temp,box2,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,box2,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_box2_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_box2_m200_geral[0]:.3f}$\pm${np.sqrt(cov_box2_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_box2_m200_geral[1]:.3f}$\pm${np.sqrt(cov_box2_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$(a_4/a)_2$')
	plt.savefig(f'{save_path}/analise_m200/box/m200_box2.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=box2,lim_region,lim_vec,config,r'$(a_4/a)_2$',('box',f'{faixas_label}'),'box2'
		redshift_cut_loop(coeff_entry)

	plots_entry=box2,lim_region,cores,names_simples,region_names,l07_regions,(r'$(a_4/a))_2$','box','box2')
	param_plots(plots_entry)
	#######################################
	#######################################
	#######################################
	# ETA
	os.makedirs(f'{save_path}/analise_m200/eta',exist_ok=True)

	y_lim=(-0.02,0.25)

	ajust_eta_m200_geral,cov_eta_m200_geral=np.polyfit(m200_temp,eta,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,eta,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_eta_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_eta_m200_geral[0]:.3f}$\pm${np.sqrt(cov_eta_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_eta_m200_geral[1]:.3f}$\pm${np.sqrt(cov_eta_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} (M_\odot)$')
	plt.ylabel(r'$\eta$')
	plt.savefig(f'{save_path}/analise_m200/eta/m200_eta.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=eta,lim_region,lim_vec,config,r'$\eta$',('eta',f'{faixas_label}'),'eta'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/eta/{pasta_save}',exist_ok=True)
		eta_linspace=np.linspace(0.01,0.1,3000)
		par_stripe_entry=eta,eta_linspace,('eta (RFF - A1)',r'$\eta$'),('eta','eta')
		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_labels,save_labels=('eta',r'$\eta$'),('eta','eta')
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=eta,lim_region,cores,names_simples,region_names,l07_regions,(r'$\eta$','eta','eta')
	param_plots(plots_entry)
	############################################
	############################################
	############################################
	# ASS
	os.makedirs(f'{save_path}/analise_m200/ass',exist_ok=True)

	y_lim=(-0.05,0.15)

	ajust_ass_m200_geral,cov_ass_m200_geral=np.polyfit(m200_temp,ass,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,ass,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_ass_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_ass_m200_geral[0]:.3f}$\pm${np.sqrt(cov_ass_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_ass_m200_geral[1]:.3f}$\pm${np.sqrt(cov_ass_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$A_1$')
	plt.savefig(f'{save_path}/analise_m200/ass/m200_ass.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=ass,lim_region,lim_vec,config,r'$A_1$',('ass',f'{faixas_label}'),'ass'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/ass/{pasta_save}',exist_ok=True)
		ass_linspace=np.linspace(-0.05,0.15,3000)
		par_stripe_entry=ass,ass_linspace,('A1',r'$A_1$'),('ass','ass')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_labels,save_labels=('A1',r'$A_1$'),('ass','ass')
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=ass,lim_region,cores,names_simples,region_names,l07_regions,(r'$A_1$','ass','ass')
	param_plots(plots_entry)
	############################################
	############################################
	############################################
	# RFF
	os.makedirs(f'{save_path}/analise_m200/rff_s',exist_ok=True)

	y_lim=(-0.01,0.18)

	ajust_rff_s_m200_geral,cov_rff_s_m200_geral=np.polyfit(m200_temp,rff_s,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,rff_s,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_rff_s_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_rff_s_m200_geral[0]:.3f}$\pm${np.sqrt(cov_rff_s_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_rff_s_m200_geral[1]:.3f}$\pm${np.sqrt(cov_rff_s_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$RFF_s$')
	plt.savefig(f'{save_path}/analise_m200/rff_s/m200_rff_s.png')
	plt.close()

	if flag == 'redshift':
		coeff_entry=rff_s,lim_region,lim_vec,config,r'$RFF$',('rff_s',f'{faixas_label}'),'rff_s'
		redshift_cut_loop(coeff_entry)

	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/rff_s/{pasta_save}',exist_ok=True)
		rff_s_linspace=np.linspace(0.0,0.08,3000)
		par_stripe_entry=rff_s,rff_s_linspace,('RFF (S)',r'$RFF$'),('rff_s','rff_s')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_labels,save_labels=('RFF (S)',r'$RFF$'),('rff_s','rff_s')
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=rff_s,lim_region,cores,names_simples,region_names,l07_regions,(r'$RFF_s$','rff_s','rff_s')
	param_plots(plots_entry)

	############################################
	############################################
	############################################
	#BT
	os.makedirs(f'{save_path}/analise_m200/bt',exist_ok=True)

	y_lim=(0,1.)

	ajust_bt_m200_geral,cov_bt_m200_geral=np.polyfit(m200_temp,bt_vec_corr,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,bt_vec_corr,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_bt_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_bt_m200_geral[0]:.3f}$\pm${np.sqrt(cov_bt_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_bt_m200_geral[1]:.3f}$\pm${np.sqrt(cov_bt_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} (M_\odot)$')
	plt.ylabel(r'$B/T$')
	plt.savefig(f'{save_path}/analise_m200/bt/m200_bt.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=bt_vec_corr,lim_region,lim_vec,config,r'$B/T$',('bt',f'{faixas_label}'),'bt'
		redshift_cut_loop(coeff_entry)

	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/bt/{pasta_save}',exist_ok=True)
		bt_linspace=np.linspace(0.0,0.1,3000)
		_,_,par_labels,save_labels=par_stripe_entry=bt_vec_corr,bt_linspace,('Razão BT',r'$B/T$'),('bt','bt')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=bt_vec_corr,lim_region,cores,names_simples,region_names,l07_regions,(r'$B/T$','bt','bt')
	param_plots(plots_entry)
	############################################
	############################################
	############################################
	# RFF_RATIO

	os.makedirs(f'{save_path}/analise_m200/rff_ratio',exist_ok=True)

	y_lim=(-0.01,1.5)

	ajust_rff_ratio_m200_geral,cov_rff_ratio_m200_geral=np.polyfit(m200_temp,rff_ratio,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_temp,rff_ratio,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_rff_ratio_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_rff_ratio_m200_geral[0]:.3f}$\pm${np.sqrt(cov_rff_ratio_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_rff_ratio_m200_geral[1]:.3f}$\pm${np.sqrt(cov_rff_ratio_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$RFF_ss/RFF_s$')
	plt.savefig(f'{save_path}/analise_m200/rff_ratio/m200_rff_ratio.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=rff_ratio,lim_region,lim_vec,config,r'$RFF_ss/RFF_s$',('rff_ratio',f'{faixas_label}'),'rff_ratio'
		redshift_cut_loop(coeff_entry)

	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/rff_ratio/{pasta_save}',exist_ok=True)
		rff_ratio_linspace=np.linspace(0.0,0.9,3000)
		par_stripe_entry=rff_ratio,rff_ratio_linspace,('rff_ratio (SS/S)',r'$RFF_{SS}/RFF_{S}$'),('rff_ratio','rff_ratio')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_labels,save_labels=('Razão de RFF (SS/S)',r'$RFF_{SS}/RFF_{S}$'),('rff_ratio','rff_ratio')
		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=rff_ratio,lim_region,cores,names_simples,region_names,l07_regions,(r'$RFF_{SS}/RFF_{S}$','rff_ratio','rff_ratio')
	param_plots(plots_entry)

	#####################################################
	#####################################################
	#####################################################
	#PHOTUTILS
	m200_phot=m200_temp[lim_photutils]
	m200_vet=m200_phot
	m200_linspace=np.linspace(min(m200_phot),max(m200_phot),3000)
	
	#A3
	os.makedirs(f'{save_path}/analise_m200/a3',exist_ok=True)
	a3_vec=[med_a3,slope_a3]
	a3_save=['med_a3','slope_a3']
	a3_label=[r'$\bar{a_3}$',r'$\alpha_{a_3}$']
	coef_titles=['médio','slope']
	if flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/a3/{pasta_save}',exist_ok=True)		
		for j,vet in enumerate(a3_vec):
			par_labels,save_labels=(f'a3 - {coef_titles[j]}',a3_label[j]),('a3',f'a3_{coef_titles[j]}')
			a3_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
			par_stripe_entry=vet,a3_linspace,par_labels,save_labels

			medias,sems=[],[]
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)
			par_only_tracer_plots(medias,sems,par_labels,save_labels)

	for j,vet in enumerate(a3_vec):
		y_lim=(np.percentile(vet,1),np.percentile(vet,99))

		ajust_a3_m200_geral,cov_a3_m200_geral=np.polyfit(m200_phot,vet,1,cov=True)

		fig1=plt.figure(figsize=(9,7))
		plt.scatter(m200_phot,vet,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
		plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_a3_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_a3_m200_geral[0]:.3f}$\pm${np.sqrt(cov_a3_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_a3_m200_geral[1]:.3f}$\pm${np.sqrt(cov_a3_m200_geral[1,1]):.3f}')
		xlim_m200=plt.xlim()
		plt.ylim(y_lim)
		plt.margins(y=0.1)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(a3_label[j])
		plt.savefig(f'{save_path}/analise_m200/a3/m200_{a3_save[j]}.png')
		plt.close()

		####
		if flag == 'redshift':
			coeff_entry=vet,lim_region_photutils,lim_vec_photutils,config,a3_label[j],('a3',f'{faixas_label}'),a3_save[j]
			redshift_cut_loop(coeff_entry)
		plots_entry=vet,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(a3_label[j],'a3',a3_save[j])
		param_plots(plots_entry)
	############################################
	# A4
	os.makedirs(f'{save_path}/analise_m200/a4',exist_ok=True)
	a4_vec=[med_a4,slope_a4]
	a4_save=['med_a4','slope_a4']
	a4_label=[r'$\bar{a_4}$',r'$\alpha_{a_4}$']
	coef_titles=['médio','slope']
	if flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/a4/{pasta_save}',exist_ok=True)
		for j,vet in enumerate(a4_vec):
			par_labels,save_labels=(f'a4 - {coef_titles[j]}',a4_label[j]),('a4',f'a4_{coef_titles[j]}')
			a4_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
			par_stripe_entry=vet,a4_linspace,par_labels,save_labels

			medias,sems=[],[]
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)
			par_only_tracer_plots(medias,sems,par_labels,save_labels)

	for j,vet in enumerate(a4_vec):
		y_lim=(np.percentile(vet,1),np.percentile(vet,99))

		ajust_a4_m200_geral,cov_a4_m200_geral=np.polyfit(m200_phot,vet,1,cov=True)

		fig1=plt.figure(figsize=(9,7))
		plt.scatter(m200_phot,vet,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
		plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_a4_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_a4_m200_geral[0]:.3f}$\pm${np.sqrt(cov_a4_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_a4_m200_geral[1]:.3f}$\pm${np.sqrt(cov_a4_m200_geral[1,1]):.3f}')
		xlim_m200=plt.xlim()
		plt.ylim(y_lim)
		plt.margins(y=0.1)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(a4_label[j])
		plt.savefig(f'{save_path}/analise_m200/a4/m200_{a4_save[j]}.png')
		plt.close()

		####
		if flag == 'redshift':
			coeff_entry=vet,lim_region_photutils,lim_vec_photutils,config,a4_label[j],('a4',f'{faixas_label}'),a4_save[j]
			redshift_cut_loop(coeff_entry)
		plots_entry=vet,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(a4_label[j],'a4',a4_save[j])
		param_plots(plots_entry)
	############################################
	# B3
	os.makedirs(f'{save_path}/analise_m200/b3',exist_ok=True)
	b3_vec=[med_b3,slope_b3]
	b3_save=['med_b3','slope_b3']
	b3_label=[r'$\bar{b_3}$',r'$\alpha_{b_3}$']
	coef_titles=['médio','slope']
	if flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/b3/{pasta_save}',exist_ok=True)
		
		for j,vet in enumerate(b3_vec):
			par_labels,save_labels=(f'b3 - {coef_titles[j]}',b3_label[j]),('b3',f'b3_{coef_titles[j]}')
			b3_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
			par_stripe_entry=vet,b3_linspace,par_labels,save_labels

			medias,sems=[],[]
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)
			par_only_tracer_plots(medias,sems,par_labels,save_labels)

	for j,vet in enumerate(b3_vec):
		y_lim=(np.percentile(vet,1),np.percentile(vet,99))

		ajust_b3_m200_geral,cov_b3_m200_geral=np.polyfit(m200_phot,vet,1,cov=True)

		fig1=plt.figure(figsize=(9,7))
		plt.scatter(m200_phot,vet,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
		plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_b3_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_b3_m200_geral[0]:.3f}$\pm${np.sqrt(cov_b3_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_b3_m200_geral[1]:.3f}$\pm${np.sqrt(cov_b3_m200_geral[1,1]):.3f}')
		xlim_m200=plt.xlim()
		plt.ylim(y_lim)
		plt.margins(y=0.1)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(b3_label[j])
		plt.savefig(f'{save_path}/analise_m200/b3/m200_{b3_save[j]}.png')
		plt.close()

		####
		if flag == 'redshift':
			coeff_entry=vet,lim_region_photutils,lim_vec_photutils,config,b3_label[j],('b3',f'{faixas_label}'),b3_save[j]
			redshift_cut_loop(coeff_entry)
		plots_entry=vet,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(b3_label[j],'b3',b3_save[j])
		param_plots(plots_entry)
	##############################
	# B4
	os.makedirs(f'{save_path}/analise_m200/b4',exist_ok=True)
	b4_vec=[med_b4,slope_b4]
	b4_save=['med_b4','slope_b4']
	b4_label=[r'$\bar{b_4}$',r'$\alpha_{b_4}$']
	coef_titles=['médio','slope']
	if flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/b4/{pasta_save}',exist_ok=True)
		for j,vet in enumerate(b4_vec):
			par_labels,save_labels=(f'b4 - {coef_titles[j]}',b4_label[j]),('b4',f'b4_{coef_titles[j]}')
			b4_linspace=np.linspace(np.percentile(vet,5),np.percentile(vet,95),3000)
			par_stripe_entry=vet,b4_linspace,par_labels,save_labels

			medias,sems=[],[]
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)
			par_only_tracer_plots(medias,sems,par_labels,save_labels)

	for j,vet in enumerate(b4_vec):
		y_lim=(np.percentile(vet,1),np.percentile(vet,99))

		ajust_b4_m200_geral,cov_b4_m200_geral=np.polyfit(m200_phot,vet,1,cov=True)

		fig1=plt.figure(figsize=(9,7))
		plt.scatter(m200_phot,vet,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
		plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_b4_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_b4_m200_geral[0]:.3f}$\pm${np.sqrt(cov_b4_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_b4_m200_geral[1]:.3f}$\pm${np.sqrt(cov_b4_m200_geral[1,1]):.3f}')
		xlim_m200=plt.xlim()
		plt.ylim(y_lim)
		plt.margins(y=0.1)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(b4_label[j])
		plt.savefig(f'{save_path}/analise_m200/b4/m200_{b4_save[j]}.png')
		plt.close()

		####
		if flag == 'redshift':
			coeff_entry=vet,lim_region_photutils,lim_vec_photutils,config,b4_label[j],('b4',f'{faixas_label}'),b4_save[j]
			redshift_cut_loop(coeff_entry)
		plots_entry=vet,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(b4_label[j],'b4',b4_save[j])
		param_plots(plots_entry)	
	##############################
	# SLOPE_GR
	os.makedirs(f'{save_path}/analise_m200/test_gr',exist_ok=True)

	y_lim=(np.percentile(slope_gr,1),np.percentile(slope_gr,99))

	ajust_slope_gr_m200_geral,cov_slope_gr_m200_geral=np.polyfit(m200_phot,slope_gr,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_phot,slope_gr,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_slope_gr_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_slope_gr_m200_geral[0]:.3f}$\pm${np.sqrt(cov_slope_gr_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_slope_gr_m200_geral[1]:.3f}$\pm${np.sqrt(cov_slope_gr_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\alpha_{g-r}$')
	plt.savefig(f'{save_path}/analise_m200/test_gr/m200_slope_gr.png')
	plt.close()
	####

	if flag == 'redshift':
		coeff_entry=slope_gr,lim_region_photutils,lim_vec_photutils,config,r'$\alpha_{g-r}$',('test_gr',f'{faixas_label}'),'slope_gr'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/test_gr/{pasta_save}',exist_ok=True)

		slope_gr_linspace=np.linspace(np.percentile(slope_gr,5),np.percentile(slope_gr,95),3000)
		_,_,par_labels,save_labels=par_stripe_entry=slope_gr,slope_gr_linspace,('slope g-r',r'$\alpha g-r$'),('test_gr','slope_gr')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=slope_gr,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(r'$\alpha g-r$','test_gr','slope_gr')
	param_plots(plots_entry)

	###########################################################
	# GRAD_E
	os.makedirs(f'{save_path}/analise_m200/grad_e',exist_ok=True)

	y_lim=(np.percentile(grad_e,1),np.percentile(grad_e,99))

	ajust_grad_e_m200_geral,cov_grad_e_m200_geral=np.polyfit(m200_phot,grad_e,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_phot,grad_e,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_grad_e_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_grad_e_m200_geral[0]:.3f}$\pm${np.sqrt(cov_grad_e_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_grad_e_m200_geral[1]:.3f}$\pm${np.sqrt(cov_grad_e_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\nabla_{e}$')
	plt.savefig(f'{save_path}/analise_m200/grad_e/m200_grad_e.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=grad_e,lim_region_photutils,lim_vec_photutils,config,r'$\nabla_{e}$',('grad_e',f'{faixas_label}'),'grad_e'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/grad_e/{pasta_save}',exist_ok=True)

		grad_e_linspace=np.linspace(min(grad_e),max(grad_e),3000)
		_,_,par_labels,save_labels=par_stripe_entry=grad_e,grad_e_linspace,('Gradiente de Elipticidade',r'$\nabla_e$'),('grad_e','grad_e')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=grad_e,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(r'$\nabla e$','grad_e','grad_e')
	param_plots(plots_entry)

	##################################################
	# GRAD_PA
	os.makedirs(f'{save_path}/analise_m200/grad_pa',exist_ok=True)

	y_lim=(np.percentile(grad_pa,1),np.percentile(grad_pa,99))

	ajust_grad_pa_m200_geral,cov_grad_pa_m200_geral=np.polyfit(m200_phot,grad_pa,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_phot,grad_pa,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_grad_pa_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_grad_pa_m200_geral[0]:.3f}$\pm${np.sqrt(cov_grad_pa_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_grad_pa_m200_geral[1]:.3f}$\pm${np.sqrt(cov_grad_pa_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.margins(y=0.1)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\nabla_{PA}$')
	plt.savefig(f'{save_path}/analise_m200/grad_pa/m200_grad_pa.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=grad_pa,lim_region_photutils,lim_vec_photutils,config,r'$\nabla_{PA}$',('grad_pa',f'{faixas_label}'),'grad_pa'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/grad_pa/{pasta_save}',exist_ok=True)
		grad_pa_linspace=np.linspace(min(grad_pa),max(grad_pa),3000)
		_,_,par_labels,save_labels=par_stripe_entry=grad_pa,grad_pa_linspace,('Gradiente de Posição angular',r'$\nabla_{PA}$'),('grad_pa','grad_pa')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_photutils[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=grad_pa,lim_region_photutils,cores,names_simples,region_names,l07_regions_photutils,(r'$\nabla_{PA}$','grad_pa','grad_pa')
	param_plots(plots_entry)

	###############################################
	###############################################
	###############################################
	#CASJOBS
	m200_casjobs=m200_temp[lim_casjobs]
	m200_linspace=np.linspace(min(m200_casjobs),max(m200_casjobs),3000)
	m200_vet=m200_casjobs
	#############################
	# AGE
	os.makedirs(f'{save_path}/analise_m200/age',exist_ok=True)

	ajust_age_m200_geral,cov_age_m200_geral=np.polyfit(m200_casjobs,age,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,age,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_age_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_age_m200_geral[0]:.3f}$\pm${np.sqrt(cov_age_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_age_m200_geral[1]:.3f}$\pm${np.sqrt(cov_age_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\tau \ (Gyr)$')
	plt.savefig(f'{save_path}/analise_m200/age/m200_age.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=age,lim_region_casjobs,lim_vec_casjobs,config,r'$\tau \ (Gyr)$',('age',f'{faixas_label}'),'age'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/age/{pasta_save}',exist_ok=True)
		medias,sems=[],[]

		age_linspace=np.linspace(min(age),max(age),3000)
		_,_,par_labels,save_labels=par_stripe_entry=age,age_linspace,('Idade estelar',r'$\tau \ (Gyr)$'),('age','age')

		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)
		par_only_tracer_plots(medias,sems,par_labels,save_labels)

	plots_entry=age,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$\tau \ (Gyr)$','age','age')
	param_plots(plots_entry)

	#############################
	# conc
	os.makedirs(f'{save_path}/analise_m200/conc',exist_ok=True)

	ajust_conc_m200_geral,cov_conc_m200_geral=np.polyfit(m200_casjobs,conc,1,cov=True)
	y_lim=(1.9,4.2)
	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,conc,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_conc_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_conc_m200_geral[0]:.3f}$\pm${np.sqrt(cov_conc_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_conc_m200_geral[1]:.3f}$\pm${np.sqrt(cov_conc_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	plt.ylim(y_lim)
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$R_{90}/R_{50}$')
	plt.savefig(f'{save_path}/analise_m200/conc/m200_conc.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=conc,lim_region_casjobs,lim_vec_casjobs,config,r'$R_{90}/R_{50}$',('conc',f'{faixas_label}'),'conc'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/conc/{pasta_save}',exist_ok=True)
		conc_linspace=np.linspace(min(conc),4,3000)
		_,_,par_labels,save_labels=par_stripe_entry=conc,conc_linspace,('Concentração R90/R50',r'$C_{90,50}$'),('conc','conc')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=conc,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$C_{90,50}$','conc','conc')
	param_plots(plots_entry)
	#################################################
	#STARMASS
	os.makedirs(f'{save_path}/analise_m200/starmass',exist_ok=True)

	ajust_starmass_m200_geral,cov_starmass_m200_geral=np.polyfit(m200_casjobs,starmass,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,starmass,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_starmass_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_starmass_m200_geral[0]:.3f}$\pm${np.sqrt(cov_starmass_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_starmass_m200_geral[1]:.3f}$\pm${np.sqrt(cov_starmass_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\log M_{\bigstar} \ (M_\odot)$')
	plt.savefig(f'{save_path}/analise_m200/starmass/m200_starmass.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=starmass,lim_region_casjobs,lim_vec_casjobs,config,r'$\log M_{\bigstar} \ (M_\odot)$',('starmass',f'{faixas_label}'),'starmass'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/starmass/{pasta_save}',exist_ok=True)
		starmass_linspace=np.linspace(10.8,max(starmass),3000)
		_,_,par_labels,save_labels=par_stripe_entry=starmass,starmass_linspace,('Massa Estelar',r'$\log M_{\bigstar} \ (M_\odot)$'),('starmass','starmass')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=starmass,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$\log M_{\bigstar} \ (M_\odot)$','starmass','starmass')
	param_plots(plots_entry)

	#################################################
	#MAGABS	
	os.makedirs(f'{save_path}/analise_m200/magabs',exist_ok=True)

	ajust_magabs_m200_geral,cov_magabs_m200_geral=np.polyfit(m200_casjobs,magabs,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,magabs,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_magabs_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_magabs_m200_geral[0]:.3f}$\pm${np.sqrt(cov_magabs_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_magabs_m200_geral[1]:.3f}$\pm${np.sqrt(cov_magabs_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$Mag_{r,bol}$')
	plt.savefig(f'{save_path}/analise_m200/magabs/m200_magabs.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=magabs,lim_region_casjobs,lim_vec_casjobs,config,r'$Mag_{r,bol}$',('magabs',f'{faixas_label}'),'magabs'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/magabs/{pasta_save}',exist_ok=True)
		magabs_linspace=np.linspace(min(magabs),max(magabs),3000)
		_,_,par_labels,save_labels=par_stripe_entry=magabs,magabs_linspace,('Magnitude Absoluta',r'$Mag_{r,bol}$'),('magabs','magabs')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=magabs,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$Mag_{r,bol}$','magabs','magabs')
	param_plots(plots_entry)

	#################################################
	info_need=bt_vec_corr,starmass
	mass_c1,mass_c2=mass_calc(info_need)
	#################################################
	#MASS C1
	os.makedirs(f'{save_path}/analise_m200/mass_c1',exist_ok=True)

	ajust_mass_c1_m200_geral,cov_mass_c1_m200_geral=np.polyfit(m200_casjobs,mass_c1,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,mass_c1,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_mass_c1_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_mass_c1_m200_geral[0]:.3f}$\pm${np.sqrt(cov_mass_c1_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_mass_c1_m200_geral[1]:.3f}$\pm${np.sqrt(cov_mass_c1_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\log M_{1,\bigstar} \ (M_\odot)$')
	plt.savefig(f'{save_path}/analise_m200/mass_c1/m200_mass_c1.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=mass_c1,lim_region_casjobs,lim_vec_casjobs,config,r'$\log M_{1,\bigstar} \ (M_\odot)$',('mass_c1',f'{faixas_label}'),'mass_c1'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/mass_c1/{pasta_save}',exist_ok=True)
		mass_c1_linspace=np.linspace(min(mass_c1),max(mass_c1),3000)
		_,_,par_labels,save_labels=par_stripe_entry=mass_c1,mass_c1_linspace,('Massa Estelar Comp. 1',r'$\log M_{1,\bigstar} \ (M_\odot)$'),('mass_c1','mass_c1')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=mass_c1,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$\log M_{1,\bigstar} \ (M_\odot)$','mass_c1','mass_c1')
	param_plots(plots_entry)
	#################################################
	#MASS C2
	os.makedirs(f'{save_path}/analise_m200/mass_c2',exist_ok=True)

	ajust_mass_c2_m200_geral,cov_mass_c2_m200_geral=np.polyfit(m200_casjobs,mass_c2,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_casjobs,mass_c2,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_mass_c2_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_mass_c2_m200_geral[0]:.3f}$\pm${np.sqrt(cov_mass_c2_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_mass_c2_m200_geral[1]:.3f}$\pm${np.sqrt(cov_mass_c2_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$\log M_{2,\bigstar} \ (M_\odot)$')
	plt.savefig(f'{save_path}/analise_m200/mass_c2/m200_mass_c2.png')
	plt.close()

	####
	if flag == 'redshift':
		coeff_entry=mass_c2,lim_region_casjobs,lim_vec_casjobs,config,r'$\log M_{2,\bigstar} \ (M_\odot)$',('mass_c2',f'{faixas_label}'),'mass_c2'
		redshift_cut_loop(coeff_entry)
	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/mass_c2/{pasta_save}',exist_ok=True)
		mass_c2_linspace=np.linspace(min(mass_c2),max(mass_c2),3000)
		_,_,par_labels,save_labels=par_stripe_entry=mass_c2,mass_c2_linspace,('Massa Estelar Comp. 2',r'$\log M_{2,\bigstar} \ (M_\odot)$'),('mass_c2','mass_c2')

		medias,sems=[],[]
		for i,stripe in enumerate(stripes):
			bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_casjobs[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
			medias.append(bin_stats[0])
			sems.append(bin_stats[1])
		medias,sems=np.asarray(medias),np.asarray(sems)

		par_only_tracer_plots(medias,sems,par_labels,save_labels)
	plots_entry=mass_c2,lim_region_casjobs,cores,names_simples,region_names,l07_regions_casjobs,(r'$\log M_{2,\bigstar} \ (M_\odot)$','mass_c2','mass_c2')
	param_plots(plots_entry)

	#################################################
	#################################################
	#################################################
	#HALPHA
	if sample == 'WHL':
		m200_halpha=m200_temp[lim_halpha]
		m200_vet=m200_halpha
		m200_linspace=np.linspace(min(m200_halpha),max(m200_halpha),3000)

		os.makedirs(f'{save_path}/analise_m200/halpha',exist_ok=True)

		ajust_halpha_m200_geral,cov_halpha_m200_geral=np.polyfit(m200_halpha,h_line,1,cov=True)

		fig1=plt.figure(figsize=(9,7))
		plt.scatter(m200_halpha,h_line,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
		plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_halpha_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_halpha_m200_geral[0]:.3f}$\pm${np.sqrt(cov_halpha_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_halpha_m200_geral[1]:.3f}$\pm${np.sqrt(cov_halpha_m200_geral[1,1]):.3f}')
		xlim_m200=plt.xlim()
		y_lim=plt.ylim()
		y_lim=(y_lim[0],4.5)
		plt.legend()
		plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
		plt.ylabel(r'$H_\alpha$')
		plt.savefig(f'{save_path}/analise_m200/magabs/m200_halpha.png')
		plt.close()

		####
		if flag == 'redshift':
			coeff_entry=h_line,lim_region_halpha,lim_vec_halpha,config,r'$H_\alpha$',('halpha',f'{faixas_label}'),'halpha'
			redshift_cut_loop(coeff_entry)
		elif flag == 'm200':
			os.makedirs(f'{save_path}/analise_m200/halpha/{pasta_save}',exist_ok=True)
			halpha_linspace=np.linspace(min(h_line),4.5,3000)
			_,_,par_labels,save_labels=par_stripe_entry=h_line,halpha_linspace,('Linha do h_alpha',r'$H_\alpha$'),('halpha','halpha')

			medias,sems=[],[]
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_halpha[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)

			par_only_tracer_plots(medias,sems,par_labels,save_labels)
		plots_entry=h_line,lim_region_halpha,cores,names_simples,region_names,l07_regions_casjobs,(r'$H_\alpha$','halpha','halpha')
		param_plots(plots_entry)
	#################################################
	#################################################
	#################################################
	m200_cor_gr=m200_temp[lim_cor_gr]
	m200_vet=m200_cor_gr
	m200_linspace=np.linspace(min(m200_cor_gr),max(m200_cor_gr),3000)
	# COR INTEGRADA
	os.makedirs(f'{save_path}/analise_m200/gr_rest',exist_ok=True)

	ajust_gr_rest_m200_geral,cov_gr_rest_m200_geral=np.polyfit(m200_cor_gr,gr_rest,1,cov=True)

	fig1=plt.figure(figsize=(9,7))
	plt.scatter(m200_cor_gr,gr_rest,c='white',edgecolors='black',alpha=0.5,label=f'{sample}')
	plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_gr_rest_m200_geral),color='black',lw=2,label=fr'$\alpha$={ajust_gr_rest_m200_geral[0]:.3f}$\pm${np.sqrt(cov_gr_rest_m200_geral[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_gr_rest_m200_geral[1]:.3f}$\pm${np.sqrt(cov_gr_rest_m200_geral[1,1]):.3f}')
	xlim_m200=plt.xlim()
	y_lim=plt.ylim()
	plt.legend()
	plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
	plt.ylabel(r'$(g-r)_{rest}$')
	plt.savefig(f'{save_path}/analise_m200/gr_rest/m200_gr_rest.png')
	plt.close()
	####

	if flag == 'redshift':
		coeff_entry=gr_rest,lim_region_cor_gr,lim_vec_cor_gr,config,r'$(g-r)_{rest}$',('gr_rest',f'{faixas_label}'),'gr_rest'
		redshift_cut_loop(coeff_entry)

	elif flag == 'm200':
		os.makedirs(f'{save_path}/analise_m200/gr_rest/{pasta_save}',exist_ok=True)
		gr_rest_linspace=np.linspace(min(gr_rest),max(gr_rest),3000)
		_,_,par_labels,save_labels=par_stripe_entry=gr_rest,gr_rest_linspace,('g-r (z=0)',r'$g-r (z=0)$'),('gr_rest','gr_rest')

		medias,sems=[],[]
		if sample == 'WHL':
			for i,stripe in enumerate(stripes):
				bin_stats=par_only_kde_stripes(i,stripes_med[i],lim_vec_cor_gr[i],to_do_test,pasta_analise,pasta_save,par_stripe_entry)
				medias.append(bin_stats[0])
				sems.append(bin_stats[1])
			medias,sems=np.asarray(medias),np.asarray(sems)

	plots_entry=gr_rest,lim_region_cor_gr,cores,names_simples,region_names,l07_regions_cor_gr,(r'$g-r (z=0)$','gr_rest','gr_rest')
	param_plots(plots_entry)
	return
def m200_coeff_stripes(xy_mesh,stripe_entry,coeff_entry,m200_vet):
	"""Helper for `analise_m200`: for a single parameter in a single bin,
	compute the KDE peak and the linear fit parameters for the three
	morphological classes.

	Args:
		xy_mesh      (tuple): (X, Y) meshgrid for 2D KDE.
		stripe_entry (tuple): (stripe_index, lim_vec) where `lim_vec`
		is a list of three boolean masks (one per
		morphological class) for this bin.
		coeff_entry  (tuple): (coeff, config, coeff_label, coeff_dir, coeff_save)
		`coeff` is the parameter array for the current parent sample.
		m200_vet      (array): M200 (or binning variable) values for this
		parent sample.

	Returns:
		tuple: (x_peak_vec, y_peak_vec, alpha_values, alpha_incs,
		beta_values, beta_incs) – each a list of three values
		(one per morphological class).  If a class has too few
		objects, zeros are returned.
	"""

	os.makedirs(f'{save_path}/analise_m200/redshift_cut',exist_ok=True)
	####
	m200_linspace=np.linspace(min(m200_vet),max(m200_vet),3000)
	X,Y=xy_mesh

	stripe,lim_vec=stripe_entry
	coeff,config,coeff_label,coeff_dir,coeff_save=coeff_entry

	names_simples=config['names_simples']
	alpha_vec=config['alpha_vec']
	cores=config['cores']
	line_width=config['line_width']
	region_names=config['save_names']
	###
	alpha_values=[]
	beta_values=[]
	alpha_incs=[]
	beta_incs=[]
	x_peak_vec=[]
	y_peak_vec=[]
	for i in range(3):
		try:
			ajust_coeff_m200,cov_coeff_m200=np.polyfit(m200_vet[lim_vec[i]],coeff[lim_vec[i]],1,cov=True)
			m200_coeff_kde = kde(np.vstack([m200_vet[lim_vec[i]], coeff[lim_vec[i]]]))
			m200_coeff_Z = m200_coeff_kde(np.vstack([X.ravel(), Y.ravel()]))
			m200_coeff_Z = m200_coeff_Z.reshape(X.shape) 
			idx = np.unravel_index(np.argmax(m200_coeff_Z), m200_coeff_Z.shape)
			x_peak,y_peak = X[idx],Y[idx]

			alpha_values.append(ajust_coeff_m200[0])
			beta_values.append(ajust_coeff_m200[1])
			alpha_incs.append(cov_coeff_m200[0,0])
			beta_incs.append(cov_coeff_m200[1,1])
			x_peak_vec.append(x_peak)
			y_peak_vec.append(y_peak)

			plt.figure(figsize=(9,7))
			plt.scatter(m200_vet[lim_vec[i]],coeff[lim_vec[i]],c=cores[i],edgecolors='black',alpha=0.3,label=f'{names_simples[i]}')
			plt.plot(m200_linspace,linfunc(m200_linspace,*ajust_coeff_m200),color='black',lw=2,label=fr'$\alpha$={ajust_coeff_m200[0]:.3f}$\pm${np.sqrt(cov_coeff_m200[0,0]):.3f}'+'\n'+fr'$\beta$={ajust_coeff_m200[1]:.3f}$\pm${np.sqrt(cov_coeff_m200[1,1]):.3f}')
			plt.contour(X, Y, m200_coeff_Z,cmap='Reds',levels=40)
			plt.scatter(x_peak, y_peak,marker='*', color='white',edgecolors='black', s=10)
			plt.xlabel(r'$\log M_{200} \ (M_\odot)$')
			plt.ylabel(f'{coeff_label}')
			plt.legend()
			plt.savefig(f'{save_path}/analise_m200/{coeff_dir[0]}/{coeff_dir[1]}/m200_{coeff_save}_z_{region_names[i]}_{stripe}.png')
			plt.close()
		except:
			alpha_values.append(0)
			alpha_incs.append(0)
			beta_values.append(0)
			beta_incs.append(0)
			x_peak_vec.append(0)
			y_peak_vec.append(0)
			plt.figure(figsize=(9,7))
			plt.scatter([],[],label='DADOS INSUFICIENTES')
			plt.legend()
			plt.savefig(f'{save_path}/analise_m200/{coeff_dir[0]}/{coeff_dir[1]}/m200_{coeff_save}_z_{region_names[i]}_{stripe}.png')
			plt.close()
	return x_peak_vec,y_peak_vec,alpha_values,alpha_incs,beta_values,beta_incs
###################################################################################
###################################################################################
#DEFINIÇÕES DOS LIMITES E DAS SUB-AMOSTRAS
def lim_sample_builder():
	"""Read all input data files, apply initial quality cuts, and return
	the fundamental boolean masks.

	The returned masks indicate which objects survive the basic quality
	filters (finite effective radii, reasonable Sérsic indices, etc.) and
	which objects have additional data from CASJOBS, Hα, velocity
	dispersion, Photutils, and galaxy‑g‑r colour.

	Returns:
		tuple: (lim_geral, lim_casjobs, lim_halpha, lim_veldisp,lim_photutils, lim_cor_gr)
	"""

	#HEADERS
	header_data=np.loadtxt(f'profit_observation.header',dtype=str)
	header_mue=['cluster','mue_med_s','mue_med_1','mue_med_2']
	header_cor_gr=['cluster','modelMag_g','modelMag_r','extinction_g','extinction_r','kcorrG','kcorrG01','kcorrR','kcorrR01']
	header_casjobs=['cluster','magabs','logmass','age','metal','conc']
	header_halpha=['cluster','halpha','halpha_err']
	header_veldisp=['cluster','veldisp','veldisp_err']
	header_photutils=np.loadtxt(f'graph_stats_WHL_sky.header',dtype=str)
	#DATA FILES
	data_cor_gr_temp=np.loadtxt(f'{sample}_files/{sample}_cor_gr.dat',dtype=str).T
	data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky.dat',dtype=str).T
	data_mue_comp_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_comp_{sample}.dat',dtype=str).T
	data_casjobs_temp=np.loadtxt(f'{sample}_files/casjobs_data_clean_{sample}.dat',dtype=str).T
	data_halpha_temp=np.loadtxt(f'{sample}_files/casjobs_halpha_{sample}.dat',dtype=str).T
	data_veldisp_temp=np.loadtxt(f'{sample}_files/casjobs_veldisp_{sample}.dat',dtype=str).T
	data_photutils_temp=np.loadtxt(f'{sample}_files/graph_stats_{sample}_sky_v2.dat',dtype=str).T
	###############
	data_obs=dict(zip(header_data,data_obs_temp))
	data_casjobs=dict(zip(header_casjobs,data_casjobs_temp))
	data_halpha=dict(zip(header_halpha,data_halpha_temp))
	data_veldisp=dict(zip(header_veldisp,data_veldisp_temp))
	data_photutils=dict(zip(header_photutils,data_photutils_temp))
	data_mue_comp=dict(zip(header_mue,data_mue_comp_temp))
	data_cor_gr=dict(zip(header_cor_gr,data_cor_gr_temp))

	mue_med_s,mue_med_comp_1,mue_med_comp_2=data_mue_comp['mue_med_s'].astype(float),data_mue_comp['mue_med_1'].astype(float),data_mue_comp['mue_med_2'].astype(float)
	re1,re2=data_obs['RE_1'].astype(float),data_obs['RE_2'].astype(float)
	n1,n2=data_obs['NSER_1'].astype(float),data_obs['NSER_2'].astype(float)

	lim_finite=np.isfinite(mue_med_s) & np.isfinite(mue_med_comp_1) & np.isfinite(mue_med_comp_2)

	n1_lim=(n1==0.5) | (n1>9.9)
	n2_lim=(n2==0.5) | (n2>14.9)

	re1_lim=(re1<=1.05)
	re2_lim=(re2<=1.05)

	cut_lim=~(n1_lim | n2_lim | re1_lim | re2_lim)

	magabs=data_casjobs['magabs'].astype(float)[lim_finite & cut_lim]
	h_line=data_halpha['halpha'].astype(float)[lim_finite & cut_lim]
	vel_disp=data_veldisp['veldisp'].astype(float)[lim_finite & cut_lim]
	slope_gr=data_photutils['slope_gr_fix_log'].astype(float)[lim_finite & cut_lim]
	model_mag_g=data_cor_gr['modelMag_g'].astype(float)[lim_finite & cut_lim]

	lim_casjobs=magabs!=0
	lim_halpha=h_line!=0
	lim_veldisp=vel_disp!=0
	lim_photutils=np.isfinite(slope_gr)
	lim_cor_gr=model_mag_g!=0
	
	lim_vec=(lim_finite & cut_lim),lim_casjobs,lim_halpha,lim_veldisp,lim_photutils,lim_cor_gr
	return lim_vec
def lim_analise_builder(mode):
	"""Build the set of morphological sub‑sample masks for a given
	analysis mode.

	Different modes define different partitions of the RFF‑η plane or
	different selection criteria based on ΔBIC.  The function returns a
	configuration dictionary containing the masks, a base `save_path`,
	and the visual styling elements (names, colours, line widths, etc.).

	Args:
		mode (str): One of:
			'delta_bic'              : split using ΔBIC threshold.
			'rff_eta_mestrado'       : split using the master's thesis RFF‑η divisions (high/low_left/low_right).
			'rff_eta_lines_resid'    : split using lines of constant residual fraction (top/mid/bot).
			'rff_eta_lines_resid_2C' : same, but restricted to cD (2C) galaxies.
			'rff_eta_lines_resid_elip', 'rff_eta_lines_resid_big', 'rff_eta_lines_resid_small', etc.
	Returns:
		dict: Configuration with keys:
			'lim_region'     : tuple of three masks (main sample).
			'lim_region_casjobs', 'lim_region_halpha', …
			'names_simples'  : list of three class labels.
			'cores', 'alpha_vec', 'line_width', 'save_names',
			'save_path'      : string, root directory for output.
	"""
	pov=[650.88187589,1177.47577485]
	bic_sersic_obs=data_obs['BIC_s'].astype(float)
	redshift=data_z['redshift'].astype(float)
	psf_fwhm=data_psf['psf_fwhm'].astype(float)
	re1=data_obs['RE_1'].astype(float)
	config={}
	if mode == 'delta_bic':
		cd_lim=delta_bic_obs < log_model(redshift,*pov)
		elip_lim= ~cd_lim
		lim_cd_small=cd_lim & (np.divide(re1,psf_fwhm)<=1.)
		lim_cd_big=cd_lim & (np.divide(re1,psf_fwhm)>1.)
		titulo=' '

		cd_lim_casjobs=cd_lim[lim_casjobs]
		cd_lim_casjobs_small=lim_cd_small[lim_casjobs]
		cd_lim_casjobs_big=lim_cd_big[lim_casjobs]
		elip_lim_casjobs=elip_lim[lim_casjobs]

		cd_lim_halpha=cd_lim[lim_halpha]
		cd_lim_halpha_small=lim_cd_small[lim_halpha]
		cd_lim_halpha_big=lim_cd_big[lim_halpha]
		elip_lim_halpha=elip_lim[lim_halpha]

		cd_lim_veldisp=cd_lim[lim_veldisp]
		cd_lim_veldisp_small=lim_cd_small[lim_veldisp]
		cd_lim_veldisp_big=lim_cd_big[lim_veldisp]
		elip_lim_veldisp=elip_lim[lim_veldisp]

		cd_lim_photutils=cd_lim[lim_photutils]
		cd_lim_photutils_small=lim_cd_small[lim_photutils]
		cd_lim_photutils_big=lim_cd_big[lim_photutils]
		elip_lim_photutils=elip_lim[lim_photutils]

		cd_lim_cor_gr=cd_lim[lim_cor_gr]
		cd_lim_cor_gr_small=lim_cd_small[lim_cor_gr]
		cd_lim_cor_gr_big=lim_cd_big[lim_cor_gr]
		elip_lim_cor_gr=elip_lim[lim_cor_gr]
		config.update(
			lim_region=(cd_lim,lim_cd_small,lim_cd_big,elip_lim),
			lim_region_casjobs=(cd_lim_casjobs,cd_lim_casjobs_small,cd_lim_casjobs_big,elip_lim_casjobs),
			lim_region_halpha=(cd_lim_halpha,cd_lim_halpha_small,cd_lim_halpha_big,elip_lim_halpha),
			lim_region_veldisp=(cd_lim_veldisp,cd_lim_veldisp_small,cd_lim_veldisp_big,elip_lim_veldisp),
			lim_region_photutils=(cd_lim_photutils,cd_lim_photutils_small,cd_lim_photutils_big,elip_lim_photutils),
			lim_region_cor_gr=(cd_lim_cor_gr,cd_lim_cor_gr_small,cd_lim_cor_gr_big,elip_lim_cor_gr),
			names_simples=['2C','E(EL)','cD','E'],
			cores=['black','blue','red','green'],
			line_width=[4,1,1,1],
			alpha_vec=[0.4,1,1,1],
			save_path=f'{sample}_stats_observation/geral'
		)
	if mode == 'rff_eta_mestrado':
		lim_rff_eta_high=(eta/rff_s > 0.5) & (rff_s >= (x0[idx_split_rff][-1]))
		lim_rff_eta_low_left=rff_s < (x0[idx_split_rff][-1])
		lim_rff_eta_low_right=(eta/rff_s < 0.5) & (rff_s >= (x0[idx_split_rff][-1]))

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_low_left_casjobs=lim_rff_eta_low_left[lim_casjobs]
		lim_rff_eta_low_right_casjobs=lim_rff_eta_low_right[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_low_left_halpha=lim_rff_eta_low_left[lim_halpha]
		lim_rff_eta_low_right_halpha=lim_rff_eta_low_right[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_low_left_veldisp=lim_rff_eta_low_left[lim_veldisp]
		lim_rff_eta_low_right_veldisp=lim_rff_eta_low_right[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_low_left_photutils=lim_rff_eta_low_left[lim_photutils]
		lim_rff_eta_low_right_photutils=lim_rff_eta_low_right[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_low_left_cor_gr=lim_rff_eta_low_left[lim_cor_gr]
		lim_rff_eta_low_right_cor_gr=lim_rff_eta_low_right[lim_cor_gr]
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_low_left,lim_rff_eta_low_right),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_low_left_casjobs,lim_rff_eta_low_right_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_low_left_halpha,lim_rff_eta_low_right_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_low_left_veldisp,lim_rff_eta_low_right_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_low_left_photutils,lim_rff_eta_low_right_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_low_left_cor_gr,lim_rff_eta_low_right_cor_gr),
			names_simples=[r'$R_{sim}$',r'$R_{low}$',r'$R_{asy}$'],
			cores=['blue','green','red'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_mestrado/',
			save_names=('high','low_left','low_right')
		)
	if mode == 'rff_eta_lines_resid':
		lim_eq=1-eta/(rff_s)
		lim_rff_eta_high=(lim_eq < 0.1)
		lim_rff_eta_middle=(lim_eq > 0.1) & (lim_eq < 0.5)
		lim_rff_eta_low=(lim_eq > 0.5)

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
			names_simples=[r'$R_{top}$',r'$R_{mid}$',r'$R_{bot}$'],
			cores=['indigo','hotpink','lime'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_lines_resid/',
			save_names=('top','mid','bot')
		)
	if mode == 'rff_eta_lines_resid_2C':
		lim_eq=1-eta/(rff_s)
		cd_lim=delta_bic_obs < log_model(redshift,*pov)

		lim_rff_eta_high=(lim_eq < 0.1) & cd_lim
		lim_rff_eta_middle=(lim_eq > 0.1) & (lim_eq < 0.5) & cd_lim
		lim_rff_eta_low=(lim_eq > 0.5) & cd_lim

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
			names_simples=[r'$R_{top}[2C]$',r'$R_{mid}[2C]$',r'$R_{bot}[2C]$'],
			cores=['indigo','hotpink','lime'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_lines_resid_2C/',
			save_names=('top_2c','mid_2c','bot_2c')
		)
	if mode == 'rff_eta_lines_resid_elip':
		lim_eq=1-eta/(rff_s)
		cd_lim=delta_bic_obs < log_model(redshift,*pov)
		elip_lim=~cd_lim

		lim_rff_eta_high=(lim_eq < 0.1) & elip_lim
		lim_rff_eta_middle=(lim_eq > 0.1) & (lim_eq < 0.5) & elip_lim
		lim_rff_eta_low=(lim_eq > 0.5) & elip_lim

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
			names_simples=[r'$R_{top}[E]$',r'$R_{mid}[E]$',r'$R_{bot}[E]$'],
			cores=['indigo','hotpink','lime'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_lines_resid_elip/',
			save_names=('top_elip','mid_elip','bot_elip')
		)
	if mode == 'rff_eta_lines_resid_big':
		lim_eq=1-eta/(rff_s)
		cd_lim=delta_bic_obs < log_model(redshift,*pov)
		lim_cd_big=cd_lim & (np.divide(re1,psf_fwhm)>1.)

		lim_rff_eta_high=(lim_eq < 0.1) & lim_cd_big
		lim_rff_eta_middle=(lim_eq > 0.1) & (lim_eq < 0.5) & lim_cd_big
		lim_rff_eta_low=(lim_eq > 0.5) & lim_cd_big

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
			names_simples=[r'$R_{top}[cD]$',r'$R_{mid}[cD]$',r'$R_{bot}[cD]$'],
			cores=['indigo','hotpink','lime'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_lines_resid_big/',
			save_names=('top_cd','mid_cd','bot_cd')
		)
	if mode == 'rff_eta_lines_resid_small':
		lim_eq=1-eta/(rff_s)
		cd_lim=delta_bic_obs < log_model(redshift,*pov)
		lim_cd_small=cd_lim & (np.divide(re1,psf_fwhm)<=1.)

		lim_rff_eta_high=(lim_eq < 0.1) & lim_cd_small
		lim_rff_eta_middle=(lim_eq > 0.1) & (lim_eq < 0.5) & lim_cd_small
		lim_rff_eta_low=(lim_eq > 0.5) & lim_cd_small

		lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		config.update(
			lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
			lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
			lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
			lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
			lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
			lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
			names_simples=[r'$R_{top}[E(EL)]$',r'$R_{mid}[E(EL)]$',r'$R_{bot}[E(EL)]$'],
			cores=['indigo','hotpink','lime'],
			line_width=[1,1,1],
			alpha_vec=[1,1,1],
			save_path=f'{sample}_stats_observation/rff_eta_lines_resid_small/',
			save_names=('top_eel','mid_eel','bot_eel')
		)
	if mode == 'rff_eta_lines_resid_multi':
		lim_eq=1-eta/(np.absolute(rff_s))
		lim_rff_eta_high=(lim_eq < 0.0)
		lim_rff_eta_middle0=(lim_eq > 0.05) & (lim_eq < 0.1)
		lim_rff_eta_middle1=(lim_eq > 0.1) & (lim_eq < 0.2)
		lim_rff_eta_middle2=(lim_eq > 0.2) & (lim_eq < 0.3)
		lim_rff_eta_middle3=(lim_eq > 0.3) & (lim_eq < 0.4)
		lim_rff_eta_middle4=(lim_eq > 0.4) & (lim_eq < 0.5)
		lim_rff_eta_middle5=(lim_eq > 0.5) & (lim_eq < 0.6)
		lim_rff_eta_low=(lim_eq > 0.6)

		print(np.sum(lim_rff_eta_high))
		print(np.sum(lim_rff_eta_middle0))
		print(np.sum(lim_rff_eta_middle1))
		print(np.sum(lim_rff_eta_middle2))
		print(np.sum(lim_rff_eta_middle3))
		print(np.sum(lim_rff_eta_middle4))
		print(np.sum(lim_rff_eta_middle5))
		print(np.sum(lim_rff_eta_low))

		# lim_rff_eta_high_casjobs=lim_rff_eta_high[lim_casjobs]
		# lim_rff_eta_middle_casjobs=lim_rff_eta_middle[lim_casjobs]
		# lim_rff_eta_low_casjobs=lim_rff_eta_low[lim_casjobs]

		# lim_rff_eta_high_halpha=lim_rff_eta_high[lim_halpha]
		# lim_rff_eta_middle_halpha=lim_rff_eta_middle[lim_halpha]
		# lim_rff_eta_low_halpha=lim_rff_eta_low[lim_halpha]

		# lim_rff_eta_high_veldisp=lim_rff_eta_high[lim_veldisp]
		# lim_rff_eta_middle_veldisp=lim_rff_eta_middle[lim_veldisp]
		# lim_rff_eta_low_veldisp=lim_rff_eta_low[lim_veldisp]

		# lim_rff_eta_high_photutils=lim_rff_eta_high[lim_photutils]
		# lim_rff_eta_middle_photutils=lim_rff_eta_middle[lim_photutils]
		# lim_rff_eta_low_photutils=lim_rff_eta_low[lim_photutils]

		# lim_rff_eta_high_cor_gr=lim_rff_eta_high[lim_cor_gr]
		# lim_rff_eta_middle_cor_gr=lim_rff_eta_middle[lim_cor_gr]
		# lim_rff_eta_low_cor_gr=lim_rff_eta_low[lim_cor_gr]
		
		# config.update(
		# 	lim_region=(lim_rff_eta_high,lim_rff_eta_middle,lim_rff_eta_low),
		# 	lim_region_casjobs=(lim_rff_eta_high_casjobs,lim_rff_eta_middle_casjobs,lim_rff_eta_low_casjobs),
		# 	lim_region_halpha=(lim_rff_eta_high_halpha,lim_rff_eta_middle_halpha,lim_rff_eta_low_halpha),
		# 	lim_region_veldisp=(lim_rff_eta_high_veldisp,lim_rff_eta_middle_veldisp,lim_rff_eta_low_veldisp),
		# 	lim_region_photutils=(lim_rff_eta_high_photutils,lim_rff_eta_middle_photutils,lim_rff_eta_low_photutils),
		# 	lim_region_cor_gr=(lim_rff_eta_high_cor_gr,lim_rff_eta_middle_cor_gr,lim_rff_eta_low_cor_gr),
		# 	names_simples=[r'$R_{top}$',r'$R_{mid}$',r'$R_{bot}$'],
		# 	cores=['indigo','hotpink','lime'],
		# 	line_width=[1,1,1],
		# 	alpha_vec=[1,1,1],
		# 	save_path=f'{sample}_stats_observation/rff_eta_lines_resid/',
		# 	save_names=('top','mid','bot')
		# )

	return config 
###################################################################################

if __name__ == '__main__':
	import sys
	sample=str(sys.argv[1])
	os.makedirs(f'{sample}_stats_observation',exist_ok=True)
	############################################################
	############################################################
	#LIMITES AMOSTRAIS GERAIS
	lim_geral,lim_casjobs,lim_halpha,lim_veldisp,lim_photutils,lim_cor_gr=lim_sample_builder()
	############################################################
	#BLOCO DE ABERTURA DOS DADOS
	#HEADERS
	header_data_z=['cluster','redshift','morfologia']
	header_eta=['cluster','assimetria']
	header_data=np.loadtxt(f'profit_observation.header',dtype=str)
	header_mue=['cluster','mue_med_s','mue_med_1','mue_med_2']
	header_psf=['cluster','npix','psf_fwhm']
	header_chi2=['cluster','chi2_s','chi2_ss']
	header_casjobs=['cluster','magabs','logmass','age','metal','conc']
	header_halpha=['cluster','halpha','halpha_err']
	header_veldisp=['cluster','veldisp','veldisp_err']
	header_photutils=np.loadtxt(f'graph_stats_WHL_sky.header',dtype=str)
	header_erro_photutils=['cluster','grad_e_err','grad_pa_err','slope_a3_err','slope_a4_err','slope_b3_err','slope_b4_err','slope_disk_err','slope_gr_fix_log_err','slope_gr_fix_err','slope_gr_free_log_err','slope_gr_free_err']
	header_kron=['cluster','kron_r']
	header_corr=['cluster','corr_1','corr_2']
	header_cor_gr=['cluster','modelMag_g','modelMag_r','extinction_g','extinction_r','kcorrG','kcorrG01','kcorrR','kcorrR01']
	#DATA FILES
	data_cor_gr_temp=np.loadtxt(f'{sample}_files/{sample}_cor_gr.dat',dtype=str)[lim_geral].T
	data_chi2_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky_chi2.dat',dtype=str)[lim_geral].T
	data_psf_temp=np.loadtxt(f'{sample}_files/{sample}_psf_data.dat',dtype=str)[lim_geral].T
	data_z_temp=np.loadtxt(f'{sample}_files/{sample}_clean_redshift.dat',dtype=str)[lim_geral].T
	data_eta_temp=np.loadtxt(f'{sample}_files/ass_{sample}.dat')[lim_geral].T
	data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky.dat',dtype=str)[lim_geral].T
	data_mue_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_{sample}.dat',dtype=str)[lim_geral].T
	data_mue_comp_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_comp_{sample}.dat',dtype=str)[lim_geral].T
	data_casjobs_temp=np.loadtxt(f'{sample}_files/casjobs_data_clean_{sample}.dat',dtype=str)[lim_geral].T
	data_halpha_temp=np.loadtxt(f'{sample}_files/casjobs_halpha_{sample}.dat',dtype=str)[lim_geral].T
	data_veldisp_temp=np.loadtxt(f'{sample}_files/casjobs_veldisp_{sample}.dat',dtype=str)[lim_geral].T
	data_photutils_temp=np.loadtxt(f'{sample}_files/graph_stats_{sample}_sky_v2.dat',dtype=str)[lim_geral].T
	data_erro_photutils_temp=np.loadtxt(f'{sample}_files/graph_errors_{sample}_sky.dat',dtype=str)[lim_geral].T
	data_200_temp=np.loadtxt(f'{sample}_files/{sample}_r200.dat',dtype=str)[lim_geral].T
	data_corr_temp=np.loadtxt(f'{sample}_files/{sample}_corr_bt_v2.dat',dtype=str)[lim_geral].T
	data_kron_temp=np.loadtxt(f'{sample}_files/kron_radius_{sample}_v2.dat')[lim_geral].T
	###############
	data_z=dict(zip(header_data_z,data_z_temp))
	data_eta=dict(zip(header_eta,data_eta_temp))
	data_obs=dict(zip(header_data,data_obs_temp))
	data_psf=dict(zip(header_psf,data_psf_temp))
	data_chi2=dict(zip(header_chi2,data_chi2_temp))
	data_casjobs=dict(zip(header_casjobs,data_casjobs_temp))
	data_halpha=dict(zip(header_halpha,data_halpha_temp))
	data_veldisp=dict(zip(header_veldisp,data_veldisp_temp))
	data_photutils=dict(zip(header_photutils,data_photutils_temp))
	data_mue=dict(zip(header_mue,data_mue_temp))
	data_mue_comp=dict(zip(header_mue,data_mue_comp_temp))
	data_erro_photutils=dict(zip(header_erro_photutils,data_erro_photutils_temp))
	data_corr=dict(zip(header_corr,data_corr_temp))
	data_kron=dict(zip(header_kron,data_kron_temp))
	data_cor_gr=dict(zip(header_cor_gr,data_cor_gr_temp))
	rff_ss=np.loadtxt(f'{sample}_files/{sample}_rff_duplo.dat',dtype=float,usecols=[1])[lim_geral].T

	############################################################
	############################################################

	cluster=data_obs['cluster']
	rff_s=data_obs['rff'].astype(float)
	ass=data_eta['assimetria'].astype(float)

	eta=rff_s-np.absolute(ass)

	limx=[-2.5,-0.5]
	limy=[-0.02,0.1]
	label_x=r'$\log\,RFF$'
	label_y=r'$\eta$'


	#RFF - ETA / PONTOS COLORIDOS
	fig1=plt.figure(figsize=(9,7))
	x=np.linspace(0.001,0.3,1000)
	y=np.linspace(0.001,0.3,1000)
	vec=[0.1,0.5]
	for item in vec:
		plt.plot(np.log10(x),x-item*x,label=str(item))
	# plt.scatter(np.log10(rff_s),eta,c='white',edgecolors='black',alpha=0.2,label='Amostra')
	plt.scatter(np.log10(rff_s[ass<0]),eta[ass<0],c='red',edgecolors='black',label='Assimetria < 0')
	plt.legend()
	plt.xlim(limx)
	plt.ylim(limy)
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.show()
	plt.close()


	rff_ratio=np.divide(rff_ss,rff_s)
	#####
	bic_sersic_obs=data_obs['BIC_s'].astype(float)
	bic_sersic_duplo_obs=data_obs['BIC_ss'].astype(float)
	delta_bic_obs=bic_sersic_duplo_obs - bic_sersic_obs
	#####
	chi2_s=data_chi2['chi2_s'].astype(float)
	chi2_ss=data_chi2['chi2_ss'].astype(float)
	psf_fwhm=data_psf['psf_fwhm'].astype(float)
	chi2_ratio=chi2_s/chi2_ss

	redshift=data_z['redshift'].astype(float)
	sky=data_obs['SKY_s'].astype(float)
	raio_kron=data_kron['kron_r'].astype(float)

	mag1,mag2,mag_s=data_obs['MAG_1'].astype(float),data_obs['MAG_2'].astype(float),data_obs['MAG'].astype(float)
	re1,re2,re_s=data_obs['RE_1'].astype(float),data_obs['RE_2'].astype(float),data_obs['RE'].astype(float)
	n1,n2,n_s=data_obs['NSER_1'].astype(float),data_obs['NSER_2'].astype(float),data_obs['NSER'].astype(float)
	e1,e2,e_s=data_obs['AXRAT_1'].astype(float),data_obs['AXRAT_2'].astype(float),data_obs['AXRAT'].astype(float)
	box1,box2,box_s=data_obs['BOX_1'].astype(float),data_obs['BOX_2'].astype(float),data_obs['BOX'].astype(float)

	re_1_kpc=dist_pc(re1,redshift)
	re_2_kpc=dist_pc(re2,redshift)
	re_s_kpc=dist_pc(re_s,redshift)

	corr_1,corr_2=data_corr['corr_1'].astype(float),data_corr['corr_2'].astype(float)
	bt_vec_corr=bt_ratio_corr(mag1,mag2,corr_1,corr_2)

	mue_med_s,mue_med_comp_1,mue_med_comp_2=data_mue_comp['mue_med_s'].astype(float),data_mue_comp['mue_med_1'].astype(float),data_mue_comp['mue_med_2'].astype(float)

	mue_med_ratio_12=np.divide(mue_med_comp_1,mue_med_comp_2)
	mue_med_ratio_1s=np.divide(mue_med_comp_1,mue_med_s)
	mue_med_ratio_2s=np.divide(mue_med_comp_2,mue_med_s)

	re_ratio_12=np.divide(re1,re2)
	re_ratio_1s=np.divide(re1,re_s)
	re_ratio_2s=np.divide(re2,re_s)
		
	n_ratio_12=np.divide(n1,n2)
	n_ratio_1s=np.divide(n1,n_s)
	n_ratio_2s=np.divide(n2,n_s)
	
	axrat_ratio_12=np.divide(e1,e2)
	axrat_ratio_1s=np.divide(e1,e_s)
	axrat_ratio_2s=np.divide(e2,e_s)
	
	dbox=0.0001
	box_ratio_12=np.divide(box1,np.clip(box2,dbox,None))
	box_ratio_1s=np.divide(box1,np.clip(box_s,dbox,None))
	box_ratio_2s=np.divide(box2,np.clip(box_s,dbox,None))

	magabs_temp=data_casjobs['magabs'].astype(float)
	starmass_temp=data_casjobs['logmass'].astype(float)
	age_temp=data_casjobs['age'].astype(float)
	conc_temp=data_casjobs['conc'].astype(float)

	magabs=data_casjobs['magabs'].astype(float)[lim_casjobs]
	starmass=data_casjobs['logmass'].astype(float)[lim_casjobs]
	age=data_casjobs['age'].astype(float)[lim_casjobs]
	conc=data_casjobs['conc'].astype(float)[lim_casjobs]

	model_mag_g=data_cor_gr['modelMag_g'].astype(float)[lim_cor_gr]
	model_mag_r=data_cor_gr['modelMag_r'].astype(float)[lim_cor_gr]
	extinction_r=data_cor_gr['extinction_r'].astype(float)[lim_cor_gr]
	extinction_g=data_cor_gr['extinction_g'].astype(float)[lim_cor_gr]
	k_corr_g_0=data_cor_gr['kcorrG'].astype(float)[lim_cor_gr]
	k_corr_g_01=data_cor_gr['kcorrG01'].astype(float)[lim_cor_gr]
	k_corr_r_0=data_cor_gr['kcorrR'].astype(float)[lim_cor_gr]
	k_corr_r_01=data_cor_gr['kcorrR01'].astype(float)[lim_cor_gr]

	gr_rest=(model_mag_g - extinction_g - k_corr_g_0)-(model_mag_r - extinction_r - k_corr_r_0)
	gr_rest_01=(model_mag_g - extinction_g - k_corr_g_01)-(model_mag_r - extinction_r - k_corr_r_01)

	h_line=data_halpha['halpha'].astype(float)[lim_halpha]
	vel_disp=data_veldisp['veldisp'].astype(float)[lim_veldisp]

	grad_e=data_photutils['ellipgrad'].astype(float)[lim_photutils]
	grad_pa=np.abs(data_photutils['pagrad'].astype(float)[lim_photutils])
	chi2_s=data_photutils['chi_s'].astype(float)[lim_photutils]
	chi2_ss=data_photutils['chi_ss'].astype(float)[lim_photutils]

	med_a3=data_photutils['med_a3'].astype(float)[lim_photutils]
	med_a4=data_photutils['med_a4'].astype(float)[lim_photutils]
	med_b3=data_photutils['med_b3'].astype(float)[lim_photutils]
	med_b4=data_photutils['med_b4'].astype(float)[lim_photutils]
	med_disk=data_photutils['med_diskness'].astype(float)[lim_photutils]

	slope_a3=data_photutils['slope_slow_a3'].astype(float)[lim_photutils]
	slope_a4=data_photutils['slope_slow_a4'].astype(float)[lim_photutils]
	slope_b3=data_photutils['slope_slow_b3'].astype(float)[lim_photutils]
	slope_b4=data_photutils['slope_slow_b4'].astype(float)[lim_photutils]
	slope_disk=data_photutils['slope_slow_diskness'].astype(float)[lim_photutils]
	slope_gr=data_photutils['slope_gr_fix_log'].astype(float)[lim_photutils]

	pov=[650.88187589,1177.47577485]

	cmap=plt.colormaps['hot']
	cmap_r=plt.colormaps['hot_r']

	###################################################################################
	###################################################################################
	############################################################
	############################################################
	if sample == 'WHL':
		header_200=['cluster','r200','richness','n200']
		data_200=dict(zip(header_200,data_200_temp))
		ra,dec=np.loadtxt(f'data_indiv_WHL_clean.dat',dtype=float,usecols=[1,2]).T

		bins = np.arange(0,0.1,0.0005)
		dlpov=[1.54622724e-02,4.61969720e-03,1.23642039e+02,-3.93661732e+00,-4.81476825e-01,7.57926648e+01]
		converter=(7057*0.0005)
		x0=np.linspace(0,0.1,10000)
		idx_split_rff = np.argwhere(np.diff(np.sign(diff(x0,dlpov)))).flatten()

		r200=data_200['r200'].astype(float)
		cl_rich=data_200['richness'].astype(float)
		m200_temp=np.log10(np.power(10,12.51)*np.power(cl_rich,1.17))
	elif sample=='L07':
		header_200=['cluster','r200','vel']
		data_200=dict(zip(header_200,data_200_temp))
		ra,dec=np.loadtxt(f'L07_files/data_indiv_clean_L07.dat',dtype=float,usecols=[5,6]).T

		vel=data_200['vel'].astype(float)
		omega_0=0.3
		omega_v=0.7
		h100=0.7
		m200_temp=np.log10(12*np.power(vel/1000.,3.)*(1./(np.sqrt(omega_v+omega_0*np.power((1+redshift),3.))*h100)))

		bins = np.arange(0,0.1,0.0017)
		dlpov=[0.017835,0.005812,42.214771,-3.522754,0.328877,14.599428]
		converter=1

		tipo_morf=data_z['morfologia']
		e_cut=tipo_morf=='E'
		cd_cut=tipo_morf=='cD'
		ecd_cut=tipo_morf=='E/cD'
		cde_cut=tipo_morf=='cD/E'

		e_cut_casjobs=e_cut[lim_casjobs]
		cd_cut_casjobs=cd_cut[lim_casjobs]
		ecd_cut_casjobs=ecd_cut[lim_casjobs]
		cde_cut_casjobs=cde_cut[lim_casjobs]

		e_cut_photutils=e_cut[lim_photutils]
		cd_cut_photutils=cd_cut[lim_photutils]
		ecd_cut_photutils=ecd_cut[lim_photutils]
		cde_cut_photutils=cde_cut[lim_photutils]

		e_cut_halpha=e_cut[lim_halpha]
		cd_cut_halpha=cd_cut[lim_halpha]
		ecd_cut_halpha=ecd_cut[lim_halpha]
		cde_cut_halpha=cde_cut[lim_halpha]

		e_cut_veldisp=e_cut[lim_veldisp]
		cd_cut_veldisp=cd_cut[lim_veldisp]
		ecd_cut_veldisp=ecd_cut[lim_veldisp]
		cde_cut_veldisp=cde_cut[lim_veldisp]

		e_cut_cor_gr=e_cut[lim_cor_gr]
		cd_cut_cor_gr=cd_cut[lim_cor_gr]
		ecd_cut_cor_gr=ecd_cut[lim_cor_gr]
		cde_cut_cor_gr=cde_cut[lim_cor_gr]
		x0=np.linspace(0,0.1,10000)
		idx_split_rff = np.argwhere(np.diff(np.sign(diff(x0,dlpov)))).flatten()
	############################################################
	###################################################################################
	###################################################################################
	mode='rff_eta_lines_resid_multi'
	config=lim_analise_builder(mode)
	# save_path=config['save_path']
	# os.makedirs(f'{save_path}',exist_ok=True)
	# lim_region,lim_region_photutils,lim_region_casjobs,lim_region_halpha,lim_region_veldisp,lim_region_cor_gr=region_vec
	names_morf=['cD','E/cD & cD/E','E']
	#englobar os modes aqui, que vai facilitar minha vida, talvez passar eles como entrada seja interessante
	#REARRANJAR OS TERMOS ABAIXO PARA QUE ENGLOBE TODO E QUALQUER GRÁFICO
	'''
	names_simples=config['names_simples']
	cores=config['cores']
	line_width=config['line_width']
	alpha_vec=config['alpha_vec']
	# #ANALISE M200
	cut_data=m200_temp,5,'m200_5cuts',config,'m200'
	analise_m200(cut_data)
	cut_data=redshift,5,'redshift_5cuts',config,'redshift'
	analise_m200(cut_data)
	# ##############################
	#PLOTS KDE - HISTOGRAMAS - TESTE KS - VALORES MÉDIOS
	# cut_data=[],5,'redshift_5cuts',config,None
	cut_data=redshift,5,'redshift_5cuts',config,'redshift'
	plots_kde(cut_data)
	# # ##############################
	#INVESTIGAÇÃO DO KORMENDY
	kormendy_plots(config)
	# ##############################
	# #HISTOGRAMA DE RFF COM A LINHA DE CORTE
	rff_eta_plots(config)
	# # ##############################
	# #PLANO DE RAZÃO DE RAIOS (RE1/RE2) X N_2
	re_ratio_n2_plots(config)
	# ##############################
	# #PLANO DE N1 X NS
	# plano_n1_ns()
	# ##############################
	# #PLANO DE RAZÃO DE RAIOS (N1/N2) x N2
	n_ratio_n2_plots(config)
	'''
