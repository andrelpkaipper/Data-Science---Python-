import math
import pyprofit

from scipy import signal
from scipy import stats

import numpy as np
"""
profit_optim_v4.py

Model creation, likelihood evaluation, and data setup for the ``pyprofit``
Sérsic‑fitting pipeline.

This module provides:

- Functions to build single‑Sérsic and double‑Sérsic model images from
  parameter vectors.
- Routines to convert between the internal (normalised and possibly
  log‑scaled) parameter representation used by the optimiser and the
  physical parameters expected by ``pyprofit``.
- Likelihood functions based on the Student‑t distribution (robust
  against outliers) or the Gaussian distribution, including a “cleaned”
  likelihood that excludes a small central region.
- A convenience function ``profit_setup_data`` that packages all
  information needed for optimisation into a single ``Data`` object.

The module is designed to work with the ``profit``-based fitting workflow
and relies on the external ``pyprofit`` package for image rendering.
"""

class Data(object):
    """Simple container for profit optimisation data.

    Attributes are added dynamically by ``profit_setup_data``.  The
    following attributes are typically present after setup:

    - image, mask, sigim, psf: the data arrays.
    - magzero: magnitude zero‑point.
    - names: list of parameter names.
    - model0: initial physical parameter vector.
    - tofit: boolean array indicating which parameters are free.
    - tolog: boolean array indicating which parameters are fitted in
      log10 space.
    - sigmas: prior standard deviations.
    - priors: list of prior functions.
    - region: boolean mask defining the galaxy region.
    - calcregion: convolution region mask.
    - mask_center: additional central mask (e.g., 1 if masked).
    - init: initial parameter vector in the transformed (optimisation)
      space.
    - bounds: parameter bounds in the transformed space.
    - verbose: bool flag for printing debug information.
    - dof: estimated degrees of freedom for the Student‑t likelihood.
    - check_model: flag set when the likelihood evaluates to NaN.
    """


    pass
#######################################################
# ----------------------------------------------------------------------
# Model construction (for verification / initial guesses)
# ----------------------------------------------------------------------
#######################################################
def build_model_simples(allparams, data, use_mask=True,sky=True):
    """Build a single‑Sérsic model image using the current parameter vector.

    This function is mainly used for testing and initial evaluation;
    the main optimisation uses ``to_pyprofit_image_simples`` which
    handles the transformed parameter space.

    The parameter vector ``allparams`` is expected to contain 9 values
    in physical units: xcen, ycen, mag, re, nser, ang, axrat, box, sky.

    Args:
        allparams (array_like): Length‑9 array of physical parameters.
        data (Data): ``Data`` object containing ``image.shape``,
            ``magzero``, and ``psf``.
        use_mask (bool): If True, a ``calcmask`` of ones is added to the
            profit model, which is required when a bad‑pixel mask is
            present (not directly used here, passed to ``pyprofit``).
        sky (bool): If True, a sky component is included; if False, no
            sky is modelled.

    Returns:
        tuple: (sparams, modelim)
            - sparams (list of dict): The Sérsic profile parameters as
              a list of dictionaries, suitable for logging.
            - modelim (2D ndarray): The rendered model image.
    """


    fields = ['xcen','ycen','mag','re','nser','ang','axrat','box']
    sparams = [x for i,x in enumerate(allparams)]
    # if hasattr(data, 'psf') and len(data.psf) > 0:
    #     fields.append('convolve')
    #     sparams.append(True)
    
    sparams = [{name: val for name, val in zip(fields, sparams)}]
    sky_params=[{'bg':allparams[-1]}]
    if data.verbose:
        print(sparams)
    if sky:
        profit_model = {'width':  data.image.shape[1],
                        'height': data.image.shape[0],
                        'magzero': data.magzero,
                        'psf': data.psf,
                        'profiles': {'sersic': sparams,'sky':sky_params}
                       }
    elif sky==False:
        profit_model = {'width':  data.image.shape[1],
                        'height': data.image.shape[0],
                        'magzero': data.magzero,
                        'psf': data.psf,
                        'profiles': {'sersic': sparams}
                       }
        print(profit_model)
    if use_mask:
        profit_model['calcmask'] = np.ones_like(data.image)
    image, _ = pyprofit.make_model(profit_model)
    return sparams, np.array(image)
def build_model_duplo(shuffle_params, data, use_mask=True,sky=True):
    """Build a double‑Sérsic model image from a parameter vector.

    This function accepts a parameter vector ``shuffle_params`` of length
    15, ordered as:
    xc1, yc1, mag1, re1, n1, pa1, ax1, box1,
    mag2, re2, n2, pa2, ax2, box2, sky.
    Both components share the same centre (xc1, yc1).

    The function rearranges the parameters into two Sérsic profile
    dictionaries and passes them to ``pyprofit.make_model``.

    Args:
        shuffle_params (array_like): Length‑15 physical parameter array.
        data (Data): ``Data`` object with image dimensions, magzero, psf.
        use_mask (bool): If True, adds a ``calcmask`` of ones.
        sky (bool): If True, includes a sky component.

    Returns:
        tuple: (allparams, modelim)
            - allparams (ndarray): The parameter vector rearranged into
              the standard double‑Sérsic order used by ``to_pyprofit_image_duplo``.
            - modelim (2D ndarray): The model image.
    """



    xc1,yc1,mag1,re1,n1,pa1,ax1,box1,mag2,re2,n2,pa2,ax2,box2,sky=shuffle_params
    params=[xc1,xc1,yc1,yc1,mag1,mag2,re1,re2,n1,n2,pa1,pa2,ax1,ax2,box1,box2,sky]
    allparams=np.array(params)

    fields = ['xcen','ycen','mag','re','nser','ang','axrat','box']
    s1params = [x for i,x in enumerate(allparams[:-1]) if i%2 == 0]
    s2params = [x for i,x in enumerate(allparams[:-1]) if i%2 != 0]
    # if hasattr(data, 'psf') and len(data.psf) > 0:
    #     fields.append('convolve')
    #     s1params.append(True)
    #     s2params.append(True)
    ##
    sky_params=[{'bg':allparams[-1]}]
    sparams = [{name: val for name, val in zip(fields, params)} for params in (s1params, s2params)]
    if data.verbose:
        print(sparams)
    if sky:
        profit_model = {'width':  data.image.shape[1],
                        'height': data.image.shape[0],
                        'magzero': data.magzero,
                        'psf': data.psf,
                        'profiles': {'sersic': sparams,'sky':sky_params}
                       }
    elif sky == False:
        profit_model = {'width':  data.image.shape[1],
                'height': data.image.shape[0],
                'magzero': data.magzero,
                'psf': data.psf,
                'profiles': {'sersic': sparams}
               }
    print(profit_model)
    if use_mask:
        profit_model['calcmask'] = np.ones_like(data.image)
    image, _ = pyprofit.make_model(profit_model)
    return allparams, np.array(image)
############################################
# ----------------------------------------------------------------------
# Transformation between optimisation space and physical parameters
# ----------------------------------------------------------------------
############################################
def to_pyprofit_image_duplo(params, data, use_mask=True):
    """Convert double‑Sérsic parameters from optimisation space to physical
    space and render the model image.

    The input ``params`` is the vector of free parameters in the
    transformed space (normalised and possibly log‑transformed).
    The function reconstructs the full physical parameter vector by
    applying the inverse transformations stored in ``data``, then builds
    the model image using ``pyprofit``.

    Args:
        params (ndarray): Free parameters in the optimisation space,
            length = number of free parameters (typically 15 for a
            double‑Sérsic model with all parameters free).
        data (Data): Data object containing the full initial vector,
            transformation flags, and scaling factors.
        use_mask (bool): Passed to ``pyprofit`` to add a calcmask.

    Returns:
        tuple: (allparams, modelim)
            - allparams (ndarray): The full physical parameter vector.
            - modelim (2D ndarray): The model image.
    """


    sigmas_tofit = data.sigmas[data.tofit]
    allparams = data.model0.copy()
    allparams[data.tofit] = params * sigmas_tofit
    allparams[data.tolog] = 10**allparams[data.tolog]
   
    fields = ['xcen','ycen','mag','re','nser','ang','axrat','box']
    s1params = [x for i,x in enumerate(allparams[2:-1]) if i%2 == 0]
    s2params = [x for i,x in enumerate(allparams[2:-1]) if i%2 != 0]
    if hasattr(data, 'psf') and len(data.psf) > 0:
        fields.append('convolve')
        s1params.append(True)
        s2params.append(True)
    ##
    s1params.insert(0,allparams[1])
    s1params.insert(0,allparams[0])
    ##
    s2params.insert(0,allparams[1])
    s2params.insert(0,allparams[0])

    sky_params=[{'bg':allparams[-1]}]
    sparams = [{name: val for name, val in zip(fields, params)} for params in (s1params, s2params)]

    if data.verbose:
        print(sparams)

    profit_model = {'width':  data.image.shape[1],
                    'height': data.image.shape[0],
                    'magzero': data.magzero,
                    'psf': data.psf,
                    'profiles': {'sersic': sparams,'sky':sky_params}
                   }
    if use_mask:
        profit_model['calcmask'] = np.ones_like(data.image)
    image, _ = pyprofit.make_model(profit_model)
    return allparams, np.array(image)
def to_pyprofit_image_simples(params, data, use_mask=True):

    """Convert single‑Sérsic parameters from optimisation space to physical
    space and render the model image.

    Analogous to ``to_pyprofit_image_duplo`` but for the 9‑parameter
    single‑Sérsic case.

    Args:
        params (ndarray): Free parameters in optimisation space (length 9
            when all are free).
        data (Data): Data object.
        use_mask (bool): Passed to ``pyprofit``.

    Returns:
        tuple: (allparams, modelim)
    """


    sigmas_tofit = data.sigmas[data.tofit]
    allparams = data.model0.copy()
    allparams[data.tofit] = params * sigmas_tofit
    allparams[data.tolog] = 10**allparams[data.tolog]
    fields = ['xcen','ycen','mag','re','nser','ang','axrat','box']
    sparams = [x for i,x in enumerate(allparams[:-1])] 
    if hasattr(data, 'psf') and len(data.psf) > 0:
        fields.append('convolve')
        sparams.append(True)
    sparams = [{name: val for name, val in zip(fields, params)} for params in (sparams,)]
    sky_params=[{'bg':allparams[-1]}]
    #print(sparams)
    if data.verbose:
        print(sparams)

    profit_model = {'width':  data.image.shape[1],
                    'height': data.image.shape[0],
                    'magzero': data.magzero,
                    'psf': data.psf,
                    'profiles': {'sersic': sparams,'sky':sky_params}
                   }
    if use_mask:
        profit_model['calcmask'] = np.ones_like(data.image)
    image, _ = pyprofit.make_model(profit_model)
    return allparams, np.array(image)
#############################################
# ----------------------------------------------------------------------
# Likelihood functions
# ----------------------------------------------------------------------
#############################################
def profit_like_model_original(params, data):

    """(Legacy) Student‑t log‑likelihood with explicit prior term.

    This function is kept for reference; the current pipeline typically
    uses ``profit_like_model``.  It evaluates a Student‑t log‑likelihood
    with a variable degrees‑of‑freedom parameter estimated from the
    variance of the normalised residuals, and adds a Gaussian prior
    computed from ``data.priors``.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object.

    Returns:
        float: Negative log‑posterior (to be minimised).

    Raises:
        ValueError: If the objective function evaluates to NaN.
    """



    # Get the priors sum
    priorsum = 0
    sigmas_tofit = data.sigmas[data.tofit]
    for i, p in enumerate(data.priors):
        priorsum += p(data.init[i] - params[i]*sigmas_tofit[i])
    # Calculate the new model
    if len(params) == 9:
        allparams, modelim = to_pyprofit_image_simples(params, data)
    elif len(params) != 9:
        allparams, modelim = to_pyprofit_image_duplo(params, data)
    # Scale and stuff
    scaledata = (data.image[data.region] - modelim[data.region])/data.sigim[data.region]
    variance = scaledata.var()
    if variance > 1:
        dof = 2*variance/(variance-1)
        dof = max(min(dof,float('inf')),0)
    #print(dof,variance,scaledata)
    else:
        dof=1000
    ll = np.sum(stats.t.logpdf(scaledata, dof))
    lp = ll #+ priorsum
    lp = -lp

    #print(ll,lt)

    if np.isnan(lp):
        print('NaN deu')
        raise ValueError('Função objetiva deu NaN')
    if data.verbose:
        print(lp, {name: val for name, val in zip(data.names, allparams)})
    return lp
def profit_like_model_gaussiano(params, data):
    """Gaussian log‑likelihood (simple χ²) for the profit model.

    This likelihood assumes that the residuals are normally distributed
    with the provided sigma map.  It is faster and more stable than the
    Student‑t version, but less robust to outliers.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object.

    Returns:
        float: Negative log‑likelihood (half the reduced χ² up to a
            constant).
    """

    if len(params) == 9:
        allparams, modelim = to_pyprofit_image_simples(params, data)
    elif len(params) != 9:
        allparams, modelim = to_pyprofit_image_duplo(params, data)

    residuals=data.image[data.region]-modelim[data.region]
    scale=data.sigim[data.region]

    ll=np.sum(stats.norm.logpdf(residuals,scale=scale))
    lp = ll
    lp = -lp

    if data.verbose:
        print(lp, {name: val for name, val in zip(data.names, allparams)})
    return lp
def profit_like_model(params, data):
    """Student‑t log‑likelihood with automatic degrees‑of‑freedom estimation.

    This is the workhorse likelihood function used in the optimisation
    and MCMC sampling.  It computes the Student‑t log‑likelihood of the
    normalised residuals, where the degrees of freedom `dof` is
    estimated from the variance of the normalised residuals:

        dof = 2 * variance / (variance - 1)   (if variance > 1)
        dof = 1000                            (otherwise)

    The dof value is stored in ``data.dof`` for later use (e.g., by
    ``clean_model``).  If the log‑likelihood evaluates to NaN, the
    attribute ``data.check_model`` is set to 1 and a large penalty
    (1e20) is returned.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object.

    Returns:
        float: Negative log‑likelihood.
    """

    if len(params) == 9:
        allparams, modelim = to_pyprofit_image_simples(params, data)
    elif len(params) != 9:
        allparams, modelim = to_pyprofit_image_duplo(params, data)

    residuals=data.image[data.region]-modelim[data.region]
    scale=data.sigim[data.region]

    scaledata=residuals/scale
    variance = scaledata.var()
    if variance > 1:
        dof = 2*variance/(variance-1)
        dof = max(min(dof,float('inf')),0)
    else:
        dof=1000
    data.dof=dof
        
    ll=np.sum(stats.t.logpdf(scaledata,df=dof))
    lp = ll
    lp = -lp
    if np.isnan(ll):
        data.check_model=1
        lp=1e20
    if data.verbose:
        print(lp, {name: val for name, val in zip(data.names, allparams)})
    return lp
def profit_like_model_2(params, data):
    """Alternative Student‑t likelihood with a slightly different dof
    estimation.

    Identical to ``profit_like_model`` except for the treatment when
    variance < 1: the dof is set to ``2 * variance / (1 - variance)``.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object.

    Returns:
        float: Negative log‑likelihood.
    """

    if len(params) == 9:
        allparams, modelim = to_pyprofit_image_simples(params, data)
    elif len(params) != 9:
        allparams, modelim = to_pyprofit_image_duplo(params, data)

    residuals=data.image[data.region]-modelim[data.region]
    scale=data.sigim[data.region]

    scaledata=residuals/scale
    variance = scaledata.var()
    
    if variance > 1.:
        dof = 2*variance/(variance-1)
    elif variance < 1.:
        dof = 2*variance/(1-variance)

    dof = max(min(dof,float('inf')),0)
    data.dof=dof
    
    ll=np.sum(stats.t.logpdf(scaledata,df=dof))
    lp = ll
    lp = -lp
    if np.isnan(ll):
        data.check_model=1
        lp=1e20
    if data.verbose:
        print(lp, {name: val for name, val in zip(data.names, allparams)})
    return lp
#############################################
# ----------------------------------------------------------------------
# Cleaned likelihood (excluding central region)
# ----------------------------------------------------------------------
#############################################
def clean_model(params, data):
    """Evaluate the Student‑t log‑likelihood excluding the central mask.

    This function is used to assess the model fit while ignoring the very
    central pixels (which may be affected by AGN or strong PSF features).
    It returns both the “cleaned” log‑likelihood (only pixels outside
    the centre) and the “uncleaned” (all‑pixel) log‑likelihood, as well
    as a 2D map of the log‑PDF.

    The degrees of freedom for the Student‑t distribution are taken from
    ``data.dof``, which should have been set by a prior call to
    ``profit_like_model``.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object containing ``mask_center`` and ``region``.

    Returns:
        tuple: (lp_clean, lp_all, logpdf_map)
            - lp_clean (float): Negative log‑likelihood for the clean region.
            - lp_all (float): Negative log‑likelihood for all pixels.
            - logpdf_map (2D ndarray): Log‑PDF values at each pixel,
              with NaN outside the data region.
    """

    if len(params) == 9:
        try:
            _, modelim = to_pyprofit_image_simples(params, data)
        except:
            _,modelim=build_model_simples(params,data)
    else:
        try:
            _, modelim = to_pyprofit_image_duplo(params, data)
        except:
            _,modelim=build_model_duplo(params,data)

    mask_center = data.mask_center.astype(bool)
    data_region = data.region.astype(bool)
    clean_region = data_region & ~mask_center

    residuals = data.image[data_region] - modelim[data_region]
    scale = data.sigim[data_region]
    scaledata = residuals / scale
    dof=float(data.dof)

    logpdf_all = stats.t.logpdf(scaledata, df=dof)  

    logpdf_map = np.full(data.image.shape, np.nan)
    
    logpdf_map[data_region] = logpdf_all

    indices_data_region = np.where(data_region)  # Coordenadas dos pixels em data.region
    clean_submask = clean_region[indices_data_region]  # Máscara 1D dos pixels válidos

    logpdf_filtered = logpdf_all[clean_submask]
    
    ll_all=np.sum(logpdf_all)
    lp_all=-ll_all

    if np.isnan(lp_all):
        lp_all=0

    ll = np.sum(logpdf_filtered)
    lp = -ll
    if np.isnan(lp):
        lp=0
    return lp,lp_all,logpdf_map

def clean_model_gaussiano(params, data):
    """Gaussian version of ``clean_model``.

    Computes the Gaussian log‑likelihood both for the full data region
    and for the region excluding the central mask.  No degrees‑of‑freedom
    parameter is needed.

    Args:
        params (ndarray): Free parameters in optimisation space.
        data (Data): Data object.

    Returns:
        tuple: (lp_clean, lp_all)
            - lp_clean: negative log‑likelihood for the clean region.
            - lp_all: negative log‑likelihood for all pixels.
    """

    if len(params) == 9:
        _, modelim = to_pyprofit_image_simples(params, data)
    else:
        _, modelim = to_pyprofit_image_duplo(params, data)

    mask_center = data.mask_center.astype(bool)
    data_region = data.region.astype(bool)
    clean_region = data_region & ~mask_center

    residuals = data.image[data_region] - modelim[data_region]
    scale = data.sigim[data_region]

    logpdf_all=stats.norm.logpdf(residuals,scale=scale)

    indices_data_region = np.where(data_region)  # Coordenadas dos pixels em data.region
    clean_submask = clean_region[indices_data_region]  # Máscara 1D dos pixels válidos

    logpdf_filtered=logpdf_all[clean_submask]
    
    ll_all=np.sum(logpdf_all)
    lp_all=-ll_all

    ll = np.sum(logpdf_filtered)
    lp = -ll

    return lp,lp_all

#############################################
# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
#############################################
def profit_setup_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center):
    """Package all input data and initial parameter information into a
    single ``Data`` object ready for optimisation.

    The function performs the following steps:
    1. Determines the galaxy region: all pixels that belong to the
       de‑blended source (based on the segmentation map) plus the
       psf‑shaped border.
    2. Computes the ``calcregion`` mask used by ``pyprofit`` for
       convolution boundaries.
    3. Transforms the initial parameter vector into the optimisation
       space: values flagged by ``tolog`` are log10‑transformed, then
       the whole vector is divided by ``sigmas``.  Only the parameters
       marked by ``tofit`` are kept in ``data.init``.
    4. Similarly transforms the lower and upper bounds into the
       optimisation space and stores them in ``data.bounds``.

    Args:
        cluster (str): Galaxy identifier (currently unused, may be used
            for logging in future versions).
        image (2D ndarray): Galaxy stamp image.
        mask (2D bool array): Bad‑pixel mask (1 = masked).
        sigim (2D ndarray): Sigma (noise) image.
        segim (2D ndarray): Segmentation map from SExtractor.
        psf (2D ndarray): PSF image.
        magzero (float): Magnitude zero‑point.
        names (list of str): Parameter names.
        model0 (ndarray): Initial physical parameter vector.
        tofit (bool array): Which parameters to fit.
        tolog (bool array): Which parameters to fit in log10 space.
        sigmas (ndarray): Prior standard deviations.
        priors (list of callables): Prior functions.
        lowers (ndarray): Lower bounds in physical units.
        uppers (ndarray): Upper bounds in physical units.
        mask_center (2D bool array): Additional mask for the central
            region (e.g., to exclude AGN light).

    Returns:
        Data: An instance of the ``Data`` class populated with all
        the necessary attributes for profit optimisation.
    """

    im_w, im_h = image.shape
    psf_w, psf_h = psf.shape
    xc,yc=model0[:2]
    # All the center containing the PSF is considered, as well
    # as the section of the image containing the galaxy
    region = np.zeros(image.shape, dtype=bool)
    region[(im_w - psf_w)//2:(im_w + psf_w)//2][(im_h - psf_h)//2:(im_h + psf_h)//2] = True
    segim_center_pix = segim[int(yc)][int(xc)]
    region[segim == segim_center_pix] = True
    # Use the PSF to calculate 'calcregion', which is where we
    # effectively calculate the sersic profile
    psf[psf<0] = 0
    calcregion = signal.convolve2d(region.copy(), psf+1, mode='same')
    calcregion = calcregion > 0

    data = Data()
    data.magzero = magzero
    data.names = names
    data.model0 = model0
    data.tolog = np.logical_and(tolog, tofit)
    data.tofit = tofit
    data.sigmas = sigmas
    data.image = image
    data.sigim = sigim
    data.psf = psf
    data.priors = priors[tofit]
    data.region = region
    data.calcregion = calcregion
    data.mask_center=mask_center
    data.verbose = False
    data.dof=0
    data.check_model=0

    # copy initial parameters
    # log some, /sigma all, filter
    data.init = data.model0.copy()
    data.init[tolog] = np.log10(data.model0[tolog])
    data.init = data.init/sigmas
    data.init = data.init[tofit]

    # Boundaries are scaled by sigma values as well
    data.bounds = np.array(list(zip(lowers/sigmas,uppers/sigmas)))[tofit]

    return data
