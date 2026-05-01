"""
isofota_coef_gr_gain_multiprocess.py

Galaxy isophotal analysis and g‑r colour gradient measurement using Photutils.

This script performs elliptical isophote fitting on r‑band and g‑band images
of BCGs in the WHL sample.  For each cluster it:

- Reads existing GalFit fit information to obtain the galaxy centre, effective
  radius, Sérsic index, ellipticity, and position angle.
- Computes the sky background level by fitting an exponential model to the
  sky pixels as a function of radius.
- Runs Photutils ``Ellipse`` with free geometry (ellipticity and PA free) on
  the r‑band image to measure the radial profiles of intensity, ellipticity,
  PA, and Fourier coefficients.
- Performs a second, fixed‑geometry ellipse analysis on both bands using the
  single average ellipticity and PA from GalFit to extract colours and the
  colour gradient.
- Generates diagnostic plots (isophotes overlayed on the image) and saves
  the full isophote tables.

The script uses multiprocessing to process many clusters in parallel.

Global dependency: input paths are hard‑coded for the WHL sample and
must be adjusted for other datasets.
"""

from math import sin, cos, tan, pi, floor, log10, sqrt, atan2, exp
import numpy as np
from subprocess import call
import os
import os.path
import scipy.optimize as scp
import photutils.isophote as phi
import photutils.aperture as php
import numpy.ma as ma
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")
import multiprocess as mp
import sys

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def skyradfunc(x, a, b, c):
    """Exponential model for radial sky background variation.

    Used to describe how the remaining sky level changes as a function of
    distance from the galaxy centre (due to large‑scale gradients or
    imperfect flat‑fielding).

    Args:
        x (float or array_like): Distance from galaxy centre (pixels).
        a (float): Amplitude (the excess at x=0).
        b (float): Scale radius.
        c (float): Asymptotic constant (sky level at large radii).

    Returns:
        array_like: a * exp(-x / b) + c.
    """
    return a * np.exp(-x / b) + c


def calc_sky(image, mask, maskb, xcenter, ycenter, cluster):
    """Estimate the true sky background level by fitting an exponential model.

    Pixels that belong to the sky (mask==0, maskb==0) are used for the fit.
    The function collects the pixel values and their distances from the
    galaxy centre, then fits the model ``skyradfunc``.  The sky value is
    taken as the asymptotic level at the largest radius.

    Args:
        image    (2D array): Sky‑subtracted or background image.
        mask     (2D bool array): Galaxy + object mask (1 = masked).
        maskb    (2D bool array): Additional inner mask (1 = masked).
        xcenter  (float): X coordinate of galaxy centre.
        ycenter  (float): Y coordinate of galaxy centre.
        cluster  (str): Cluster identifier (currently unused; kept for
                        compatibility with logging).

    Returns:
        float: Estimated sky background value.  If fewer than 4 sky pixels
               are available, returns 0.
    """
    vsky = []
    dsky = []
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if mask[j, i] == 0 and maskb[j, i] == 0:
                vsky.append(image[j, i])
                dsky.append(((j - ycenter)**2 + (i - xcenter)**2)**0.5)
    if len(dsky) <= 3:
        return 0
    popt, pcov = scp.curve_fit(skyradfunc, dsky, vsky, p0=[200, 100, 100])
    return skyradfunc(np.max(dsky), *popt)


def testgr(cluster, ellgalfit, pagalfit, imgr, imgg,
           sigskyg, sigskyr, c1, c2, c3, c4):
    """Perform fixed‑geometry isophote extraction and compute g‑r colour profiles.

    This function:
    1. Reads the free‑geometry isophote table previously saved to disk.
    2. For each free‑geometry semi‑major axis, extracts the intensity and
       error in the g band using the free‑geometry (r‑band) ellipse parameters.
    3. Also extracts g‑ and r‑band intensities using a fixed geometry defined
       by the GalFit ellipticity and position angle, on a geometrically
       spaced radius grid.
    4. Generates overlay plots of the isophotes on the r‑band and g‑band images.
    5. Writes a new extended table (``iso_table_gr.dat``) that combines the
       free‑geometry parameters with the g‑band free intensities and the
       fixed‑geometry g and r intensities and errors.

    Args:
        cluster    (str): Cluster identifier.
        ellgalfit  (float): Ellipticity (1 - b/a) from GalFit.
        pagalfit   (float): Position angle (radians) from GalFit.
        imgr       (2D array): r‑band image (sky‑subtracted, masked).
        imgg       (2D array): g‑band image (sky‑subtracted).
        sigskyg    (float): Sky RMS in g band.
        sigskyr    (float): Sky RMS in r band.
        c1, c2, c3, c4 (float): Conversion factors (EXPTIME/NMGY) for the
                                two bands (currently unused directly in the
                                function body but passed for potential future
                                use).

    Returns:
        tuple: (free_g, fixgal_r) – the Photutils ``IsophoteList`` objects
               for the free‑geometry g‑band and the fixed‑geometry r‑band.
               These are returned for possible further analysis (they are not
               saved to disk here, only the table is).

    Note:
        The function assumes that the free‑geometry isophote table exists as
        ``WHL_gr/{cluster}/iso_table.dat`` and that the output directory
        ``WHL_gr/{cluster}/`` already exists.
    """
    import matplotlib.pyplot as plt

    # Read the previously saved isophote table
    temp = [[] for _ in range(17)]
    with open(f'WHL_gr/{cluster}/iso_table.dat', 'r') as iso_table:
        for item in iso_table.readlines():
            if len(item.split()) == 3:
                extval_ellip = float(item.split()[0])
                extval_pa = float(item.split()[1])
                maxrad = float(item.split()[2])
            else:
                for i in range(17):
                    temp[i].append(float(item.split()[i]))

    vec = [np.asarray(temp[i]) for i in range(17)]
    (x0, y0, sma, pa, eps, intens,
     a3, b3, a4, b4,
     ellip_err, pa_err, int_err,
     a3_err, b3_err, a4_err, b4_err) = vec

    xc = x0[0]
    yc = y0[0]

    # Prepare storage for the different isophote extractions
    isofree_g = [[], []]    # [intensities, errors]
    isofree_r = [[], []]
    galfix_r  = [[], []]
    galfix_g  = [[], []]
    isovec    = [[], [], [], []]   # will hold Isophote objects

    # Semi‑major axes in arcseconds (free geometry uses same radii)
    smagr = sma / 0.396
    smagal = np.geomspace(5, maxrad / 0.396, len(smagr))

    for i in range(len(intens)):
        # ---- Free geometry, g band and r band ----
        freegeo_g = phi.EllipseGeometry(
            x0=xc, y0=yc, sma=float(smagr[i]),
            eps=float(eps[i]), pa=float(pa[i]),
            fix_center=True, fix_eps=True, fix_pa=True)
        freesamp_g = phi.EllipseSample(imgg, sma=float(smagr[i]),
                                       sclip=3.0, nclip=5,
                                       geometry=freegeo_g)
        freesamp_g.update()
        freeiso_g = phi.Isophote(freesamp_g, 0, True, 0)
        isovec[0].append(freeiso_g)
        isofree_g[0].append(freeiso_g.intens)
        isofree_g[1].append(freeiso_g.int_err)

        freegeo_r = phi.EllipseGeometry(
            x0=xc, y0=yc, sma=float(smagr[i]),
            eps=float(eps[i]), pa=float(pa[i]),
            fix_center=True, fix_eps=True, fix_pa=True)
        freesamp_r = phi.EllipseSample(imgr, sma=float(smagr[i]),
                                       sclip=3.0, nclip=5,
                                       geometry=freegeo_r)
        freesamp_r.update()
        freeiso_r = phi.Isophote(freesamp_r, 0, True, 0)
        isovec[1].append(freeiso_r)
        isofree_r[0].append(freeiso_r.intens)
        isofree_r[1].append(freeiso_r.int_err)

        # ---- Fixed geometry (using GalFit global shape) ----
        galgeo_g = phi.EllipseGeometry(
            x0=xc, y0=yc, sma=float(smagal[i]),
            eps=ellgalfit, pa=pagalfit,
            fix_center=True, fix_eps=True, fix_pa=True)
        galsamp_g = phi.EllipseSample(imgg, sma=float(smagal[i]),
                                      sclip=3.0, nclip=5,
                                      geometry=galgeo_g)
        galsamp_g.update()
        galfot_g = phi.Isophote(galsamp_g, 0, True, 0)
        isovec[2].append(galfot_g)
        galfix_g[0].append(galfot_g.intens)
        galfix_g[1].append(galfot_g.int_err)

        galgeo_r = phi.EllipseGeometry(
            x0=xc, y0=yc, sma=float(smagal[i]),
            eps=ellgalfit, pa=pagalfit,
            fix_center=True, fix_eps=True, fix_pa=True)
        galsamp_r = phi.EllipseSample(imgr, sma=float(smagal[i]),
                                      sclip=3.0, nclip=5,
                                      geometry=galgeo_r)
        galsamp_r.update()
        galfot_r = phi.Isophote(galsamp_r, 0, True, 0)
        isovec[3].append(galfot_r)
        galfix_r[0].append(galfot_r.intens)
        galfix_r[1].append(galfot_r.int_err)

    free_g = phi.IsophoteList(isovec[0])
    free_r = phi.IsophoteList(isovec[1])
    fixgal_g = phi.IsophoteList(isovec[2])
    fixgal_r = phi.IsophoteList(isovec[3])

    list_iso = [free_g, free_r, fixgal_g, fixgal_r]
    name_iso = ['free_g', 'free_r', 'fixgal_g', 'fixgal_r']
    # conversion factors not actually used to display the images
    image_iso = [(imgg * c3) / c4, (imgr * c1) / c2,
                 (imgg * c3) / c4, (imgr * c1) / c2]

    # Diagnostic plots: isophotes overlaid on the respective images
    for item in list_iso:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_iso[list_iso.index(item)],
                  vmin=0, vmax=1500, origin='lower')
        paircont = 0
        for raio in item.sma:
            iso = item.get_closest(raio)
            x, y = iso.sampled_coordinates()
            if paircont % 2 == 0:
                plt.plot(x, y, color='white', linewidth=1)
            paircont += 1
        plt.xlabel(r'$X (pixel)$')
        plt.ylabel(r'$Y (pixel)$')
        plt.tight_layout()
        plt.savefig(f'WHL_gr/{cluster}/{cluster}_{name_iso[list_iso.index(item)]}.png')
        plt.close(fig)

    # Write the combined isophote table (free geometry + fixed‑geometry colours)
    with open(f'WHL_gr/{cluster}/iso_table_gr.dat', 'w') as iso_table_vg:
        iso_table_vg.write(f'{extval_ellip:f}\t{extval_pa:f}\t{maxrad:f}\t'
                           f'{sigskyg:f}\t{sigskyr:f}\n')
        for r in range(len(sma)):
            iso_table_vg.write(
                f'{x0[r]:f}\t{y0[r]:f}\t{sma[r]:f}\t{pa[r]:f}\t{eps[r]:f}\t'
                f'{intens[r]:f}\t{a3[r]:f}\t{b3[r]:f}\t{a4[r]:f}\t{b4[r]:f}\t'
                f'{ellip_err[r]:f}\t{pa_err[r]:f}\t{int_err[r]:f}\t'
                f'{a3_err[r]:f}\t{b3_err[r]:f}\t{a4_err[r]:f}\t{b4_err[r]:f}\t'
                f'{isofree_g[0][r]:f}\t{isofree_g[1][r]:f}\t'
                f'{isofree_r[0][r]:f}\t{isofree_r[1][r]:f}\t'
                f'{galfix_r[0][r]:f}\t{galfix_r[1][r]:f}\t'
                f'{galfix_g[0][r]:f}\t{galfix_g[1][r]:f}\n'
            )
    return free_g, fixgal_r


# ----------------------------------------------------------------------
# Main per‑cluster pipeline
# ----------------------------------------------------------------------

def isobuilder(cluster, flagmask, flagdelta, ok, okk, good_path, bad_path):
    """Full isophote analysis pipeline for a single cluster.

    This function:
    1. Checks whether the cluster should be processed (skips if already done
       or flagged as bad).
    2. Reads the r‑band and g‑band images, masks, and GalFit results.
    3. Computes sky background in both bands using ``calc_sky``.
    4. Runs an initial free‑geometry Ellipse fit on the r‑band image.
    5. If the fit fails to converge (< 3 isophotes), attempts retries with
       random starting position angles up to 500 times.
    6. Once a good fit is obtained, saves the free‑geometry isophote table
       and calls ``testgr`` to perform the fixed‑geometry colour extraction.
    7. Success or failure is recorded in ``good_path`` or ``bad_path``,
       respectively.

    Args:
        cluster     (str): Cluster identifier.
        flagmask    (int): Flag from the parent catalogue (0 = good).
        flagdelta   (int): Second flag (0 = good).
        ok          (list of str): List of clusters already successfully
                       processed (skip list).
        okk         (list of str): List of clusters that previously failed
                       (will be removed).
        good_path   (str): File path where successful cluster IDs are appended.
        bad_path    (str): File path where failed cluster IDs are appended.

    Returns:
        None.  All outputs are written to disk under ``WHL_gr/{cluster}/``.

    Note:
        File paths are hard‑coded for the WHL sample and the original
        directory structure on the author's machine.  They must be adapted
        for other environments.
    """
    import matplotlib.pyplot as plt

    if [flagmask, flagdelta] != [0, 0] or cluster in okk:
        call(f'rm -rf WHL_gr/{cluster}', shell=True)
        return
    elif cluster in ok:
        return
    else:
        call(f'mkdir -p WHL_gr/{cluster}', shell=True)

        # Read input images and headers
        ajust1 = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/ajust-bcg-r.fits')[1].data
        ajust2 = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/ajust-bcg-r.fits')[2].data
        header = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/ajust-bcg-r.fits')[2].header
        headerr = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/ajust-bcg-r.fits')[1].header
        mask = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_r_mask.fits')[0].data
        maskb = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_r_mask_b.fits')[0].data

        datar0 = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_r.fits')[0].data
        datag0 = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_g.fits')[0].data
        headerg = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_g.fits')[0].header
        maskg = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_r_mask_g.fits')[0].data
        maskbg = fits.open(f'/home/andrelpk/Documentos/Projetos/WHL/{cluster}/bcg_r_mask_b_g.fits')[0].data

        xc = float(header['1_XC'].split()[0].replace('*', ''))
        yc = float(header['1_YC'].split()[0].replace('*', ''))
        re = float(header['1_RE'].split()[0].replace('*', ''))
        mag = float(header['1_MAG'].split()[0].replace('*', ''))
        n = float(header['1_N'].split()[0].replace('*', ''))
        sky = float(header['2_SKY'].split()[0].replace('*', ''))
        pa0 = (float(header['1_PA'].split()[0].replace('*', '')) + 90.) * np.pi / 180.
        ell0 = 1. - float(header['1_AR'].split()[0].replace('*', ''))
        chisq_galfit = float(header['CHI2NU'])

        # If the galaxy centre falls on a masked pixel, skip
        if mask[int(yc), int(xc)] == 1:
            with open(bad_path, 'a') as outfail:
                outfail.write(f'{cluster}\n')
            call(f'rm -r WHL_gr/{cluster}', shell=True)
            return

        # Conversion factors to flux units
        NMGYr = float(headerr['NMGY'])
        EXPTIMEr = float(headerr['EXPTIME'])
        NMGYg = float(headerg['NMGY'])
        EXPTIMEg = float(headerg['EXPTIME'])

        datar = (datar0 * NMGYr) / EXPTIMEr
        datag = (datag0 * NMGYg) / EXPTIMEg

        # Sky subtraction and RMS estimation
        skyvalueg = calc_sky(datag, maskg, maskbg, xc, yc, cluster)
        imageg = datag - skyvalueg
        negpixsg = imageg[(imageg < skyvalueg) & (imageg > -10000.)]
        sigmaskyg = np.std(negpixsg) / np.sqrt(1. - 2. / np.pi)
        sigmaskyg /= 4.

        skyvaluer = calc_sky(datar, mask, maskb, xc, yc, cluster)
        imager = datar - skyvaluer
        negpixsr = imager[(imager < skyvaluer) & (imager > -10000.)]
        sigmaskyr = np.std(negpixsr) / np.sqrt(1. - 2. / np.pi)
        sigmaskyr /= 4.

        # Masked images for Photutils (pixels with mask==1 are excluded)
        isoimage_g = ma.masked_where(maskg == 1, imageg)
        isoimage_r = ma.masked_where(mask == 1, imager)

        # If the free‑geometry table already exists but not the colour table,
        # run testgr directly
        if os.path.isfile(f'WHL_gr/{cluster}/iso_table.dat'):
            temp = [[] for _ in range(17)]
            with open(f'WHL_gr/{cluster}/iso_table.dat', 'r') as iso_table:
                for item in iso_table.readlines():
                    if len(item.split()) == 3:
                        extval_ellip = float(item.split()[0])
                        extval_pa = float(item.split()[1])
                        maxrad = float(item.split()[2])
                    else:
                        for i in range(17):
                            temp[i].append(float(item.split()[i]))
            vec = [np.asarray(temp[i]) for i in range(17)]
            (x0, y0, sma, pa, eps, intens, a3, b3, a4, b4,
             ellip_err, pa_err, int_err, a3_err, b3_err, a4_err, b4_err) = vec
            if os.path.isfile(f'WHL_gr/{cluster}/iso_table_gr.dat'):
                with open(good_path, 'a') as output:
                    output.write(f'{cluster}\n')
                return
            else:
                testcor = testgr(cluster, ell0, pa0, isoimage_r, isoimage_g,
                                 sigmaskyg, sigmaskyr,
                                 EXPTIMEr, NMGYr, EXPTIMEg, NMGYg)
                with open(good_path, 'a') as output:
                    output.write(f'{cluster}\n')
                return

        # Otherwise, perform the free‑geometry Ellipse fit
        else:
            isogal = phi.EllipseGeometry(x0=xc, y0=yc, sma=20,
                                         eps=ell0, pa=pa0)
            isomodel = phi.Ellipse(isoimage_r, isogal)

            # Quick‑look plot of the images
            fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            axs[0, 0].imshow((isoimage_r * EXPTIMEr) / NMGYr,
                             vmin=0, vmax=1500, origin='lower')
            axs[0, 0].set_ylabel(r'$Y (pixel)$')
            axs[0, 1].imshow((datar * EXPTIMEr) / NMGYr,
                             vmin=0, vmax=1500, origin='lower')
            axs[1, 0].imshow((isoimage_g * EXPTIMEg) / NMGYg,
                             vmin=0, vmax=1500, origin='lower')
            axs[1, 0].set_xlabel(r'$X (pixel)$')
            axs[1, 0].set_ylabel(r'$Y (pixel)$')
            axs[1, 1].imshow((datag * EXPTIMEg) / NMGYg,
                             vmin=0, vmax=1500, origin='lower')
            axs[1, 1].set_xlabel(r'$X (pixel)$')
            plt.tight_layout()
            plt.savefig(f'WHL_gr/{cluster}/{cluster}_bcg.png')
            plt.close(fig)

            try:
                isolist = isomodel.fit_image(
                    minsma=5,
                    maxsma=np.max(isoimage_r.shape) / 2.,
                    step=0.02,
                    fix_center=True,
                    sclip=3.0,
                    nclip=5,
                    conver=0.1,
                    fflag=0.5)

                # Plot the isophotes on the r‑band image
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow((isoimage_r * EXPTIMEr) / NMGYr,
                          vmin=0, vmax=1500, origin='lower')
                paircont = 0
                for sma_val in isolist.sma:
                    if isolist.intens[np.where(isolist.sma == sma_val)][0] > sigmaskyr:
                        iso = isolist.get_closest(sma_val)
                        x, y = iso.sampled_coordinates()
                        if paircont % 2 == 0:
                            plt.plot(x, y, color='white', linewidth=1)
                        paircont += 1
                plt.xlabel(r'$X (pixel)$')
                plt.ylabel(r'$Y (pixel)$')
                plt.tight_layout()
                plt.savefig(f'WHL_gr/{cluster}/{cluster}_iso_free.png')
                plt.close(fig)

            except Exception:
                # If the fit raised an exception, try to check if len(isolist.sma) exists
                try:
                    print(len(isolist.sma))
                    pass
                except NameError:
                    with open(bad_path, 'a') as outfail:
                        outfail.write(f'{cluster}\n')
                    call(f'rm -r WHL_gr/{cluster}', shell=True)
                    return

            n_isofotas = len(isolist.sma)
            isotry = 0
            # Retry loop if too few isophotes were found
            if n_isofotas < 3:
                while isotry <= 500 and len(isolist.sma) < 3:
                    isotry += 1
                    try:
                        isogal = phi.EllipseGeometry(
                            x0=xc, y0=yc, sma=20,
                            eps=ell0,
                            pa=np.random.uniform(-180., 180.))
                        isomodel = phi.Ellipse(isoimage_r, isogal)
                        isolist = isomodel.fit_image(
                            minsma=5,
                            maxsma=np.max(isoimage_r.shape) / 2.,
                            step=0.02,
                            fix_center=True,
                            sclip=3.0, nclip=5,
                            conver=0.1, fflag=0.5)
                        # Plot each attempt (overwrites previous plot)
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow((isoimage_r * EXPTIMEr) / NMGYr,
                                  vmin=0, vmax=1500, origin='lower')
                        paircont = 0
                        for sma_val in isolist.sma:
                            if isolist.intens[np.where(isolist.sma == sma_val)][0] > sigmaskyr:
                                iso = isolist.get_closest(sma_val)
                                x, y = iso.sampled_coordinates()
                                if paircont % 2 == 0:
                                    plt.plot(x, y, color='white', linewidth=1)
                                paircont += 1
                        plt.xlabel(r'$X (pixel)$')
                        plt.ylabel(r'$Y (pixel)$')
                        plt.tight_layout()
                        plt.savefig(f'WHL_gr/{cluster}/{cluster}_iso_free.png')
                        plt.close(fig)
                    except Exception:
                        isotry += 1

            # If we still don't have enough isophotes after 200 tries, mark as failed
            if isotry >= 200 and len(isolist.sma) <= 3:
                with open(bad_path, 'a') as outfail:
                    outfail.write(f'{cluster}\n')
                call(f'rm -r WHL_gr/{cluster}', shell=True)
                return

            # ---- Extract parameters above the sky RMS ----
            valid = isolist.intens > sigmaskyr
            x0 = isolist.x0[valid]
            y0 = isolist.y0[valid]
            sma = isolist.sma[valid] * 0.396        # convert to arcsec
            pa = isolist.pa[valid]
            eps = isolist.eps[valid]
            intens = isolist.intens[valid]
            a3, b3 = isolist.a3[valid], isolist.b3[valid]
            a4, b4 = isolist.a4[valid], isolist.b4[valid]
            ellip_err = isolist.ellip_err[valid]
            pa_err = isolist.pa_err[valid]
            int_err = isolist.int_err[valid]
            a3_err, b3_err = isolist.a3_err[valid], isolist.b3_err[valid]
            a4_err, b4_err = isolist.a4_err[valid], isolist.b4_err[valid]

            # Weighted average ellipticity and PA for the fixed‑geometry step
            extval_ellip = np.average(eps, weights=np.power(sma, 2))
            extval_pa = np.average(pa, weights=np.power(sma, 2))
            maxrad = np.max(sma)

            # Write free‑geometry table
            with open(f'WHL_gr/{cluster}/iso_table.dat', 'w') as inptrue:
                inptrue.write(f'{extval_ellip:f}\t{extval_pa:f}\t{maxrad:f}\n')
                for r in range(len(sma)):
                    inptrue.write(
                        f'{x0[r]:f}\t{y0[r]:f}\t{sma[r]:f}\t{pa[r]:f}\t'
                        f'{eps[r]:f}\t{intens[r]:f}\t'
                        f'{a3[r]:f}\t{b3[r]:f}\t{a4[r]:f}\t{b4[r]:f}\t'
                        f'{ellip_err[r]:f}\t{pa_err[r]:f}\t{int_err[r]:f}\t'
                        f'{a3_err[r]:f}\t{b3_err[r]:f}\t'
                        f'{a4_err[r]:f}\t{b4_err[r]:f}\n'
                    )

            # Run the colour extraction (fixed geometry)
            testcor = testgr(cluster, ell0, pa0, isoimage_r, isoimage_g,
                             sigmaskyg, sigmaskyr,
                             EXPTIMEr, NMGYr, EXPTIMEg, NMGYg)
            with open(good_path, 'a') as output:
                output.write(f'{cluster}\n')
            return

    return


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Parallel isophote analysis for the WHL sample.

    Reads the parent catalogue, determines which clusters are already
    processed (good_path) or known to fail (bad_path), and distributes
    the remaining clusters across 16 processes using ``multiprocess.Pool``.
    """
    good_path = 'checkiso_WHL.dat'
    bad_path = 'iso_fail_10k.dat'

    ok = []
    with open(good_path, 'r+') as output:
        for item in output.readlines():
            ok.append(item.split()[0])

    okk = []
    with open(bad_path, 'r+') as outfail:
        for galaxy in outfail.readlines():
            okk.append(galaxy.split()[0])

    data_info = []
    with open('/home/andrelpk/Documentos/Projetos/WHL/pargal_WHL_compact_astro_vfix.dat', 'r') as inp1:
        for obj in inp1.readlines():
            ll1 = obj.split()
            cluster = ll1[0]
            flagmask = int(float(ll1[11]))
            flagdelta = int(float(ll1[12]))
            data_info.append((cluster, flagmask, flagdelta))

    with mp.Pool(processes=16) as pool:
        pool.starmap(isobuilder, [(*data, ok, okk, good_path, bad_path)
                                  for data in data_info])