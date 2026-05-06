import requests
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from astropy.io import fits, ascii
import numpy as np
from astropy.wcs import WCS
import time
from collections import defaultdict
from astropy.modeling.models import Moffat2D
import subprocess
import multiprocessing as mp

"""
query_desi_v3.py

Download and prepare DESI Legacy Survey cutouts for a list of target galaxies.

This script performs the following steps for a sample of galaxies given in
``desi_indiv_WHL.dat``:

1. Associates each galaxy with the corresponding Legacy Survey *brick*
   using a pre‑downloaded brick catalogue (``survey-bricks.fits.gz``).
   This step is parallelised via ``multiprocessing``.
2. Groups galaxies by brick, downloads the brick‑level images (science,
   inverse variance, PSF size) from the public NOIRLab archive, and
   extracts postage‑stamp cutouts around each target.
3. For each cutout, saves three files:
   * ``stamp.fits`` – the science image.
   * ``sigma-{band}.fits`` – the pixel‑wise standard deviation derived
     from the inverse variance.
   * ``psf_b.fits`` – a local Moffat (or Gaussian fallback) PSF model
     normalised to unit sum, with the PSF FWHM recorded in the header.
4. Maintains a log file (``datalog.dat``) so that the script can be
   restarted without re‑processing already‑completed galaxies.

Global configuration constants:
- ``data_release`` : `'dr9'` (Legacy Survey Data Release 9)
- ``band``         : `'r'`
- ``pixscale``     : 0.262 arcsec/pixel
- ``cutout_size``  : 500 pixels

The script is written for the WHL sample but can be adapted to other
catalogues by changing the input file and adjusting paths.
"""

# -- Configuration --
data_release = 'dr9'
band = 'r'
pixscale = 0.262
cutout_size = 500

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def find_brickname_for_radec(target_ra, target_dec, bricks_data):
    """Find the Legacy Survey brick that contains a given RA/Dec.

    The brick catalogue is expected to provide RA1, RA2, DEC1, DEC2 for
    each brick.  This function handles the case where the brick crosses
    RA = 0° (wrap‑around).

    Args:
        target_ra  (float): Right Ascension in degrees (will be taken
                            modulo 360).
        target_dec (float): Declination in degrees.
        bricks_data: FITS table data containing columns 'RA1', 'RA2',
                     'DEC1', 'DEC2', and 'BRICKNAME'.

    Returns:
        str or None: The brick name if found, else None.
    """
    target_ra = target_ra % 360
    for brick in bricks_data:
        ra1, ra2 = brick['RA1'], brick['RA2']
        dec1, dec2 = brick['DEC1'], brick['DEC2']
        if ra1 <= ra2:
            ra_in_brick = (ra1 <= target_ra <= ra2)
        else:
            ra_in_brick = (target_ra >= ra1) or (target_ra <= ra2)
        dec_in_brick = (dec1 <= target_dec <= dec2)
        if ra_in_brick and dec_in_brick:
            return brick['BRICKNAME']
    return None
def download_brick_file(url, filename, max_retries=3):
    """Download a file using ``wget`` with resume and retry capability.

    This method is preferred over ``requests`` for large FITS files because
    it handles timeouts and partial downloads more robustly.

    Args:
        url         (str): Full URL of the file to download.
        filename    (str): Local output path.
        max_retries (int): Maximum number of download attempts (default 3).

    Returns:
        bool: True if the download was successful, False otherwise.
    """

    """Use system wget which is more robust for large files."""
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt+1} for {filename}")
            
            # Use wget with resume capability and timeout
            result = subprocess.run([
                'wget', '-c',  # continue partial downloads
                '-O', filename,
                '--timeout=30',
                '--tries=3',
                '--show-progress',
                url
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully downloaded: {filename}")
                return True
            else:
                print(f"wget failed (attempt {attempt+1}): {result.stderr}")
                
        except Exception as e:
            print(f"Download attempt {attempt+1} failed: {e}")
        
        wait_time = 2 ** attempt
        print(f"Waiting {wait_time} seconds before retry...")
        time.sleep(wait_time)
    
    return False
def create_moffat_psf(fwhm_arcsec, pixscale, beta=3.5, kernel_size=None):
    """Create a 2D circular Moffat PSF kernel from the local FWHM.

    The Moffat profile is defined as:
        I(r) = [1 + (r/α)²]^{-β}
    with α = FWHM / (2 * sqrt(2^{1/β} - 1)).

    The kernel is constructed on a grid of size ``kernel_size`` and
    normalised to sum = 1.

    If the regular construction fails, a Gaussian fallback is returned.

    Args:
        fwhm_arcsec (float): Full‑width at half‑maximum in arcseconds.
        pixscale    (float): Pixel scale in arcseconds per pixel.
        beta        (float): Moffat slope parameter (default 3.5).
        kernel_size (int, optional): Side length of the kernel in pixels.
                     If None, it is set to min(8 * FWHM_pix, 151) and
                     forced to be odd.

    Returns:
        ndarray: 2D normalised PSF kernel.
    """

    """
    Create a 2D circular Moffat PSF kernel with proper bounds checking.
    """
    # Convert FWHM from arcseconds to pixels
    fwhm_pix = fwhm_arcsec / pixscale
    
    # Calculate alpha (gamma in Astropy)
    alpha_pix = fwhm_pix / (2 * np.sqrt(2**(1/beta) - 1))
    
    # Set reasonable kernel size with bounds checking
    if kernel_size is None:
        kernel_size = int(8 * fwhm_pix)  # Reduced from 10 to 8
        # Ensure kernel_size is reasonable
        kernel_size = min(kernel_size, 151)  # Max 151x151 pixels
        kernel_size = max(kernel_size, 15)   # Min 15x15 pixels
    
    # Make sure it's odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    print(f"Creating PSF kernel: {kernel_size}x{kernel_size} pixels")
    
    try:
        # Create the grid - CORRECT BOUNDS
        y, x = np.mgrid[-kernel_size//2: kernel_size//2 + 1, 
                        -kernel_size//2: kernel_size//2 + 1]
        
        # Verify shapes
        assert x.shape == y.shape, f"Shape mismatch: x{x.shape} vs y{y.shape}"
        
        # Create Moffat model
        moffat_model = Moffat2D(amplitude=1.0, x_0=0, y_0=0, 
                               gamma=alpha_pix, alpha=beta)
        
        # Evaluate model
        kernel_array = moffat_model(x, y)
        
        # Normalize
        kernel_sum = np.sum(kernel_array)
        if kernel_sum > 0:
            kernel_array /= kernel_sum
        else:
            raise ValueError("PSF kernel sum is zero or negative")
            
        print(f"PSF created successfully: FWHM={fwhm_arcsec:.2f}\", size={kernel_size}x{kernel_size}")
        return kernel_array
        
    except Exception as e:
        print(f"Error creating PSF: {e}")
        # Fallback: create a simple Gaussian-like kernel
        print("Creating fallback PSF...")
        return create_fallback_psf(kernel_size)
def create_fallback_psf(kernel_size, sigma=2.0):
    """Create a simple Gaussian PSF kernel as a fallback.

    Args:
        kernel_size (int): Side length in pixels (must be odd).
        sigma       (float): Gaussian sigma in pixels (default 2.0).

    Returns:
        ndarray: 2D normalised Gaussian kernel.
    """

    """Create a simple Gaussian PSF as fallback."""
    y, x = np.mgrid[-kernel_size//2: kernel_size//2 + 1,
                    -kernel_size//2: kernel_size//2 + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel
def cleanup_files(brickname, galaxy_indices):
    """Remove all temporary files associated with a brick and its galaxies.

    Deletes the brick‑level FITS files and the individual galaxy
    cutout/PSF files from the current working directory.

    Args:
        brickname       (str): Brick identifier.
        galaxy_indices  (list): List of galaxy index values used in the
                                temporary filenames.

    Returns:
        None.
    """

    """Limpeza simples e direta para Linux"""
    files_to_remove = [
        f"brick_{brickname}_sci.fits.fz",
        f"brick_{brickname}_invvar.fits.fz", 
        f"brick_{brickname}_psfsize.fits.fz"
    ]
    
    # Adiciona arquivos temporários de cada galáxia
    for idx in galaxy_indices:
        files_to_remove.extend([
            f"galaxy_{idx}_sci_brick_{brickname}.fits",
            f"galaxy_{idx}_wt_brick_{brickname}.fits",
            f"galaxy_{idx}_sigma_brick_{brickname}.fits.fz",
            f"galaxy_{idx}_psfmoffatbeta3.5_brick_{brickname}.fits"
        ])
    
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Removed: {filename}")
            except Exception as e:
                print(f"Warning: Could not remove {filename}: {e}")
def process_galaxy(args):
    """Worker function for multiprocessing: assign a brick to one galaxy.

    This function is called in parallel for each galaxy.  It checks
    whether the galaxy has already been fully processed (by looking for
    ``psf_b.fits``) and, if not, calls ``find_brickname_for_radec``.
    Results are written to the log file ``datalog.dat``.

    Args:
        args (tuple): (index, name, ra, dec, bricks_data)

    Returns:
        dict: A status dictionary with keys:
            - 'status' (str): 'skip', 'fail', or 'success'
            - 'name'   (str): galaxy name
            - additional keys depending on status (e.g., 'brickname',
              'survey_region').
    """

    """Processa uma única galáxia para encontrar seu brick."""
    i, name, ra, dec, bricks_data = args
    print(name)
    # Verificar se já foi processada
    if os.path.isfile(f'{name}/psf_b.fits'):
        return {'status': 'skip', 'name': name, 'reason': 'already_processed'}
            
    brickname = find_brickname_for_radec(ra, dec, bricks_data)
    with open('datalog.dat', 'a') as data_exist:
        if brickname is None:
            data_exist.write(f'{name} fail\n')
            data_exist.close()
            return {'status': 'fail', 'name': name, 'reason': 'no_brick'}
            print(name,'não deu')
        else:
            print(name,'feito')
            data_exist.write(f'{name} {brickname}\n')
            region = 'south' if dec < 32.375 else 'north'
            data_exist.close()
            return {
                'status': 'success', 
                'name': name, 
                'brickname': brickname,
                'idx': i, 
                'ra': ra, 
                'dec': dec, 
                'survey_region': region
            }
def load_existing_log():
    """Read the log file ``datalog.dat`` to retrieve previously
    processed galaxies and their brick assignments.

    This allows the script to resume after interruption.

    Returns:
        tuple:
            processed_galaxies (set): Set of galaxy names already logged.
            brick_assignments  (dict): Mapping ``name -> brickname`` for
                                       galaxies that were successfully
                                       assigned a brick.
    """
    """Carrega o log existente para continuar de onde parou."""
    processed_galaxies = set()
    brick_assignments = {}
    
    if os.path.exists('datalog.dat'):
        print("Loading existing log file...")
        with open('datalog.dat', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    galaxy_name = parts[0]
                    brick_info = parts[1]
                    processed_galaxies.add(galaxy_name)
                    if brick_info != 'fail':
                        brick_assignments[galaxy_name] = brick_info
        print(f"Found {len(processed_galaxies)} galaxies in existing log")
    
    return processed_galaxies, brick_assignments
def multiprocess_find_bricks(name_list, ra_list, dec_list, bricks_data, num_cores=18):
    """Run the brick‑finding step in parallel over a list of galaxies.

    Only galaxies that are not already in the log are processed.

    Args:
        name_list   (list of str): Galaxy names.
        ra_list     (list of float): Right Ascensions.
        dec_list    (list of float): Declinations.
        bricks_data: FITS table with brick boundaries.
        num_cores   (int): Number of processes to use (default 18).

    Returns:
        list of dict: Results from ``process_galaxy`` for each input
                      galaxy.
    """

    """Processa a busca de bricks em paralelo."""
    # Preparar argumentos apenas para galáxias não processadas
    args_list = []
    for i, (name, ra, dec) in enumerate(zip(name_list, ra_list, dec_list)):
        args_list.append((i, name, ra, dec, bricks_data))
    
    print(f"Processing {len(args_list)} galaxies with {num_cores} cores...")
    
    brick_dict = defaultdict(list)
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_galaxy, args_list,chunksize=1)
    
    return results

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    """Main execution routine.

    1. Read the input catalogue ``desi_indiv_WHL.dat``.
    2. Load the brick boundary catalogue.
    3. Run Phase 1 (multiprocessing) to associate galaxies with bricks,
       using the log file to skip already‑processed ones.
    4. Run Phase 2: for each brick, download the brick‑level images,
       extract cutouts, compute sigma maps, and generate PSF models.
    5. All output is written into sub‑directories named by the galaxy
       identifier.

    Returns:
        None.  Prints progress information to stdout.
    """

    # Ler dados
    data = np.loadtxt('desi_indiv_WHL.dat',skiprows=1,dtype=str).T
    name_list = data[0]
    ra_list = data[1].astype(float)
    dec_list = data[2].astype(float)
    
    # Carregar log existente
    processed_galaxies, brick_assignments = load_existing_log()
    
    # Carregar bricks_data
    bricks_hdu = fits.open('survey-bricks.fits.gz')
    bricks_data = bricks_hdu[1].data
    bricks_hdu.close()
    
    # Abrir arquivo de log para append
    data_exist = open('datalog.dat', 'a')
    brick_dict = defaultdict(list)
    
    # FASE 1: Multiprocessamento para encontrar bricks
    print("=== FASE 1: Encontrando bricks (Multiprocessamento) ===")
    
    # Filtrar galáxias não processadas
    unprocessed_indices = []
    unprocessed_data = []
    
    for i, (name, ra, dec) in enumerate(zip(name_list, ra_list, dec_list)):
        if name in processed_galaxies:
            # Já está no log, usar a informação existente
            if name in brick_assignments:
                brickname = brick_assignments[name]
                region = 'south' if dec < 32.375 else 'north'
                brick_dict[brickname].append({
                    'idx': i, 
                    'name': name, 
                    'ra': ra, 
                    'dec': dec, 
                    'survey_region': region
                })
                print(f"Galaxy {name} -> Brick {brickname} (from log)")
        else:
            # Não processada ainda, adicionar à lista
            unprocessed_indices.append(i)
            unprocessed_data.append((name, ra, dec))
    
    print(f"Galaxies to process: {len(unprocessed_data)}")
    print(f"Galaxies from log: {len(processed_galaxies)}")
    
    # Processar apenas as não processadas com multiprocessamento
    if unprocessed_data:
        unprocessed_names, unprocessed_ra, unprocessed_dec = zip(*unprocessed_data)
        results = multiprocess_find_bricks(
            unprocessed_names, unprocessed_ra, unprocessed_dec, bricks_data, num_cores=18
        )
        
        # Processar resultados do multiprocessamento
        for result in results:
            if result['status'] == 'skip':
                print(f'Galaxy already processed: {result["name"]}')
            elif result['status'] == 'fail':
                data_exist.write(f'{result["name"]} fail\n')
                print(f'Galaxy {result["name"]} -> No brick found')
            elif result['status'] == 'success':
                data_exist.write(f'{result["name"]} {result["brickname"]}\n')
                # Encontrar o índice original
                original_idx = unprocessed_indices[results.index(result)]
                brick_dict[result['brickname']].append({
                    'idx': original_idx,
                    'name': result['name'],
                    'ra': result['ra'],
                    'dec': result['dec'],
                    'survey_region': result['survey_region']
                })
                print(f"Galaxy {result['name']} -> Brick {result['brickname']}")
    
    data_exist.close()
    print(f"Found {len(brick_dict)} unique bricks for {len(name_list)} galaxies.")

    # FASE 2: Processamento normal dos bricks (igual ao seu código original)
    print("\n=== FASE 2: Processando bricks e baixando dados ===")
    
    # Processar cada brick
    for brickname, galaxy_list in brick_dict.items():
        print(f"\nProcessing brick {brickname} with {len(galaxy_list)} galaxies...")
        galaxy_indices = [galaxy['idx'] for galaxy in galaxy_list]

        # Download dos arquivos do brick
        base_url = f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/{data_release}/{galaxy_list[0]['survey_region']}/coadd"
        subdir = brickname[:3]
        base_path = f"{base_url}/{subdir}/{brickname}"

        file_specs = [
            (f"{base_path}/legacysurvey-{brickname}-image-{band}.fits.fz", f"brick_{brickname}_sci.fits.fz"),
            (f"{base_path}/legacysurvey-{brickname}-invvar-{band}.fits.fz", f"brick_{brickname}_invvar.fits.fz"),
            (f"{base_path}/legacysurvey-{brickname}-psfsize-{band}.fits.fz", f"brick_{brickname}_psfsize.fits.fz"),
            (f"{base_path}/legacysurvey-{brickname}-image-g.fits.fz", f"brick_{brickname}_sci_g.fits.fz")
        ]

        # Fazer download
        download_success = True
        for url, filename in file_specs:
            if not os.path.exists(filename):
                if not download_brick_file(url, filename):
                    print(f"Failed to download {filename}. Skipping brick.")
                    download_success = False
                    break

        if not download_success:
            continue

        # Processar brick
        try:
            with fits.open(f"brick_{brickname}_sci.fits.fz") as sci_hdu, \
                 fits.open(f"brick_{brickname}_invvar.fits.fz") as invvar_hdu, \
                 fits.open(f"brick_{brickname}_psfsize.fits.fz") as psf_hdu:
                
                wcs = WCS(sci_hdu[1].header)
                sci_data = sci_hdu[1].data
                invvar_data = invvar_hdu[1].data
                fwhm_map = psf_hdu[1].data

                for galaxy in galaxy_list:
                    idx, name, ra, dec = galaxy['idx'], str(galaxy['name']), galaxy['ra'], galaxy['dec']
                    
                    #Pular se já foi processada
                    if os.path.isfile(f'{name}/psf_b.fits'):
                        print(f'  Galaxy already processed: {name}')
                        continue
                    
                    print(f"  Processing {name} at ({ra:.5f}, {dec:.5f})")
                    
                    # Criar diretório
                    os.makedirs(str(name), exist_ok=True)
                    
                    # Coordenadas e recorte
                    x_center, y_center = wcs.all_world2pix(ra, dec, 0)
                    x0 = int(np.round(x_center - cutout_size / 2))
                    y0 = int(np.round(y_center - cutout_size / 2))
                    x1, y1 = x0 + cutout_size, y0 + cutout_size
                    
                    sci_cutout = sci_data[y0:y1, x0:x1]
                    invvar_cutout = invvar_data[y0:y1, x0:x1]
                    
                    # Salvar stamp.fits
                    hdu_sci = fits.PrimaryHDU(data=sci_cutout)
                    hdu_sci.header.update(wcs[y0:y1, x0:x1].to_header())
                    hdu_sci.writeto(os.path.join(name, "stamp.fits"), overwrite=True)
                    
                    # Salvar sigma-{band}.fits
                    sigma_data = 1.0 / np.sqrt(invvar_cutout)
                    sigma_data[invvar_cutout <= 0] = 0
                    hdu_sigma = fits.PrimaryHDU(data=sigma_data)
                    hdu_sigma.header.update(wcs[y0:y1, x0:x1].to_header())
                    hdu_sigma.header['BUNIT'] = 'nanomaggies'
                    hdu_sigma.writeto(os.path.join(name, f"sigma-{band}.fits"), overwrite=True)
                    
                    # Salvar psf_b.fits
                    local_fwhm = fwhm_map[int(y_center), int(x_center)]
                    print(f"  Local FWHM: {local_fwhm:.2f} arcsec")
                    try:                        
                        psf_kernel = create_moffat_psf(local_fwhm, pixscale=0.262, beta=3.5)
                    except Exception as e:
                        print(f"  Error creating PSF for galaxy {name}: {e}")
                        # Create a reasonable fallback
                        psf_kernel = create_fallback_psf(31)
                    
                    hdu_psf = fits.PrimaryHDU(data=psf_kernel)
                    hdu_psf.header['PSFFWHM'] = (local_fwhm, 'PSF FWHM [arcsec]')
                    hdu_psf.writeto(os.path.join(name, "psf_b.fits"), overwrite=True)

                    print(f"    Saved files for {name}")

        except Exception as e:
            print(f"Error processing brick {brickname}: {e}")
        
        # Limpeza (opcional - descomente se quiser)
        # cleanup_files(brickname, galaxy_indices)

    print("\nAll galaxies processed!")

if __name__ == "__main__":
    main()