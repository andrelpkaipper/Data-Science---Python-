import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

def two_line_model(params, x, y):
    """
    Model with two lines and a dividing point
    params = [slope1, intercept1, slope2, intercept2, x_div]
    """
    slope1, intercept1, slope2, intercept2, x_div = params
    
    # Create predictions based on which side of x_div the point falls
    y_pred = np.zeros_like(x)
    mask1 = x <= x_div
    mask2 = x > x_div
    
    y_pred[mask1] = slope1 * x[mask1] + intercept1
    y_pred[mask2] = slope2 * x[mask2] + intercept2
    
    return y_pred

def chi_square(params, x, y, y_err=None):
    """
    Calculate chi-square for the two-line model
    """
    if y_err is None:
        y_err = np.ones_like(y)  # Assume equal errors if not provided
    
    y_pred = two_line_model(params, x, y)
    chi2_val = np.sum(((y - y_pred) / y_err) ** 2)
    return chi2_val

def find_optimal_division(x, y, y_err=None, initial_guess=None):
    """
    Find optimal parameters for two-line model
    """
    if initial_guess is None:
        # Make reasonable initial guesses
        x_min, x_max = np.min(x), np.max(x)
        x_mid = 0.5*(x_min+x_max)
        
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

def plot_results(x, y, params, y_err=None, ax=None):
    """
    Plot the data and fitted two-line model
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    if y_err is not None:
        ax.errorbar(x, y, yerr=y_err, fmt='o', alpha=0.7, label='Data')
    else:
        ax.scatter(x, y, alpha=0.7, label='Data')
    
    # Plot fitted lines
    slope1, intercept1, slope2, intercept2, x_div = params
    
    # Create fine grid for plotting lines
    x_fine = np.linspace(np.min(x), np.max(x), 1000)
    y_pred_fine = two_line_model(params, x_fine, np.zeros_like(x_fine))
    
    # Plot individual lines in different colors
    mask1 = x_fine <= x_div
    mask2 = x_fine > x_div
    
    ax.plot(x_fine[mask1], y_pred_fine[mask1], 'r-', 
            label=f'Line 1: y = {slope1:.3f}x + {intercept1:.3f}', linewidth=2)
    ax.plot(x_fine[mask2], y_pred_fine[mask2], 'g-', 
            label=f'Line 2: y = {slope2:.3f}x + {intercept2:.3f}', linewidth=2)
    
    # Plot dividing line
    ax.axvline(x=x_div, color='k', linestyle='--', 
               label=f'Division at x = {x_div:.3f}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def bootstrap_uncertainty(x, y, params, y_err=None, n_bootstrap=100):
    """
    Estimate uncertainties using bootstrap resampling
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

# Example usage with synthetic data
def generate_example_data():
    """Generate example data with two linear regions"""
    
    # First linear region
    n1 = 150
    x1 = np.linspace(0, 2, n1)
    y1 = -2.0 * x1 + 1 + np.random.normal(0, 1.0, n1)
    
    # Second linear region
    n2 = 200
    x2 = np.linspace(2, 6, n2)
    y2 = -5.5 * x2 + 7 + np.random.normal(0, 1.0, n2)
    
    # Combine
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    
    return x, y

# Main execution
if __name__ == "__main__":
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

    #DATA FILES
    data_chi2_temp=np.loadtxt(f'{sample}_profit_observation_SE_sky_chi2.dat',dtype=str).T
    data_psf_temp=np.loadtxt(f'{sample}_psf_data.dat',dtype=str).T
    data_z_temp=np.loadtxt(f'{sample}_clean_redshift.dat',dtype=str).T
    data_eta_temp=np.loadtxt(f'ass_{sample}.dat').T
    data_obs_temp=np.loadtxt(f'{sample}_profit_observation_SE_sky.dat',dtype=str).T
    data_mue_temp=np.loadtxt(f'mue_med_dimm_{sample}.dat',dtype=str).T
    data_mue_comp_temp=np.loadtxt(f'mue_med_dimm_comp_{sample}.dat',dtype=str).T
    data_casjobs_temp=np.loadtxt(f'casjobs_data_clean_{sample}.dat',dtype=str).T
    data_halpha_temp=np.loadtxt(f'casjobs_halpha_{sample}.dat',dtype=str).T
    data_veldisp_temp=np.loadtxt(f'casjobs_veldisp_{sample}.dat',dtype=str).T
    data_photutils_temp=np.loadtxt(f'graph_stats_WHL_sky.dat',dtype=str).T
    data_simul_s_temp=np.loadtxt(f'{sample}_profit_simulation_SE_sky.dat',dtype=str).T
    data_simul_ss_temp=np.loadtxt(f'{sample}_profit_simulation_s_duplo_SE_sky.dat',dtype=str).T


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
    data_simul_s=dict(zip(header_data,data_simul_s_temp))
    data_simul_ss=dict(zip(header_data,data_simul_ss_temp))
    data_mue=dict(zip(header_mue,data_mue_temp))
    data_mue_comp=dict(zip(header_mue,data_mue_comp_temp))

    rff_ss=np.loadtxt(f'{sample}_rff_duplo.dat',dtype=float,usecols=[1]).T
    mue_med_s,mue_med_1,mue_med_2=data_mue['mue_med_s'].astype(float),data_mue['mue_med_1'].astype(float),data_mue['mue_med_2'].astype(float)
    mue_med_s,mue_med_comp_1,mue_med_comp_2=data_mue_comp['mue_med_s'].astype(float),data_mue_comp['mue_med_1'].astype(float),data_mue_comp['mue_med_2'].astype(float)
    re1,re2=data_obs['RE_1'].astype(float),data_obs['RE_2'].astype(float)
    n1,n2=data_obs['NSER_1'].astype(float),data_obs['NSER_2'].astype(float)

    lim_finite=np.isfinite(mue_med_s) & np.isfinite(mue_med_comp_1) & np.isfinite(mue_med_comp_2)

    n1_lim=(n1==0.5) | (n1>9.9)
    n2_lim=(n2==0.5) | (n2>14.9)

    re1_lim=(re1<=1.05)
    re2_lim=(re2<=1.05)

    cut_lim=~(n1_lim | n2_lim | re1_lim | re2_lim)
    
    # You can provide measurement errors if available
    y_err = None  # or np.array of errors
    
    print("Fitting two-line model...")
    
    # Find optimal parameters
    result = find_optimal_division(x, y, y_err)
    
    if result.success:
        params_opt = result.x
        chi2_min = result.fun
        dof = len(x) - 5  # degrees of freedom
        
        print(f"\nOptimization successful!")
        print(f"Minimum chi-square: {chi2_min:.3f}")
        print(f"Degrees of freedom: {dof}")
        print(f"Reduced chi-square: {chi2_min/dof:.3f}")
        
        print(f"\nOptimal parameters:")
        print(f"Slope 1: {params_opt[0]:.3f}")
        print(f"Intercept 1: {params_opt[1]:.3f}")
        print(f"Slope 2: {params_opt[2]:.3f}")
        print(f"Intercept 2: {params_opt[3]:.3f}")
        print(f"Dividing x: {params_opt[4]:.3f}")
        
        # Estimate uncertainties
        uncertainties = bootstrap_uncertainty(x, y, params_opt, y_err)
        print(f"\nEstimated uncertainties:")
        print(f"Slope 1 uncertainty: ±{uncertainties[0]:.3f}")
        print(f"Intercept 1 uncertainty: ±{uncertainties[1]:.3f}")
        print(f"Slope 2 uncertainty: ±{uncertainties[2]:.3f}")
        print(f"Intercept 2 uncertainty: ±{uncertainties[3]:.3f}")
        print(f"Dividing x uncertainty: ±{uncertainties[4]:.3f}")
        
        # Create figure with subplots - FIXED VERSION
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot main results on first subplot
        plot_results(x, y, params_opt, y_err, ax=ax1)
        ax1.set_title('Two-Line Fit')
        
        # Plot residuals on second subplot
        y_pred = two_line_model(params_opt, x, y)
        residuals = y - y_pred
        ax2.scatter(x, residuals, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Optimization failed!")
        print(result.message)
