import os
import os.path
import numpy as np
from subprocess import call
import itertools as tools
# ##################################################################################
sample='WHL'

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
header_200=['cluster','r200','richness','n200']

#DATA FILES
data_chi2_temp=np.loadtxt(f'{sample}_profit_observation_SE_sky_chi2.dat',dtype=str).T
data_psf_temp=np.loadtxt(f'{sample}_psf_data.dat',dtype=str).T
data_z_temp=np.loadtxt(f'{sample}_clean_redshift.dat',dtype=str).T
data_eta_temp=np.loadtxt(f'ass_{sample}.dat',dtype=str).T
data_obs_temp=np.loadtxt(f'{sample}_profit_observation_SE_sky.dat',dtype=str).T
data_mue_temp=np.loadtxt(f'mue_med_dimm_{sample}.dat',dtype=str).T
data_mue_comp_temp=np.loadtxt(f'mue_med_dimm_comp_{sample}.dat',dtype=str).T
data_casjobs_temp=np.loadtxt(f'casjobs_data_clean_{sample}.dat',dtype=str).T
data_halpha_temp=np.loadtxt(f'casjobs_halpha_{sample}.dat',dtype=str).T
data_veldisp_temp=np.loadtxt(f'casjobs_veldisp_{sample}.dat',dtype=str).T
data_photutils_temp=np.loadtxt(f'graph_stats_WHL_sky_v2.dat',dtype=str).T
data_simul_s_temp=np.loadtxt(f'{sample}_profit_simulation_SE_sky.dat',dtype=str).T
data_simul_ss_temp=np.loadtxt(f'{sample}_profit_simulation_s_duplo_SE_sky.dat',dtype=str).T
data_erro_photutils_temp=np.loadtxt(f'graph_errors_{sample}_sky.dat',dtype=str).T
data_200_temp=np.loadtxt('WHL_r200.dat',dtype=str).T
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
data_200=dict(zip(header_200,data_200_temp))

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


dict_vec=[data_z['cluster'],data_eta['cluster'],data_obs['cluster'],data_psf['cluster'],data_chi2['cluster'],data_casjobs['cluster'],data_halpha['cluster'],data_veldisp['cluster'],data_photutils['cluster'],data_simul_s['cluster'],data_simul_ss['cluster'],data_mue['cluster'],data_mue_comp['cluster'],data_200['cluster']]
name_vec=['data_z','data_eta','data_obs','data_psf','data_chi2','data_casjobs','data_halpha','data_veldisp','data_photutils','data_simul_s','data_simul_ss','data_mue','data_mue_comp','data_200']
lim_photutils=np.isfinite(data_photutils['cluster'][lim_finite & cut_lim].astype(float))
test=[data_obs['cluster'][lim_finite & cut_lim][lim_photutils],data_photutils['cluster'][lim_finite & cut_lim][lim_photutils]]#list(tools.combinations(dict_vec,2))
name_test=list(tools.combinations(name_vec,2))
for i,item in enumerate(test):
	for j,obj in enumerate(item):
		if np.array_equal(obj,test[i+1][j])==False:
			print(obj,test[i+1][j],np.array_equal(obj,test[i+1][j]))
	# if np.array_equal(item,item) == False:
		# print('ok')#name_test[i],np.array_equal(item[0],item[1]))