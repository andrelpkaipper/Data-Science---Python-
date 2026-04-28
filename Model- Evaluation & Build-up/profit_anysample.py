import itertools
import os
from subprocess import call
from astropy.io import fits
from scipy import optimize
from scipy import stats
import numpy as np
from profit_optim_v4 import profit_setup_data, profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
import multiprocessing as mp
import warnings
import numpy.ma as ma
warnings.filterwarnings("ignore")


def make_callback(cluster,data,model):
	hist_likelihood=open(f'{sample}/{cluster}/hist_likelihood_{model}.dat','a')
	hist_params=open(f'{sample}/{cluster}/hist_params_{model}.dat','a')
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		hist_likelihood.write(f'{ll}\n')
		hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_s(cluster,data,model):
	hist_likelihood=open(f'{sample}/{cluster}/hist_likelihood_simul_s_{model}.dat','a')
	hist_params=open(f'{sample}/{cluster}/hist_params_simul_s_{model}.dat','a')
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		hist_likelihood.write(f'{ll}\n')
		hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_se(cluster,data,model):
	hist_likelihood=open(f'{sample}/{cluster}/hist_likelihood_simul_se_{model}.dat','a')
	hist_params=open(f'{sample}/{cluster}/hist_params_simul_se_{model}.dat','a')
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		hist_likelihood.write(f'{ll}\n')
		hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def rff_calc(sample,cluster,figname):
	
	ajust1 = fits.getdata(f'{sample}/{cluster}/{figname}',1)
	ajust3 = fits.getdata(f'{sample}/{cluster}/{figname}',3)

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
	def norm_with_fixed_sigma(x):
		return stats.norm.logpdf(x, 0, s)
	return norm_with_fixed_sigma
def centro_magzero(sample,cluster):
	import photutils.psf as ppsf
	import photutils.aperture as phta

	mask_b=fits.open(f'../{sample}/{cluster}/bcg_r_mask_b.fits',memmap=True)[0].data
	data2=fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[1].data
	header = fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[1].header
	model_header=fits.open(f'../{sample}/{cluster}/ajust-bcg-r.fits',memmap=True)[2].header
	sigma_img=fits.open(f'../{sample}/{cluster}/sigma-r.fits',memmap=True)[0].data
	mask=fits.open(f'../{sample}/{cluster}/bcg_r_mask.fits',memmap=True)[0].data
	psf=fits.open(f'../{sample}/{cluster}/bcg_r_psf_b.fits',memmap=True)[0].data
	#######################################
	NMGY = float(header['NMGY'])
	EXPTIME = float(header['EXPTIME'])
	data = data2*NMGY/EXPTIME
	vec = data[np.where(mask_b == 1)]

	magzero=float(model_header['MAGZPT'])+2.5*np.log10(EXPTIME)

	sky=data2[(mask==0) & (mask_b==0)]
	sigma_sky=np.std(sky)
	median_sky=np.median(sky)

	xc=float(model_header['1_XC'].split()[0].replace('*',''))#xy_center[0][0]
	yc=float(model_header['1_YC'].split()[0].replace('*',''))#xy_center[1][0]
	mag=22.5-2.5*np.log10(sum(vec)) 
	pa=float(model_header['1_PA'].split()[0].replace('*',''))
	ax=float(model_header['1_AR'].split()[0].replace('*',''))
	re=np.max(data2.shape)/1.8
	
	psf_fit=np.ceil(ppsf.fit_fwhm(psf))
	rad_mask=phta.CircularAperture((xc,yc),int(1.5*psf_fit)).to_mask()
	mask_center=rad_mask.to_image(data2.shape).astype(int)

	return xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,mask_center
def infotype(modeltype):
	if modeltype==0:
		name_s_ajust=f'ajust-sersic-llh.fits'
		name_se_ajust=f'ajust-sersic-exp-llh.fits'
		name_ss_ajust=f'ajust-sersic-duplo-llh.fits'
		save_file='output_obs.dat'
		return name_s_ajust,name_se_ajust,name_ss_ajust,save_file
	elif modeltype==1:
		name_s_ajust=f'ajust-simul-s-sersic-llh.fits'
		name_se_ajust=f'ajust-simul-s-sersic-exp-llh.fits'
		name_ss_ajust=f'ajust-simul-s-sersic-duplo-llh.fits'
		save_file='output_simul_s.dat'
		return name_s_ajust,name_se_ajust,name_ss_ajust,save_file
	elif modeltype==2:
		name_s_ajust=f'ajust-simul-se-sersic-llh.fits'
		name_se_ajust=f'ajust-simul-se-sersic-exp-llh.fits'
		name_ss_ajust=f'ajust-simul-se-sersic-duplo-llh.fits'
		save_file='output_simul_se.dat'
		return name_s_ajust,name_se_ajust,name_ss_ajust,save_file
##############################################################################################
#CONSTRUÇÃO INICIAL DOS MODELOS
def sersic_unico(sample,cluster):
	#####
	xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,_=centro_magzero(sample,cluster)
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
def sersic_duplo(sample,cluster,data_entry,img_siz):

	xc,yc,mag,re,n,pa,ax,box,sky = data_entry
	sigma_sky=centro_magzero(sample,cluster)[7]
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
	return names, model0, tofit, tolog, sigmas, priors, lowers, uppers
###########################################################################################
#CONSTRUÇÃO DOS MODELOS

def sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype):
	print(f'{cluster} SERSIC')
	name_s_ajust=infotype(modeltype)[0]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{name_s_ajust}',2)		
		sersic_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY']
		all_params_sersic=[]
		for item in sersic_names:
			all_params_sersic.append(float(ajuste[item].split()[0].replace('*','')))

		magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster)
		print(f'{cluster} SERSIC FINALIZADO')
		return all_params_sersic,magzero,model0
	except:
		magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster)
		mask_center=centro_magzero(sample,cluster)[-1]
		data_sersic = profit_setup_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic.verbose = False

		if modeltype == 0:
			callback=make_callback(cluster,data_sersic,'sersic')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic,'sersic')
		elif modeltype == 2:
			callback=make_callback_simul_se(cluster,data_sersic,'sersic')

		result_sersic = optimize.minimize(profit_like_model, data_sersic.init, args=(data_sersic,), method='L-BFGS-B', bounds=data_sersic.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic = to_pyprofit_image_simples(data_sersic.init, data_sersic, use_mask=True)
		
		all_params_sersic, modelim_sersic = to_pyprofit_image_simples(result_sersic.x, data_sersic, use_mask=True)

		clean_max_llh_sersic,unclean_max_llh_sersic,img_likelihood_sersic=clean_model(result_sersic.x,data_sersic)
		fits.writeto(f'{sample}/{cluster}/likelihood_{name_s_ajust}',img_likelihood_sersic,overwrite=True)
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
		model_img.header['LLH_1']=clean_max_llh_sersic
		model_img.header['LLH_0']=unclean_max_llh_sersic
		model_img.header['FLAG']=data_sersic.check_model
		model_img.header['DOF']=data_sersic.dof

		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
		hdulist.writeto(f'{sample}/{cluster}/{name_s_ajust}',overwrite=True)

	print(f'{cluster} SERSIC FINALIZADO')
	return all_params_sersic,magzero,model0
def sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,magzero,fit_params_sersic,modeltype):
	print(f'{cluster} SERSIC + EXP')
	name_se_ajust=infotype(modeltype)[1]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{name_se_ajust}',2)

		sersic_exp_names=['XCEN_S', 'YCEN_S', 'MAG_S', 'RE_S', 'NSER_S', 'ANG_S', 'AXRAT_S', 'BOX_S', 'MAG_E', 'RE_E', 'NSER_E', 'ANG_E', 'AXRAT_E', 'BOX_E', 'SKY']
		sersic_exp_header=fits.getheader(f'{sample}/{cluster}/{name_se_ajust}',2)
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
		mask_center=centro_magzero(sample,cluster)[-1]
		data_sersic_exp = profit_setup_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic_exp.verbose = False
		if modeltype == 0:
			callback=make_callback(cluster,data_sersic_exp,'sersic_exp')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic_exp,'sersic_exp')
		elif modeltype == 2:
			callback=make_callback_simul_se(cluster,data_sersic_exp,'sersic_exp')

		result_sersic_exp = optimize.minimize(profit_like_model, data_sersic_exp.init, args=(data_sersic_exp,), method='L-BFGS-B', bounds=data_sersic_exp.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic_exp = to_pyprofit_image_duplo(data_sersic_exp.init, data_sersic_exp, use_mask=True)
		all_params_sersic_exp, modelim_sersic_exp  = to_pyprofit_image_duplo(result_sersic_exp.x, data_sersic_exp, use_mask=True)

		clean_max_llh_sersic_exp,unclean_max_llh_sersic_exp,img_likelihood_sersic_exp=clean_model(result_sersic_exp.x,data_sersic_exp)
		fits.writeto(f'{sample}/{cluster}/likelihood_{name_se_ajust}',img_likelihood_sersic_exp,overwrite=True)

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
		model_img.header['LLH_1']=clean_max_llh_sersic_exp
		model_img.header['LLH_0']=unclean_max_llh_sersic_exp
		model_img.header['FLAG']=data_sersic_exp.check_model
		model_img.header['DOF']=data_sersic_exp.dof
		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])

		hdulist.writeto(f'{sample}/{cluster}/{name_se_ajust}',overwrite=True)

		print(f'{cluster} SERSIC + EXP FINALIZADO')
	return
def sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,magzero,fit_params_sersic,modeltype):
	print(f'{cluster} SERSIC 2 + SERSIC 1')
	name_ss_ajust=infotype(modeltype)[2]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{name_ss_ajust}',2)
		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO')
		return
	except:
		if fit_params_sersic[0] != -1000:
			data_entry=fit_params_sersic
		else:
			data_entry=fit_params_sersic

		names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_duplo(sample,cluster,data_entry,image.shape)
		mask_center=centro_magzero(sample,cluster)[-1]
		data_sersic_duplo = profit_setup_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic_duplo.verbose = False
		if modeltype == 0:
			callback=make_callback(cluster,data_sersic_duplo,'sersic_duplo')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic_duplo,'sersic_duplo')
		elif modeltype == 2:
			callback=make_callback_simul_se(cluster,data_sersic_duplo,'sersic_duplo')
		result_sersic_duplo = optimize.minimize(profit_like_model, data_sersic_duplo.init, args=(data_sersic_duplo,), method='L-BFGS-B', bounds=data_sersic_duplo.bounds, options={'disp':True},callback=callback)
		_, modelim0_sersic_duplo = to_pyprofit_image_duplo(data_sersic_duplo.init, data_sersic_duplo, use_mask=True)
		all_params_sersic_duplo, modelim_sersic_duplo  = to_pyprofit_image_duplo(result_sersic_duplo.x, data_sersic_duplo, use_mask=True)
		####
		clean_max_llh_sersic_duplo,unclean_max_llh_sersic_duplo,img_likelihood_sersic_duplo=clean_model(result_sersic_duplo.x,data_sersic_duplo)
		fits.writeto(f'{sample}/{cluster}/likelihood_{name_ss_ajust}',img_likelihood_sersic_duplo,overwrite=True)

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
		model_img.header['LLH_1']=clean_max_llh_sersic_duplo
		model_img.header['LLH_0']=unclean_max_llh_sersic_duplo
		model_img.header['FLAG']=data_sersic_duplo.check_model
		model_img.header['DOF']=data_sersic_duplo.dof
		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])

		hdulist.writeto(f'{sample}/{cluster}/{name_ss_ajust}',overwrite=True)

		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO')
	return 
############################################################################################
def sample_setup(sample,cluster):
	call(f'mkdir {sample}/{cluster}',shell=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	mask_center=centro_magzero(sample,cluster)[-1]

	image = np.array(fits.getdata(f'../{sample}/{cluster}/ajust-bcg-r.fits',1))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype)
	ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	return 
def simul_s_setup(sample,cluster):
	##############################################################
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	model_s_img=fits.getdata(f'{sample}/{cluster}/observation/ajust-sersic-llh.fits',2)
	noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
	image = model_s_img+noise_img
	modeltype=1
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	# ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)

	return 
def simul_se_setup(sample,cluster):
	##############################################################
	##IMAGENS
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))

	model_se_img=fits.getdata(f'{sample}/{cluster}/ajust-sersic-exp-llh.fits',2)
	noise_img=np.random.normal(loc=0,scale=sigim,size=model_se_img.shape)
	image = model_se_img+noise_img

	modeltype=2
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype)
	ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)

	with open(checkpath,'a') as check:
		check.write(f"{cluster}\n")

	return 

def finish_details(sample,cluster,modeltype,rff_galfit,ass):
	#old_data=np.loadtxt('profit_L07_output_obs.dat',dtype=str,usecols=[0,-6]).T

	#cl_id=old_data[0].index(cluster)
	#bm_old=old_data[1][cl_id]

	model_names=infotype(modeltype)

	sersic_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY', 'MAX_LLH', 'BIC','LLH_1','FLAG','DOF']
	sersic_exp_names=['XCEN_S', 'YCEN_S', 'MAG_S', 'RE_S', 'NSER_S', 'ANG_S', 'AXRAT_S', 'BOX_S', 'MAG_E', 'RE_E', 'NSER_E', 'ANG_E', 'AXRAT_E', 'BOX_E', 'SKY', 'MAX_LLH', 'BIC','LLH_1','FLAG','DOF']
	sersic_duplo_names=['XCEN_1', 'YCEN_1', 'MAG_1', 'RE_1', 'NSER_1', 'ANG_1', 'AXRAT_1', 'BOX_1', 'MAG_2', 'RE_2', 'NSER_2', 'ANG_2', 'AXRAT_2', 'BOX_2', 'SKY', 'MAX_LLH', 'BIC','LLH_1','FLAG','DOF']

	rff=rff_calc(sample,cluster,model_names[0])
	with open('rff_WHL.dat','a') as rff_data:
		rff_data.write(f'{cluster} {rff} {ass}\n')
	'''
	image = np.array(fits.getdata(f'../{sample}/{cluster}/ajust-bcg-r.fits',1))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	img_likelihood_sersic=np.array(fits.getdata(f'{sample}/{cluster}/likelihood_{model_names[0]}'))
	img_likelihood_sersic_exp=np.array(fits.getdata(f'{sample}/{cluster}/likelihood_{model_names[1]}'))
	img_likelihood_sersic_duplo=np.array(fits.getdata(f'{sample}/{cluster}/likelihood_{model_names[2]}'))

	#################################
	#RESULTADOS SERSIC	
	sersic_header=fits.getheader(f'{sample}/{cluster}/{model_names[0]}',2)
	sersic_dict=dict(sersic_header)
	result_sersic=[]
	for param in sersic_names:
		result_sersic.append(str(sersic_dict[param]).split()[0])	
	#################################
	#RESULTADOS SERSIC+EXP
	sersic_exp_header=fits.getheader(f'{sample}/{cluster}/{model_names[1]}',2)
	sersic_exp_dict=dict(sersic_exp_header)
	result_sersic_exp=[]
	for param in sersic_exp_names:
		result_sersic_exp.append(str(sersic_exp_dict[param]).split()[0])
	#################################
	#RESULTADOS SERSIC+SERSIC
	sersic_duplo_header=fits.getheader(f'{sample}/{cluster}/{model_names[2]}',2)
	sersic_duplo_dict=dict(sersic_duplo_header)
	result_sersic_duplo=[]
	for param in sersic_duplo_names:
		result_sersic_duplo.append(str(sersic_duplo_dict[param]).split()[0])
	#################################
	#ESCOLHA DO BIC E CONFECÇÃO DOS DELTAS

	bic_str=['S','S+E','S+S']

	## TODOS OS PIXEIS
	bic_sersic=float(result_sersic[-4])
	bic_sersic_exp=float(result_sersic_exp[-4])
	bic_sersic_duplo=float(result_sersic_duplo[-4])

	max_llh_sersic=-float(result_sersic[-5])
	max_llh_sersic_exp=-float(result_sersic_exp[-5])
	max_llh_sersic_duplo=-float(result_sersic_duplo[-5])

	bic_vec=np.array((bic_sersic,bic_sersic_exp,bic_sersic_duplo))
	best_model=bic_str[np.argmin(bic_vec)]

	#CONFECÇÃO DAS IMAGENS DE DELTAS

	img_vec=[img_likelihood_sersic,img_likelihood_sersic_exp,img_likelihood_sersic_duplo]
	print(cluster,best_model)
	# delta_img=img_likelihood_sersic-img_vec[np.argmin(bic_vec)]
	# fits.writeto(f'{sample}/{cluster}/delta_S_bm.fits',delta_img,overwrite=True)

	# delta_s=max_llh_sersic_exp-max_llh_sersic
	# delta_se=max_llh_sersic_duplo-max_llh_sersic_exp

	# output=open(f'{sample}_{model_names[3]}','a')
	# output.write(f'%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s \n'%(cluster,*result_sersic,*result_sersic_exp,*result_sersic_duplo,rff,ass,best_model,delta_s,delta_se))
	# output.close()
	'''
	return #best_model
##########################################
if __name__ == '__main__':
	import sys
	sample=sys.argv[1]
	ok=[]

	with open(f'check_done_{sample}.dat','r') as inp2:
		for item in inp2.readlines():   
			ok.append(item.split()[0])
	save_path=f'check_done_{sample}.dat'
	catalogo=[]
	with open(f'../{sample}/pargal_{sample}_compact_astro_vfix.dat','r') as inp1:
		data_sample=inp1.readlines()
		for obj in data_sample:
			flagmask=int(float(obj.split()[11]))
			flagdelta=int(float(obj.split()[12]))
			cluster=obj.split()[0]
			if [flagmask,flagdelta] == [0,0] and cluster != '2102' and cluster!='3338':
				catalogo.append(cluster)
				rff_galfit=float(obj.split()[7])
				ass=float(obj.split()[9])

				# modeltype=[0,1,2]
					# for type in modeltype:
					# 	finish_details(sample,cluster,type,rff_galfit,ass)
				#try:
				# finish_details(sample,cluster,0,rff_galfit,ass)
				#except:
				#	pass
	with mp.Pool(processes=15) as pool:
		chunksize=3
		pool.starmap(sample_setup,[(sample,galaxia) for galaxia in catalogo],chunksize=chunksize)

	# with mp.Pool(processes=16) as pool:
	# 	chunksize=1
	# 	pool.starmap(simul_s_setup,[(sample,galaxia) for galaxia in catalogo],chunksize=chunksize)

	# with mp.Pool(processes=16) as pool:
	# 	chunksize=1
	# 	pool.starmap(simul_se_setup,[(sample,galaxia,ok,save_path) for galaxia in catalogo],chunksize=chunksize)

	# call(f'rm -r L07/{cluster}/ajust-sersic-exp-llh.*',shell=True)
	# call(f'rm -r L07/{cluster}/ajust-sersic-duplo-llh.*',shell=True)