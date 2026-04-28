import itertools
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import shutil
from subprocess import call
from astropy.io import fits
from scipy import optimize
from scipy import stats
import numpy as np
import multiprocessing as mp
import warnings
import numpy.ma as ma

warnings.filterwarnings("ignore")
def make_callback(cluster, data, save_value, model):
	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		with open(f'{sample}/{cluster}/{save_value}/hist_likelihood_{model}_{save_value}.dat','a') as hist_likelihood:
			hist_likelihood.write(f'{ll}\n')
		with open(f'{sample}/{cluster}/{save_value}/hist_params_{model}_{save_value}.dat','a') as hist_params:
			hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_s(cluster,data,save_value,model):
	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		with open(f'{sample}/{cluster}/{save_value}/hist_likelihood_{model}_{save_value}.dat','a') as hist_likelihood:
			hist_likelihood.write(f'{ll}\n')
		with open(f'{sample}/{cluster}/{save_value}/hist_params_{model}_{save_value}.dat','a') as hist_params:
			hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
def make_callback_simul_se(cluster,data,save_value,model):
	hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_simul_se_{model}_{save_value}.dat','a')
	hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_simul_se_{model}_{save_value}.dat','a')
	def callback_fn(params_iter):
		ll = -profit_like_model(params_iter, data)
		hist_likelihood.write(f'{ll}\n')
		hist_params.write(f'{" ".join(str(x) for x in params_iter.copy())}\n')
	return callback_fn
#####################################################
def rff_calc(sample,cluster,figname,save_value):

	ajust1 = fits.getdata(f'{sample}/{cluster}/{save_value}/{figname}',1)
	ajust3 = fits.getdata(f'{sample}/{cluster}/{save_value}/{figname}',3)


	mask = fits.getdata(f'../{sample}_desi/{cluster}/bcg_r_mask.fits')
	mask_b = fits.getdata(f'../{sample}_desi/{cluster}/bcg_r_mask_b.fits')

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
def centro_magzero(sample,cluster,save_value,ra,dec):#DESI LEVANTAMENTO
	import photutils.psf as ppsf
	import photutils.aperture as phta
	from math import floor
	from astropy.wcs import WCS
	from astropy.coordinates import SkyCoord

	data2_path=f'../{sample}_desi/{cluster}/bcg_r.fits'
	cluster_path=f'{sample}/{cluster}/{save_value}'

	mask_b=fits.open(f'../{sample}_desi/{cluster}/bcg_r_mask_b.fits',memmap=True)[0].data
	data=fits.open(f'../{sample}_desi/{cluster}/bcg_r.fits',memmap=True)[0].data
	header=fits.open(f'../{sample}_desi/{cluster}/bcg_r.fits',memmap=True)[0].header
	sigma_img=fits.open(f'../{sample}_desi/{cluster}/sigma-r.fits',memmap=True)[0].data
	mask=fits.open(f'../{sample}_desi/{cluster}/bcg_r_mask.fits',memmap=True)[0].data
	psf=fits.open(f'../{sample}_desi/{cluster}/psf_b.fits',memmap=True)[0].data
	#####################
	vec = data[np.where(mask_b == 1)]
	vec_1= data[np.where(mask == 0)]

	magzero=22.5

	sky=data[(mask==0) & (mask_b==0)]
	sigma_sky=np.std(sky)
	median_sky=np.median(sky)

	pixel_finder=WCS(header)
	pos_bcg = SkyCoord(ra, dec, unit='deg', frame='fk5')
	x0,y0=pixel_finder.world_to_pixel(pos_bcg)

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
				pa=float(lla[11])
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
def split_view_desi(sample,cluster,modeltype,save_value,ra,dec):
	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model, build_model_simples, build_model_duplo

	image = np.array(fits.getdata(f'../{sample}_desi/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../{sample}_desi/{cluster}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}_desi/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}_desi/{cluster}/psf_b.fits'))

	name_model=infotype(modeltype,save_value)[2]
	param_models=['XCEN_1', 'YCEN_1', 'MAG_1', 'RE_1', 'NSER_1', 'ANG_1', 'AXRAT_1', 'BOX_1', 'MAG_2', 'RE_2', 'NSER_2', 'ANG_2', 'AXRAT_2', 'BOX_2','SKY']

	ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_model}',2)
	xc,yc,mag1,re1,n1,pa1,ax1,box1,mag2,re2,n2,pa2,ax2,box2,sky=param_models

	comp1=[xc,yc,mag1,re1,n1,pa1,ax1,box1,sky]
	comp2=[xc,yc,mag2,re2,n2,pa2,ax2,box2,sky]
	param_comp1=[]
	param_comp2=[]
	param_comp_all=[]
	for item in comp1:
		param_comp1.append(float(ajuste[item].split()[0].replace('*','')))
	for item in comp2:
		param_comp2.append(float(ajuste[item].split()[0].replace('*','')))
	for item in param_models:
		param_comp_all.append(float(ajuste[item].split()[0].replace('*','')))

	param_comp1=np.asarray(param_comp1)
	param_comp2=np.asarray(param_comp2)

	####
	mask_center=centro_magzero(sample,cluster,save_value,ra,dec)[-2]
	magzero, names, model_entry_s, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster,save_value,ra,dec)

	#S+S
	names, model_entry_duplo, tofit, tolog, sigmas, priors, lowers, uppers = sersic_duplo(sample,cluster,model_entry_s,image.shape,save_value,ra,dec)
	data_sersic=build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model_entry_duplo, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)
	###
	sky=param_comp1[-1]
	all_params_comp1, modelim_comp1 = build_model_simples(param_comp1, data_sersic, use_mask=True,sky=True)
	all_params_comp2, modelim_comp2 = build_model_simples(param_comp2, data_sersic, use_mask=True,sky=True)
	ajuste_img=fits.getdata(f'{sample}/{cluster}/{save_value}/{name_model}',2)
	#print(np.sum(modelim_comp1-sky),np.sum(modelim_comp2),np.sum(modelim_comp1-sky)+np.sum(modelim_comp2),np.sum(ajuste_img))
	#FAZ A IMG DE AJUST DO PROFIT
	bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
	model_img=fits.ImageHDU(modelim_comp1,name='MODEL')
	resid_img=fits.ImageHDU(image - modelim_comp1,name='RESIDUAL')
	hdu0=fits.PrimaryHDU()
	hdulist0=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
	hdulist0.writeto(f'{sample}/{cluster}/{save_value}/comp_1_{save_value}.fits',overwrite=True)

	bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
	model_img=fits.ImageHDU(modelim_comp2,name='MODEL')
	resid_img=fits.ImageHDU(image - modelim_comp2,name='RESIDUAL')
	hdu1=fits.PrimaryHDU()
	hdulist1=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
	hdulist1.writeto(f'{sample}/{cluster}/{save_value}/comp_2_{save_value}.fits',overwrite=True)
	# bcg_img=fits.ImageHDU(image,name='BCG_STAMP')
	# model_img=fits.ImageHDU(modelim_comp,name='MODEL')
	# resid_img=fits.ImageHDU(image - modelim_comp,name='RESIDUAL')
	# hdu0=fits.PrimaryHDU()
	# hdulist2=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
	# hdulist2.writeto(f'{sample}/{cluster}/{save_value}/comp_{double_type}_nosky.fits',overwrite=True)

	return
##############################################################################################
#CONSTRUÇÃO INICIAL DOS MODELOS
def sersic_unico(sample,cluster,save_value,ra,dec):
	#####
	xc,yc,mag,pa,ax,re,magzero,sigma_sky,median_sky,_,_=centro_magzero(sample,cluster,save_value,ra,dec)
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
def sersic_duplo(sample,cluster,data_entry,img_siz,save_value,ra,dec):

	xc,yc,mag,re,n,pa,ax,box,sky = data_entry

	sigma_sky=centro_magzero(sample,cluster,save_value,ra,dec)[7]
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
def infotype(modeltype,save_value):
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
		name_s_ajust=f'ajust-simul-se-sersic-llh-{save_value}.fits'
		name_se_ajust=f'ajust-simul-se-sersic-exp-llh-{save_value}.fits'
		name_ss_ajust=f'ajust-simul-se-sersic-duplo-llh-{save_value}.fits'
		save_file='output_simul_se.dat'
	return	name_s_ajust,name_se_ajust,name_ss_ajust,save_file
def build_data(cluster, image, mask, sigim, segim, psf,	magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center=None):
	from profit_optim_v4 import profit_setup_data

	data = profit_setup_data(
	cluster, image.copy(), mask.copy(), sigim.copy(), segim.copy(), psf.copy(),
	magzero, names[:], model0.copy(), tofit.copy(), tolog.copy(),
	sigmas.copy(), priors.copy(), lowers.copy(), uppers.copy(),
	mask_center if mask_center is not None else np.zeros_like(image, dtype=bool))

	return data
def sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value,ra,dec):
	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	print(f'{cluster} SERSIC')
	name_s_ajust=infotype(modeltype,save_value)[0]
	try:#{sample}/{cluster}
		ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',2)		
		sersic_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY']
		all_params_sersic=[]
		for item in sersic_names:
			all_params_sersic.append(float(ajuste[item].split()[0].replace('*','')))

		magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster,save_value,ra,dec)
		print(f'{cluster} SERSIC FINALIZADO')
		return all_params_sersic,magzero,model0
	except:
		try:
			magzero, names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_unico(sample,cluster,save_value,ra,dec)
			mask_center=centro_magzero(sample,cluster,save_value,ra,dec)[-2]
			data_sersic = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

			data_sersic.verbose = False
			if modeltype == 0:
				hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_sersic_{save_value}.dat','w')
				hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_sersic_{save_value}.dat','w')
				callback=make_callback(cluster,data_sersic,save_value,'sersic')
			elif modeltype == 1:
				callback=make_callback_simul_s(cluster,data_sersic,save_value,'sersic')
			elif modeltype == 2:
				callback=make_callback_simul_se(cluster,data_sersic,save_value,'sersic')
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
			model_img.header['FLAG2']=result_sersic.success
			hdu0=fits.PrimaryHDU()

			hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
			hdulist.writeto(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',overwrite=True)

			print(f'{cluster} SERSIC FINALIZADO -- CONVERGENCIA {result_sersic.success}')
			return all_params_sersic,magzero,model0
		except:
			with open(f'{sample}_fail_log.dat','a') as fail_log:
				fail_log.write(f'{cluster}\n')
			return None,None,None
def sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,magzero,fit_params_sersic,modeltype,save_value):
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
def sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,magzero,fit_params_sersic,modeltype,save_value,ra,dec):
	from profit_optim_v4 import profit_like_model, to_pyprofit_image_simples, to_pyprofit_image_duplo, clean_model
	print(f'{cluster} SERSIC 2 + SERSIC 1')
	name_ss_ajust=infotype(modeltype,save_value)[2]
	try:
		ajuste=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}',2)
		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO')
		return
	except:
		if fit_params_sersic == None:
			return
		name_s_ajust=infotype(modeltype,save_value)[0]
		if os.path.isfile(f'{sample}/{cluster}/{save_value}/{name_s_ajust}'):
			ajuste_sersic=fits.getheader(f'{sample}/{cluster}/{save_value}/{name_s_ajust}',2)		
			data_entry = [float(ajuste_sersic[k].split()[0].replace('*','')) for k in ('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')]
		else:
			data_entry = [0.0 for k in ('XCEN','YCEN','MAG','RE','NSER','ANG','AXRAT','BOX','SKY')]
		
		names, model0, tofit, tolog, sigmas, priors, lowers, uppers = sersic_duplo(sample,cluster,data_entry,image.shape,save_value,ra,dec)
		mask_center=centro_magzero(sample,cluster,save_value,ra,dec)[-2]
		data_sersic_duplo = build_data(cluster,image, mask, sigim, segim, psf,magzero,names, model0, tofit, tolog, sigmas, priors, lowers, uppers,mask_center)

		data_sersic_duplo.verbose = False
		if modeltype == 0:
			hist_likelihood=open(f'{sample}/{cluster}/{save_value}/hist_likelihood_sersic_duplo_{save_value}.dat','a')
			hist_params=open(f'{sample}/{cluster}/{save_value}/hist_params_sersic_duplo_{save_value}.dat','a')
			callback=make_callback(cluster,data_sersic_duplo,save_value,'sersic_duplo')
		elif modeltype == 1:
			callback=make_callback_simul_s(cluster,data_sersic_duplo,save_value,'sersic_duplo')
		elif modeltype == 2:
			callback=make_callback_simul_se(cluster,data_sersic_duplo,save_value,'sersic_duplo')
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
		model_img.header['FLAG2']=result_sersic_duplo.success
		hdu0=fits.PrimaryHDU()

		hdulist=fits.HDUList([hdu0,bcg_img,model_img,resid_img])
		hdulist.writeto(f'{sample}/{cluster}/{save_value}/{name_ss_ajust}',overwrite=True)

		print(f'{cluster} SERSIC 2 + SERSIC 1 FINALIZADO -- CONVERGENCIA {result_sersic_duplo.success}')
		return 
############################################################################################
def sample_setup(sample,cluster,save_value):
	call(f'mkdir {sample}/{cluster}',shell=True)
	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	image = np.array(fits.getdata(f'../{sample}/{cluster}/ajust-bcg-r.fits',1))
	# image = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)	
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)
	return 
def simul_s_setup(sample,cluster,save_value):
	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################
	sigim = np.array(fits.getdata(f'../{sample}/{cluster}/sigma-r.fits'))
	mask  = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r_psf_b.fits'))
	model_s_img=fits.getdata(f'{sample}/{cluster}/observation_SE_v2/ajust-sersic-llh-observation_SE_v2.fits',2)
	noise_img=np.random.normal(loc=0,scale=sigim,size=model_s_img.shape)
	image = model_s_img+noise_img
	modeltype=1
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype)
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)

	return 
def desi_setup(sample,cluster,save_value,ra,dec):
	call(f'mkdir {sample}/{cluster}/{save_value}',shell=True)
	##############################################################{sample}/{cluster}
	##IMAGENS
	image = np.array(fits.getdata(f'../{sample}_desi/{cluster}/bcg_r.fits'))
	# image = np.array(fits.getdata(f'../{sample}/{cluster}/bcg_r.fits'))
	sigim = np.array(fits.getdata(f'../{sample}_desi/{cluster}/sigma-r.fits'))	
	mask  = np.array(fits.getdata(f'../{sample}_desi/{cluster}/bcg_r_mask.fits')).astype(np.bool_)
	segim = np.logical_not(mask)
	psf   = np.array(fits.getdata(f'../{sample}_desi/{cluster}/psf_b.fits'))
	modeltype=0
	################################################################
	ajuste_sersic = sersic_fit(sample,cluster,image, mask, sigim, segim, psf,modeltype,save_value,ra,dec)
	# ajuste_sersic_exp = sersic_exp_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value)	
	ajuste_sersic_duplo = sersic_duplo_fit(sample,cluster,image, mask, sigim, segim, psf,ajuste_sersic[1],ajuste_sersic[0],modeltype,save_value,ra,dec)
	return 
def finish_details(sample,cluster,modeltype,rff_galfit,ass,save_value):
	#results_dir=f'{sample}/{cluster}/{save_value}'#f'{sample}/{cluster}'
	ok=[]
	with open(f'{sample}_profit_{save_value}.dat','r') as inp2:
		for item in inp2.readlines():   
			ok.append(item.split()[0])

	if cluster in ok:
		pass
	else:
		# try:
		model_names=infotype(modeltype,save_value)

		sersic_names=['XCEN', 'YCEN', 'MAG', 'RE', 'NSER', 'ANG', 'AXRAT', 'BOX', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']
		sersic_exp_names=['XCEN_S', 'YCEN_S', 'MAG_S', 'RE_S', 'NSER_S', 'ANG_S', 'AXRAT_S', 'BOX_S', 'MAG_E', 'RE_E', 'NSER_E', 'ANG_E', 'AXRAT_E', 'BOX_E', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']
		sersic_duplo_names=['XCEN_1', 'YCEN_1', 'MAG_1', 'RE_1', 'NSER_1', 'ANG_1', 'AXRAT_1', 'BOX_1', 'MAG_2', 'RE_2', 'NSER_2', 'ANG_2', 'AXRAT_2', 'BOX_2', 'SKY', 'MAX_LLH', 'BIC','CL_LLH']

		rff=rff_calc(sample,cluster,model_names[0],save_value)	
		img_likelihood_sersic=np.array(fits.getdata(f'{sample}/{cluster}/{save_value}/likelihood_{model_names[0]}'))
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
		bic_str=['S','S+E','S+S']

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
def run_sample_setup(args):
	sample, cl, i= args
	return sample_setup(sample, cl, i)
def run_simul_setup(args):
	sample, cl, i= args
	return simul_s_setup(sample, cl, i)
def run_desi_setup(args):
	sample, cl, i,ra,dec= args
	return desi_setup(sample, cl, i,ra,dec)
def zelador(sample,cluster,pasta):
	# Processamento direto
	caminho=f'{sample}/{cluster}/{pasta}'

	if os.path.isdir(f'{sample}/{cluster}/{pasta}'):
		novo_caminho=f'{caminho}_old'
		shutil.move(caminho,novo_caminho)
		print(f"Renomeado: {caminho} -> {novo_caminho}")
	# call(f'rm -r {sample}/{cluster}/{pasta}/*',shell=True)
	return
if __name__ == '__main__':
	import sys
	sample=sys.argv[1]
	save_value=sys.argv[2]
	ok=[]
	# run_desi_setup((sample,,'desi_obs',154.934471448436,-0.638302587847237))
	# inp2=open(f'ass_{sample}.dat','a')
	# # inp3=open(f'data_desi_{sample}_entry.dat','a')
	# with open(f'ass_{sample}.dat','r') as inp2:
	# 	for item in inp2.readlines():   
	# 		ok.append(item.split()[0])
	# output=open(f'{sample}_profit_{save_value}.dat','a')
	args_list = []
	bcg_el=[]
	# data_indiv=np.loadtxt('data_indiv_ecd.dat',usecols=[5,6])
	#fail_data_whl=['020769','048016','087041','088670','024079','020769','048016','087041','088670','074086','068819','023456','122252', '067541', '029053', '066300', '020277', '025771', '023651', '067136', '027682', '025790', '023852', '105603', '031605']
	fail_data=[]
	# with open(f'../data_analysis/data_indiv_E_EL_{sample}.dat','r') as inp1:
	# 	data_check=inp1.readlines()
	# 	for i,obj in enumerate(data_check):
	# 		if obj.split()[0] not in fail_data:
	# 			bcg_el.append(obj.split()[0])
	done=0
	all_done=0
	with open(f'../{sample}_desi/data_indiv_desi_{sample}_clean.dat','r') as inp1:#
		data_sample=inp1.readlines()
	for i,obj in enumerate(data_sample):
		cl=obj.split()[0]
		ra=float(obj.split()[1])
		dec=float(obj.split()[2])
		if os.path.exists(f'{sample}/{cl}/desi_obs/ajust-sersic-duplo-llh-desi_obs.fits'):
			# all_done+=1
			if os.path.isfile(f'{sample}/{cl}/desi_obs/comp_2_desi_obs.fits'):
				# done+=1		
				pass
			# k_rad_desi=centro_magzero_desi(sample,cl,'desi_obs',ra,dec)
			# inp3.write(f'{cl} {k_rad_desi}\n')
			else:
				# pass
				split_view_desi(sample,cl,0,'desi_obs',ra,dec)
		else:
			pass
	# print(all_done,done)

	# done=0
	# with open(f'../{sample}_desi/data_indiv_desi_{sample}_clean.dat','r') as inp1:
	# 	data_sample=inp1.readlines()
	# 	for i,obj in enumerate(data_sample):
	# 		cl=obj.split()[0]
	# 		ra=float(obj.split()[1])
	# 		dec=float(obj.split()[2])
	# 		if os.path.exists(f'{sample}/{cl}/{save_value}/ajust-sersic-duplo-llh-{save_value}.fits'):
	# 			finish_details(sample,cl,0,0,0,save_value)
	# 			done+=1
	# 		if cl not in fail_data and os.path.exists(f'../{sample}_desi/{cl}/bcg_r.fits'):
	# 			args_list.append((sample, cl, save_value,ra,dec))
	# 			rff_galfit=0#float(obj.split()[7])
	# 			ass=0#float(obj.split()[9])
	# 			# img=fits.open(f'../{sample}/{cluster}/bcg_r.fits',memmap=True)[0].data
	# 			# inp3.write(f'{cluster} {data_indiv[i][0]} {data_indiv[i][1]} {img.shape[0]} {img.shape[1]}\n')
	# 			# finish_details(sample,cl,0,rff_galfit,ass,save_value)
	# #			zelador(sample,cl,save_value)
	# 		else:
	# 			print(cl)
	# print(done)
	#run_desi_setup(args_list[0])
	# with mp.Pool(processes=18) as pool:
	# 	chunksize=1
	# 	for _ in pool.imap_unordered(run_desi_setup, args_list,chunksize=chunksize):
	# 		pass
