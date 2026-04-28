from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp
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
#############################

def skyradfunc(x,a,b,c):
	return a*np.exp(-x/b)+c

def calc_sky(image,mask,maskb,xcenter,ycenter,cluster):
	vsky=[]
	dsky=[]
	for j in range(0,image.shape[0]):
		for i in range(0,image.shape[1]):
			if mask[j,i]==0 and maskb[j,i]==0:
				vsky.append(image[j,i])
				dsky.append(((j-ycenter)**2+(i-xcenter)**2)**0.5)
	if len(dsky) <= 3:
		skyvalue=+0
	else:
		popt,pcov=scp.curve_fit(skyradfunc,dsky,vsky,p0=[200,100,100])
		skyvalue =+ skyradfunc(np.max(dsky),*popt)
	return skyvalue
	
def testgr(cluster,ellgalfit,pagalfit,imgr,imgg,sigskyg,sigskyr,c1,c2,c3,c4):
	import matplotlib.pyplot as plt
	#LEITURA DO ISOTABLE	
	temp=[[] for i in range(17)]
	iso_table=open(f'WHL_gr/{cluster}/iso_table.dat','r')
	for item in iso_table.readlines():
		if len(item.split()) == 3:
			extval_ellip=float(item.split()[0])
			extval_pa=float(item.split()[1])
			maxrad=float(item.split()[2])
		else:
			for i in range(17):
				temp[i].append(float(item.split()[i]))

	vec=[]
	for i in range(17):
		vec.append(np.asarray(temp[i]))
	x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err=vec				
	#####################################################################
	#TESTE G-R COM ISOFOTAS LIVRES & GALFIT
	xc=x0[0]
	yc=y0[0]
	isofree_g,isofree_r,galfix_r,galfix_g=[[[],[]],[[],[]],[[],[]],[[],[]]]
	isovec=[[],[],[],[]]
	smagr=sma/0.396
	smagal=np.geomspace(5,maxrad/0.396,len(smagr))
	for i in range(len(intens)):
		#####
		#ELIPSES LIVRE BANDA G E R
		#
		freegeo_g=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagr[i]), eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
		freesamp_g=phi.EllipseSample(imgg,sma=float(smagr[i]),sclip=3.0, nclip=5,geometry=freegeo_g)
		freesamp_g.update()					
		freeiso_g=phi.Isophote(freesamp_g,0,True,0)
		isovec[0].append(freeiso_g)
		isofree_g[0].append(freeiso_g.intens)
		isofree_g[1].append(freeiso_g.int_err)
		#
		freegeo_r=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagr[i]), eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
		freesamp_r=phi.EllipseSample(imgr,sma=float(smagr[i]),sclip=3.0, nclip=5,geometry=freegeo_r)
		freesamp_r.update()
		freeiso_r=phi.Isophote(freesamp_r,0,True,0)
		isovec[1].append(freeiso_r)
		isofree_r[0].append(freeiso_r.intens)
		isofree_r[1].append(freeiso_r.int_err)
		#####

		#ELIPSES DO GALFIT BANDA G E R RESPECTIVAMENTE
		#
		galgeo_g=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagal[i]),eps=ellgalfit,pa=pagalfit, fix_center=True,fix_eps=True,fix_pa=True)
		galsamp_g=phi.EllipseSample(imgg,sma=float(smagal[i]), sclip=3.0, nclip=5,geometry=galgeo_g)
		galsamp_g.update()
		galfot_g=phi.Isophote(galsamp_g,0,True,0)
		isovec[2].append(galfot_g)
		galfix_g[0].append(galfot_g.intens)
		galfix_g[1].append(galfot_g.int_err)
		#	
		galgeo_r = phi.EllipseGeometry(x0=xc, y0=yc,sma=float(smagal[i]),eps=ellgalfit,pa=pagalfit, fix_center=True,fix_eps=True,fix_pa=True)
		galsamp_r=phi.EllipseSample(imgr,sma=float(smagal[i]), sclip=3.0, nclip=5,geometry=galgeo_r)
		galsamp_r.update()
		galfot_r=phi.Isophote(galsamp_r,0,True,0)
		isovec[3].append(galfot_r)
		galfix_r[0].append(galfot_r.intens)
		galfix_r[1].append(galfot_r.int_err)
		#
	free_g=phi.IsophoteList(isovec[0])
	free_r=phi.IsophoteList(isovec[1])
	fixgal_g=phi.IsophoteList(isovec[2])
	fixgal_r=phi.IsophoteList(isovec[3])
	
	list_iso=[free_g,free_r,fixgal_g,fixgal_r]
	name_iso=['free_g','free_r','fixgal_g','fixgal_r']
	image_iso=[(imgg*c3)/c4,(imgr*c1)/c2,(imgg*c3)/c4,(imgr*c1)/c2]
	
	for item in list_iso:
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.imshow(image_iso[list_iso.index(item)],vmin=0,vmax=1500,origin='lower')
		paircont=0
		for raio in item.sma:
			iso = item.get_closest(raio)
			x, y, = iso.sampled_coordinates()
			if paircont%2==0:
				plt.plot(x, y, color='white',linewidth=1)
			paircont+=1
		plt.xlabel(r'$X (pixel)$')
		plt.ylabel(r'$Y (pixel)$')
		plt.tight_layout()
		plt.savefig(f'WHL_gr/{cluster}/{cluster}_{name_iso[list_iso.index(item)]}.png')
		plt.close(fig)

	#################################################################################################################
	
	iso_table_vg=open(f'WHL_gr/{cluster}/iso_table_gr.dat','w')
	iso_table_vg.write('%f \t %f \t %f \t %f \t %f\n'%(extval_ellip,extval_pa,maxrad,sigskyg,sigskyr))		
	for r in range(len(sma)):
		iso_table_vg.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],a3[r],b3[r],a4[r],b4[r], ellip_err[r],pa_err[r],int_err[r],a3_err[r],b3_err[r], a4_err[r],b4_err[r],isofree_g[0][r],isofree_g[1][r],isofree_r[0][r],isofree_r[1][r],galfix_r[0][r],galfix_r[1][r],galfix_g[0][r],galfix_g[1][r]))
	iso_table_vg.close()		

	return free_g,fixgal_r


########################################################################################################################################################
def isobuilder(cluster,flagmask,flagdelta,ok,okk,good_path,bad_path):
	print(cluster)
	import matplotlib.pyplot as plt

	if [flagmask,flagdelta] != [0,0] or cluster in okk:
		call(f'rm -rf WHL_gr/{cluster}',shell=True)
		return
	elif cluster in ok:
		return
	else:
		call(f'mkdir -p WHL_gr/{cluster}',shell=True)	

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

		xc=float(header['1_XC'].split()[0].replace('*',''))
		yc=float(header['1_YC'].split()[0].replace('*',''))
		re = float(header['1_RE'].split()[0].replace('*',''))
		mag = float(header['1_MAG'].split()[0].replace('*',''))
		n = float(header['1_N'].split()[0].replace('*',''))
		sky=float(header['2_SKY'].split()[0].replace('*',''))
		pa0=(float(header['1_PA'].split()[0].replace('*',''))+90.)*np.pi/180.
		ell0=1.-float(header['1_AR'].split()[0].replace('*',''))
		chisq_galfit=float(header['CHI2NU'])

		if mask[int(yc),int(xc)] == 1:
			with open(bad_path,'a') as outfail:
				outfail.write('%s \n'%(cluster))
			call(f'rm -r WHL_gr/{cluster}',shell=True)
			return

		NMGYr=float(headerr['NMGY'])
		EXPTIMEr=float(headerr['EXPTIME'])
		
		NMGYg=float(headerg['NMGY'])
		EXPTIMEg=float(headerg['EXPTIME'])
		
		datar=(datar0*NMGYr)/EXPTIMEr
		datag=(datag0*NMGYg)/EXPTIMEg

		# sky calculation
		########################################33
		#CALCULO DA BANDA G E R 

		skyvalueg=calc_sky(datag,maskg,maskbg,xc,yc,cluster)
		imageg = datag - skyvalueg
		negpixsg = imageg[np.where((imageg<skyvalueg) & (imageg>-10000.))]
		sigmaskyg = np.std(negpixsg)/np.sqrt(1.-2./np.pi)
		sigmaskyg/=4.

		#

		skyvaluer=calc_sky(datar,mask,maskb,xc,yc,cluster)
		imager = datar - skyvaluer
		negpixsr = imager[np.where((imager<skyvaluer) & (imager>-10000.))]
		sigmaskyr = np.std(negpixsr)/np.sqrt(1.-2./np.pi)
		sigmaskyr/=4.
	##########################LEITURA DO ISOTABLE################################################

		isoimage_g=ma.masked_where(maskg==1,imageg)
		isoimage_r=ma.masked_where(mask==1,imager)
		
		if os.path.isfile(f'WHL_gr/{cluster}/iso_table.dat'):
			temp=[[] for i in range(17)]
			iso_table=open(f'WHL_gr/{cluster}/iso_table.dat','r')
			for item in iso_table.readlines():
				if len(item.split()) == 3:
					extval_ellip=float(item.split()[0])
					extval_pa=float(item.split()[1])
					maxrad=float(item.split()[2])
				else:
					for i in range(17):
						temp[i].append(float(item.split()[i]))

			vec=[]
			for i in range(17):
				vec.append(np.asarray(temp[i]))
			x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err=vec
			if os.path.isfile(f'WHL_gr/{cluster}/iso_table_gr.dat'):
				with open(good_path,'a') as output:
					output.write('%s \n'%(cluster))
				return							
			else:
				testcor=testgr(cluster,ell0,pa0,isoimage_r,isoimage_g,sigmaskyg, sigmaskyr,EXPTIMEr,NMGYr,EXPTIMEg,NMGYg)
				with open(good_path,'a') as output:
					output.write('%s \n'%(cluster))
				return					
	#####################################################################################################################################

		else:
			isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=pa0)  
			isomodel=phi.Ellipse(isoimage_r,isogal) 

			fig, axs = plt.subplots(2,2,sharey=True,sharex=True)
			plt.subplots_adjust(hspace=0.01, wspace=0.01)
			axs[0,0].imshow((isoimage_r*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')
			axs[0,0].set_ylabel(r'$Y (pixel)$')
			
			axs[0,1].imshow((datar*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')

			axs[1,0].imshow((isoimage_g*EXPTIMEg)/NMGYg,vmin=0,vmax=1500,origin='lower')
			axs[1,0].set_xlabel(r'$X (pixel)$')
			axs[1,0].set_ylabel(r'$Y (pixel)$')
			
			axs[1,1].imshow((datag*EXPTIMEg)/NMGYg,vmin=0,vmax=1500,origin='lower')
			axs[1,1].set_xlabel(r'$X (pixel)$')
			plt.tight_layout()
			plt.savefig(f'WHL_gr/{cluster}/{cluster}_bcg.png')
			#plt.savefig(f'iso_fail{cluster}_bcg.png')
			plt.close(fig)

			try:
				isolist=isomodel.fit_image(minsma=5,maxsma=np.max(isoimage_r.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)	
				######################################################################################
				#VERIFICACAO DE CONVERGENCIA
			
				fig, ax = plt.subplots(figsize=(6, 6))
				ax.imshow((isoimage_r*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')
				paircont=0
				for sma in isolist.sma:
					if isolist.intens[np.where(isolist.sma==sma)][0]>sigmaskyr:
						iso = isolist.get_closest(sma)
						x, y, = iso.sampled_coordinates()
						if paircont%2==0:
							plt.plot(x, y, color='white',linewidth=1)
						paircont+=1
				plt.xlabel(r'$X (pixel)$')
				plt.ylabel(r'$Y (pixel)$')
				plt.tight_layout()
				plt.savefig(f'WHL_gr/{cluster}/{cluster}_iso_free.png')
				plt.close(fig)


			except:
				try:
					print(len(isolist.sma))
					pass
				except:
					with open(bad_path,'a') as outfail:
						outfail.write('%s \n'%(cluster))
					call(f'rm -r WHL_gr/{cluster}',shell=True)
					return
			n_isofotas=len(isolist.sma)
			isotry=0
			if n_isofotas < 3:
				while isotry<=500 and len(isolist.sma) < 3: # caso nao haja convergencia, repetir
					isotry+=1
					try:
						isogal= phi.EllipseGeometry(x0=xc, y0=yc, sma=20, eps=ell0,pa=np.random.uniform(-180.,180.))
						isomodel=phi.Ellipse(isoimage_r,isogal)
						isolist=isomodel.fit_image(minsma=5,maxsma=np.max(isoimage_r.shape)/2.,step=0.02,fix_center=True,sclip=3.0, nclip=5,conver=0.1,fflag=0.5)

						fig, ax = plt.subplots(figsize=(6, 6))
						ax.imshow((isoimage_r*EXPTIMEr)/NMGYr,vmin=0,vmax=1500,origin='lower')

						paircont=0
						for sma in isolist.sma:
							if isolist.intens[np.where(isolist.sma==sma)][0]>sigmaskyr:
								iso = isolist.get_closest(sma)
								x, y, = iso.sampled_coordinates()
								if paircont%2==0:
									plt.plot(x, y, color='white',linewidth=1)
								paircont+=1
						plt.xlabel(r'$X (pixel)$')
						plt.ylabel(r'$Y (pixel)$')
						plt.tight_layout()
						plt.savefig(f'WHL_gr/{cluster}/{cluster}_iso_free.png')
						plt.close(fig)
					except:
						isotry+=1
			if isotry<200 or len(isolist.sma) > 3:
				print(cluster,'ok')
			###########################################################################################
			#VALORES DE INTERESSE
				x0=isolist.x0[np.where(isolist.intens>sigmaskyr)]
				y0=isolist.y0[np.where(isolist.intens>sigmaskyr)]
				sma=isolist.sma[np.where(isolist.intens>sigmaskyr)]*0.396
				pa=isolist.pa[np.where(isolist.intens>sigmaskyr)]
				eps=isolist.eps[np.where(isolist.intens>sigmaskyr)]
				intens=isolist.intens[np.where(isolist.intens>sigmaskyr)]
				a3=isolist.a3[np.where(isolist.intens>sigmaskyr)]
				b3=isolist.b3[np.where(isolist.intens>sigmaskyr)]
				a4=isolist.a4[np.where(isolist.intens>sigmaskyr)]
				b4=isolist.b4[np.where(isolist.intens>sigmaskyr)]
				#
				ellip_err=isolist.ellip_err[np.where(isolist.intens>sigmaskyr)]
				pa_err=isolist.pa_err[np.where(isolist.intens>sigmaskyr)]
				int_err=isolist.int_err[np.where(isolist.intens>sigmaskyr)]
				a3_err=isolist.a3_err[np.where(isolist.intens>sigmaskyr)]
				b3_err=isolist.b3_err[np.where(isolist.intens>sigmaskyr)]
				a4_err=isolist.a4_err[np.where(isolist.intens>sigmaskyr)]
				b4_err=isolist.b4_err[np.where(isolist.intens>sigmaskyr)]
				#
				extval_ellip=np.average(eps,weights=np.power(sma,2))
				extval_pa=np.average(pa,weights=np.power(sma,2))
				maxrad=np.max(sma)
				#########################################################################################
				inptrue=open(f'WHL_gr/{cluster}/iso_table.dat','w')
				inptrue.write('%f \t %f \t %f \n'%(extval_ellip,extval_pa,maxrad))		
				for r in range(len(sma)):
					inptrue.write('%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n'%(x0[r],y0[r],sma[r],pa[r],eps[r],intens[r],a3[r],b3[r],a4[r],b4[r],ellip_err[r],pa_err[r],int_err[r],a3_err[r],b3_err[r],a4_err[r],b4_err[r]))
				inptrue.close()
				#
				testcor=testgr(cluster,ell0,pa0,isoimage_r,isoimage_g,sigmaskyg, sigmaskyr,EXPTIMEr,NMGYr,EXPTIMEg,NMGYg)
				with open(good_path,'a') as output:
					output.write('%s \n'%(cluster))
				return
			else:
				with open(bad_path,'a') as outfail:
					outfail.write('%s \n'%(cluster))
				call(f'rm -r WHL_gr/{cluster}',shell=True)
				return
	# ############################################################################################
	return


if __name__ == "__main__":

	good_path='checkiso_WHL.dat'
	bad_path='iso_fail_10k.dat'

	ok=[]
	with open(good_path,'r+') as output:
		for item in output.readlines():
			ok.append(item.split()[0])

	okk=[]
	with open(bad_path,'r+') as outfail:
		for galaxy in outfail.readlines():
			okk.append(galaxy.split()[0])

	data_info=[]
	with open(f'/home/andrelpk/Documentos/Projetos/WHL/pargal_WHL_compact_astro_vfix.dat','r') as inp1:
		ninp1=inp1.readlines()
		for obj in ninp1:
			ll1=obj.split()	
			cluster=ll1[0]
			flagmask=int(float(ll1[11]))
			flagdelta=int(float(ll1[12]))
			data_info.append((cluster,flagmask,flagdelta))


	with mp.Pool(processes=16) as pool:
		pool.starmap(isobuilder,[(*data,ok,okk,good_path,bad_path) for data in data_info])


