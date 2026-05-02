from math import sin,cos,tan,pi,floor,log10,sqrt,atan2,exp
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt
import scipy.optimize as scp 
import scipy.stats as sst
import numpy.ma as ma
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer as kes
import pandas as pd
import photutils.isophote as phi
import warnings
warnings.filterwarnings("ignore")

"""
graficos_kmeans_last.py

K‑means clustering analysis and visualisation of galaxy structural
parameters (ellipticity ε, position angle PA, and flux‑weighted Sérsic
index n) derived from isophote fitting.

This module:
- Reads pre‑computed isophote tables and surface‑brightness profile fits.
- Computes a flux‑weighted, radially‑averaged Sérsic index from the
  Sérsic+Exponential and double‑Sérsic model components.
- Applies K‑means clustering to the three‑dimensional space
  (ε, PA, n) using normalised / perturbed data and error weighting.
- Evaluates the optimal number of clusters via the elbow method,
  silhouette score, and the gap statistic.
- Produces 3D scatter plots, 2D projection maps, and structural
  parameter vs. radius diagrams for each cluster solution.
- Generates diagnostic isophote overlay images (via ``isomaker``).

The code is designed for the L07 sample and expects specific
directory structures under ``L07_coef_gain/``.

Global variables:
- ``ok``, ``info``, ``covec`` are loaded in the main block and used
  inside ``bcgfigs`` to retrieve pre‑fitted model parameters.
"""

# ----------------------------------------------------------------------
# Isophote visualisation
# ----------------------------------------------------------------------
def isomaker(cluster,tipo,imgr,xc,yc,sma,intens,eps,pa):
	"""Overlay free‑geometry elliptical isophotes on the r‑band image.

	This diagnostic function reconstructs elliptical isophotes using the
	parameters from a previously saved isophote table and plots them
	on top of the original image.  Two versions are shown: one with
	lines and one with scatter points.

	Args:
		cluster (str): Galaxy identifier.
		tipo    (str): Galaxy morphological type (e.g., 'eliptica').
		imgr    (2D array): r‑band image (sky‑subtracted).
		xc, yc  (float): Galaxy centre (pixels).
		sma     (array): Semi‑major axis values (arcsec).
		intens  (array): Isophotal intensities.
		eps     (array): Ellipticity profile.
		pa      (array): Position angle (radians).

	Returns:
		None.  Displays the plots using ``plt.show()`` (the save
		lines are currently commented out).

	Note:
		The function uses a pixel scale of 0.396 arcsec/pixel.
	"""


	print(cluster,tipo)
	#####################################################################
	#TESTE G-R COM ISOFOTAS LIVRES & GALFIT
	
	isofree_r=[[],[]]
	isovec=[]
	smagr=sma/0.396
	for i in range(len(intens)):
		#####
		#ELIPSES BANDA R
		freegeo_r=phi.EllipseGeometry(x0=xc,y0=yc,sma=float(smagr[i]), eps=float(eps[i]),pa=float(pa[i]),fix_center=True,fix_eps=True,fix_pa=True)
		freesamp_r=phi.EllipseSample(imgr,sma=float(smagr[i]),sclip=3.0, nclip=5,geometry=freegeo_r)
		freesamp_r.update()
		freeiso_r=phi.Isophote(freesamp_r,0,True,0)
		isovec.append(freeiso_r)
		isofree_r[0].append(freeiso_r.intens)
		isofree_r[1].append(freeiso_r.int_err)
		#####

	free_r=phi.IsophoteList(isovec)
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(imgr,vmin=0,vmax=1500,origin='lower')
	paircont=0
	for raio in free_r.sma:
		iso = free_r.get_closest(raio)
		x, y, = iso.sampled_coordinates()
		print(x,y)
		if paircont%2==0:
			plt.plot(x, y, color='white',linewidth=1)
		paircont+=1
	plt.xlabel('X (pix)')
	plt.ylabel('Y (pix)')
	plt.tight_layout()
	plt.show()
	# plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_free_r.png')
	# plt.savefig('L07_coef_gain/'+tipo+'/'+iso_imgs+'/'+cluster+'_free_r.png')
	# plt.close(fig)

	# plt.figure(figsize=(6, 6))
	# paircont=0
	# for raio in free_r.sma:
	# 	iso = free_r.get_closest(raio)
	# 	x, y, = iso.sampled_coordinates()
	# 	if paircont%2==0:
	# 		plt.scatter(x, y, color='black',s=1)#linewidth=1)
	# 	paircont+=1
	# plt.xlabel('X (pix)')
	# plt.ylabel('Y (pix)')
	# plt.tight_layout()
	# plt.show()
	# plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_free_r.png')
	# plt.savefig('L07_coef_gain/'+tipo+'/'+iso_imgs+'/'+cluster+'_free_r.png')
	# plt.close(fig)

	# return 

# ----------------------------------------------------------------------
# Clustering helper functions
# ----------------------------------------------------------------------
def optimalK(data,data_inc,maxClusters):
	"""Determine the optimal number of clusters using the gap statistic.

	For each candidate number of clusters k (from 1 to maxClusters-1),
	the function:
	- Generates ``nrefs`` synthetic reference datasets by drawing from
	  a uniform distribution (currently using ``np.random.random_sample``).
	- Computes the within‑cluster dispersion (inertia) for the real
	  data and for the reference data.
	- Calculates gap(k) = mean(log(W_ref)) - log(W_data).
	The optimal k is the one that maximises the gap.

	Args:
		data        (2D array): Real data (n_samples, n_features).
		data_inc    (array): Uncertainties (used to compute reference
					  spread in the original code, but here the reference
					  is drawn uniformly; the argument is present for
					  compatibility).
		maxClusters (int): Maximum number of clusters to test.

	Returns:
		tuple:
			optimal_k (int): Number of clusters that maximises the gap.
			resultsdf (DataFrame): Pandas DataFrame with columns
				'clusterCount' and 'gap'.
	"""


	nrefs=10
	gaps = np.zeros((len(range(1, maxClusters)),))
	resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
	for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
		refDisps = np.zeros(nrefs)
		
		
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
		for i in range(nrefs):

			
			randomReference = np.random.random_sample(size=data.shape)
			
			#np.random.normal(data,data_inc)#np.column_stack((x,y,z))
			#print(randomReference)
			fig=plt.figure()
			ax=fig.add_subplot(projection='3d')
			ax.scatter(randomReference[:,0],randomReference[:,1],randomReference[:,2],s=30)
			plt.close()
			#plt.show()
            # Fit to it
			km = KMeans(k)
			km.fit(randomReference)
			refDisp = km.inertia_
			refDisps[i] = refDisp
			print(refDisps)
        # Fit cluster to original data and create dispersion
		km = KMeans(k)
		km.fit(data)
        
		origDisp = km.inertia_

        # Calculate gap statistic
		gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
		gaps[gap_index] = gap
        
		resultsdf = resultsdf._append({'clusterCount':k, 'gap':gap}, ignore_index=True)

	return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

# ----------------------------------------------------------------------
# Clustering visualisation
# ----------------------------------------------------------------------
def analise_comp(xyz,comp,comp_center,x,y,z,sma,mag,cluster,tipo,test_kmeans,which_model):
	"""Generate structural‑parameter plots colour‑coded by K‑means
	cluster membership.

	For each cluster component, the function creates:
	- A 3D scatter plot (ε, PA, n) with cluster centres marked.
	- 2D projection maps (PA vs ε, PA vs n, ε vs n).
	- Radial profiles of Sérsic index n, surface brightness μ,
	  ellipticity ε, and position angle PA, colour‑coded by
	  cluster.

	Plots are saved to the cluster directory and to a summary
	directory under ``L07_coef_gain/{tipo}/{test_kmeans}/``.

	Args:
		xyz          (2D array): Normalised coordinates (n_samples, 3)
					  corresponding to (ε, PA, n).
		comp         (list of boolean arrays): One mask per cluster.
		comp_center  (list of arrays): Coordinates of cluster centres.
		x, y, z      (arrays): Raw ε, PA (radians), and n values.
		sma          (array): Semi‑major axis (arcsec).
		mag          (array): Surface brightness μ (mag/arcsec²).
		cluster      (str): Galaxy ID.
		tipo         (str): Galaxy morphological type.
		test_kmeans  (str): Label for the K‑means test (used in file names).
		which_model  (str): 'S+E' or 'S+S' – indicates which model was
					  used to compute the normalised n.

	Returns:
		None.  Saves PNG figures.
	"""


	
	dist=[sma[item] for item in comp]
	mu=[mag[item] for item in comp]
	#
	eps=[xyz[:,0][item] for item in comp]
	pa=[xyz[:,1][item] for item in comp]
	n=[xyz[:,2][item] for item in comp]
	#	
	eps_v=[x[item] for item in comp]
	pa_v=[y[item]/np.pi*180. for item in comp]
	n_v=[z[item] for item in comp]
	#
	eps_center=[item[:,][0] for item in comp_center]
	pa_center=[item[:,][1] for item in comp_center]
	n_center=[item[:,][2] for item in comp_center]
	#

	ncolors=['b','g','r','c','m','y','k']
	nmarkers=['o','v','p','*','P','s','D']
	
	fig=plt.figure()
	ax=fig.add_subplot(projection='3d')
	ax.set_title(r'Projeção 3D $\epsilon$ - PA - n'+which_model+'-'+tipo)
	for i in range(len(comp)):
		ax.scatter(eps[i],pa[i],n[i],s=12,c=ncolors[i],marker=nmarkers[i],edgecolors='black')
		ax.scatter(eps_center[i],pa_center[i],n_center[i],s=50,c='black', marker=nmarkers[i],edgecolors='black',label='centro comp '+str(i))
	ax.set_xlabel(r'$\epsilon$')
	ax.set_ylabel(r'$PA$')
	ax.set_zlabel(r'$n$')
	ax.view_init(elev=20, azim=30)
	plt.legend()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/ '+cluster+'_bojo_env_3d_norm_'+test_kmeans+'.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_bojo_env_3d_norm.png')
	plt.close(fig)

	fig=plt.figure()
	ax=fig.add_subplot(projection='3d')
	ax.set_title(r'Projeção 3D $\epsilon$ - PA - n'+which_model+'-'+tipo)
	for i in range(len(comp)):
		ax.scatter(eps_v[i],pa_v[i],n_v[i],s=12,c=ncolors[i],marker=nmarkers[i],edgecolors='black')

	ax.set_xlabel(r'$\epsilon$')
	ax.set_ylabel(r'$PA$')
	ax.set_zlabel(r'$n$')
	ax.view_init(elev=20, azim=30)
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/ '+cluster+'_bojo_env_3d_'+test_kmeans+'.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_bojo_env_3d.png')
	plt.close(fig)
	
	########################################################################################

	plt.figure()
	plt.suptitle(r'Mapa 2D de $PA - \epsilon$'+which_model+'-'+tipo)

	for i in range(len(comp)):
		plt.scatter(pa_v[i],eps_v[i],s=20,c=ncolors[i], marker=nmarkers[i],edgecolors='black')		
	
	plt.xlim([np.min(y/np.pi*180.)-(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))*0.1,np.max(y/np.pi*180.)+(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))/10.])
	plt.ylim([0,np.max(x)+np.max(x)*0.1])
	plt.xlabel(r'$PA$')
	plt.ylabel(r'$\epsilon$')
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_2dmaps_pa_e_'+test_kmeans+'.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_2dmaps_pa_e.png')	
	plt.close()


	plt.figure()
	plt.suptitle(r'Mapa 2D de $PA - n$'+which_model+'-'+tipo)
	for i in range(len(comp)):
		plt.scatter(pa_v[i],n_v[i],s=20,c=ncolors[i], marker=nmarkers[i],edgecolors='black')		

	plt.xlim([np.min(y/np.pi*180.)-(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))*0.1,np.max(y/np.pi*180.)+(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))/10.])
	plt.ylim([np.min(z)-(np.max(z)-np.min(z))*0.1,np.max(z)+(np.max(z)-np.min(z))/10.])
	plt.xlabel(r'$PA$')
	plt.ylabel(r'$n$')
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_2dmaps_pa_n_'+test_kmeans+'.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_2dmaps_pa_n.png')	
	plt.close()


	plt.figure()
	plt.suptitle(r'Mapa 2D de $\epsilon - n$'+which_model+'-'+tipo)
	for i in range(len(comp)):
		plt.scatter(eps_v[i],n_v[i],s=20,c=ncolors[i], marker=nmarkers[i],edgecolors='black')		
	plt.xlabel(r'$\epsilon$')
	plt.ylabel(r'$n$')
	plt.xlim([0,np.max(x)+np.max(x)*0.1])
	plt.ylim([np.min(z)-(np.max(z)-np.min(z))*0.1,np.max(z)+(np.max(z)-np.min(z))/10.])
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_2dmaps_e_n.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_2dmaps_e_n.png')	
	plt.close()
	
	#################################################################################################################
	
	fig,axs=plt.subplots(4,1,figsize=(6,12),sharex=True)
	plt.subplots_adjust(hspace=0.35, wspace=0.35)
	plt.suptitle('Análise estrutural'+which_model+'-'+tipo)
	for i in range(len(comp)):
		axs[0].scatter(np.power(dist[i],0.25),n_v[i],s=15,marker=nmarkers[i],color=ncolors[i],edgecolors='black')
	axs[0].set_ylim([np.min(z)-(np.max(z)-np.min(z))*0.1,np.max(z)+(np.max(z)-np.min(z))/10.])
	axs[0].set_ylabel(r'$n$')
	
	for i in range(len(comp)):
		axs[1].scatter(np.power(dist[i],0.25),mu[i],s=15,marker=nmarkers[i],color=ncolors[i],edgecolors='black')
	axs[1].set_ylim([np.min(mag)-(np.max(mag)-np.min(mag))/10.,np.max(mag)+(np.max(mag)-np.min(mag))/10.])
	axs[1].set_ylabel(r'$\mu$ (mag)')
	axs[1].invert_yaxis()
	
	for i in range(len(comp)):
		axs[2].scatter(np.power(dist[i],0.25),eps_v[i],s=15,marker=nmarkers[i],color=ncolors[i],edgecolors='black')
	axs[2].set_ylim([0,np.max(x)+np.max(x)*0.1])
	axs[2].set_ylabel(r'$\epsilon$')	
	
	for i in range(len(comp)):
		axs[3].scatter(np.power(dist[i],0.25),pa_v[i],s=15,marker=nmarkers[i],color=ncolors[i],edgecolors='black')
	axs[3].set_ylim([np.min(y/np.pi*180.)-(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))*0.1,np.max(y/np.pi*180.)+(np.max(y/np.pi*180.)-np.min(y/np.pi*180.))/10.])
	axs[3].set_xlabel(r'$R^{1/4}$ (arcsec)')
	axs[3].set_ylabel(r'$PA$ (deg)')
	plt.tight_layout()
	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_iso_info_'+test_kmeans+'.png')	
	plt.savefig('L07_coef_gain/'+tipo+'/'+test_kmeans+'/ncluster_'+str(len(comp))+'/'+cluster+'_iso_info.png')	
	plt.close()	
	
	return
####################################################################################################################
def testkmeans(cluster,tipo,x,x_inc,y,y_inc,z,z_inc,sma,mag,tipo_kmeans,which_model):
	"""Run a complete K‑means clustering analysis on the 3D parameter
	space (ε, PA, n).

	Steps:
	1. Normalise the input data with ``scipy.stats.zscore``.
	2. Construct a combined uncertainty vector and use
	   inverse‑variance weights in the clustering.
	3. Generate a raw 3D scatter plot (no clusters).
	4. Use the ``yellowbrick`` elbow visualiser to find the optimal
	   number of clusters based on the distortion score.
	5. Compute the silhouette score for k = 2 … 8.
	6. Call ``optimalK`` to obtain the gap statistic optimal k.
	7. For each candidate k, create a 3D scatter plot with clusters
	   colour‑coded (these are currently commented out in the code).
	8. Finally, select the elbow‑determined best k, fit a final
	   K‑means model, and (optionally) call ``analise_comp`` to
	   produce the final set of diagnostic figures.

	Args:
		cluster      (str): Galaxy ID.
		tipo         (str): Morphological type.
		x, y, z      (arrays): Raw ε, PA (radians), n values.
		x_inc, y_inc, z_inc (arrays): Uncertainties/errors for x, y, z.
		sma          (array): Semi‑major axis.
		mag          (array): Surface brightness.
		tipo_kmeans  (str): Label for the K‑means run (used in file paths).
		which_model  (str): 'S+E' or 'S+S'.

	Returns:
		None.  Saves figures and, if uncommented, calls ``analise_comp``.
	"""


	
	x_norm=sst.zscore(x)
	y_norm=sst.zscore(y)
	z_norm=sst.zscore(z)
	
	xyz=np.column_stack((x_norm,y_norm,z_norm))
	
	x_inc_temp=np.power(x_inc,2)
	y_inc_temp=np.power(y_inc,2)
	z_inc_temp=np.power(z_inc,2)

	xyz_inc_test=np.column_stack((x_inc,y_inc,z_inc))

	xyz_inc=np.sqrt(x_inc + y_inc + z_inc)
	xyz_err=1/(x_inc + y_inc + z_inc)

	#PLOT INICIAL SEM FIT
	fig=plt.figure()
	ax=fig.add_subplot(projection='3d')
	ax.set_title(r'Projeção 3D $\epsilon$ - PA - n'+which_model+'-'+tipo)
	ax.scatter(x_norm,y_norm,z_norm,s=12,c='black',edgecolors='black')
	ax.set_xlabel(r'$\epsilon$')
	ax.set_ylabel(r'$PA$')
	ax.set_zlabel(r'$n$')
	ax.view_init(elev=20, azim=30)
	plt.show()
	#plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/ '+cluster+'_bojo_env_3d.png')	
	#plt.savefig('L07_coef_gain/'+tipo+'/'+tipo_kmeans+'/'+cluster+'_bojo_env_3d.png')
	plt.close(fig)
	##############################################################################	
	##############################################################################
	
	kmeans_model=KMeans(n_init='auto')#,random_state=42)
	test_elbow=kes(kmeans_model,k=(1,8),title=cluster+' '+'Elbow score'+which_model+'-'+tipo)
	test_elbow.fit(xyz,sample_weight=xyz_err)
	test_elbow.show()
	print(test_elbow.k_scores_)
	
	#outkelbow.write('%s \t %i \n'%(cluster,test_elbow.elbow_value_))
	#test_elbow.show(outpath='L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_elbow_'+tipo_kmeans+'.png')
	#test_elbow.show(outpath='L07_coef_gain/'+tipo+'/'+tipo_kmeans+'/ncluster_'+str(test_elbow.elbow_value_)+'/'+cluster+'_elbow.png')
	plt.close()
	
	#############################################################################
	#############################################################################	
	
	nclusters=[2,3,4,5,6,7,8]#[i+1 for i in range(test_elbow.elbow_value_-1)]
	siltest=[]
	dbtest=[]
	for item in nclusters:
		ktest=KMeans(n_clusters=item,n_init='auto')
		ktest.fit(xyz,sample_weight=xyz_err)
		siltest.append(silhouette_score(xyz,ktest.labels_))
	
		#dbtest.append(davies_bouldin_score(xyz,ktest.labels_))
	print(siltest)
	print(max(siltest))
	
	k, gapdf = optimalK(xyz,xyz_inc_test,maxClusters=8)
	print('Optimal k is: ', k)
	
	plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
	plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
	plt.grid(True)
	plt.xlabel('Cluster Count')
	plt.ylabel('Gap Value')
	plt.title('Gap Values by Cluster Count')
	plt.show()
	'''
	#plt.plot(nclusters,siltest)
	#plt.show()
	#plt.close()
	
	ncolors=['b','g','r','c','m','y','k','w']
	nmarkers=['o','v','p','*','P','s','D','X']
	
	for item in nclusters:
		ktest=KMeans(n_clusters=item,n_init='auto',random_state=42)
		ktest.fit(xyz)
		print(ktest.inertia_)
		fig=plt.figure()
		ax=fig.add_subplot(projection='3d')
		ax.set_title('Projeção 3D para solução de '+str(item)+' conjuntos' +which_model+'-'+tipo)
		ax.view_init(elev=20, azim=30)
		ax.set_xlabel(r'$\epsilon$')
		ax.set_ylabel(r'$PA$')
		ax.set_zlabel(r'$n$')
		for i in range(len(ktest.labels_)):
			conj=ktest.labels_[i]
			ax.scatter(xyz[:,0][i],xyz[:,1][i],xyz[:,2][i],s=30,c=ncolors[conj], marker=nmarkers[conj],edgecolors='black')
		
	#	plt.savefig('L07_coef_gain/'+tipo+'/'+cluster+'/'+cluster+'_'+str(item)+'_'+tipo_kmeans+'.png')	
		plt.close()
################################################################################################################
################################################################################################################

	best_ncluster=test_elbow.elbow_value_
	best_kmeans=KMeans(n_clusters=best_ncluster,n_init='auto',random_state=42)
	best_kmeans.fit_predict(xyz)
	#	
	comp=[]
	comp_center=[]
	_,idx=np.unique(best_kmeans.labels_,return_index=True)
	ordercl=(best_kmeans.labels_[np.sort(idx)])
	for cl in ordercl:
		comp.append(best_kmeans.labels_==cl)
		comp_center.append(best_kmeans.cluster_centers_[cl])

	#analise_comp(xyz,comp,comp_center,x,y,z,sma,mag,cluster,tipo,tipo_kmeans,which_model)
	'''	
	return
####################################################################################################################
# ----------------------------------------------------------------------
# Standard surface‑brightness models (duplicated from previous scripts)
# ----------------------------------------------------------------------
def linfunc(x,a,b):
	"""Linear function y = a * x + b.

	Args:
		x (float or array_like): Independent variable.
		a (float): Slope.
		b (float): Intercept.

	Returns:
		array_like: a*x + b.
	"""


	return a*x+b
def skyradfunc(x,a,b,c):
	"""Exponential model for the sky radial profile.

	y = a * exp(-x / b) + c

	Args:
		x (float or array_like): Distance from centre.
		a (float): Amplitude at x=0.
		b (float): Scale radius.
		c (float): Asymptotic constant.

	Returns:
		array_like: a * exp(-x/b) + c.
	"""


	return a*np.exp(-x/b)+c
def calc_sky(image,mask,maskb,xcenter,ycenter,cluster):
	"""Estimate sky background by fitting an exponential model to
	sky pixels as a function of radius.

	Args:
		image   (2D array): Background image.
		mask    (2D bool): Galaxy+object mask (1 = masked).
		maskb   (2D bool): Additional inner mask.
		xcenter, ycenter (float): Galaxy centre.
		cluster (str): Cluster ID (unused inside the function).

	Returns:
		float: Sky value at large radii, or 0 if too few sky pixels.
	"""


	vsky=[]
	dsky=[]
	for j in range(image.shape[0]):
		for i in range(image.shape[1]):
			if mask[j,i]==0 and maskb[j,i]==0:
				vsky.append(image[j,i])
				dsky.append(((j-ycenter)**2+(i-xcenter)**2)**0.5)
	if len(dsky) <= 3:
		skyvalue=+0
	else:
		popt,pcov=scp.curve_fit(skyradfunc,dsky,vsky,p0=[200,100,100])
		skyvalue =+ skyradfunc(np.max(dsky),*popt)
	return skyvalue
def sersic(x,ie,re,n):
	"""Sérsic surface‑brightness profile (mag/arcsec²).

	Args:
		x  (array_like): Radius (arcsec).
		ie (float): Central intensity (flux/arcsec²).
		re (float): Effective radius (arcsec).
		n  (float): Sérsic index.

	Returns:
		array_like: μ(x) in mag/arcsec².
	"""


	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.)))+22.5
	return i
def sersicexp(x,ie,re,n,i0,rd):
	"""Sérsic + exponential disk profile (mag/arcsec²).

	Args:
		x  (array_like): Radius.
		ie (float): Central intensity of Sérsic.
		re (float): Effective radius of Sérsic.
		n  (float): Sérsic index.
		i0 (float): Central intensity of exponential.
		rd (float): Exponential scale length.

	Returns:
		array_like: Total μ.
	"""


	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	b1=2.*abs(1.)-1./3+4./405/abs(1.)+46./25515/abs(1.)**2+131./1148175/abs(1.)**3-2194697./30690717750/abs(1.)**4
	i=-2.5*np.log10(abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))+abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
def doublesersic(x,ie,re,n,i0,rd,nd):
	"""Double Sérsic (Sérsic+Sérsic) profile.

	Args:
		x  (array_like): Radius.
		ie, re, n: Parameters for component 1.
		i0, rd, nd: Parameters for component 2.

	Returns:
		array_like: Total μ.
	"""


	bn=2.*abs(n)-1./3+4./405/abs(n)+46./25515/abs(n)**2+131./1148175/abs(n)**3-2194697./30690717750/abs(n)**4
	bnd=2.*abs(nd)-1./3+4./405/abs(nd)+46./25515/abs(nd)**2+131./1148175/abs(nd)**3-2194697./30690717750/abs(nd)**4
	s1=abs(ie)*np.exp(-bn*((np.divide(x,abs(re)))**(1./abs(n))-1.))
	s2=abs(i0)*np.exp(-bnd*((np.divide(x,abs(rd)))**(1./abs(nd))-1.))
	i=-2.5*np.log10(s1+s2)+22.5
	return i
def envel(x,i0,rd):
	"""Exponential disk profile alone (mag/arcsec²).

	Args:
		x  (array_like): Radius.
		i0 (float): Central intensity.
		rd (float): Scale length.

	Returns:
		array_like: μ(x).
	"""

	i=-2.5*np.log10(abs(i0)*np.exp(-np.divide(x,abs(rd))))+22.5
	return i
#############################################################################################################################################
# ----------------------------------------------------------------------
# Master pipeline for one galaxy
# ----------------------------------------------------------------------
def bcgfigs(cluster,tipo):
	"""Main analysis pipeline for a single galaxy: read data, compute
	flux‑weighted Sérsic index, run K‑means clustering.

	Detailed steps:
	1. Read the extended isophote table (``iso_table_gr.dat``) and
	   apply PA continuity correction to avoid discontinuities at ±π.
	2. Retrieve pre‑fitted model parameters (Sérsic, S+E, S+S) and
	   their covariance from the global lists ``info`` and ``covec``.
	3. Calculate the radially‑varying, flux‑weighted normalised Sérsic
	   index for the S+E and S+S models.  The index at each radius is
	   the intensity‑weighted average of the component Sérsic indices.
	   Uncertainties are propagated accordingly.
	4. Perturb the observed ellipticity, PA, and the normalised Sérsic
	   index by adding Gaussian noise based on their errors to create
	   a Monte‑Carlo realisation for the clustering.
	5. Call ``testkmeans`` on the perturbed parameters (and, in
	   commented lines, on the unperturbed S+E and S+S indices).

	Args:
		cluster (str): Galaxy identifier.
		tipo    (str): Morphological type (e.g., 'eliptica').

	Returns:
		None.  All outputs are saved to disk.

	Note:
		The function uses global variables ``ok``, ``info``, and
		``covec`` which must be populated before calling.
	"""


	print(cluster,tipo)
	#####################################################################################################################	
	temp=[[] for i in range(25)]
	iso_table=open('L07_coef_gain/'+cluster+'/iso_table_gr.dat','r')
	for item in iso_table.readlines():
		if len(item.split()) == 5:
			extval_ellip=float(item.split()[0])
			extval_pa=float(item.split()[1])
			maxrad=float(item.split()[2])
			sigmaskyg=float(item.split()[3])
			sigmaskyr=float(item.split()[4])

		else:
			for i in range(25):
				temp[i].append(float(item.split()[i]))
	vec_dirty=[]
	for i in range(25):
		vec_dirty.append(np.asarray(temp[i]))
	#

	#
	ext_cut=ma.masked_where(vec_dirty[3] == vec_dirty[3][-1],vec_dirty[3])
	ext_slice=ma.clump_masked(ext_cut)
	for piece in ext_slice:
		if piece.stop == len(vec_dirty[3]):
			ext_cleaner=piece.start
	
	#
	vec=[]
	cutoffs=(vec_dirty[2]<vec_dirty[2][ext_cleaner]) & (vec_dirty[10]!=0) & (vec_dirty[11]!=0)
	for conj in vec_dirty:
		vec.append(conj[cutoffs])
	x0,y0,sma,pa,eps,intens,a3,b3,a4,b4,ellip_err,pa_err,int_err,a3_err,b3_err,a4_err,b4_err,intens_free_g,int_free_err_g, intens_free_r,int_free_err_r,intens_fix_r,int_fix_err_r,intens_fix_g,int_fix_err_g=vec


	mag_iso = 22.5-2.5*np.log10(intens) + 2.5*(log10(0.1569166))
	mag_err = 2*2.5*np.log10(np.exp(1.))*int_err/intens

	fig,ax=plt.subplots(1,3,figsize=(9,3),layout='constrained')
	pa0=np.copy(pa)
	ax[0].scatter(np.power(sma,0.25),pa/np.pi*180.,c='black',edgecolors='black')
	ax[0].set_ylabel('PA (photutils)')
	ax[0].set_xlabel(r'$R^{1/4}$ (arcsec)')	
	#
	
	for i in range(1,len(pa)):
		while pa[i] - pa[i-1] >np.pi/2:
			pa[i]-=np.pi
		while pa[i] - pa[i-1] <-np.pi/2:
			pa[i]+=np.pi
	pa_temp=np.copy(pa)
	for i in range(3,len(pa)):
		while pa[i] - np.average([pa[i-1],pa[i-2],pa[i-3]]) >np.pi/2:
			pa[i]-=np.pi
		while pa[i] - np.average([pa[i-1],pa[i-2],pa[i-3]])<-np.pi/2:
			pa[i]+=np.pi

	ax[1].scatter(sma,pa/np.pi*180.,c='black',edgecolors='black')
	ax[1].set_ylabel('PA (corrigido)')
	ax[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	ax[2].scatter(np.power(sma[pa==pa0],0.25),pa[pa==pa0]/np.pi*180.,c='white',edgecolors='black',label='posição mantida')
	ax[2].scatter(np.power(sma[pa!=pa0],0.25),pa0[pa!=pa0]/np.pi*180.,c='red',edgecolors='black',label='posição original')
	ax[2].scatter(np.power(sma[pa!=pa0],0.25),pa[pa!=pa0]/np.pi*180.,c='black',edgecolors='black',label='posição corrigida')
	ax[2].set_ylabel('PA (final)')
	ax[2].set_xlabel(r'$R^{1/4}$ (arcsec)')
	fig.legend(loc=('outside upper right'))
	#plt.savefig('L07_coef_gain/'+tipo+'/pa_corrigido/'+cluster+'.png')
	plt.close()

	fig,ax=plt.subplots(1,2,figsize=(9,6),layout='constrained')
	ax[0].scatter(np.power(sma[pa_temp==pa0],0.25),pa_temp[pa_temp==pa0]/np.pi*180.,c='white',edgecolors='black',label='posição mantida')
	ax[0].scatter(np.power(sma[pa_temp!=pa0],0.25),pa0[pa_temp!=pa0]/np.pi*180.,c='red',edgecolors='black',label='posição original')
	ax[0].scatter(np.power(sma[pa_temp!=pa0],0.25),pa_temp[pa_temp!=pa0]/np.pi*180.,c='black',edgecolors='black',label='posição corrigida')
	ax[0].set_ylabel('PA (corrigido)')
	ax[0].set_xlabel(r'$R^{1/4}$ (arcsec)')
	ax[0].set_title('PA semi-corrigido')
	ax[1].scatter(np.power(sma[pa==pa0],0.25),pa[pa==pa0]/np.pi*180.,c='white',edgecolors='black',label='posição mantida')
	ax[1].scatter(np.power(sma[pa!=pa0],0.25),pa0[pa!=pa0]/np.pi*180.,c='red',edgecolors='black',label='posição original')
	ax[1].scatter(np.power(sma[pa!=pa0],0.25),pa[pa!=pa0]/np.pi*180.,c='black',edgecolors='black',label='posição corrigida')
	ax[1].set_ylabel('PA (final)')
	ax[1].set_xlabel(r'$R^{1/4}$ (arcsec)')
	ax[1].set_title('PA corrigido')
	fig.legend(loc=('outside upper right'))
	#plt.savefig('L07_coef_gain/'+tipo+'/pa_corrigido_comp/'+cluster+'.png')
	plt.close()
	
	vec=info[ok.index(cluster)].split()
	vec_err=covec[ok.index(cluster)].split()
	
	spopt = np.abs([float(vec[7]),float(vec[8]),float(vec[9])])#7,8,9]]
	sepopt=np.abs([float(vec[10]),float(vec[11]),float(vec[12]),float(vec[13]),float(vec[14])])#10,11,12,13,14
	sspopt=np.abs([float(vec[15]),float(vec[16]),float(vec[17]), float(vec[18]),float(vec[19]),float(vec[20])])#15,16,17,18,19,20
	
	spcov = np.abs([float(vec_err[0]),float(vec_err[1]),float(vec_err[2])])#7,8,9]]
	sepcov=np.abs([float(vec_err[3]),float(vec_err[4]),float(vec_err[5]),float(vec_err[6]),float(vec_err[7])])#10,11,12,13,14
	sspcov=np.abs([float(vec_err[8]),float(vec_err[9]),float(vec_err[10]),float(vec_err[11]),float(vec_err[12]),float(vec_err[13])])#15,16,17,18,19,20

	chisq_s,chisq_se,chisq_ss= float(vec[4]),float(vec[5]),float(vec[6])
	
	print('Sersic fit results:\n    Ie =',format(abs(spopt[0]), '.3E'),'\n    Re =',format(spopt[1], '.3E'),'\n    n =',format(abs(spopt[2]), '.3E'))
	print('S+E fit results:\n    Ie =',format(abs(sepopt[0]), '.3E'),'\n    Re =',format(abs(sepopt[1]), '.3E'),'\n    n =',format(abs(sepopt[2]), '.3E'),'\n    I0 =',format(abs(sepopt[3]), '.3E'),'\n    Rd =',format(abs(sepopt[4]), '.3E'))
	print('S+S fit results:\n    Ie =',format(abs(sspopt[0]), '.3E'),'\n    Re =',format(abs(sspopt[1]), '.3E'),'\n    n =',format(abs(sspopt[2]), '.3E'),'\n    I0 =',format(abs(sspopt[3]), '.3E'),'\n    Rd =',format(abs(sspopt[4]), '.3E'),'\n    nd =',format(abs(sspopt[5]), '.3E'))


	#################################################################################################################
	###################

	intens_sersic=np.power(10,np.divide(22.5 - sersic(sma,sepopt[0],sepopt[1],sepopt[2]),2.5))
	intens_env=np.power(10,np.divide(22.5 - envel(sma,*sepopt[3:5]),2.5))
	
	n_se_1=sepopt[2]
	n_se_2=1.
	
	n_se_1_err=sepcov[2]

	n_se_norm_temp=[]
	n_se_inc_temp=[]
	for i in range(len(sma)):
		n_intens=(n_se_1*intens_sersic[i])+(n_se_2*intens_env[i])
		n_int_err=n_se_1_err*intens_sersic[i]
		intens_weight=intens_sersic[i]+intens_env[i]
		n_se_norm_temp.append(np.divide(n_intens,intens_weight))
		n_se_inc_temp.append(np.divide(n_int_err,intens_weight))
	n_se_norm=np.array(n_se_norm_temp)
	n_se_inc=np.array(n_se_inc_temp)

	#################################################################################################################
	###################
	
	intens_sersic_1=np.power(10,np.divide(22.5 - sersic(sma,sspopt[0],sspopt[1],sspopt[2]),2.5))
	intens_sersic_2=np.power(10,np.divide(22.5 - sersic(sma,sspopt[3],sspopt[4],sspopt[5]),2.5))
	
	n_ss_1=sspopt[2]
	n_ss_2=sspopt[5]
	
	n_ss_1_err=sspcov[2]
	n_ss_2_err=sspcov[5]
	
	n_ss_norm_temp=[]
	n_ss_inc_temp=[]
	for i in range(len(sma)):
		n_intens=(n_ss_1*intens_sersic_1[i])+(n_ss_2*intens_sersic_2[i])
		intens_weight=intens_sersic_1[i]+intens_sersic_2[i]
		n_int_err=np.power(n_ss_1_err*(intens_sersic_1[i]/intens_weight),2)+np.power(n_ss_2_err*(intens_sersic_2[i]/intens_weight),2)
		n_ss_norm_temp.append(np.divide(n_intens,intens_weight))
		n_ss_inc_temp.append(np.sqrt(n_int_err))
		
	n_ss_norm=np.array(n_ss_norm_temp)
	n_ss_inc=np.array(n_ss_inc_temp)

	###############################################################################################################################
	#PERTURBAÇÃO DOS PARÂMETROS 
	
	pa_perturb= np.random.normal(loc=pa,scale=pa_err)
	eps_perturb=np.random.normal(loc=eps,scale=ellip_err)
	n_se_perturb=np.random.normal(loc=n_se_norm,scale=n_se_inc)
	n_ss_perturb=np.random.normal(loc=n_ss_norm,scale=n_ss_inc)
	for i in range(len(pa_perturb)):
		print(pa[i],pa_err[i],pa_perturb[i],'-----',eps[i],ellip_err[i],eps_perturb[i],'-------',n_se_norm[i],n_se_inc[i],n_se_perturb[i],'-------',n_ss_norm[i],n_ss_inc[i],n_ss_perturb[i])
	
	#testkmeans(cluster,tipo,eps,ellip_err,pa,pa_err,n_se_norm,n_se_inc,sma,mag_iso,'test_kmeans_se',' S+E')
	testkmeans(cluster,tipo,eps_perturb,ellip_err,pa_perturb,pa_err,n_se_perturb,n_se_inc,sma,mag_iso,'test_kmeans_se',' S+E')
	#testkmeans(cluster,tipo,eps,ellip_err,pa,pa_err,n_ss_norm,n_ss_inc,sma,mag_iso,'test_kmeans_ss',' S+S')
	
	'''
	plt.figure()
	plt.suptitle(cluster)
	plt.scatter(np.power(sma,0.25),n_se_norm,edgecolors='black',label='n bojo='+str(n_se_1)+'\n'+'n env='+str(n_se_2))
	plt.ylim([np.min(n_se_norm)-(np.max(n_se_norm)-np.min(n_se_norm))*0.1,np.max(n_se_norm)+(np.max(n_se_norm)-np.min(n_se_norm))/10.])
	plt.legend()
	plt.ylabel(r'$n$')
	plt.xlabel('sma')
	plt.savefig('L07_coef_gain/'+tipo+'/test_n_se/'+cluster+'.png')
	plt.close()
	
	plt.figure()
	plt.suptitle(cluster)
	plt.scatter(np.power(sma,0.25),n_ss_norm,edgecolors='black',label='n bojo='+str(n_ss_1)+'\n'+'n env='+str(n_ss_2))
	plt.ylim([np.min(n_ss_norm)-(np.max(n_ss_norm)-np.min(n_ss_norm))*0.1,np.max(n_ss_norm)+(np.max(n_ss_norm)-np.min(n_ss_norm))/10.])
	plt.legend()
	plt.ylabel(r'$n$')
	plt.xlabel('sma')
	plt.savefig('L07_coef_gain/'+tipo+'/test_n_ss/'+cluster+'.png')
	plt.close()
	'''
	return 


############################################################################################
# ----------------------------------------------------------------------
# Main execution block
# ----------------------------------------------------------------------
############################################################################################

ok=[]
info=[]
covec=[]
output=open('iso_geral_values_coef_v3.dat','r+')
outcov=open('iso_geral_errors_v4.dat','r+')
#outkelbow=open('elbow_check.dat','r')
#outkelbow.write('cluster \t numero de cluster \n\n')
for item in output.readlines():
	ok.append(item.split()[0])
	info.append(item[4:])
for error in outcov.readlines():
	covec.append(error[4:])
with open('iso_geral_values_coef_astro_tipo.dat','r') as inp1:
	ninp1=len(inp1.readlines())
inp1=open('iso_geral_values_coef_astro_tipo.dat','r')

for ik in range(1,ninp1):
	ls1=inp1.readline()
	ll1=ls1.split()
	#if ll1[0] == '1090':
	if ll1[-1] == 'eliptica':	
		coef_comp=bcgfigs(ll1[0],ll1[-1])
		break
		#print(covec[ok.index(ll1[0])])
		#print(ll1[0])
output.close()
#outkelbow.close()
############################################################################################

