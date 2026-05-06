import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
import scipy.optimize as scp
from subprocess import call
import numpy.ma as ma
from sklearn.mixture import GaussianMixture

def gauss(x,mu,sigma,amp):
	return amp*np.exp(-((x-mu)**2)/(2*(sigma)**2))
def bigauss(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
	return gauss(x,mu1,sigma1,amp1) + gauss(x,mu2,sigma2,amp2)
def lognormal(x,mu,sigma,amp):
	return (amp)*np.exp(-((np.log(x)-mu)**2)/(2*(sigma)**2))
def triforce(x,mu1,sigma1,amp1,mu2,sigma2,amp2,mu3,sigma3,amp3):
	return gauss(x,mu1,sigma1,amp1) + gauss(x,mu2,sigma2,amp2) + lognormal(x,mu3,sigma3,amp3)
def dlognorm(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
    px=amp1*np.exp(-((x-mu1)**2)/(2*(sigma1)**2))+amp2*np.exp(-(np.log(x)-mu2)**2/(2*sigma2**2))
    return px
def dlognorm3(x,mu1,sigma1,amp1,mu2,sigma2,amp2,nn2,mu3,sigma3,amp3):
    px=(amp1*np.exp(-(item-mu1)**2/(2*sigma1**2))+amp2*np.exp(-(np.log(item*nn2)-mu2)**2/(2*sigma2**2))/(item*nn2)+amp3*np.exp(-(item-mu3)**2/(2*sigma3**2)))
    return px

sample=str(sys.argv[1])
#data_rff=np.loadtxt(f'WHL_profit_observation_SE_sky.dat',dtype=str).T
header_mue=['cluster','mue_med_s','mue_med_1','mue_med_2']
header_data=np.loadtxt(f'profit_observation.header',dtype=str)
data_obs_temp=np.loadtxt(f'{sample}_files/{sample}_profit_observation_SE_sky.dat',dtype=str).T
data_obs=dict(zip(header_data,data_obs_temp))
data_mue_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_{sample}.dat',dtype=str).T
data_mue_comp_temp=np.loadtxt(f'{sample}_files/mue_med_dimm_comp_{sample}.dat',dtype=str).T
data_mue=dict(zip(header_mue,data_mue_temp))
data_mue_comp=dict(zip(header_mue,data_mue_comp_temp))

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


rff=data_obs['rff'].astype(float)[lim_finite & cut_lim]

bins = np.arange(0,0.1,0.0018)
# bins=[]
# wid=np.arange(0.001,0.002,0.0001)
# for i,space in enumerate(wid):
# 	bins.append(np.arange(0,0.1,space))
# print(bins)
# # converter=(len(rff)*0.0005)
# converter=(7057*0.0005)
# rff_vec=rff.reshape(-1,1)

output=[]
chi2_vec=[]
test_out=[]
table=open('diagram_trials_L07.dat','w')
for j in range(100):
	print(j)
	fig0=plt.figure()
	y,x,_ = plt.hist(rff,bins=bins,density=True,histtype='step',color='white',edgecolor='black')
	x = (x[1:]+x[:-1])/2
	x0=np.linspace(0,0.1,10000)
	p0=[np.random.normal(0.017,0.005),np.random.normal(0.005,0.001),np.random.normal(42,10),np.random.normal(-3.5,0.005),np.random.normal(0.3,0.05),np.random.normal(15,5)]
	# dlpov=[1.54622724e-02,4.61969720e-03,1.23642039e+02,-3.93661732e+00,-4.81476825e-01,7.57926648e+01]
	#0.019113 0.005185 16.049583 -4.001696 0.497914 30.278537 [0.837686]
	#0.017835 0.005812 42.214771 -3.522754 0.328877 14.599428 1.050523

	dlpov,dlcov=scp.curve_fit(dlognorm,x,y,p0=p0,maxfev=10000000)
	# dlpov,dlcov=scp.curve_fit(dlognorm,x,y,maxfev=10000000)
	lim=y!=0.0
	y=y[lim]
	x=x[lim]
	sigma_y = np.sqrt(y)
	ynorm=np.power(y-(dlognorm(x,*dlpov)),2)/(sigma_y**2)
	chi2_norm=np.sum(ynorm)/(len(y)-7)
	table.write('%i %f %f %f %f %f %f %f\n'%(j,*dlpov,chi2_norm))
	plt.plot(x0,dlognorm(x0,*dlpov),linewidth=2, color='b',label='G+log')
	plt.plot([],[],color='b',linewidth=2,label = r'$\chi^{2}$ '+str(format(abs(chi2_norm), '.5'))+'')
	plt.plot(x0,gauss(x0,*dlpov[:3]),linewidth=1,ls=':',color='black',label='G')
	plt.plot(x0,lognormal(x0,*dlpov[3:6]),linewidth=1,ls='--',color='g',label='log')
	# # plt.axvline(0.01842,label='RFF=0.0184',color='b')
	# # plt.axvline(0.0198,label='RFF=0.0198',color='r')
	plt.legend()
	plt.xlabel('RFF')
	plt.ylabel('Objetos')
	# plt.show()
	plt.savefig('L07_diagram_dump/'+str(j)+'.png')
	plt.close(fig0)
