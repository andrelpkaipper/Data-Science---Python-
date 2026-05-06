import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure()
gs = gridspec.GridSpec(2, 3, width_ratios=[1,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
ax_center = fig.add_subplot(gs[1,1])
ax_topx = fig.add_subplot(gs[0,1],sharex=ax_center)
ax_righty = fig.add_subplot(gs[1,2],sharey=ax_center)
ax_lefty = fig.add_subplot(gs[1,0])

sc=ax_center.scatter([],[],c=[],cmap=plt.colormaps['hot'],label='aqui')

cbar = plt.colorbar(sc, ax=ax_lefty,location='left')
cbar.set_label(r'$n$')


# fig = plt.figure()
# gs = gridspec.GridSpec(2, 4, width_ratios=[1,5,5,1], height_ratios=[1,5],hspace=0.05, wspace=0.05)
# ax_center_left = fig.add_subplot(gs[1,1])
# ax_center_right = fig.add_subplot(gs[1,2])
# ax_topx_left = fig.add_subplot(gs[0,1],sharex=ax_center_left)
# ax_topx_right = fig.add_subplot(gs[0,2],sharex=ax_center_right)
# ax_righty = fig.add_subplot(gs[1,3],sharey=ax_center_right)
# ax_lefty = fig.add_subplot(gs[1,0])

# sc=ax_center_left.scatter([],[],c=[],cmap=plt.colormaps['hot'],label='aqui')

# cbar = plt.colorbar(sc, ax=ax_lefty,location='left')
# cbar.set_label(r'$n$')
# ax_center.plot(linspace_re, linha_sersic, color='red', linestyle='-', label=linha_elip_label+'\n'+fr'$\alpha={alpha_med_ser}$'+'\n'+fr'$\beta={beta_med_ser}$')
# ax_center.scatter(re_sersic_cd_kpc,mue_sersic_cd,marker='s',edgecolor='black',alpha=0.3,label=label_2c,color='black')
# ax_center.plot(linspace_re, linha_cd, color='black', linestyle='-', label=linha_2c_label+'\n'+fr'$\alpha={alpha_cd_label}$'+'\n'+fr'$\beta={beta_cd_label}$')
ax_center.legend(fontsize='small')
# ax_center.set_ylim(y2ss,y1ss)
# ax_center.set_xlim(x1ss,x2ss)
# ax_center.set_xlabel(xlabel)
# ax_center.set_ylabel(ylabel)

# ax_topx.plot(re_linspace,re_kde_sersic_kpc(re_linspace),color='green')
# ax_topx.plot(re_linspace,re_kde_cd_kpc(re_linspace),color='black')
# ax_topx.axvline(np.average(re_sersic_kpc),ls='--',color='green',label=fr'$\mu={format(np.average(re_sersic_kpc),'.3')}$')
# ax_topx.axvline(np.average(re_sersic_cd_kpc),ls='--',color='black',label=fr'$\mu={format(np.average(re_sersic_cd_kpc),'.3')}$')
# ax_topx.legend(fontsize='small')
# ax_topx.tick_params(labelbottom=False)
# ax_righty.plot(mue_kde_sersic_kpc(mue_linspace),mue_linspace,color='green')
# ax_righty.plot(mue_kde_cd_kpc(mue_linspace),mue_linspace,color='black')
# ax_righty.axhline(np.average(mue_sersic),ls='--',color='green',label=fr'$\mu={format(np.average(mue_sersic),'.3')}$')
# ax_righty.axhline(np.average(mue_sersic_cd),ls='--',color='black',label=fr'$\mu={format(np.average(mue_sersic_cd),'.3')}$')
# ax_righty.legend(fontsize='small')
# ax_righty.tick_params(labelleft=False)
plt.show()
plt.close()
