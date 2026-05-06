import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2


def lnt(cluster,X,Y):
    ajust1 = fits.getdata(cluster+'/bcg_r.fits',1)
    mask = fits.getdata(cluster+'bcg_r_mask.fits')
    mask_b = fits.getdata(cluster+'bcg_r_mask_b.fits')
    ##############################################################
    # INICIO DA ASSIMETRIA

    #IMG SMALL 
    xxc = X-5
    xxc2 = X+5
    yyc = Y-5
    yyc2 = Y+5
    
    bcg_small = ajust1[int(yyc):int(yyc2),int(xxc):int(xxc2)]# img bcg
    mask_small = mask[int(yyc):int(yyc2),int(xxc):int(xxc2)]# mask normal 
    mask_b_small = mask_b[int(yyc):int(yyc2),int(xxc):int(xxc2)]# mask bcg
    
    dxb=0
    dyb=0
    speb=0
    matrix = ([1,0],[0,1])
    # CALCULO DA ASSIMETRIA 10X10

    for a in range(1000):
        #
        dx = np.random.normal(dxb,0.1)
        dy = np.random.normal(dyb,0.1)          
        #
        bcg_small_temp = ssn.affine_transform(bcg_small,matrix,offset=[dx,dy],mode='constant')# img bcg
        mask_b_small_temp = ssn.affine_transform(mask_b_small,matrix,offset=[dx,dy],mode='constant')# mask bcg
        mask_small_temp = ssn.affine_transform(mask_small,matrix,offset=[dx,dy],mode='constant')# mask normal
        #
        bcg_small_rot=np.rot90(bcg_small_temp,2)#img bcg rot
        mask_b_small_rot=np.rot90(mask_b_small_temp,2)#mask bcg rot
        mask_small_rot=np.rot90(mask_small_temp,2)#mask normal rot

        #############
        vas=bcg_small_temp[np.where(((bcg_small_temp != 0.0) & (bcg_small_rot != 0.0)) & ((mask_b_small_temp==1) & (mask_small_temp == 0) & (mask_b_small_rot == 1) & (mask_small_rot == 0)))] #bcg normal
        
        var= bcg_small_rot[np.where(((bcg_small_temp != 0.0) & (bcg_small_rot != 0.0)) & ((mask_b_small_temp==1) & (mask_small_temp == 0) & (mask_b_small_rot == 1) & (mask_small_rot == 0)))] #bcg rotacionada
        
    
        spe = (1-sst.spearmanr(vas,var)[0])
        if spe < speb or speb == 0:
            speb = spe
            dxb = dx
            dyb = dy
    ################################################################
    # ASSIMETRIA GRANDE

    bcg_data = ssn.affine_transform(ajust1,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# img bcg
    mask_b_data = ssn.affine_transform(mask_b,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask bcg
    mask_data = ssn.affine_transform(mask,matrix,offset=[(Y+dxb)-ajust1.shape[0]/2.,(X+dyb)-ajust1.shape[1]/2.],mode='constant')# mask normal
    
    bcg_rot= np.rot90((bcg_data),2)#img bcg rot
    mask_b_rot=np.rot90((mask_b_data),2)#mask bcg rot
    mask_rot=np.rot90((mask_data),2)#mask normal rot

    
    vas=bcg_data[np.where(((bcg_data!=0.0) & (bcg_rot!=0.0)) & ((mask_data == 0) & (mask_rot == 0)) & ((mask_b_data == 1) | (mask_b_rot == 1)))] #bcg normal
    
    var=bcg_rot[np.where(((bcg_data!=0.0) & (bcg_rot!=0.0)) & ((mask_data == 0) & (mask_rot == 0)) & ((mask_b_data == 1) | (mask_b_rot == 1)))] #bcg rotacionada

    res=np.sum(np.absolute(vas-var))

    vas1=np.sum(vas)

    nn2=len(vas)
        
    A1=((res-1.127*sbk*nn2)/vas1)/2.
    A0 = (1-sst.spearmanr(vas,var)[0])

    return A1

with open('data_indiv_desi_L07.dat','r') as inp1:
    ninp1=len(inp1.readlines())
inp1=open('data_indiv_desi_L07.dat','r')
save_file=open('L07_ass_desi.dat','a')
for ik in range(0,ninp1):
    ls1=inp1.readline()
    ll1=ls1.split()
    cluster=ll1[0]
    ra=ll1[1]
    dec=ll1[2]
    if os.path.isfile(f'{cluster}/bcg_r.fits'):
        with open('base_default.sex','r') as inp2:
            ninp2=len(inp2.readlines())
        inp2=open('base_default.sex','r')
        out1=open(cluster+'/base_default.sex','w')
        for j in range(0,ninp2):
            ls2=inp2.readline()
            ll2=ls2.split()
            if len(ll2)>0 and ll2[0]=='CATALOG_NAME':

                ll2[1]=ll1[0]+'/out_sex_large.cat'
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
                ll2[1]=cluster+'/check1_large.fits,'+cluster+'/check2_large.fits,'+cluster+'/check3_large.fits'
                lstrin=' '
                for k in range(0,len(ll2)):
                    lstrin+=ll2[k]+' '
                out1.write('%s\n' % lstrin[1:len(lstrin)])
            else:
                out1.write('%s' % ls2)
        inp2.close()
        out1.close()
        call(f'sex {cluster}/stamp.fits -c {cluster}/base_default.sex',shell=True)

        header = fits.open(f'{cluster}/stamp.fits',memmap=True)[0].header
        data = fits.open(f'{cluster}/stamp.fits',memmap=True)[0].data

        pixel_finder=WCS(header)
        pos_bcg = SkyCoord(ra, dec, unit='deg', frame='fk5')
        x0,y0=pixel_finder.world_to_pixel(pos_bcg)
        ass=lnt(cluster,x0,y0)
        save_file.write('%s %f \n'%(cluster,ass))
save_file.close()