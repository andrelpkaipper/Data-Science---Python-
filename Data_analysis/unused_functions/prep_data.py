import numpy as np
import os
import shutil
"""
# output=np.loadtxt(f'{sample}_stats_observation/plano_re_n2/svc_results_el_cd_WHL.dat',usecols=[0,-1],dtype=str).T
# print(output.T,'\n #########')
# output=np.loadtxt(f'{sample}_stats_observation/plano_rff_eta/svc_results_el_cd_WHL.dat',usecols=[0,-1],dtype=str).T
# print(output.T,'\n #########')
# output=np.loadtxt(f'{sample}_stats_observation/plano_n_ratio_n2/svc_results_el_cd_WHL.dat',usecols=[0,-1],dtype=str).T
# print(output.T)
"""
# data_halpha_temp=np.loadtxt(f'casjobs_halpha_{sample}.dat',dtype=str).T
# data_veldisp_temp=np.loadtxt(f'casjobs_veldisp_{sample}.dat',dtype=str).T
# output=open(f'{sample}_files/data_indiv_clean_{sample}.dat','a')
# line=[]
# bcg=[]
# with open(f'data_indiv_ecd.dat','r') as inp_casjobs:
# 	data_casjobs=inp_casjobs.readlines()
# 	for i,obj in enumerate(data_casjobs):
# 		data_line=obj.split()
# 		cl=data_line[0]
# 		line.append(' '.join(data_line))
# 		bcg.append(cl)
# with open(f'{sample}_files/ass_{sample}.dat','r') as inp1:
# 	data_sample=inp1.readlines()
# 	for i,obj in enumerate(data_sample):
# 		cluster=obj.split()[0]
# 		if cluster in bcg:
# 			cl_idx=bcg.index(cluster)
# 			output.write(f'{line[cl_idx]}\n')
# 		else:
# 			vec=['0' for _ in range(4)]
# 			lines=' '.join(vec)
# 			output.write(f'{cluster} {lines}\n')

# output.close()
# sample = 'L07'
# diretorio = f'{sample}_stats_observation/test_ks/'
# destino = f'{diretorio}all_data'

# os.makedirs(destino, exist_ok=True)

# for item in os.listdir(diretorio):
#     path_item = os.path.join(diretorio, item)

#     if os.path.isdir(path_item) and item != 'all_data':
#         for arquivo in os.listdir(path_item):
#             if arquivo.endswith(('.png', '.pdf', '.jpg', '.jpeg')):
#                 origem = os.path.join(path_item, arquivo)
#                 shutil.copy2(origem, destino)

sample='L07'

output=open(f'{sample}_files/{sample}_rff_duplo_desi.dat','w')
bcg=[]
line=[]
with open(f'{sample}_rff_duplo_desi.dat','r') as inp_casjobs:
	data_casjobs=inp_casjobs.readlines()
	for i,obj in enumerate(data_casjobs):
		data_line=obj.split()
		cl=data_line[0]
		line.append(' '.join(data_line))
		bcg.append(cl)

with open(f'{sample}_files/ass_{sample}.dat','r') as inp1:
	data_sample=inp1.readlines()
	for i,obj in enumerate(data_sample):
		cluster=obj.split()[0]
		if cluster in bcg:
			cl_idx=bcg.index(cluster)
			# vec=['0' for i in range(len(line[1].split()[1:]))]
			# if line[cl_idx].split()[1] == '-1000':
			# 	output.write(f'{cluster} {' '.join(vec)}\n')
			# else:
			output.write(f'{line[cl_idx]}\n')
		else:
			vec=['0' for i in range(len(line[1].split()[1:]))]
			output.write(f'{cluster} {' '.join(vec)}\n')