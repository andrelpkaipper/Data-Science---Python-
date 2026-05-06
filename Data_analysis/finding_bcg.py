import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

"""
finding_bcg.py

Cross‑match between the L07 BCG catalogue and the full WHL cluster catalogue.

This script reads two fixed text files:
- ``L07_files/data_indiv_clean_L07.dat`` : contains the BCG sample with columns
  cluster, ?, ?, ?, ?, RA, Dec (columns 0,5,6).
- ``L07_type_description.dat`` : contains the full WHL cluster description with
  columns RA, Dec, redshift, velocity dispersion (columns 2,3,4,5).

It performs a coordinate cross‑match within a hard‑coded tolerance of 0.003°
(~10.8 arcsec) using a simple spherical distance calculation (no wrapping
around RA = 0° is needed for this dataset).  For each matched cluster, it
computes R₂₀₀ from the velocity dispersion using the formula:

    R₂₀₀ = 1.73 * (σ / 1000 km/s) / (0.7 + 0.3 * (1+z)³)   [Mpc]

and writes the output to ``L07_r200.dat`` with columns:
    cluster_ID   R₂₀₀ [Mpc]   velocity_dispersion [km/s]

Galaxies that cannot be matched are collected in a list (assigned to the
variable ``fail``, which is currently not used beyond that but could be
written out if needed).

Note: The variable ``fail`` is used in the script but never defined outside
the conditional block, causing a ``NameError`` if any galaxy fails to match.
This is a known bug in the original code.
"""

data_whl = np.loadtxt('L07_files/data_indiv_clean_L07.dat', dtype=str).T
data_full = np.loadtxt('L07_type_description.dat', dtype=str, usecols=[2,3,4,5]).T

# Converte para float
cluster=data_whl[0]

ra_meu  = data_whl[5].astype(float)
dec_meu = data_whl[6].astype(float)
ra_whl  = data_full[0].astype(float)
dec_whl = data_full[1].astype(float)

vel=data_full[3].astype(float)
z=data_full[2].astype(float)
r200=1.73*(vel/1000.)*(1/(0.7+0.3*(1+z)**3))

ra1  = np.radians(ra_meu)
dec1 = np.radians(dec_meu)
ra2  = np.radians(ra_whl)
dec2 = np.radians(dec_whl)

matches = open('L07_r200.dat','w')
tol_deg = 0.003
for i in range(len(ra1)):
    cos_d = np.sin(dec1[i])*np.sin(dec2) + np.cos(dec1[i])*np.cos(dec2)*np.cos(ra1[i] - ra2)
    sep = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))  # em graus
    idx = np.where(sep <= tol_deg)[0]
    if len(idx) > 0:
        for j in idx:
            # print(cluster[i],data_full[0][j],sep[j]*3600,len(idx))
            matches.write(f'{cluster[i]} {r200[j]} {vel[j]} \n')  # em arcsec
    elif len(idx) == 0:
        fail.append((cluster[i],min(sep)))
matches.close()
# for m in matches:
#     print(f"Obj {m[0]} ↔ {m[1]} : {m[2]:.4f} arcsec")


# # Cria coordenadas
# coord_meu  = SkyCoord(ra_meu*u.deg, dec_meu*u.deg)
# coord_whl  = SkyCoord(ra_whl*u.deg, dec_whl*u.deg)

# # Tolerância (ex: 1 arcsec)
# tol = 1.0 * u.arcsec
# print(tol)
# # Loop para encontrar matches
# matches = []
# for i, c in enumerate(coord_meu):
#     sep = c.separation(coord_whl) # distância angular para todos
#     print(sep)
#     idx = np.where(sep == min(sep))[0]
#     if len(idx) > 0:
#         for j in idx:
#             print(cluster[i],data_full[0][j],sep[j].arcsec,len(idx))
#             matches.append((i, j, sep[j].arcsec))
# print(len(matches))
# Exibe
# for m in matches:
#     print(f"Obj {m[0]} ↔ {m[1]} : {m[2]:.4f} arcsec")