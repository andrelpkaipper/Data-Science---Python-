import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma, gammainc

def sersic_flux_total(I0, Re, n):
    """Calcula fluxo total para perfil de Sérsic"""
    k = 2*n - 1/3 + 4/(405*n)  # Aproximação de Ciotti
    b_n = k * n
    return I0 * Re**2 * 2 * np.pi * n * np.exp(b_n) * gamma(2*n) / (b_n**(2*n))

def sersic_flux_within_radius(I0, Re, n, r_max):
    """Calcula fluxo dentro de um raio r_max"""
    k = 2*n - 1/3 + 4/(405*n)
    b_n = k * n
    x = b_n * (r_max/Re)**(1/n)
    return I0 * Re**2 * 2 * np.pi * n * np.exp(b_n) * gamma(2*n) * gammainc(2*n, x) / (b_n**(2*n))

# Parâmetros típicos de galáxia cD
I0_bojo = 1000    # Intensidade central do bojo
Re_bojo = 3.0     # Raio efetivo do bojo (kpc)
n_bojo = 4.0      # Índice do bojo

I0_env = 500      # Intensidade central do envelope  
Re_env = 60.0     # Raio efetivo do envelope (kpc)

# Calculando B/T para diferentes n do envelope
n_envelope_valores = [1.0, 2.0]
resultados = []

for n_env in n_envelope_valores:
    # Fluxo total de cada componente
    fluxo_bojo_total = sersic_flux_total(I0_bojo, Re_bojo, n_bojo)
    fluxo_env_total = sersic_flux_total(I0_env, Re_env, n_env)
    
    # Razão B/T
    BT = fluxo_bojo_total / (fluxo_bojo_total + fluxo_env_total)
    
    resultados.append({
        'n_envelope': n_env,
        'fluxo_bojo': fluxo_bojo_total,
        'fluxo_envelope': fluxo_env_total,
        'BT': BT
    })

# Mostrando resultados
print("=== COMPARAÇÃO n_envelope = 1 vs 2 ===")
for res in resultados:
    print(f"n_env = {res['n_envelope']}: B/T = {res['BT']:.3f}")
    print(f"  Fluxo bojo: {res['fluxo_bojo']:.2e}")
    print(f"  Fluxo envelope: {res['fluxo_envelope']:.2e}")