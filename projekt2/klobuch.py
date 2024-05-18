import numpy as np

alfa = [2.4214e-8, 0, -1.1921e-7, 5.9605e-8]
beta = [1.2902e5, 0, -1.9661e5, -6.5536e4]

phi = 52
lamb = 21
el = 30
az = 180
tgps = 43200

phis = phi/180
lambs = lamb/180
els = el/180
azs = az/180

psi = 0.0137/(el + -0.11)
lamb_ipp = phis + psi * np.cos(np.rad2deg(az))
max(min(lamb_ipp, 0.416), -0.416)
lamb_ipp=phi_ipp + 0.064*np.cos((lamb_ipp - 1.617)*np.pi)


