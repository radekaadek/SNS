import numpy as np

def jono(tow, phi, lam, el, az):
    '''

    Parameters
    ----------
    tow : float/int
        Time of Week - czas tygodnia GPS obserwacji
    phi : float [degrees]
        szerokosć geodezyjna odbiornika
    lam : float [degrees]
        długosć geodezyjna odbiornika
    el : float [degrees]
        elewacja satelity
    az : float [degrees]
        azymut satelity

    Returns
    -------
    dI : float
        poprawka jonosferyczna w kierunku satelity

    '''
    alfa = [1.6764e-08,  7.4506e-09, -1.1921e-07,  0.0000e+00]
    beta = [1.1059e+05,  0.0000e+00, -2.6214e+05,  0.0000e+00]
    sem_in_deg = 1/180
    sem_in_rad = np.pi
    els = el * sem_in_deg
    psi = 0.0137/(els + 0.11) - 0.022

    phi_ipp = (phi*sem_in_deg) + psi * np.cos(np.deg2rad(az))
    if phi_ipp > 0.416:
        phi_ipp = 0.416
    elif phi_ipp < -0.416:
        phi_ipp = -0.416

    lam_ipp = (lam*sem_in_deg) + (psi*np.sin(np.deg2rad(az))) / np.cos(phi_ipp*sem_in_rad)

    phi_m = phi_ipp + 0.064*np.cos((lam_ipp-1.617)*sem_in_rad)

    local_time = 43200*lam_ipp + tow
    local_time = np.mod(local_time, 86400)
    if local_time >= 86400:
        local_time = local_time - 86400
    elif local_time < 0:
        local_time = local_time+86400

    iono_amplitude = alfa[0] + alfa[1]*phi_m + alfa[2]*phi_m**2 + alfa[3]*phi_m**3

    if iono_amplitude < 0:
        iono_amplitude = 0

    pion = beta[0] + beta[1]*phi_m + beta[2]*phi_m**2 + beta[3]*phi_m**3
    if pion < 72000:
        pion = 72000

    phi_ion = 2*np.pi*(local_time - 50400)/pion

    mapper = 1 + 16 * (0.53 - els)**3

    c = 299792458
    if abs(phi_ion) <= np.pi/2:
        dI = c * mapper * (5*10**(-9) + iono_amplitude *(1 - (phi_ion**2)/2 + (phi_ion**4)/24))
    elif abs(phi_ion) > np.pi/2:
        dI = c*mapper*(5*10**(-9))
    else:
        dI = 0
    return dI

