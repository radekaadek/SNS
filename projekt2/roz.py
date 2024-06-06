from readrnx_studenci import *
import math as mat
import datetime

a = 6378137
e2 = 0.00669438002290

def Np(B):
    N = a/(1-e2*(np.sin(B)**2))**0.5
    return N

def hirvonen(X,Y,Z):
    r = (X**2 + Y**2)**0.5
    B = mat.atan(Z/(r*(1-e2)))
    
    while 1:
        N = Np(B)
        H = r/np.cos(B) - N
        Bst = B
        B = mat.atan(Z/(r*(1-(e2*(N/(N+H))))))    
        if abs(Bst-B)<(0.00001/206265):
            break
    L = mat.atan(Y/X)
    N = Np(B)
    H = r/np.cos(B) - N
    return B, L, H

nav_file = 'BRDC00WRD_R_20240650000_01D_GN.rnx'

nav, inav = readrnxnav(nav_file)


healthy = nav[:, 30] == 0

nav = nav[healthy]
inav = inav[healthy]

data_obliczen = [2024,3,5,11,15,0]
week, tow, _ = date2tow(data_obliczen)


observed_satelites = [2,5,8,10,24]

observed_data = []
for sat in observed_satelites:
    satelite_index = inav == sat
    nav_sat = nav[satelite_index]
    dt = np.abs(nav_sat[:,17] - tow)
    index = np.argmin(dt)
    nav_sat = nav_sat[index]
    observed_data.append(nav_sat)

observed_data = np.array(observed_data)

def julday(y,m,d,h=0):
    '''
    Simplified Julian Date generator, valid only between
    1 March 1900 to 28 February 2100
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
    # A = np.trunc(y/100)
    # B = 2-A+np.trunc(A/4)
    # C = np.trunc(365.25*y)
    # D = np.trunc(30.6001 * (m+1))
    # jd = B + C + D + d + 1720994.5
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5
    return jd
def get_gps_time(y, m, d, h, mnt, s):
    '''
    Convert date to GPS time
    '''
    days = julday(y, m, d) - julday(1980, 1, 6)
    weeks = days // 7
    day = days % 7
    secOfWeek = day * 86400 + h * 3600 + mnt * 60 + s

    return weeks, secOfWeek
micro = 3.986005 * 10**14
Wec = 7.2921151467e-5
Uc = 3.986005e14
def satpos(time, nav):
    toc = get_gps_time(nav[0], nav[1], nav[2], nav[3], nav[4], nav[5])[1]
    toe = nav[17]
    tk = time - toe
    # if __name__ == '__main__':
    #     print(f'tk={tk}')
    a = nav[16] ** 2
    # if __name__ == '__main__':
    #     print(f'a={a}')
    n0 = np.sqrt(Uc / a ** 3)
    # if __name__ == '__main__':
    #     print(f'n0={n0}')
    n = n0 + nav[11]
    # if __name__ == '__main__':
    #     print(f'n={n}')
    # print(nav[11])
    Mk = n * tk + nav[12]
    # if __name__ == '__main__':
    #     print(f'Mk={Mk}')
    Ek = Mk
    e = nav[14]
    # if __name__ == '__main__':
    #     print(f'e={e}')
    Ek_old = 0
    while abs(Ek - Ek_old) > 1e-12:
        Ek_old = Ek
        Ek = Mk + e * np.sin(Ek)
    Vk = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(Ek), np.cos(Ek) - e)
    # if __name__ == '__main__':
    #     print(f'Vk={Vk}')
    Phi_k = Vk + nav[23]
    # if __name__ == '__main__':
    #     print(f'Phi_k={Phi_k}')
    Cuc = nav[13]
    Cus = nav[15]
    Cic = nav[18]
    Cis = nav[20]
    Crc = nav[22]
    Crs = nav[10]
    delta_u = Cuc * np.cos(2 * Phi_k) + Cus * np.sin(2 * Phi_k)
    # if __name__ == '__main__':
    #     print(f'delta_u={delta_u}')
    delta_i = Cic * np.cos(2 * Phi_k) + Cis * np.sin(2 * Phi_k)
    # if __name__ == '__main__':
    #     print(f'delta_i={delta_i}')
    delta_r = Crc * np.cos(2 * Phi_k) + Crs * np.sin(2 * Phi_k)
    # if __name__ == '__main__':
    #     print(f'delta_r={delta_r}')
    u = Phi_k + delta_u
    # if __name__ == '__main__':
    #     print(f'u={u}')
    r = a * (1 - e * np.cos(Ek)) + delta_r
    # if __name__ == '__main__':
    #     print(f'r={r}')
    i = nav[21] + delta_i + nav[25] * tk
    # if __name__ == '__main__':
    #     print(f'i={i}')
    x = r * np.cos(u)
    y = r * np.sin(u)
    omega0 = nav[19]
    # print(f'omega0={omega0}')
    omegaC = nav[24]
    # print(f'omegaC={omegaC}')
    Omega = omega0 + (omegaC - Wec) * tk - Wec * toe
    # if __name__ == '__main__':
    #     print(f'Omega={Omega}')
    X = x * np.cos(Omega) - y * np.cos(i) * np.sin(Omega)
    Y = x * np.sin(Omega) + y * np.cos(i) * np.cos(Omega)
    Z = y * np.sin(i)
    alphaf0 = nav[6]
    alphaf1 = nav[7]
    alphaf2 = nav[8]
    c = 299792458
    gt2 = alphaf0 + alphaf1 * (time - toc) + alphaf2 * (time - toc) ** 2
    gtrel = -2 * np.sqrt(Uc * a) * e * np.sin(Ek) / (c ** 2)
    gtrel2 = gt2 + gtrel
    return np.array([X, Y, Z, gtrel2])

# if __name__ == '__main__':
#     Xk, Yk, Zk, dt_s_rel = sat_position(tow, observed_data[0])
#     print(f"Xk: {Xk}, Yk: {Yk}, Zk: {Zk}, dt_s_rel: {dt_s_rel}")
def satelite_position(observed_data, tow):
    # Obliczenie różnicy czasu między czasem obserwacji a czasem obserwacji satelity
# czas jaki upłynął od epoki wyznaczenia efemerydy
    tk = tow - observed_data[17]

# Obliczenie dużej półosi orbity
    a = observed_data[16]**2

# Wyznaczenie średniej prędkości kątowej n, znanej jako ruch średni (ang. mean motion) na podstawie
# III prawa Keplera: (a – duża półoś orbity)
    n0 = np.sqrt(micro / a**3)
    n = n0 + observed_data[11]

# Wyznaczenie poprawionego ruchu średniego:
    Mk = observed_data[12] + n * tk

# Poprawiona anomalia średnia na epokę tk :
    Ek = Mk
    Ek_old = 0
    while np.abs(Ek - Ek_old) > 1e-12:
        Ek_old = Ek
        Ek = Mk + observed_data[14] * np.sin(Ek)

    vk = np.arctan2(np.sqrt(1 - observed_data[14]**2) * np.sin(Ek), np.cos(Ek) - observed_data[14])

# Wyznaczenie argumentu szerokości:
    fik = vk + observed_data[23]

# Wyznaczenie poprawek do argumentu szerokości ∆uk , promienia orbity ∆rk i inklinacji orbity ∆ik :
    delta_uk = observed_data[13] * np.cos(2*fik) + observed_data[15] * np.sin(2*fik)
    delta_ik = observed_data[18] * np.cos(2*fik) + observed_data[20] * np.sin(2*fik)
    delta_rk = observed_data[22] * np.cos(2*fik) + observed_data[10] * np.sin(2*fik)

# Wyznaczenie poprawionych wartości argumentu szerokości uk , promienia orbity rk i inklinacji orbity ik :
    uk = fik + delta_uk
    rk = a * (1 - observed_data[14] * np.cos(Ek)) + delta_rk
    ik = observed_data[21] + delta_ik

# Wyznaczenie pozycji satelity w układzie orbity:
    xk = rk * np.cos(uk)
    yk = rk * np.sin(uk)

# KONTROLA
    # print(f"Kontrola: {np.allclose(rk, np.sqrt(xk**2 + yk**2))}")

# Poprawiona długość węzła wstępującego:
    omk = observed_data[19] + (observed_data[24] - Wec) * tk - Wec * observed_data[17]

# Wyznaczenie pozycji satelity w układzie geocentrycznym ECEF:
    Xk = xk * np.cos(omk) - yk * np.cos(ik) * np.sin(omk)
    Yk = xk * np.sin(omk) + yk * np.cos(ik) * np.cos(omk)
    Zk = yk * np.sin(ik)
    # obliczenie błędu synchronizacji zegara satelity na podstawie wielomianu drugiego stopnia:
    delta_ts = observed_data[6] + observed_data[7] * (tow - observed_data[17]) + observed_data[8] * (tow - observed_data[17])**2
    # obliczenie poprawki relatywistycznej i dodanie jej do wartości błędu synchronizacji zegara satelity:
    c = 299792458.0
    dt_rel = -2 * np.sqrt(micro * a) / c**2 * observed_data[14] * np.sin(Ek)
    dt_s_rel = delta_ts + dt_rel
    return Xk, Yk, Zk, dt_s_rel

# Xk, Yk, Zk, dt_s_rel = sat_position(tow, observed_data[0])
# print(f"Xk: {Xk}, Yk: {Yk}, Zk: {Zk}, dt_s_rel: {dt_s_rel}")
# print(f"Xk: {Xk}")
# print(f"Yk: {Yk}")
# print(f"Zk: {Zk}")
#
# # Kontrola
# print(f"Kontrola: {np.allclose(rk, np.sqrt(Xk**2 + Yk**2 + Zk**2))}")
#
# # Obliczenie błędu synchronizacji zegara satelity na podstawie wielomianu drugiego stopnia:
# delta_ts = observed_data[:,6] + observed_data[:,7] * (tow - observed_data[:,17]) + observed_data[:,8] * (tow - observed_data[:,17])**2
# print(f"delta_ts: {delta_ts}")
#
# # Obliczenie poprawki relatywistycznej i dodanie jej do wartości błędu synchronizacji zegara satelity:
# c = 299792458.0
# dt_rel = -2 * np.sqrt(micro * a) / c**2 * observed_data[:,14] * np.sin(Ek)
# dt_s_rel = delta_ts + dt_rel
# print(f"dt_s_rel: {dt_s_rel}")
#
#
# print('Satelity: ', observed_satelites)
# print('Pozycje satelitów: ')
# print('Xk: ', Xk)
# print('Yk: ', Yk)
# print('Zk: ', Zk)
# print('Błąd synchronizacji zegara satelity: ', dt_s_rel)


