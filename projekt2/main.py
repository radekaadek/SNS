from readrnx_studenci import readrnxnav, readrnxobs, date2tow, hirvonen
from roz import satpos, satelite_position
from top import topo
from jon import jono
import numpy as np
import argparse

# parse mask, tropospheric boolean, ionospheric boolean, reference coordinates, observation file, navigation file
parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=int, default=10, help='Elevation mask')
parser.add_argument('--tropo', type=bool, default=True, help='Tropospheric correction')
parser.add_argument('--iono', type=bool, default=True, help='Ionospheric correction')
parser.add_argument('--obs', type=str, default='JOZ200POL_R_20240650000_01D_30S_MO.rnx', help='Observation file')
parser.add_argument('--nav', type=str, default='BRDC00WRD_R_20240650000_01D_GN.rnx', help='Navigation file')
parser.add_argument('--x0', type=float, default=3660000., help='X coordinate of receiver')
parser.add_argument('--y0', type=float, default=1400000., help='Y coordinate of receiver')
parser.add_argument('--z0', type=float, default=5000000., help='Z coordinate of receiver')

args = parser.parse_args()

# elevation mask
mask = args.mask
tropo = args.tropo
iono = args.iono
xr0 = [args.x0, args.y0, args.z0]
# observation file path
obs_file = args.obs
nav_file = args.nav


# cieżka do pliku nawigacyjnego
# nav_file = 'BRDC00WRD_R_20240650000_01D_GN.rnx'
# cieżka do pliku obserwacyjnego
# obs_file = 'JOZ200POL_R_20240650000_01D_30S_MO.rnx'

# zdefiniowanie czasu obserwacji: daty początkowej i końcowej
# dla pierwszej epoki z pliku będzie to:
time_start =  [2024, 3, 5, 0, 0, 0]  
time_end =    [2024, 3, 5, 23, 59, 59] 

# odczytanie danych z pliku obserwacyjnego
obs, iobs = readrnxobs(obs_file, time_start, time_end, 'G')
# odczytanie danych z pliku nawigacyjnego:
nav, inav = readrnxnav(nav_file)

healthy = nav[:, 30] == 0
nav = nav[healthy]
inav = inav[healthy]


def Rneu(phi, lamb):
    '''
    Parameters
    ----------
    phi : float
        latitude [rad].
    lamb : float
        longitude [rad].
    Returns
    -------
    R : numpy array
        rotation matrix.
    '''
    R = np.array([[-np.sin(phi)*np.cos(lamb), -np.sin(lamb), np.cos(phi)*np.cos(lamb)],
                    [-np.sin(phi)*np.sin(lamb), np.cos(lamb), np.cos(phi)*np.sin(lamb)],
                    [np.cos(phi), 0, np.sin(phi)]])
    return R

#%%
"""
zdefiniowanie współrzędnych przybliżonych odbiornika - mogą to być współrzędne z nagłówka 
pliku obserwacyjnego, skopiowane "z palca" lub pobierane automatycznie z treci nagłówka pliku Rinex
"""
# xr0 = [3660000.,  1400000.,  5000000.]

"""
Przeliczenie daty początkowej i końcowej do sekund tygodnia GPS - niezbędne w celu
poprawnej definicji pętli związanej z czasem obserwacji w ciągu całej doby
"""
week, tow = date2tow(time_start)[0:2]
week_end, tow_end = date2tow(time_end)[0:2]
# tow = 213300
# week = 2304
#%% Obliczenia


# t = 213300
# idx_t = np.where(iobs[:, -1] == t)[0]
# Pobs = obs[idx_t, 0]
# print(f"{Pobs=}\n")
# satelity = iobs[idx_t, 0]
omega = 7.2921151467*10**-5

tau = 0.07
dtr = 0
c = 299792458
observed_satelites = [25, 31, 32, 29, 28, 24, 20, 11, 12, 6]

observed_data = []
# for sat in observed_satelites:
#     satelite_index = inav == sat
#     nav_sat = nav[satelite_index]
#     dt = np.abs(nav_sat[:,17] - tow)
#     index = np.argmin(dt)
#     nav_sat = nav_sat[index]
#     observed_data.append(nav_sat)
# # sats = satelity
# observed_data = np.array(observed_data)

# sats = iobs[idx_t, 0]
# Pobs = obs[idx_t, 0]
iters = 5
final_xyz = []
for time in range(tow, tow_end+1, 30):
    # print(f"{time=}\n")
    idx_t = np.where(iobs[:, -1] == time)[0]
    Pobs = obs[idx_t, 0]
    sats = iobs[idx_t, 0]
    # print(f"{sats=}\n")
    observed_satelites = sats
    observed_data = []
    for sat in observed_satelites:
        satelite_index = inav == sat
        nav_sat = nav[satelite_index]
        dt = np.abs(nav_sat[:,17] - time)
        index = np.argmin(dt)
        nav_sat = nav_sat[index]
        observed_data.append(nav_sat)
    observed_data = np.array(observed_data)
    rho = np.array([0]*len(sats))
    xyz = np.array(xr0) if time == tow else np.array(final_xyz[-1])
    el = np.array([np.pi/2]*len(sats))
    # dtr = 0
    for j in range(iters):
        # print(f"-------------------{j}-------------------\n")
        A = []
        ys = []
        for i, sat in enumerate(sats):
            # 1. Obliczenie współrzędnych satelity (współrzędnych xyz oraz błędu zegara satelity, z uwzględnieniem
            # poprawki relatywistycznej, δts ) na czas emisji sygnału ts :
            # ts = tr + δtr − τ(1)
            # xs0 , y0s , z0s , δts = satpos(tr , nav)(2)
            # *uwaga: czas propagacji sygnału τ w pierwszej iteracji jest wartością znaną, przybliżoną, taką samą
            # dla każdego satelity. W kolejnych iteracjach, wartość ta będzie zależeć od obliczonej w poprzedniej
            # iteracji odległości geometrycznej: τ = ρs0 /c i będzie różna dla każdego satelity!
            curr_tau = 0.07 if j == 0 else rho[i]/c
            # print(f"{curr_tau=}\n")
            ts = time - curr_tau + dtr
            # print(f"{ts=}\n")
            # xyzdts = satelite_position(observed_data[i], ts)
            xyzdts = satpos(ts, observed_data[i])
            xs = xyzdts[0:3]
            # print(f"{xs=}\n")
            dts = xyzdts[3]
            # print(f"{dts=}\n")
            # 2. Transformacja współrzędnych satelity do chwilowego układu współrzędnych, na moment odbioru
            # sygnału:
            xs_rot = np.array([[np.cos(omega*curr_tau), np.sin(omega*curr_tau), 0],
                            [-np.sin(omega*curr_tau), np.cos(omega*curr_tau), 0],
                            [0, 0, 1]]).dot(xs)
            # print(f"{xs_rot=}\n")
            # 3. Obliczenie odległości geometrycznej między satelitą a odbiornikiem:
            # p
            # ρs0 = (xs − x0 )2 + (y s − y0 )2 + (z s − z0 )2
            # s
            # s
            # (4)
            # s
            # * uwaga: za współrzędne satelity x , y , z przyjmujemy współrzędne “obrócone” do chwilowego
            # układu współrzędnych → wynik równania (3);
            # *uwaga: z obliczonych odległości geometrycznych (lub czasu propagacji sygnału τ = ρs0 /c) dla
            # danego satelity będziemy musieli skorzystać w kolejnej iteracji, dlatego trzeba te wartości zapisać
            # do jakiejś zmiennej.
            p0 = np.sqrt((xs_rot[0] - xyz[0])**2 + (xs_rot[1] - xyz[1])**2 + (xs_rot[2] - xyz[2])**2)
            # print(f"{p0=}\n")
            # 4. Wyznaczenie elewacji (i azymutu, niezbędnego do wyznaczenia opóźnienia jonosferycznego) satelity
            # oraz odrzucenie satelitów znajdujących się poniżej maski;
            # *uwaga! Współrzędne przybliżone odbiornika x0 w pierwszej iteracji mogą być bardzo odległe od
            # rzeczywistej pozycji odbiornika. Dlatego w pierwszej iteracji, za wartość elewacji można przyjąć np.
            # 90◦ dla każdego satelity. W kolejnych iteracjach należy obliczyć odpowiednie kierunki do satelitów,
            # najpierw przeliczając współrzędne odbiornika x0 do współrzędnych krzywoliniowych, wykorzystując
            # algorytm Hirvonena.

            sat_odb = np.array([xs_rot[0] - xyz[0], xs_rot[1] - xyz[1], xs_rot[2] - xyz[2]])
            # print(f"{sat_odb=}\n")

            b,l,h = hirvonen(xyz[0], xyz[1], xyz[2])

            # x_neu = np.dot(Rneu(b, l), xs_rot)
            # az = np.rad2deg(np.arctan2(x_neu[1], x_neu[0]))
            # az = az if az>0 else az + 360
            # el = np.rad2deg(np.arcsin(x_neu[2]/np.linalg.norm(x_neu)))
            R = Rneu(b, l)
            neu = R.T.dot(sat_odb)
            az = np.rad2deg(np.arctan2(neu[1], neu[0]))
            az = az if az>0 else az + 360
            el = np.rad2deg(np.arcsin(neu[2]/np.linalg.norm(neu)))
            # print(f"{el=}, {az=}\n")

            # 5. Wyznaczenie opóźnienia troposferycznego δTrs i jonosferycznego δIrs dla danego satelity (wzory w
            # odpowiednich prezentacjach).
            rho[i] = p0
            if el>mask:
                a_el = np.array([-(xs_rot[0] - xyz[0])/p0, -(xs_rot[1] - xyz[1])/p0, -(xs_rot[2] - xyz[2])/p0, 1])
                # print(f"{a_el=}\n")
                A.append(a_el)
                if j == 0:
                    dT = 0
                    dJ = 0
                else:
                    if tropo:
                        dT = topo(h, el)
                    else:
                        dT = 0
                    if iono:
                        dJ = jono(time, b, l, el, az)
                    else:
                        dJ = 0
                # print(f"{dT=}, {dJ=}\n")
                # cdts = c * dts
                Pcalc = rho[i] - c*dts + c*dtr + dT + dJ
                # print(f"{Pcalc=}\n")
                # print(f"{Pobs[i]=}\n")
                y = Pobs[i] - Pcalc
                # print(f"{y=}\n")
                ys.append(y)
        A = np.array(A)
        # print(f"{A=}\n")
        ys = np.array(ys)
        # print(f"{ys=}\n")
        x = np.linalg.inv(np.dot(A.T, A)).dot(np.dot(A.T, ys))
        # print(f"{x=}\n")
        xyz[0] += x[0]
        xyz[1] += x[1]
        xyz[2] += x[2]
        dtr += x[3]/c
        # print(f"{dtr=}\n")
    # print(f"{xyz=}\n")
    final_xyz.append(xyz)

xr_true = [3664880.9100,  1409190.3850,  5009618.2850]
import matplotlib.pyplot as plt
final_xyz = np.array(final_xyz)
# show 3 axis with xerror, yerror, zerror and RMS
plot, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(final_xyz[:, 0] - xr_true[0], label='X error')
ax[0].plot([0, len(final_xyz)], [0, 0], 'r--')
RMS = np.sqrt(np.mean((final_xyz[:, 0] - xr_true[0])**2))
RMS = round(RMS, 2)
stdev_x = np.std(final_xyz[:, 0] - xr_true[0])
ax[0].set_title(f'X error, RMS = {RMS}m, Standard deviation = {round(stdev_x, 2)}m')
# add RMS to the legend
ax[0].legend()
ax[1].plot(final_xyz[:, 1] - xr_true[1], label='Y error')
ax[1].plot([0, len(final_xyz)], [0, 0], 'r--')
RMS = np.sqrt(np.mean((final_xyz[:, 1] - xr_true[1])**2))
RMS = round(RMS, 2)
stdev_y = np.std(final_xyz[:, 1] - xr_true[1])
ax[1].set_title(f'Y error, RMS = {RMS}m, Standard deviation = {round(stdev_y, 2)}m')
ax[1].legend()
ax[2].plot(final_xyz[:, 2] - xr_true[2], label='Z error')
ax[2].plot([0, len(final_xyz)], [0, 0], 'r--')
RMS = np.sqrt(np.mean((final_xyz[:, 2] - xr_true[2])**2))
RMS = round(RMS, 2)
stdev_z = np.std(final_xyz[:, 2] - xr_true[2])
ax[2].set_title(f'Z error, RMS = {RMS}m, Standard deviation = {round(stdev_z, 2)}m')
ax[2].legend()

plt.show()

