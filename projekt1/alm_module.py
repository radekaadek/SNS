# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:42:24 2023

@author: mgrzy
"""


import numpy as np

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
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5;
    return jd

# compare 

import datetime

def julday_stdlib(y, m, d, h=0):
    """
    Calculate Julian date using standard library datetime module.
    """
    dt = datetime.datetime(y, m, d, hour=h)
    julian_date = dt.toordinal() + 1721425.5 + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    return julian_date

def read_yuma(almanac_file):
    ''' 
    Reading and parsing YUMA asci format
    INPUT:
        Almanac: YUMA format 
    OUTPUT:
        almanac_data -  type list of list [strings value], number of lists is equal to number of satellite
                        one list contain satellites according to the order:         
                        ['SV ID', 'Health', 'Eccentricity', 'Time of Applicability(s)', 'Inclination(rad)', 
                        'Rate of Right Ascen(r/s)', 'SQRT(A)  (m 1/2)', 'Right Ascen at Week(rad)', 
                        'Argument of Perigee(rad)', 'Mean Anom(rad)', 'Af0(s)', 'Af1(s/s)', 'Week no']
        
    '''
    
    if almanac_file:
        alm = open(almanac_file)
        
        alm_lines = alm.readlines()
        all_sat = []
        for idx, value in enumerate(alm_lines):
            # print(idx, value)
            
            if value[0:3]=='ID:':
                one_sat_block = alm_lines[idx:idx+13]
                one_sat = []
                for line in one_sat_block:
                    data = line.split(':')
                    one_sat.append(float(data[1].strip()))
                all_sat.append(one_sat)
        alm.close()
        all_sat = np.array(all_sat)
        
        return (all_sat)


def blh2xyz(fi, lam, h):
    '''
    Parameters
    ----------
    fi : float
        latitude [rad].
    lam : float
        longitude [rad].
    h : float
        height [m].
    Returns
    -------
    xyz : numpy array
        x, y, z [m].
    '''
    a = 6378137.0
    e2 = 0.00669438002290
    N = a/np.sqrt(1-e2*np.sin(fi)**2)
    x = (N+h)*np.cos(fi)*np.cos(lam)
    y = (N+h)*np.cos(fi)*np.sin(lam)
    z = (N*(1-e2)+h)*np.sin(fi)
    xyz = np.array([x, y, z])
    return xyz

def read_alm(file):
    '''
    Parameters
    ----------
    file : .alm file
    Returns
    -------
    nav_data : 
    nav_data[0] - svprn
    nav_data[1] - health
    nav_data[2] - eccentricity
    nav_data[3] - SQRT_A (square root of major axis a [m**(1/2)])
    nav_data[4] - Omega (Longitude of ascending node at the beginning of week [deg])
    nav_data[5] - omega (Argument of perigee [deg])
    nav_data[6] - M0 (Mean anomally [deg])
    nav_data[7] - ToA (Time of Almanac [second of week])
    nav_data[8] - delta_i (offset to nominal inclination angle; i = 54 + delta_i [deg])
    nav_data[9] - Omega_dot (Rate of Right Ascension [deg/s * 1000])
    nav_data[10]- Satellite clock offset [ns]
    nav_data[11]- Satellite clock drift [ns/s]
    nav_data[12]- GPS week
   
    '''
    m = 0
    with open(file, "r") as f:
        block = []
        nav_data = []
        for s in f:
            # print(s)
            
            if m<13:
                m+=1
                block.append(s)
            else:
                block_array = np.genfromtxt(block,delimiter=10).T
                nav_data.extend(block_array)
                
                m = 0
                block = []
            
    nav_data = np.array(nav_data)        
    return nav_data

def get_prn_number(nav_data):
    prns = []
    for nav in nav_data:
        nsat = nav[0]
        if 0<nsat<=37:
            prn = int(nsat)
            prns.append(prn)
        elif 38<=nsat<=64:
            prn = 100 + int(nsat-37)
            prns.append(prn)
        elif 111<=nsat<=118:
            prn = 400 + int(nsat-110)
            prns.append(prn)
        elif 201<=nsat<=263:
            prn = 200 + int(nsat-200)
            prns.append(prn)    
        elif 264<=nsat<=310:
            prn = 300 + int(nsat-263)
            prns.append(prn)
        elif 311<=nsat:
            prn = 300 + int(nsat-328)
            prns.append(prn)           
        else: 
            prn = 500 + int(nsat)
            prns.append(prn)
    return prns

def get_alm_data(file):
    nav_data = read_alm(file)
    prns = get_prn_number(nav_data)
    nav_data[:,0] = prns
    return nav_data

def create_prn_alm(nav_data):
    prns = []
    for nav in nav_data:
        nsat = nav[0]
        if 0<nsat<=37:
            prn = 'G'+str(int(nsat)).zfill(2)
            prns.append(prn)
        elif 38<=nsat<=64:
            prn = 'R'+str(int(nsat-37)).zfill(2)
            prns.append(prn)
        elif 111<=nsat<=118:
            prn = 'Q'+str(int(nsat-110)).zfill(2)
            prns.append(prn)
        elif 201<=nsat<=263:
            prn = 'E'+str(int(nsat-200)).zfill(2)
            prns.append(prn)    
        elif 264<=nsat<=310:
            prn = 'C'+str(int(nsat-263)).zfill(2)
            prns.append(prn)
        elif 311<=nsat:
            prn = 'C'+str(int(nsat-328)).zfill(2)
            prns.append(prn)           
        else: 
            prn = 'S'+str(int(nsat)).zfill(2)
            prns.append(prn)
    return prns

def get_alm_data_str(alm_file):
    alm_data = read_alm(alm_file)
    prns = create_prn_alm(alm_data) 
    return alm_data, prns    

# 5 Wyznaczenie pozycji satelity - algorytm
# Zestawienie stałych:
# μ = 3.986005 · 1014[ m3
# s2 ]
# ωE = 7.2921151467 · 10−5[ rad
# s ]
# t – czas (w sekundach GPS), na który chcemy policzyć współrzędne satelity
# 1. Czas jaki upłynął od epoki wyznaczenia almanachu (należy uwzględnić również tydzień GPS):
# tk = t − toa (1)
# 2. Obliczenie dużej półosi orbity:
# a = (√a)2 (2)
# 3. Wyznaczenie średniej prędkości kątowej n, znanej jako ruch średni (ang. mean motion) na podstawie
# III prawa Kepplera:
# n =
# √ μ
# a3 (3)
# 4. Poprawiona anomalia średnia na epokę tk:
# Mk = M0 + n · tk (4)
# 5. Wyznaczenie anomalii mimośrodowej (Równanie Kepplera):
# Ek = Mk + e sin(Ek) (5)
# Równanie to należy rozwiązać w sposób iteracyjny: (dla i=1,2,3...)
# E1 = Mk (6a)
# Ei+1 = Mk + e sin(Ei) (6b)
# Kryterium zakończenia obliczeń iteracyjnych wymaga spełnienia warunku: |Ei − Ei−1| < 10−12
# 6. Wyznaczenie anomalii prawdziwej:
# vk = arctan
# ( √1 − e2 sin(Ek)
# cos(Ek) − e
# )
# (7)
# * skorzystać z funkcji arctan2 (odpowiednie ćwiartki dla arcus tangens):
# 7. Wyznaczenie argumentu szerokości:
# Φk = vk + ω (8)
# 8. Wyznaczenie promienia orbity:
# rk = a (1 − e cos(Ek)) (9)
# 9. Wyznaczenie pozycji satelity w układzie orbity:
# xk = rk · cos(Φk) (10a)
# yk = rk · sin(Φk) (10b)
# 10. Poprawiona długość węzła wstępującego:
# Ωk = Ω0 +
# ( ˙Ω − ωE
# )
# tk − ωE toa (11)
# 11. Wyznaczenie pozycji satelity w układzie geocentrycznym ECEF:
# Xk = xk cos(Ωk) − yk cos(i) sin(Ωk) (12a)
# Yk = xk sin(Ωk) + yk cos(i) cos(Ωk) (12b)
# Zk = yk sin(i) (12c)

# Wyjaśnienie symboli
# Parametr Symbol Opis
# Id prn numer PRN satelity
# Health stan pracy satelity: 000 = usable
# Eccentricity e ekscentr (mimośród) orbity
# Time of Almanac (Aplicabi-
# lity)
# toa czas odniesienia almanachu, znacznik czasu (sekunda tygo-
# dnia GPS)
# Orbital Inclination i Kąt nachylenia płaszczyzny orbity do płaszczyzny równika
# Rate of Right Ascension ˙Ω Tempo zmiany rektascenzji węzła wstępującego
# SQRT(A) Square Root of
# Semi-Major Axis
# √a Pierwiastek z dużej półosi orbity, zdefiniowanej jako odle-
# głość od środka orbity do punktu perygeum lub apogeum.
# Right Ascension of Ascen-
# ding Node (Longitude)
# Ω0 Długość węzła wstępującego orbity na początek tygodnia
# GPS
# Argument of Perigee ω argument perygeum – Pomiar kątowy wzdłuż ścieżki orbi-
# talnej mierzony od węzła wstępującego do punktu perygeum
# (mierzony w kierunku ruchu SV).
# Mean Anomaly M0 anomalia średnia, w perygeum = 0
# Af0 αf0 opóźnienie zegara satelity
# Af1 αf1 dryft zegara satelity
# GPS Week numer tygodnia satelity, w przypadku formatu Trimble, li-
# czony od początku skali czasu GPS (6.01.1980), w przypadku
# formatu YUMA w przedziale 0000-1023 (licząc od 7.04.2019)

# 1 Cel zadania
# Opracowanie aplikacji, której celem jest przedstawienie wizualizacji parametrów związanych z
# planowaniem pomiarów GNSS. Aplikacja powinna umożliwiać wizualizacje dla całej, dowolnej
# doby (w zależności od daty wybranego almanachu). Aplikacja powinna również umożliwiać
# wybór krótszego fragmentu doby.
# 2 Schemat wykonywania obliczeń
# 2.1 Ustawienia wejściowe
# • wprowadzenie daty obliczeń – początek oraz koniec,
# • wybór almanachu danych,
# • wprowadzenie dowolnych współrzędnych miejsca obserwacji (współrzędne φ, λ, h),
# • wprowadzenie maski obserwacji,
# • zdefiniowanie interwału dla obliczeń (np. 10 minut – 10·60s),
# • ustawienia dodatkowe: np. wybór lub odrzucenie poszczególnych satelitów/systemów
# GNSS.
# 2.2 Obliczenia wstępne
# • odczytanie danych z almanachu:
# * funkcja read_alm – wynikiem funkcji jest tablica nav_data z efemerydami, w której
# każdy wiersz odpowiada osobnemu satelicie (opis i struktura danych w pliku z al-
# gorytmem obliczeń współrzędnych satelity), funkcje create_prn_alm/get_alm_data
# zwracają dodatkowo 3-znakowy identyfikator danego satelity, gdzie pierwszy znak jest
# to identyfikator danego systemu GNSS, a dwa pozostałe to numer PRN.
# * Uwaga! W danych może brakować niektórych satelitów, więc nie należy korzystać z
# numeru wiersza jako identyfikatora numeru satelity – numer satelity znajduje się w
# kolumnie 0 tablicy z efemerydami,
# • przeliczenie podanej daty obliczeń do skali czasu GPS – numer tygodnia i sekunda,
# • przeliczenie współrzędnych (φ, λ, h) do współrzędnych (X, Y, Z).
# Kolejność obliczeń
# Podstawowym elementem zadania jest obliczenie współrzędnych satelitów. Obliczenia te
# można wykonać w dwojaki sposób:
# 1. obliczenie współrzędnych wszystkich satelitów dla danej epoki i kolejno przejście do
# kolejnej epoki i powtórzenie obliczeń dla pozostałych epok,
# 2. obliczenie współrzędnych dla danego satelity dla wszystkich epok, a następnie
# powtórzenie obliczeń dla kolejnych satelitów.
# Preferowaną opcją jest opcja 1. (zdecydowanie)
# 2
# 2.3 Schemat obliczeń dla pojedynczej epoki
# Dla pierwszego satelity
# 1. pobranie danych efemerydalnych z tablicy nav dla danego satelity (jeden wiersz tablicy),
# 2. obliczenie współrzędnych satelity,
# 3. obliczenie elewacji i azymutu satelity, zgodnie ze schematem z prezentacji,
# 4. sprawdzenie, czy satelita znajduje się powyżej maski elewacji, jeśli tak:
# (a) wyznaczenie odpowiednich elementów wiersza macierzy A, odpowiadającego analizo-
# wanemu satelicie,
# (b) oznaczenie satelity jako satelita widoczny nad horyzontem,
# 5. powtórzenie obliczeń dla kolejnego satelity z tablicy nav aż do ostatniego satelity.
# • Po ukończeniu obliczeń dla wszystkich satelitów w pojedynczej epoce, należy złożyć wszyst-
# kie obliczone wiersze macierzy A w jedną macierz i wykonać obliczenia współczynników
# DOP;
# • Należy skompletować wszystkie wyznaczone parametry (elewacje, azymuty, DOP, liczba
# widocznych satelitów, ewentualnie współrzędne XYZ satelitów), co niezbędne będzie do
# wykonania wizualizacji;
# • Po ukończeniu obliczeń dla danej epoki, należy powtórzyć schemat dla kolejnej epoki, aż
# do ostatniej epoki.
import wget
import os

# constans
c1 = 3.986005*10**14
c2 = 7.2921151467*10**-5

def download_alm(url, filename):
    if os.path.exists(filename):
        os.remove(filename)
    wget.download(url, filename)
    return filename

f = download_alm('ftp://ftp.trimble.com/pub/eph/Almanac.alm', 'Almanac.alm')

nav, prn = get_alm_data_str(f)

satelity = nav[:, 0]<400
nav = nav[satelity]
prn = np.array(prn)[satelity] # S, E, C, G, Q, R

def get_gps_time(y,m,d,h=0, mnt=0,s=0):
    days = julday(y,m,d) - julday(1980,1,6)
    week = days//7
    day =  days%7
    sec_of_week = day*86400 + h*3600 + mnt*60 + s
    return week, sec_of_week


def get_satellite_position(nav, y, m, d, h=0, mnt=0, s=0):
    week, sec_of_week = get_gps_time(y,m,d,h,mnt,s)

# 2. Obliczenie dużej półosi orbity:
    a = (nav[:,3])**2
# 3. Wyznaczenie średniej prędkości kątowej n, znanej jako ruch średni (ang. mean motion) na podstawie
# III prawa Kepplera:
    n = np.sqrt(c1/a**3)
# 4. Poprawiona anomalia średnia na epokę tk:
    Mk = nav[:,6] + n*sec_of_week
# 5. Wyznaczenie anomalii mimośrodowej (Równanie Kepplera):
    E = Mk
    for i in range(100):
        E = Mk + nav[:,2]*np.sin(E)
        if np.abs(E[i] - E[i-1]) < 10**-12:
            break
# 6. Wyznaczenie anomalii prawdziwej:
    v = np.arctan(np.sqrt(1 - nav[:,2]**2)*np.sin(E)/(np.cos(E) - nav[:,2]))
# 7. Wyznaczenie argumentu szerokości:
    phi = v + nav[:,5]
# 8. Wyznaczenie promienia orbity:
    r = a*(1 - nav[:,2]*np.cos(E))
# 9. Wyznaczenie pozycji satelity w układzie orbity:
    x = r*np.cos(phi)
    y = r*np.sin(phi)
# 10. Poprawiona długość węzła wstępującego:
    Omega = nav[:,4] + (nav[:,9] - c2)*sec_of_week - c2*nav[:,3]
# 11. Wyznaczenie pozycji satelity w układzie geocentrycznym ECEF:
    X = x*np.cos(Omega) - y*np.cos(nav[:,8])*np.sin(Omega)
    Y = x*np.sin(Omega) + y*np.cos(nav[:,8])*np.cos(Omega)
    Z = y*np.sin(nav[:,8])
    return X, Y, Z

# 3 Wizualizacja
# 3.1 Wizualizacja dla całej doby
# Wizualizacja dla całej doby powinna zawierać:
# • wykres widoczności satelitów w funkcji czasu (dla każdego satelity osobno),
# • wykres DOP w funkcji czasu,
# • wykres liczby widocznych satelitów w funkcji czasu,
# • wykres współrzędnych XYZ satelitów w funkcji czasu,
# • wykres współrzędnych XYZ odbiornika w funkcji czasu,
# • wykres błędów pozycjonowania w funkcji czasu.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# find satelites in the GPS system
idxs = [i for i, prn in enumerate(prn) if prn[0] == 'G']

hours_in_day = 24
minutes_in_hour = 60
satelite = nav
positions = []
for idx in idxs:
    for i in range(minutes_in_hour):
        X, Y, Z = get_satellite_position(satelite, 2023, 4, 15, 0, i)
        positions.append([X[idx], Y[idx], Z[idx]])

positions = np.array(positions)

# show a 3d plot the positions of the first satelite during one day every hour
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('GPS satalites positions')

# draw the earth as an ellipsoid
a = 6378137.0
b = 6356752.3142
phi = np.linspace(0, 2*np.pi, 100)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
X = a*np.sin(theta)*np.cos(phi)
Y = a*np.sin(theta)*np.sin(phi)
Z = b*np.cos(theta)
ax.plot_surface(X, Y, Z, color='b', alpha=0.1)

ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
