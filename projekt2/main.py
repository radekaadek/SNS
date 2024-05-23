from readrnx_studenci import readrnxnav, readrnxobs, date2tow, hirvonen
from roz import sat_position, satelite_position
from top import topo
from jon import jono
import numpy as np

# cieżka do pliku nawigacyjnego
nav_file = 'BRDC00WRD_R_20240650000_01D_GN.rnx'
# cieżka do pliku obserwacyjnego
obs_file = 'JOZ200POL_R_20240650000_01D_30S_MO.rnx'

# zdefiniowanie czasu obserwacji: daty początkowej i końcowej
# dla pierwszej epoki z pliku będzie to:
time_start =  [2024, 3, 5, 0, 0, 0]  
time_end =    [2024, 3, 5, 23, 59, 59] 

# odczytanie danych z pliku obserwacyjnego
obs, iobs = readrnxobs(obs_file, time_start, time_end, 'G')
# odczytanie danych z pliku nawigacyjnego:
nav, inav = readrnxnav(nav_file)

zdrowe = nav[:, 30] == 0
nav = nav[zdrowe]
inav = inav[zdrowe]


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
xr0 = [3660000.,  1400000.,  5000000.]

"""
Wprowadzenie ustawień, takich jak maska obserwacji, czy typ poprawki troposferycznej
"""
el_mask = 10 # elevation mask/cut off in degrees

"""
Przeliczenie daty początkowej i końcowej do sekund tygodnia GPS - niezbędne w celu
poprawnej definicji pętli związanej z czasem obserwacji w ciągu całej doby
"""
week, tow = date2tow(time_start)[0:2]
week_end, tow_end = date2tow(time_end)[0:2]
tow = 213300
week = 2304
#%% Obliczenia


t = 213300
idx_t = np.where(iobs[:, -1] == t)[0]
Pobs = obs[idx_t, 0]
print(f"{Pobs=}\n")
satelity = iobs[idx_t, 0]
omega = 7.2921151467*10**-5

tau = 0.07
dtr = 0
c = 299792458
observed_satelites = [25, 31, 32, 29, 28, 24, 20, 11, 12, 6]
mask_rad = np.deg2rad(el_mask)

observed_data = []
for sat in observed_satelites:
    satelite_index = inav == sat
    nav_sat = nav[satelite_index]
    dt = np.abs(nav_sat[:,17] - tow)
    index = np.argmin(dt)
    nav_sat = nav_sat[index]
    observed_data.append(nav_sat)
sats = satelity
observed_data = np.array(observed_data)

# sats = iobs[idx_t, 0]
# Pobs = obs[idx_t, 0]
mask = 10
iters = 2
final_xyz = []
dtr = 0
for time in range(tow, tow_end+1, 30):
    print(f"{time=}\n")
    # idx_t = np.where(iobs[:, -1] == time)[0]
    # Pobs = obs[idx_t, 0]
    # sats = iobs[idx_t, 0]
    # observed_satelites = np.unique(sats)
    # observed_data = []
    # for sat in observed_satelites:
    #     satelite_index = inav == sat
    #     nav_sat = nav[satelite_index]
    #     dt = np.abs(nav_sat[:,17] - tow)
    #     index = np.argmin(dt)
    #     nav_sat = nav_sat[index]
    #     observed_data.append(nav_sat)
    # observed_data = np.array(observed_data)
    # print(f"Sats: {sats}\n")
    # tau = 0.07
    rho = np.array([0]*len(sats))
    xyz = np.array(xr0)
    el = np.array([np.pi/2]*len(sats))
    for j in range(iters):
        print(f"-------------------{j}-------------------\n")
        A = []
        ys = []
        seen_cnt = 0
        dts = 0
        for i, sat in enumerate(sats):
            # 1. Obliczenie współrzędnych satelity (współrzędnych xyz oraz błędu zegara satelity, z uwzględnieniem
            # poprawki relatywistycznej, δts ) na czas emisji sygnału ts :
            # ts = tr + δtr − τ(1)
            # xs0 , y0s , z0s , δts = satpos(tr , nav)(2)
            # *uwaga: czas propagacji sygnału τ w pierwszej iteracji jest wartością znaną, przybliżoną, taką samą
            # dla każdego satelity. W kolejnych iteracjach, wartość ta będzie zależeć od obliczonej w poprzedniej
            # iteracji odległości geometrycznej: τ = ρs0 /c i będzie różna dla każdego satelity!
            curr_tau = 0.07 if j == 0 else rho[i]/c
            print(f"{curr_tau=}\n")
            ts = time - curr_tau + dtr
            print(f"{ts=}\n")
            xyzdts = satelite_position(observed_data[i], ts)
            xs = xyzdts[0:3]
            print(f"{xs=}\n")
            dts = xyzdts[3]
            print(f"{dts=}\n")
            # 2. Transformacja współrzędnych satelity do chwilowego układu współrzędnych, na moment odbioru
            # sygnału:
            xs_rot = np.array([[np.cos(omega*curr_tau), np.sin(omega*curr_tau), 0],
                            [-np.sin(omega*curr_tau), np.cos(omega*curr_tau), 0],
                            [0, 0, 1]]).dot(xs)
            print(f"{xs_rot=}\n")
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
            print(f"{p0=}\n")
            # 4. Wyznaczenie elewacji (i azymutu, niezbędnego do wyznaczenia opóźnienia jonosferycznego) satelity
            # oraz odrzucenie satelitów znajdujących się poniżej maski;
            # *uwaga! Współrzędne przybliżone odbiornika x0 w pierwszej iteracji mogą być bardzo odległe od
            # rzeczywistej pozycji odbiornika. Dlatego w pierwszej iteracji, za wartość elewacji można przyjąć np.
            # 90◦ dla każdego satelity. W kolejnych iteracjach należy obliczyć odpowiednie kierunki do satelitów,
            # najpierw przeliczając współrzędne odbiornika x0 do współrzędnych krzywoliniowych, wykorzystując
            # algorytm Hirvonena.

            sat_odb = np.array([xs_rot[0] - xyz[0], xs_rot[1] - xyz[1], xs_rot[2] - xyz[2]])
            print(f"{sat_odb=}\n")

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
            print(f"{az=}, {el=}\n")

            # 5. Wyznaczenie opóźnienia troposferycznego δTrs i jonosferycznego δIrs dla danego satelity (wzory w
            # odpowiednich prezentacjach).
            rho[i] = p0
            if el>mask:
                seen_cnt += 1
                a_el = np.array([-(xs_rot[0] - xyz[0])/p0, -(xs_rot[1] - xyz[1])/p0, -(xs_rot[2] - xyz[2])/p0, 1])
                print(f"{a_el=}\n")
                A.append(a_el)
                if j == 0:
                    dT = 0
                    dJ = 0
                else:
                    dT = topo(h, el)
                    dJ = jono(tow, b, l, el, az)
                cdts = c * dts
                Pcalc = np.float64(rho[i] - cdts + c*dtr) # + dT + dJ)
                print(f"{Pcalc=}\n")
                Pobs[i] = rho[i]+c*dts-c*dtr # + dT + dJ
                print(f"{Pobs[i]=}\n")
                y = Pobs[i] - Pcalc
                ys.append(y)
        print(f"{seen_cnt=}\n")
        if seen_cnt < 4:
            break
        A = np.array(A)
        print(f"{A=}\n")
        ys = np.array(ys)
        print(f"{ys=}\n")
        x = np.linalg.inv(np.dot(A.T, A)).dot(np.dot(A.T, ys))
        print(f"{x=}\n")
        xyz[0] += x[0]
        xyz[1] += x[1]
        xyz[2] += x[2]
        dtr += x[3]/c
        print(f"{dtr=}\n")
    exit()
    final_xyz.append(xyz)

import matplotlib.pyplot as plt
final_xyz = np.array(final_xyz)
dists = []
for xyz in final_xyz:
    dist = np.sqrt((xyz[0] - xr0[0])**2 + (xyz[1] - xr0[1])**2 + (xyz[2] - xr0[2])**2)
    dists.append(dist)
plt.plot(dists)
plt.show()

"""
Otwieramy dużą pętlę
for t in range(tow, tow_end+1, dt): gdzie dt równe 30
"""
"""
Wewnątrz tej pętli, zajmujemy się obserwacjami wyłącznie dla jednej epoki (epoka t), zatem:
    1. Wybieramy obserwacje dla danej epoki, na podstawie tablicy iobs oraz naszej epoki t
    czyli, wybieramy te obserwacje z tablicy obs, dla których w tablicy iobs ostatnia kolumna 
    jest równa t - przypisujemy do zmiennej np. Pobs
    2. wybieramy satelity, obserwowane w danej epoce, na podstawie tablicy iobs - na podstawie 
    naszego t - przypisujemy do zmiennej np. sats
    3. Definiujemy wartości przybliżone błąd zegara odbiornika
    dtr = 0 oraz czasu propagacji sygnału tau = 0.07
    4. Najprawdopodobniej przyda się definicja pustych wektorów, np. zawierających 
    odległosci geometryczne (wartoci przybliżone na podstawie tau)

        
    Przechodzimy do iteracyjnego obliczenia współrzędnych odbiornika - w pierwszych testach naszego programu, zróbmy obliczenia nieiteracyjnie, 
    ale pamiętajmy o tym, że będzie trzeba przygotować kod do działania w pętli:
        
        Po weryfikacji działania programu, można zamienić pętlę for na pętle while, dopisując
        warunek zbieżnoci kolejnych współrzędnych - skróci nam to czas obliczeń, ponieważ 
        najczęściej wystarcza mniej iteracji niż 5
        
        for i in range(5):
            Wykonujemy kolejne obliczenia, niezależnie dla kolejnych satelitów, obserwowanych
            w danej epoce, czyli przechodzimy do pętli:
                for sat in sats: (przyda nam się tutaj również indeks satelity, np. for i, sat in enumerate(sats):)
                    Obliczamy czas emisji sygnału:
                        ts = t - tau + dtr
                    Kolejne kroki, znane z poprzedniego ćwiczenia:
                    wyznaczamy współrzędne satelity xs (oraz błąd zegara satelity dts) na czas ts (UWAGA, w kolejnych iteracjach
                    czas ts będzie się zmieniał i aktualizował, neizależnie dla każdego satelity!!!)
                    
                    Odległosć geometryczna:
                        1. rotacja do układu chwilowego - otrzymujemy xs_rot
                        2. Na podstawie xs_rot obliczamy odległosć geometryczną rho
                        
                    Obliczamy elewację i azymut
                    Macierz Rneu definiujemy na podstawie x0, przeliczonego do współrzędnych
                    phi, lambda, algorytmem Hirvonena
                    
                    Odrzucamy satelity znajdujące się poniżej maski
                    
                        Obliczamy poprawki atmosferyczne - dopiero wówczas, kiedy działać będzie nam program bez uwzględniania poprawek:
                            trop oraz iono
                    
                    Wyznaczamy pseudoodległosć przybliżoną (obliczoną), jako:
                        Pcalc = rho - c*dts + dtr + trop + iono
                        
                    Wyznaczamy kolejne elementy wektora wyrazów wolnych y, jako:
                        y = Pobs - Pcalc
                        
                    Budujemy kolejne wiersze macierzy A:
                
                Kończymy pętle dla kolejnych satelitów
                
                1. Łączymy ze sobą elementy wektora wyrazów wolych w jeden wektor
                2. Łączymy ze sobą kolejnę wiersze macierz współczynników A
                3. Rozwiązujemy układ równań, metodą najmniejszych kwadratów
                
                               
                Aktualizujemy wartosci przybliżone o odpowiednie elementy wektora x
                xr[0] = x0[0] + x[0]
                xr[1] = x0[1] + x[1]
                xr[2] = x0[2] + x[2]
                dtr = dtr + x[3]/c 
                
                Tak obliczone wartoci xr oraz dtr stanowią wartoci wejsciowe do kolejnej iteracji, itd 
                do skończenia piątej iteracji lub spełnienia warunku zbieżnoci współrzędncyh
            
            
            Po skończeniu 5. iteracji, zbieramy obliczone współrzędne xr - warto zebrać również
            liczby obserwowanych satelitów, obliczone wartoci współczynników DOP (przynajmniej PDOP)
            
"""


"""
Wyznaczenie pozycji użytkownika systemu GNSS na
podstawie obserwacji kodowych – model pozycjonowania
Single Point Positioning
Systemy Nawigacji Satelitarnej
Maciej Grzymała
maciej.grzymala@pw.edu.pl
Wydział Geodezji i Kartografii, Politechnika Warszawska
Warszawa, 2024
1 Cel ćwiczenia
Rozwiązanie pozycji odbiornika, na podstawie danych obserwacji kodowych, z wykorzystaniem modelu
pozycjonowania Single Point Positioning (SPP).
Zadanie należy wykonać na podstawie danych obserwacyjnych i nawigacyjnych, zapisanych w plikach
w formacie RINEX. Program umożliwiać ma obliczenie współrzędnych odbiornika dla całej doby, co 30
sekund, niezależnie dla każdej epoki.
Program powinien umożliwiać wybór maski elewacji oraz eliminacje błędów związanych z propagacją
fali przez atmosferę (przynajmniej opóźnienia troposferycznego).
Dodatkowo, program może dawać możliwość:
• wyboru typu obserwacji kodowej;
• wyboru metody eliminacji błędu troposfery i jonosfery;
• wykonania wizualizacji, przedstawiających szereg czasowy wyznaczonych współrzędnych punktu lub
wizualizacji położenia punktu "w czasie rzeczywistym";
• przeliczenia wyznaczonych współrzędnych XYZ do układów płaskich/lokalnych/krzywoliniowych;
• obliczenia współczynników DOP.
2 Kolejność wykonania zadania
Ustawienia wstępne:
• odczytanie danych obserwacyjnych i nawigacyjnych, korzystając z funkcji readrnx;
• zadeklarowanie odpowiedniej daty obliczeń;
• ustawienie maski elewacji.
2.1 Rozwiązanie pozycji dla pojedynczej epoki tr:
1. selekcja pseudoodległości, zapisanych w zmiennej obs, zarejestrowanych w danej epoce tr, i odpowiadających im satelitów, na podstawie identyfikatorów ze zmiennej iobs, w których zapisane są
czasy obserwacji oraz numery satelitów;
1
2. Zdefiniowanie wartości przybliżonych:
• zadeklarowanie przybliżonych współrzędnych odbiornika x0;
• δtr = 0s - poprawka do zegara odbiornika
• τ = 0.07s - czas propagacji sygnału
Ponieważ niewiadomymi w rozwiązaniu pozycji metodą najmniejszych kwadratów będą przyrosty
do współrzędnych, a nie same współrzędne odbiornika, toteż cały proces będzie należało wykonać w sposób iteracyjny, aż do momentu, kiedy różnica między współrzędnymi, wyznaczonymi z
kolejnych iteracji, nie będzie się różnić o więcej niż 1 mm. Zazwyczaj 5 iteracji jest całkowicie
wystarczające, dlatego zamiast definiowania warunku na spójność wyznaczonych współrzędnych
z kolejnych iteracji, obliczenia wykonać można dla z góry ustalonej liczby pięciu iteracji.
Pętla na iteracyjne wyznaczenie szukanej pozycji odbiornika (przyrostów do współrzędnych
przybliżonych). Dla kolejnej iteracji (1:5):
Dla każdego satelity obserwowanego w danej epoce:
1. Obliczenie współrzędnych satelity (współrzędnych xyz oraz błędu zegara satelity, z uwzględnieniem
poprawki relatywistycznej, δts
) na czas emisji sygnału t
s
:
t
s = tr + δtr − τ (1)
x
s
0
, ys
0
, zs
0
, δts = satpos(tr, nav) (2)
*uwaga: czas propagacji sygnału τ w pierwszej iteracji jest wartością znaną, przybliżoną, taką samą
dla każdego satelity. W kolejnych iteracjach, wartość ta będzie zależeć od obliczonej w poprzedniej
iteracji odległości geometrycznej: τ = ρ
s
0/c i będzie różna dla każdego satelity!
2. Transformacja współrzędnych satelity do chwilowego układu współrzędnych, na moment odbioru
sygnału:





x
s
y
s
z
s





=





cos(ωEτ ) sin(ωEτ ) 0
− sin(ωEτ ) cos(ωEτ ) 0
0 0 1





·





x
s
0
y
s
0
z
s
0





(3)
gdzie:
• ωE = 7.2921151467 · 10−5
[
rad
s
] prędkość obrotowa Ziemi
• τ - czas propagacji sygnału (w pierwszej iteracji wartość znana, przybliżona, w kolejnych
iteracjach, mając wyznaczoną odległość geometryczną ρ
s
r
: τ = ρ
s
0/c
• c = 299792458.0 [m/s] - prędkość światła
3. Obliczenie odległości geometrycznej między satelitą a odbiornikiem:
ρ
s
0 =
p
(x
s − x0)
2 + (y
s − y0)
2 + (z
s − z0)
2 (4)
* uwaga: za współrzędne satelity x
s
, ys
, zs przyjmujemy współrzędne “obrócone” do chwilowego
układu współrzędnych → wynik równania (3);
*uwaga: z obliczonych odległości geometrycznych (lub czasu propagacji sygnału τ = ρ
s
0/c) dla
danego satelity będziemy musieli skorzystać w kolejnej iteracji, dlatego trzeba te wartości zapisać
do jakiejś zmiennej.
4. Wyznaczenie elewacji (i azymutu, niezbędnego do wyznaczenia opóźnienia jonosferycznego) satelity
oraz odrzucenie satelitów znajdujących się poniżej maski;
*uwaga! Współrzędne przybliżone odbiornika x0 w pierwszej iteracji mogą być bardzo odległe od
rzeczywistej pozycji odbiornika. Dlatego w pierwszej iteracji, za wartość elewacji można przyjąć np.
90◦ dla każdego satelity. W kolejnych iteracjach należy obliczyć odpowiednie kierunki do satelitów,
najpierw przeliczając współrzędne odbiornika x0 do współrzędnych krzywoliniowych, wykorzystując
algorytm Hirvonena.
5. Wyznaczenie opóźnienia troposferycznego δTs
r
i jonosferycznego δIs
r dla danego satelity (wzory w
odpowiednich prezentacjach).
2
Ułożenie równań obserwacyjnych i rozwiązanie pozycji metodą najmniejszych kwadratów:
(a) Przypomnijmy, uproszczone równanie pseudoodległości dla obserwacji kodowych, z uwzględnieniem wpływu błędów pomiarowych, można przedstawić następująco:
P
s
r = ρ
s
r + cδtr − cδts + δIs
r + δTs
r
(5)
gdzie:
P
s
r
: pomierzona wartość pseudoodległości między satelitą s i odbiornikiem r
ρ
s
r
: odległość geometryczna między satelitą s i odbiornikiem r
c : prędkość światła
δtr : błąd zegara odbiornika r
δts
: błąd zegara satelity s
δIs
r
: wpływ refrakcji jonosferycznej
δTs
r
: wpływ refrakcji troposferycznej
Lewa stronę równania reprezentuje pomierzona wartość odległości (pseudoodległość) pomiędzy odbiornikiem r i satelitą s. Prawa strona równania są to parametry składające się na tę
pomierzoną wartość odległości, czyli odległość geometryczna oraz błędy pomiarowe.
(b) Zapisanie zlinearyzowanych równań obserwacyjnych w taki sposób, aby niewiadome znalazły
się po jednej stronie równania, a znane po drugiej (jako elementy znane, traktujemy również
modelowane wartości opóźnienia atmosferyczne i błąd zegara satelity):
*uwaga! Jako niewiadomą będziemy liczyć przyrost do przybliżonej wartości błędu zegara
satelity ∆(cδtr) → błąd zegara, z równania (), możemy rozwinąć do: cδtr = cδtr + ∆(cδtr)
P
s
r −ρ
s
0 + cδts−cδtr−δTs
r −δIs
r =
−(x
s − x0)
ρ
s
0
·∆x+
−(y
s−y0)
ρ
s
0
·∆y+
−(z
s − z0)
ρ
s
0
·∆z+∆(cδtr)
Lewa strona tego równania stanowić będzie element wektora wyrazów wolnych y. Współczynniki przy niewiadomych, po prawej stronie równania, będą stanowić elementy macierzy równań
obserwacyjnych A.
(c) Zbudowanie wektora wyrazów wolnych y oraz macierzy równań obserwacyjnych A. Będą one
miały tyle wierszy, ile satelitów obserwowanych w danej epoce, powyżej maski elewacji.
Dla pojedynczego satelity, elementy wektora y i macierzy A będą wyglądać następująco:
y
s
r = P
s
r − ρ
r
0 + cδts − cδtr − δTs
r − δIs
r
(6)
A
s =
␔
−(x
s − xr)
ρ
s
r
−(y
s − yr)
ρ
s
r
−(z
s − zr)
ρ
s
r
1
␕
(7)
3
(d) Układ równań w postaci macierzowej będzie wyglądał następująco:








y
s1
y
s2
.
.
.
y
sn








=











−(x
s1 − x0)
ρ
s1
0
−(y
s1 − y0)
ρ
s1
0
−(z
s1 − z0)
ρ
s1
0
1
−(x
s2 − x0)
ρ
s2
0
−(y
s2 − y0)
ρ
s2
0
−(z
s2 − z0)
ρ
s2
0
1
.
.
.
.
.
.
.
.
.
.
.
.
−(x
sn − x0)
ρ
sn
0
−(y
sn − y0)
ρ
sn
0
−(z
sn − z0)
ρ
sn
0
1



















∆x
∆y
∆z
cδtr








(8)
y A x
(e) Powyższy układ równań rozwiązujemy następująco:
x = (ATA)
−1ATy (9)
6. Poprawiamy współrzędne przybliżone odbiornika oraz przybliżoną wartość poprawki zegara odbiornika:





xr = x0 + ∆x
yr = y0 + ∆y
zr = z0 + ∆z





(10)
δtr = δtr + ∆δtr/c (11)
UWAGA!!!
Wyznaczone współrzędne odbiornika xr, yr, zr oraz błąd zegara odbiornika δtr stanowią dane
wejściowe do następnej iteracji!!! Zatem w kolejnych iteracjach, za współrzędne przybliżone
odbiornika x0 przyjmujemy obliczone w poprzedniej iteracji współrzędne xr (np. równanie
4). Natomiast w równaniach (1) i (6) podstawiamy nowy błąd zegara odbiornika δtr. Z kolei
w równaniu (1) wykorzystujemy obliczone w porpzedniej iteracji odległości geometryczne
ρ
s
0
i na ich podstawie liczymy czas propagacji sygnalu τ (dla każdego satelity odległość geometryczna i czas propagacji sygnału są inne! Tylko w pierwszej iteracji
przyjmujemy takie same dla wszystkich satelitów.).
7. Szukaną pozycją odbiornika, a także kierunkami do satelity czy współczynnikami DOP, są wartości
obliczone w ostatniej iteracji.
Obliczone współrzędne odbiornika, dla każdej epoki, należy zapisać do zmiennej i porównać ze
współrzędnymi referencyjnymi.
4
"""

