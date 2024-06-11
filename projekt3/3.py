import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from pyproj import Geod
import pyproj

# % program   : RTKPOST ver.2.4.3 b34
# % inp file  : WUTR00POL_R_20241390000_01D_30S_MO.rnx
# % inp file  : CBKA1390.24o
# % inp file  : CBKA1390.24n
# % obs start : 2024/05/18 00:00:00.0 GPST (week2314 518400.0s)
# % obs end   : 2024/05/18 23:59:30.0 GPST (week2314 604770.0s)
# % pos mode  : Kinematic
# % freqs     : L1+2
# % solution  : Forward
# % elev mask : 15.0 deg
# % dynamics  : off
# % tidecorr  : off
# % ionos opt : Broadcast
# % tropo opt : Saastamoinen
# % ephemeris : Broadcast
# % navi sys  : GPS
# % amb res   : Continuous
# % val thres : 2.5
# % antenna1  :                       ( 0.0000  0.0000  0.1010)
# % antenna2  :                       ( 0.0000  0.0000  0.0000)
# % ref pos   :  3654410.0348   1407752.4780   5017576.9176
# %
# % (x/y/z-ecef=WGS84,Q=1:fix,2:float,3:sbas,4:dgps,5:single,6:ppp,ns=# of satellites)
# %  GPST                      x-ecef(m)      y-ecef(m)      z-ecef(m)   Q  ns   sdx(m)   sdy(m)   sdz(m)  sdxy(m)  sdyz(m)  sdzx(m) age(s)  ratio
# 2024/05/18 00:00:00.000   3655334.3694   1403901.2189   5018038.3221   2   8   1.1580   0.6265   1.2637   0.5497   0.4747   0.9638   0.00    2.2
# 2024/05/18 00:00:30.000   3655333.8112   1403901.0398   5018037.9883   1   8   0.0117   0.0063   0.0128   0.0056   0.0048   0.0097   0.00    2.6
# 2024/05/18 00:01:00.000   3655333.8139   1403901.0395   5018037.9864   1   8   0.0117   0.0063   0.0128   0.0056   0.0048   0.0097   0.00    3.6
# 2024/05/18 00:01:30.000   3655333.8147   1403901.0381   5018037.9881   1   8   0.0117   0.0064   0.0128   0.0056   0.0048   0.0097   0.00    4.6
# 2024/05/18 00:02:00.000   3655333.8134   1403901.0444   5018037.9794   1   8   0.0117   0.0063   0.0128   0.0056   0.0048   0.0097   0.00    7.1
# 2024/05/18 00:02:30.000   3655333.8094   1403901.0412   5018037.9807   1   8   0.0117   0.0063   0.0128   0.0057   0.0048   0.0096   0.00    7.5
# 2024/05/18 00:03:00.000   3655333.8199   1403901.0421   5018037.9870   1   8   0.0117   0.0063   0.0127   0.0057   0.0048   0.0096   0.00    8.5
# 2024/05/18 00:03:30.000   3655333.8110   1403901.0363   5018037.9877   1   8   0.0117   0.0064   0.0127   0.0057   0.0047   0.0095   0.00    8.2
# 2024/05/18 00:04:00.000   3655333.8195   1403901.0378   5018037.9864   1   8   0.0117   0.0063   0.0127   0.0056   0.0047   0.0095   0.00    6.9
# 2024/05/18 00:04:30.000   3655333.8130   1403901.0372   5018037.9773   1   8   0.0117   0.0064   0.0127   0.0057   0.0048   0.0095   0.00    6.2
# 2024/05/18 00:05:00.000   3655333.8140   1403901.0385   5018037.9832   1   8   0.0117   0.0065   0.0127   0.0056   0.0047   0.0095   0.00    5.7
# 2024/05/18 00:05:30.000   3655333.8166   1403901.0390   5018037.9865   1   8   0.0117   0.0064   0.0127   0.0057   0.0047   0.0094   0.00    5.2

def xyz_neu(xyz, actual_pos):
    # convert xyz to blh of the reference position
    xyz_proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    blh_proj = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(xyz_proj, blh_proj)
    ref_blh = transformer.transform(actual_pos[0], actual_pos[1], actual_pos[2])
    # convert xyz to neu
    Rneu = np.array([[-np.sin(ref_blh[0])*np.cos(ref_blh[1]), -np.sin(ref_blh[0])*np.sin(ref_blh[1]), np.cos(ref_blh[0])],
                     [-np.sin(ref_blh[1]), np.cos(ref_blh[1]), 0],
                     [-np.cos(ref_blh[0])*np.cos(ref_blh[1]), -np.cos(ref_blh[0])*np.sin(ref_blh[1]), -np.sin(ref_blh[0])]])
    Rneu = Rneu.T
    neu = np.dot(Rneu, xyz - actual_pos[:, None])
    return neu



wyniki = np.loadtxt('res.obs', comments='%', usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
actual_pos = np.array([3655333.847, 1403901.067, 5018038.047])


# convert all to neu
neu = xyz_neu(wyniki[:, :3].T, actual_pos)
wyniki[:, :3] = neu.T
actual_pos_neu = np.array([0, 0, 0])

# dates
dates = []
with open('res.obs') as f:
    for line in f:
        if not line.startswith('%'):
            date = line.split()[0]
            time = line.split()[1]
            # convert to unix timestamp
            date = date.split('/')
            time = time.split(':')
            date = [int(x) for x in date]
            time = [int(float(x)) for x in time]
            obs_date = datetime.datetime(date[0], date[1], date[2], time[0], time[1], time[2])
            dates.append(obs_date)
dates = np.array(dates)

# first row is bad, remove it
wyniki = wyniki[1:, :]
dates = dates[1:]



# plot x, y, z with respect to time
fig, ax = plt.subplots(3, 1)
fig.suptitle('N, E, U with respect to time')
ax[0].plot(dates, wyniki[:, 0])
ax[0].set_title('N')
ax[1].plot(dates, wyniki[:, 1])
ax[1].set_title('E')
ax[2].plot(dates, wyniki[:, 2])
ax[2].set_title('U')
plt.show()

# show a plot of ratio with respect to time
fig, ax = plt.subplots()
ax.plot(dates, wyniki[:, -1])
ax.set_title('Ratio with respect to time')
plt.show()


# show a 2D plot of x and y and color opacity based on z
fig, ax = plt.subplots()
mean_pos = np.mean(wyniki[:, :3], axis=0)
ax.scatter(wyniki[:, 0] - actual_pos_neu[0], wyniki[:, 1] - actual_pos_neu[1], c=wyniki[:, 2], alpha=0.5)
ax.set_title('N and E with respect to actual position colored by U')
# add a legend that shows the color scale of z
cbar = plt.colorbar(ax.scatter(wyniki[:, 0] - actual_pos_neu[0], wyniki[:, 1] - actual_pos_neu[1], c=wyniki[:, 2], alpha=0.5))
cbar.set_label('U')
plt.show()

# sort observations by test ratio and show the difference to the actual position
ratio_sorted = wyniki[wyniki[:, -1].argsort()]
diff_to_actual = np.sqrt(np.sum((ratio_sorted[:, :3] - actual_pos_neu) ** 2, axis=1))
fig, ax = plt.subplots()
ax.plot(diff_to_actual)
ax.set_title('Difference to actual position sorted by test ratio')
# add labels
ax.set_xlabel('Sorted observation number')
ax.set_ylabel('Difference to actual position')
plt.show()

# now show the same plot but draw a trend line
fig, ax = plt.subplots()
# calculate the trend line
trend = np.polyfit(np.arange(diff_to_actual.size), diff_to_actual, 1)
trend_line = np.polyval(trend, np.arange(diff_to_actual.size))
ax.plot(trend_line)
ax.set_title('Least squares line of difference to actual position sorted by test ratio')
# add labels
ax.set_xlabel('Trend line')
ax.set_ylabel('Difference to actual position')
plt.show()

# show a 3D plot of x, y, z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wyniki[:, 0], wyniki[:, 1], wyniki[:, 2])
ax.set_title('N, E, U')
ax.set_xlabel('N')
ax.set_ylabel('E')
ax.set_zlabel('U')
plt.show()


# count the average error to actual for every number of ns
ns = np.unique(wyniki[:, 4])
actual_diff = []
for n in ns:
    actual_diff.append(np.mean(np.sqrt(np.sum((wyniki[wyniki[:, 4] == n, :3] - actual_pos_neu) ** 2, axis=1))))
fig, ax = plt.subplots()
ax.bar(ns, actual_diff)
ax.set_xlabel('Number of satellites visible')
ax.set_ylabel('Mean difference to actual position')
ax.set_title('Mean difference to actual position with respect to number of satellites visible')
plt.show()

# now show the same plot but with difference to mean position
fig, ax = plt.subplots()
mean_diff = []
for n in ns:
    mean_diff.append(np.mean(np.sqrt(np.sum((wyniki[wyniki[:, 4] == n, :3] - mean_pos) ** 2, axis=1))))
ax.bar(ns, mean_diff)
ax.set_xlabel('Number of satellites visible')
ax.set_ylabel('Mean difference to mean position')
ax.set_title('Mean difference to mean position with respect to number of satellites visible')
plt.show()

# show an animated 3D plot of x, y, z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# set axis limits
# ax.set_title('Animated X, Y, Z')
ax.set_xlabel('N')
ax.set_ylabel('E')
ax.set_zlabel('U')
for i in range(wyniki.shape[0]):
    ax.set_xlim(np.min(wyniki[:, 0]), np.max(wyniki[:, 0]))
    ax.set_ylim(np.min(wyniki[:, 1]), np.max(wyniki[:, 1]))
    ax.set_zlim(np.min(wyniki[:, 2]), np.max(wyniki[:, 2]))
    ax.set_title(f"N, E, U at {dates[i]}")
    ax.scatter(wyniki[:i, 0], wyniki[:i, 1], wyniki[:i, 2])
    plt.pause(0.00001)
    # set title to date
    ax.clear()

