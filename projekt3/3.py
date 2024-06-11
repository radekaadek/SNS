import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pyproj

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
diff_to_actual = np.sqrt(np.sum((wyniki[:, :3] - actual_pos_neu) ** 2, axis=1))



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
ax.set_title('Ratio test with respect to time')
plt.show()

# show a bar graph of counts of Q
qs = np.unique(wyniki[:, 3])
q_counts = []
for q in qs:
    q_counts.append(np.sum(wyniki[:, 3] == q))
fig, ax = plt.subplots()
ax.bar(qs, q_counts)
# 1 means fixed, 2 means float
ax.set_xticks([1, 2])
ax.set_xticklabels(['Fixed', 'Float'])
ax.set_title('Number of fixed and float solutions')
plt.show()

# show a plot of error to actual position with respect to time
fig, ax = plt.subplots()
ax.plot(dates, diff_to_actual)
ax.set_title('Difference to actual position with respect to time')
plt.show()


# show a bar plot of rmse, stdev, max and mean of error to actual position
rmse = np.sqrt(np.mean(diff_to_actual ** 2))
stdev = np.std(diff_to_actual)
max_diff = np.max(diff_to_actual)
mean_diff = np.min(diff_to_actual)
fig, ax = plt.subplots()
ax.bar(['RMSE', 'Stdev', 'Max', 'Mean'], [rmse, stdev, max_diff, mean_diff])
ax.set_title('Error to actual position metrics')
for i, v in enumerate([rmse, stdev, max_diff, mean_diff]):
    ax.text(i, v + 0.1, str(round(v, 2)))
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

