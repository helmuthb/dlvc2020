import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import csv

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

X = []
Y = []
Z = []
with open('../results.csv', 'r') as f:
  reader = csv.reader(f)
  for row in reader:
    X.append(float(row[0]))
    Y.append(float(row[1]))
    Z.append(float(row[2]))

# find maximum Z value
Zmax = max(Z)
Imax = [i for i in range(len(Z)) if Z[i] == Zmax][0]
# append it to the end so that the best value is plotted over
Z.append(Zmax)
X.append(X[Imax])
Y.append(Y[Imax])

# map accuracy to color
norm = colors.Normalize(vmin=min(Z), vmax=max(Z))
cmap = cm.viridis

extent = (0,1,0,1)

# plt.tricontourf(X, Y, Z, cmap=cmap, norm=norm, alpha=0.5)
plt.scatter(X, Y, c=Z, cmap=cmap, norm=norm)
plt.scatter(X[Imax], Y[Imax], c='black', marker='x', alpha=0.7, s=200)
im = plt.imshow(np.empty((0,0)), cmap=cmap, norm=norm, extent=extent)
fig.colorbar(im, cmap=cmap, norm=norm)
plt.title("Hyperparameter Optimization")
plt.xlabel("Learning Rate")
plt.ylabel("Momentum")
plt.savefig('plot.pdf', bbox_inches='tight')
