import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Load in data
npzfile = np.load('outfile1.npz')
oe = npzfile['oe']
oi = npzfile['oi']
x = npzfile['x']
re = npzfile['re']
ri = npzfile['ri']
ip = npzfile['ip']
Pr = npzfile['Pr']
Gam = int(Pr[2])
M = int(Pr[3])
NE = int(Pr[4])
dt = Pr[1]
S = Pr[0]
del(npzfile)
del(ri)
del(oi)

# Calculate decoder and decoded signal
xr = np.zeros((M, NE))
rr = np.zeros((NE, NE))
for t in range(int(Gam * 0 / 10), Gam):
    xr[:,:] += np.outer(x[:, t],re[:, t])
    rr[:,:] += np.outer(re[:, t],re[:, t])
x2 = np.zeros((M, Gam))
xr[:,:] /= Gam * 10 / 10
rr[:,:] /= Gam * 10 / 10
D=xr @ np.linalg.inv(rr + 1e-300*np.identity(NE))
#D=np.transpose(F)
for t in range(1, Gam):
    x2[:,t] = D @ re[:, t]
del(xr); del(rr); del(D)

# Ranges for the plots
t = np.arange(0, 1, dt)
t2 = np.arange(Gam)

# Error and rate (Figures)
fig, ax1 = plt.subplots()

ax1.plot(t2,
         scipy.ndimage.gaussian_filter1d(
             np.linalg.norm(x2[:,:] - x[:,:], axis=0)**2,
             sigma=int(1e-0 / dt)),
         c='b')
ax1.set_ylabel('Error', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(t2,
         scipy.ndimage.gaussian_filter1d(
             np.mean(re[:,:], axis=0),
             sigma=int(1e-0 / dt)),
         c='tab:orange')
ax2.set_ylabel('Rate', color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
fig.tight_layout()

# Input and output signal comparison (Figure 2C, 4C bottom)
lim = - int(Gam / S)
plt.figure()
plt.plot(t, x[0, lim:], c='purple')
plt.plot(t, x[1, lim:], c='purple')
plt.plot(t, x2[0, lim:], c='g')
plt.plot(t, x2[1, lim:], c='g')

# Excitory and inhibitory input comparison (Figure 4C middle)
plt.figure()
plt.plot(t, ip[0, lim:], c='b')
plt.plot(t, - ip[1, lim:], c='tab:orange')

del(x); del(x2)

# Rasterplot of firings
fig, ax=plt.subplots()
ax.eventplot([np.array(np.nonzero(row)[0]) for row in oe],
             colors='k')

plt.show()
