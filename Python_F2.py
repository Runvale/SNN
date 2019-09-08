import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import math

N, M = 20, 2
lam = 50
S= 10
dt= 1e-3
Gam=round(S/dt)
F=np.zeros((N,M,Gam))
Om=np.zeros((N,N,Gam))
eF, eO= 1e-5, 1e-4
a, b= 0.21, 1.25
mu= 0.02
T= 0.5
sigi, nu= 2e3, 6
gam, om = 0.8, -0.5
sigv, sigt= 1e-3, 2e-2
F[:,:,0]=np.random.randn(N,M)
for n in range(N):
    L=np.linalg.norm(F[n,:,0])
    for i in range(M):
        F[n,i,0]=gam * F[n,i,0]/L
    #if F[n,0,0]>0: F[n,0,0]=-F[n,0,0]
Om[:,:,0]=om*np.identity(N)

V=np.zeros((N,Gam))
o=np.zeros((N,Gam))
r=np.zeros((N,Gam))
x0=sigi*np.random.randn(M,Gam)
x=scipy.ndimage.gaussian_filter1d(x0, sigma=nu, axis=1)
#x=np.fromfunction(lambda i, j: 700 * np.sin(j/100+i * np.pi/2), (M,Gam))
x2=np.zeros((M,Gam))
epv=sigv * np.random.randn(1,Gam)
ept=sigt * np.random.randn(1,Gam)
c=np.zeros((M,Gam))
xr=np.zeros((M,N,Gam))
rr=np.zeros((N,N,Gam))

for t in range(1,Gam):
    c[:,t-1]=(x[:,t]-x[:,t-1])+dt*lam * x[:,t-1]
    V[:,t]=(1-lam * dt) * V[:,t-1]+dt * F[:,:,t-1] @ c[:,t-1]+Om[:,:,t-1] @ o[:,t-1]+epv[0,t]
    n=np.argmax(V[:,t]-T-ept[0,t])
    F[:,:,t]=F[:,:,t-1]
    Om[:,:,t]=Om[:,:,t-1]
    if V[n,t]>T:
        o[n,t]=1
        F[n,:,t]=F[n,:,t-1]+eF * (a*x[:,t-1]-F[n,:,t-1])
        Om[:,n,t]=Om[:,n,t-1]-eO * (b * (V[:,t-1]+mu * r[:,t-1])+Om[:,n,t-1])
        Om[n,n,t]=Om[n,n,t-1]-eO*mu
    r[:,t]=(1-lam * dt) * r[:,t-1]+o[:,t-1]
    xr[:,:,t]=np.outer(x[:,t],r[:,t])
    rr[:,:,t]=np.outer(r[:,t],r[:,t])


D = np.mean(xr, axis=2) @ np.linalg.inv(np.mean(rr, axis=2))
for t in range(1, Gam):
    x2[:,t] = D @ r[:,t]


t = np.arange(0, S, dt)

fig=plt.figure()
ax=plt.subplot(121)
ax.set_aspect('equal')
cir2=plt.Circle((0,0), 0.8, color='b', fill=False)
ax.add_artist(cir2)

plt.scatter(F[:,0,Gam-1],F[:,1,Gam-1], c='k')

plt.plot(0,0, c='k', marker='+')
plt.axis([-1,1,-1,1])
ax2=plt.subplot(122)
ax2.set_aspect('equal')
im = ax2.pcolormesh(Om[:,:,Gam-1], cmap=plt.get_cmap('PuOr'))
fig.colorbar(im, ax=ax2)
plt.ylim(20,0)

fig, ax1 = plt.subplots()

ax1.plot(t, scipy.ndimage.uniform_filter1d(np.linalg.norm(x2-x, axis=0)**2, size=nu), c='b')
ax1.set_ylabel('Error', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(t, np.mean(r, axis=0), c='tab:orange')
ax2.set_ylabel('Rate', color='tab:orange')
ax2.tick_params('y', colors='tab:orange')
fig.tight_layout()

plt.figure()
plt.plot(t, x[0,:], c='purple')
plt.plot(t, x[1,:], c='purple')
plt.plot(t, x2[0,:], c='g')
plt.plot(t, x2[1,:], c='g')

fig, ax=plt.subplots()
ax.eventplot([np.array(np.nonzero(row)[0]) for row in o], colors='k')

rv=np.random.randint(0,20, size=3)
plt.figure()

ax1=plt.subplot(311)
plt.plot(t, V[rv[0],:], c='k')
ax2=plt.subplot(312)
plt.plot(t, V[rv[1],:], c='k')
ax3=plt.subplot(313)
plt.plot(t, V[rv[2],:], c='k')
plt.show()
print(np.count_nonzero(o))
print(x[0,:])
print(x2[0,:])