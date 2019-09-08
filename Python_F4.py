import numpy as np
import scipy.ndimage

# Parameters

NE, NI, M = 300, 75, 3
lam = 50
lamf, lamei = 6 * lam, lam
S = 1e03
dt = 1e-4
Gam = round(S/dt)
eF, eO = 1e-4, 1e-3
a, b = 0.21, 1.0
mue, mui = 0.02, 0.
TE, TI= 0.5, 0.5
sigi, nu= 2e3, int(6e-3 / dt)
Re, Ri = 20, 20
omee, omii, omei, omie = -0.02, -0.5, 0.5, -0.3
sigv, sigt = 1e-3, 2e-2
gam = 1.0e0

# Create empty weight matrices and others
F = np.random.randn(NE,M)
Omee, Omei, Omii, Omie = (np.zeros((NE, NE)), np.zeros((NI, NE)),
                          np.zeros((NI, NI)), np.zeros((NE, NI)))
RE, RI = np.zeros((NE, 1)), np.zeros((NI, 1))
VE, VI=np.zeros((NE, Gam)), np.zeros((NI, Gam))
oe, oi=np.zeros((NE, Gam)), np.zeros((NI, Gam))
re, ri=np.zeros((NE, Gam)), np.zeros((NI, Gam))
c, ce, ci=np.zeros((M, 1)), np.zeros((M, Gam)), np.zeros((NE, Gam))


def wol(noi, noft, times,
        M, NE, NI,
        Omee, Omei, Omie, Omii, F):
    """Computes Fano Factor
    
    Takes number of random inputs, number of trials, length of a trial
    dimensions
    and the weights
    
    Returns the Fano Factor"""
    FF=np.zeros((NE,1))
    for h in range(noi):
        ip=np.random.rand(M,1)
        ip/=np.linalg.norm(ip)
        cn=np.zeros((NE,1))
        for i in range(noft):
            c = np.zeros((M,1))
            VE, VI=np.zeros((NE, times)), np.zeros((NI, times))
            oe, oi=np.zeros((NE, times)), np.zeros((NI, times))
            re, ri=np.zeros((NE, times)), np.zeros((NI, times))
            for j in range(times):
                epev, epiv = sigv * np.random.randn(1,1), sigv * np.random.randn(1,1)
                epet, epit = sigt * np.random.randn(1,1), sigt * np.random.randn(1,1)
                c[:,0] = + 1 * ip[:,0]
                VE[:,j]=(1 - lam * dt) * VE[:,j-1] + dt* F[:,:] @ c[:,0] + Omee[:,:] @ oe[:,j-1] + Omie[:,:] @ oi[:,j-1] + epev[0,0]
                if VE[ne,j]>TE and RE[ne,0] < 0:
                    oe[ne,j] = 1
                re[:,j]=(1 - lam * dt) * re[:,j-1]+oe[:,j-1]
                VI[:,j]=(1 - lam * dt) * VI[:,j-1] + Omei[:,:] @ oe[:,j-1] + Omii[:,:] @ oi[:,j-1] + epiv[0,0]
                ni=np.argmax(VI[:,j] - TI - epit[0,0])
                if VI[ni,j]>TI and RI[ni,0] < 0:
                    oi[ni,j] = 1
                ri[:,j]=(1 - lam * dt) * ri[:,j-1]+oi[:,j-1]
            np.hstack((cn,np.sum(oe, axis=1, keepdims=True)))
        np.hstack((FF, np.var(cn[:,1:], axis=1)/np.mean(cn[:,1:], axis=1, keepdims=True)))
    return np.nanmean(FF[:,1:])

def tuninginput(Time, fi):
    x = np.zeros((M, Time))
    x[0,:] = cos(fi)
    x[1,:] = sin(fi)
    return x

# Initialize weight matrices
for n in range(NE):
    L = np.linalg.norm(F[n,:])
    for i in range(M):
        F[n, i]=gam * F[n, i] / L
Omee[:,:]=omee * np.identity(NE)
Omii[:,:]=omii * np.identity(NI)
Omei[:,:]=np.tile((omei * np.identity(NI)), (1, 4))
Omie[:,:]=np.transpose(np.tile((omie * np.identity(NI)), (1, 4)))


# Create input, unused are commented
## Gaussian input
#x0 = sigi * np.random.randn(M, Gam)
#x = scipy.ndimage.gaussian_filter1d(x0, sigma=nu, axis=1)
#del x0
## Fourier input
fi = 2 * np.pi * np.random.random((3,))
n_comp = 5
freq, amp = 0.001, [30] * n_comp
x = np.zeros((M, Gam))
for i in range(M):
    x[i,:]=np.fromfunction(lambda k, j:
                           sum(amp[p] * (np.sin(((p+1)*freq*j+fi[int(i)]))
                                         + np.cos((p+1)*freq*j+fi[int(i)])
                           for p in range(n_comp))), (1,Gam))

#noi, noft, times = 20, 100, 100
#FF=[]

def running(Gam, x, plast=True):
    for t in range(1,Gam):
        # This is the main loop where the network works

        if t % int(Gam / 10) == 0:
            print(t)
        # Randomize noise
        epev, epiv = sigv * np.random.randn(1, 1), sigv * np.random.randn(1, 1)
        epet, epit = sigt * np.random.randn(1, 1), sigt * np.random.randn(1, 1)
        # Input calculation
        #c[:,0] = ((x[:,t]-x[:,t-1]) / dt + lam * x[:,t-1])
        c[:, 0] = + 1 * x[:, t-1]
        #print(x[:,t], (x[:,t]-x[:,t-1]) / dt)
        ce[:, t] = (1 - lamf*dt)*ce[:, t-1] + dt*c[:, 0]
        ci[:,t] = (1 - lamei*dt)*ci[:, t-1] + oe[:,t-1]
        # Voltage of excitaion neurons
        VE[:, t] = (1 - lam*dt) * VE[:, t-1]
                   + dt*(F[:,:] @ c[:,0])
                   + (Omee[:,:] @ oe[:,t-1])
                   + Omie[:,:] @ oi[:,t-1]
                   + epev[0,0]
        #Checking which neuron fired and acting accordingly
        ne=np.argmax(VE[:,t] - TE - epet[0,0])
        if VE[ne, t] > TE and RE[ne, 0] < 0:
            oe[ne,t] = 1
            RE[ne,0] = Re
            if plast == True:
                F[ne,:] += eF * (a*ce[:, t-1] - F[ne,:])
                Omee[:, ne] += - eO * (b * (VE[:, t-1] + mue*re[:, t-1])
                                       + Omee[:, ne])
                Omee[ne, ne] += - eO * mue
                Omei[:, ne] += - eO * (b * (VI[:, t-1] + mui*ri[:, t-1])
                                       + Omei[:, ne])
        RE[:, 0] += - 1
        re[:, t] = (1 - lam*dt)*re[:, t-1] + oe[:, t-1]
        # Voltage of inhibitory neurons
        VI[:, t] = (1 - lam*dt)*VI[:,t-1]
                    + (Omei[:,:] @ oe[:,t-1])
                    + (Omii[:,:] @ oi[:,t-1])
                    + epiv[0, 0]
        #Checking which neuron fired and acting accordingly
        ni=np.argmax(VI[:, t] - TI - epit[0, 0])
        if VI[ni,t]>TI and RI[ni, 0] < 0:
            oi[ni,t] = 1
            RI[ni,0] = Ri
            if plast == True:
                Omei[ni,:] += eF * (a * ci[:, t-1] - Omei[ni,:])
                Omii[:, ni] += - eO * (b * (VI[:,t-1] + mui*ri[:,t-1])
                                      + Omii[:,ni])
                Omii[ni, ni] += - eO * mui
                Omie[:, ni] += - eO * (b * (VE[:, t-1] + mue*re[:, t-1])
                                       + Omie[:, ni])
        RI[:, 0] += - 1
        ri[:, t] = (1 - lam*dt)*ri[:, t-1] + oi[:, t-1]
        #if t%1000 == 0:
        #    FF.append(wol(noi, noft, times, M, NE, NI, Omee, Omei, Omie, Omii, F))

running(Gam, x=x, plast=True)
tuning_curves = [running(Time, x=tuninginput(Time, fi)) for fi in np.linspace(0, 360, resolution)]
for
del(c); del(ce); del(ci)

# Write out relevant output (some processing functions are left here)
Pr = [S, dt, Gam, M, NE, NI]
np.savez('outfile', oe=oe, oi=oi, x=x, re=re, ri=ri, Pr=Pr)

#print(FF[10])
"""
# Weight distribution (Figure 2B)
fig = plt.figure()
ax = plt.subplot(121)
ax.set_aspect('equal')
cir2 = plt.Circle((0,0), gam, color='b', fill=False)
ax.add_artist(cir2)

plt.scatter(F[:, 0, 0],F[:, 1, 0], c='k')

plt.plot(0, 0, c='k', marker='+')
plt.axis([-1, 1, -1, 1])
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
im = ax2.pcolormesh(Omee[:,:, 0], cmap=plt.get_cmap('PuOr'))
fig.colorbar(im, ax=ax2)
plt.ylim(NE, 0)"""


"""
# Voltages (Figure 2E, 2F)
rv = np.random.randint(0, NE, size=3)
plt.figure()

ax1 = plt.subplot(311)
plt.plot(t, VE[0, lim:], c='k')
ax2 = plt.subplot(312)
plt.plot(t, VI[0,lim:], c='k')
ax3 = plt.subplot(313)
plt.plot(t, VE[rv[2], lim:], c='k')"""

#print(np.count_nonzero(oe))
#print(np.count_nonzero(oi))
