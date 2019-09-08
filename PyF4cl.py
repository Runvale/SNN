import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# Parameters

NE, NI, M = 300, 75, 3
lam = 50
lamf, lamei = 6 * lam, lam
S = 1e02
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
tun_resolution = 360
tun_time = 1000
nip = np.random.randint(0, NE, size=1)

class Netw:
    def __init__(self, M, NE, NI, Time,
                 F=None, Omee=None, Omei=None, Omie=None, Omii=None):
        self.F = F
        if self.F is None:
            self.F = np.random.randn(NE,M)
            for n in range(NE):
                L = np.linalg.norm(self.F[n, :])
                for i in range(M):
                    self.F[n, i] = gam * self.F[n, i] / L
        self.Omee = Omee
        if self.Omee is None:
            self.Omee = omee * np.identity(NE)
        self.Omei = Omei
        if self.Omei is None:
            self.Omei = np.tile((omei * np.identity(NI)), (1, 4))
        self.Omie = Omie
        if self.Omie is None:
            self.Omie = np.transpose(np.tile((omie * np.identity(NI)),
                                             (1, 4)))
        self.Omii = Omii
        if self.Omii is None:
            self.Omii = omii * np.identity(NI)
        self.RE = np.zeros((NE, 1))
        self.RI = np.zeros((NI, 1))
        self.VE = np.zeros((NE, Time))
        self.VI = np.zeros((NI, Time))
        self.oe = np.zeros((NE, Time))
        self.oi = np.zeros((NI, Time))
        self.re = np.zeros((NE, Time))
        self.ri = np.zeros((NI, Time))
        self.c = np.zeros((M, 1))
        self.ce = np.zeros((M, Time))
        self.ci = np.zeros((NE, Time))
        self.ip = np.zeros((2, Time))

    def runit(self, input, Time, plast=True, nip=0):
        assert input.shape[-1] >= Time
        for t in range(1, Time):
            # Write out epoch
            if t % int(Time / 10) == 0 and plast:
                print(t)

            # Randomize noise
            epev = sigv * np.random.randn(1, 1)
            epiv = sigv * np.random.randn(1, 1)
            epet = sigt * np.random.randn(1, 1)
            epit = sigt * np.random.randn(1, 1)
            # Input calculation
            # self.c[:,0] = ((input[:,t]-input[:,t-1]) / dt + lam * input[:,t-1])
            self.c[:, 0] = + 1 * input[:, t - 1]
            # print(input[:,t], (input[:,t]-input[:,t-1]) / dt)
            self.ce[:, t] = (1 - lamf * dt) * self.ce[:, t - 1] + dt * self.c[:, 0]
            self.ci[:, t] = (1 - lamei * dt) * self.ci[:, t - 1] + self.oe[:, t - 1]
            # Voltage of excitaion neurons
            ipe = self.Omee[:, :] @ self.oe[:, t - 1]
            ipi = self.Omie[:, :] @ self.oi[:, t - 1]
            ff = dt * (self.F[:, :] @ self.c[:, 0])
            for k in range(ff.shape[0]):
                if ff[k,] >= 0:
                    ipe += ff[k,]
                else:
                    ipi += ff[k,]
            self.VE[:, t] = ((1 - lam * dt) * self.VE[:, t - 1]
                            + ipe
                            + ipi
                            + epev[0, 0])
            self.ip[0, t] = ipe[nip,]
            self.ip[1, t] = ipi[nip,]
            ne = np.argmax(self.VE[:, t] - TE - epet[0, 0])
            if self.VE[ne, t] > TE and self.RE[ne, 0] < 0:
                self.oe[ne, t] = 1
                self.RE[ne, 0] = Re
                if plast == True:
                    self.F[ne, :] += eF * (a * self.ce[:, t - 1] - self.F[ne, :])
                    self.Omee[:, ne] += - eO * (b * (self.VE[:, t - 1]
                                                     + mue * self.re[:, t - 1])
                                           + self.Omee[:, ne])
                    self.Omee[ne, ne] += - eO * mue
                    self.Omei[:, ne] += - eO * (b * (self.VI[:, t - 1]
                                                     + mui * self.ri[:, t - 1])
                                           + self.Omei[:, ne])
            self.RE[:, 0] += - 1
            self.re[:, t] = (1 - lam*dt)*self.re[:, t-1] + self.oe[:, t-1]
            # Voltage of inhibitory neurons
            self.VI[:, t] = ((1 - lam * dt) * self.VI[:, t - 1]
                        + (self.Omei[:, :] @ self.oe[:, t - 1])
                        + (self.Omii[:, :] @ self.oi[:, t - 1])
                        + epiv[0, 0])
            # Checking which neuron fired and acting accordingly
            ni = np.argmax(self.VI[:, t] - TI - epit[0, 0])
            if self.VI[ni, t] > TI and self.RI[ni, 0] < 0:
                self.oi[ni, t] = 1
                self.RI[ni, 0] = Ri
                if plast == True:
                    self.Omei[ni, :] += eF * (a * self.ci[:, t - 1]
                                              - self.Omei[ni, :])
                    self.Omii[:, ni] += - eO * (b * (self.VI[:, t - 1]
                                                     + mui * self.ri[:, t - 1])
                                           + self.Omii[:, ni])
                    self.Omii[ni, ni] += - eO * mui
                    self.Omie[:, ni] += - eO * (b * (self.VE[:, t - 1]
                                                     + mue * self.re[:, t - 1])
                                           + self.Omie[:, ni])
            self.RI[:, 0] += - 1
            self.ri[:, t] = (1 - lam * dt) * self.ri[:, t - 1]\
                            + self.oi[:, t - 1]

def Fourier_input(length, n_comp = 5, freq=0.001, amp=None):
    fi = 2 * np.pi * np.random.random((M,))
    if amp == None:
        amp = [30] * n_comp
    assert len(amp) == n_comp
    x = np.zeros((M, Gam))
    for i in range(M):
        x[i, :] = np.fromfunction(lambda k, j:
                                      sum(amp[p] * (np.sin(((p + 1) * freq * j + fi[int(i)]))
                                                    + np.cos((p + 1) * freq * j + fi[int(i)]))
                                      for p in range(n_comp)), (1, length))
    return x

def Gaussian_input():
    x0 = sigi * np.random.randn(M, Gam)
    x = scipy.ndimage.gaussian_filter1d(x0, sigma=nu, axis=1)
    return x

def tuning_input(Time, fi):
    x = np.zeros((M, Time))
    for i in range(Time):
        amp = np.random.uniform(0, 60)
        x[0, i] = amp * np.cos(fi)
        x[1, i] = amp * np.sin(fi)
    return x

def create_tun(tun_resolution, tun_time, weights, ident):
    degrees = np.linspace(0, 2 * np.pi, tun_resolution)
    tun_plot = np.zeros((NE, tun_resolution))
    i = 0
    for fi in degrees:
        tuning = Netw(M, NE, NI, Time=tun_time,
                      F=weights[0], Omee=weights[1], Omei=weights[2],
                      Omie=weights[3], Omii=weights[4])
        tuning.runit(tuning_input(tun_time, fi), tun_time, plast=False)
        for n in range(NE):
            tun_plot[n, i] = np.sum(tuning.oe[n,int(tun_time / 10):]) / ((tun_time*dt)*0.9)
        i += 1
    ind = np.argpartition(np.sum(tun_plot, axis=1), -10)[-10:]
    print(ind)
    plt.figure()
    for n in ind:
        plt.plot(degrees, tun_plot[n,:], c='tab:orange')

    plt.savefig('tuning%s.png' % ident)

Pr = [S, dt, Gam, M, NE, NI]
trained = Netw(M, NE, NI, Time=Gam)
weights = [trained.F, trained.Omee, trained.Omei, trained.Omie, trained.Omii]
create_tun(tun_resolution, tun_time, weights, ident='init')

x = Fourier_input(length=Gam)
trained.runit(input=x, Time=Gam)
np.savez('outfile1', oe=trained.oe, oi=trained.oi, x=x, re=trained.re, ri=trained.ri, ip=trained.ip, Pr=Pr)
weights = [trained.F, trained.Omee, trained.Omei, trained.Omie, trained.Omii]
for q in range(9):
    del(trained)
    x = Fourier_input(length=Gam)
    trained = Netw(M, NE, NI, Time=Gam, F=weights[0], Omee=weights[1], Omei=weights[2], Omie=weights[3], Omii=weights[4])
    trained.runit(input=x, Time=Gam)
    weights = [trained.F, trained.Omee, trained.Omei, trained.Omie, trained.Omii]

np.savez('outfile2', oe=trained.oe, oi=trained.oi, x=x, re=trained.re, ri=trained.ri, ip=trained.ip, Pr=Pr)
del(trained)
create_tun(tun_resolution, tun_time, weights, ident='final')

