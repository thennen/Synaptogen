"""
A quick translation of Synaptogen into Python

The unvectorized version would be miserably slow, so I only translated the vectorized version,
which I still expect to be slower than the original Julia version.

TODO: replace unicode because it's not so normal to use it in python (e.g. Γ → Gamma)

Note: numpy is row-major by default. Julia is column-major. I did not optimize by transposing things.
"""

import numpy as np
from numpy import polyval, sqrt, float32, abs
from dataclasses import dataclass
import os
import json
from functools import partial

def array32(x):
    return np.array(x, dtype=np.float32)

def zeros32(n):
    return np.zeros(n, dtype=np.float32)

def empty32(n):
    return np.empty(n, dtype=np.float32)

rng = np.random.default_rng()
randn = partial(rng.standard_normal, dtype=np.float32)
rand = partial(rng.random, dtype=np.float32)

Uread = float32(0.2)
e = float32(1.602176634e-19)
kBT = float32(1.380649e-23 * 300)
#σClip = float32(3.5)
iHRS, iUS, iLRS, iUR = 0, 1, 2, 3

moduledir = os.path.dirname(__file__)
default_param_fp = os.path.join(moduledir, "default_params.json")

def r(R, G_HHRS, G_LLRS):
    return (G_LLRS - 1/R) / (G_LLRS - G_HHRS)

def Γ(γ, x):
    y = np.zeros_like(x)
    for γv in γ.T:
        y = y * x + γv[:, np.newaxis] # for broadcasting to work..
    return y

def Ψ(μ, σ, x):
    y = μ + σ * x
    y[iHRS, :] = 10 ** y[iHRS, :]
    return y

def Ireset(a, c, U, η, Umax):
    return a * abs(Umax - U)**η + c


@dataclass
class CellParams : 
    Umax : float           # Maximum voltage applied during the experiment.  defines the point where HRS is reached.
    U0 : float             # Voltage used in the definition of resistance R = U₀ / I(U₀)
    η : float              # Sets the curvature of the reset transition
    nfeatures : int        # Number of features for the VAR model
    p : int                # Order of the VAR model (how many cycles of history remembered)
    K : int                # How many components in the GMM for modeling device-to-device distribution
    γdeg : int             # Degree of the non-linear transformation polynomials 
    G_HHRS : float         # Conductance of the HHRS 
    G_LLRS : float         # Conductance of the LLRS
    HHRSdeg : int          # Degree of the HHRS polynomial
    LLRSdeg : int          # Degree of the LLRS polynomial
    HHRS : np.ndarray      # HHRS coefficients.  Not a vector because of polyval shenanigans
    LLRS : np.ndarray      # LLRS coefficients
    γ : np.ndarray         # non-linear transformation coefficients
    wk : np.ndarray        # weights of the GMM components
    μDtD : np.ndarray      # mean vectors for the GMM
    LDtD : np.ndarray      # Cholesky decomposition of covariance matrices for the GMM (lower triangular)
    VAR : np.ndarray       # VAR coefficients, including A and B


@dataclass
class CellArray : 
    M : int                        # scalar      (number of cells)
    Xhat : np.ndarray              # 4(p+1) × M  (feature history and εₙ for all cells)
    #Xbuf : np.ndarray             # 4(p+1) × M  (buffer to improve the speed of the partial shift operation)
    x : np.ndarray                 # 4 × M       (generated normal feature vectors ̂x*ₙ, basically also a buffer)
    σ : np.ndarray                 # 4 × M       (CtC scale vectors)
    μ : np.ndarray                 # 4 × M       (CtC offset vectors)
    y : np.ndarray                 # 4 × M       (scaled feature vector)
    r : np.ndarray                 # M × 1       (device state variables)
    n : np.ndarray                 # M × 1       (cycle numbers)
    k : np.ndarray                 # M × 1       (GMM component, not strictly necessary to store)
    UR : np.ndarray                # M × 1       (voltage thresholds for reset switching)
    #Umax : np.ndarray              # M × 1       (Vector of Umax, probably all the same value, just for vectorization of polyval)
    resetCoefs : np.ndarray        # M × 2       (polynomial coefficients for reset transitions)
    Iread : np.ndarray             # M × 1       (readout buffer)
    inHRS : np.ndarray             # Using BitVector does not save much memory and isn't faster either.
    inLRS : np.ndarray
    setMask : np.ndarray
    resetMask : np.ndarray
    fullResetMask : np.ndarray
    partialResetMask : np.ndarray
    resetCoefsCalcMask : np.ndarray
    drawVARMask : np.ndarray
    params : CellParams


def load_params(param_fp:str=default_param_fp, p:int=10):
    ### load model parameters from file
    with open(param_fp, 'r', encoding='UTF-8') as f:
        json_params = json.load(f)

    γ = array32(json_params['γ'])
    nfeatures, γdeg = γ.shape

    VAR_keys = [k for k in json_params.keys() if k.startswith('VAR')]
    available_orders = np.sort([int(k.split('_')[-1]) for k in VAR_keys])
    q = available_orders[np.where(available_orders >= p)[0][0]]
    VAR = array32(json_params[f'VAR_{q:>03}'])[:, :(p+1)*nfeatures]
    Umax = float32(json_params['Umax'])
    U0 = float32(json_params['U₀'])
    η = float32(json_params['η'])
    HHRS = array32(json_params['HHRS'])
    LLRS = array32(json_params['LLRS'])
    G_HHRS = polyval(HHRS, U0)/U0
    G_LLRS = polyval(LLRS, U0)/U0
    HHRSdeg = HHRS.shape[0]
    LLRSdeg = LLRS.shape[0]
    wk = array32(json_params['wk'])
    K = wk.shape[0]
    LDtD = np.moveaxis(array32(json_params['LDtD']).reshape(2*nfeatures, K, 2*nfeatures), 1, 2)
    μDtD = array32(json_params['μDtD'])

    return CellParams(Umax, U0, η, nfeatures, p, K, γdeg, G_HHRS, G_LLRS, HHRSdeg, LLRSdeg,
                      HHRS, LLRS, γ, wk, μDtD, LDtD, VAR)

default_params = load_params(default_param_fp)

def CellArrayCPU(M, params:CellParams=default_params):
    """
    Initialize an array of cells
    Name matches Julia version.
    """
    # No struct unpacking in python.
    VAR = params.VAR
    nfeatures = params.nfeatures
    p = params.p
    γ = params.γ
    wk = params.wk
    μDtD = params.μDtD
    LDtD = params.LDtD
    G_HHRS = params.G_HHRS
    G_LLRS = params.G_LLRS
    #Umax = params.Umax

    Xhat = zeros32((nfeatures * (p + 1), M))
    #Xhat[:nfeatures, :] = randn((nfeatures, M))
    randn((nfeatures, M), out=Xhat[:nfeatures, :])
    x = VAR @ Xhat
    Xhat[-nfeatures:, :] = x
    cs = np.cumsum(wk) / np.sum(wk)
    k = np.searchsorted(cs, rand(M))
    μσCtC = empty32((nfeatures * 2, M))
    for kk in range(len(wk)):
        mask = k == kk
        Mk = np.sum(mask)
        μσCtC[:, mask] = μDtD[:, kk, np.newaxis] + LDtD[:,:,kk] @ randn((nfeatures * 2, Mk))
    μCtC = μσCtC[:nfeatures, :]
    σCtC = μσCtC[nfeatures:, :]
    y = Ψ(μCtC, σCtC, Γ(γ, x))
    resetCoefs = empty32((2,M))
    r0 = r(y[iHRS, :], G_HHRS, G_LLRS)
    n = np.zeros(M, dtype=np.int64)
    UR = y[iUR, :]
    #Umax = np.repeat(Umax, M)
    Iread = zeros32(M)
    inHRS = np.zeros(M, dtype=bool)
    inLRS = np.zeros(M, dtype=bool)
    setMask = np.zeros(M, dtype=bool)
    resetMask = np.zeros(M, dtype=bool)
    fullResetMask = np.zeros(M, dtype=bool)
    partialResetMask = np.zeros(M, dtype=bool)
    resetCoefsCalcMask = np.zeros(M, dtype=bool)
    drawVARMask = np.zeros(M, dtype=bool)

    return CellArray(M, Xhat, x, σCtC, μCtC, y, r0, n, k, UR, resetCoefs, Iread,
                     inHRS, inLRS, setMask, resetMask, fullResetMask, partialResetMask,
                     resetCoefsCalcMask, drawVARMask, params)


def HRS(c:CellArray):
    return c.y[iHRS]

def LRS(c:CellArray):
    return c.y[iLRS]

def US(c:CellArray):
    return c.y[iUS]

def UR(c:CellArray):
    return c.y[iUR]

def VAR_sample(c:CellArray):
    """
    Draw the next VAR terms, updating the history matrix (c.Xhat)
    involves a shift operation for the subset of columns corresponding to drawVARMask == true
    """
    nfeatures = c.params.nfeatures
    VAR = c.params.VAR
    mask = c.drawVARMask
    randn((nfeatures, c.M), out=c.Xhat[:nfeatures, :])
    x = VAR @ c.Xhat
    c.Xhat[nfeatures:-nfeatures, mask] = c.Xhat[2*nfeatures:, mask]
    c.Xhat[-nfeatures:, mask] = x[:, mask]

def rIU(I, U, HHRS, LLRS):
    IHHRS_U = polyval(HHRS, U)
    ILLRS_U = polyval(LLRS, U)
    return (I - ILLRS_U) / (IHHRS_U - ILLRS_U)

def Imix(r, U, HHRS, LLRS):
    return (1 - r) * polyval(LLRS, U) + r * polyval(HHRS, U)

def I(c:CellArray, U):
    return Imix(c.r, U, c.params.HHRS, c.params.LLRS)

def resetCoefs(x1, x2, y1, y2, η):
    a = (y1 - y2) / abs(x2 - x1)**η
    c = y2
    return np.vstack((a, c))


def applyVoltage(c:CellArray, Ua):
    """
    Apply voltages from array U to the corresponding cell in the CellArray
    if U > UR or if U ≤ US, cell states will be modified
    """
    if type(Ua) is np.ndarray:
        Ua = Ua.astype(float32)
    else:
        Ua = np.repeat(float32(Ua), c.M)

    Umax = c.params.Umax
    G_HHRS = c.params.G_HHRS
    G_LLRS = c.params.G_LLRS
    HHRS = c.params.HHRS
    LLRS = c.params.LLRS
    γ = c.params.γ
    η = c.params.η
    nfeatures = c.params.nfeatures

    c.setMask = ~c.inLRS & (Ua <= US(c))
    c.resetMask = ~c.inHRS & (Ua > c.UR)
    c.fullResetMask = c.resetMask & (Ua >= Umax)
    c.partialResetMask = c.resetMask & (Ua < Umax)
    c.drawVARMask = c.inLRS & c.resetMask
    c.resetCoefsCalcMask = c.drawVARMask & ~c.fullResetMask

    if any(c.setMask):
        c.r[c.setMask] = r(LRS(c)[c.setMask], G_HHRS, G_LLRS)
        c.inLRS |= c.setMask
        c.inHRS = c.inHRS & ~c.setMask
        c.UR[c.setMask] = UR(c)[c.setMask]

    if any(c.drawVARMask):
        VAR_sample(c)
        c.n += c.drawVARMask
        c.y = Ψ(c.μ, c.σ, Γ(γ, c.Xhat[-nfeatures:, :]))

    if any(c.resetCoefsCalcMask):
        x1 = c.UR[c.resetCoefsCalcMask]
        x2 = Umax
        y1 = Imix(c.r[c.resetCoefsCalcMask], x1, HHRS, LLRS)
        r_HRS = r(HRS(c)[c.resetCoefsCalcMask], G_HHRS, G_LLRS)
        y2 = Imix(r_HRS, x2, HHRS, LLRS)
        c.resetCoefs[:, c.resetCoefsCalcMask] = resetCoefs(x1, x2, y1, y2, η)

    if any(c.resetMask):
        c.inLRS = c.inLRS & ~c.resetMask
        c.UR[c.resetMask] = Ua[c.resetMask]

    if any(c.partialResetMask):
        Itrans = Ireset(c.resetCoefs[0, c.partialResetMask], c.resetCoefs[1, c.partialResetMask], Ua[c.partialResetMask], η, Umax)
        c.r[c.partialResetMask] = rIU(Itrans, Ua[c.partialResetMask], HHRS, LLRS)

    if any(c.fullResetMask):
        c.inHRS |= c.fullResetMask
        c.r[c.fullResetMask] = r(HRS(c)[c.fullResetMask], G_HHRS, G_LLRS)

    return c

def Iread(c:CellArray, U=Uread, nbits=4, Imin=1e-6, Imax=5e-5, BW=1e8):
    """
    Return the current at Ureadout for the current cell state
    """
    if type(U) is np.ndarray:
        U = U.astype(float32)
    else:
        U = float32(U)

    Imin = float32(Imin)
    Imax = float32(Imax)
    BW = float32(BW)
    Inoiseless = I(c, U)
    johnson = 4*kBT*BW*np.abs(Inoiseless/U)
    shot = 2*e*np.abs(Inoiseless)*BW
    σ_total = np.sqrt(johnson + shot)
    Irange = Imax - Imin
    nlevels = 2**nbits
    q = Irange / nlevels
    randn(c.M, out=c.Iread)
    c.Iread = Inoiseless + c.Iread * σ_total
    c.Iread = np.clip(np.round((c.Iread - Imin) / q), 0, nlevels) * q + Imin

    return c.Iread


def test(M=2**5, N=2**6):
    """
    Continuous IV sweeping of M devices for N cycles.
    Gives visual indication whether things are working.
    Every device gets the same voltage, though this is not necessary
    """
    pts = 200 # per cycle, make it divisible by 4
    Umin = -1.5
    Umax = 1.5
    linspace32 = partial(np.linspace, dtype=float32)
    Usweep = np.concatenate((linspace32(0, Umin, pts//4),
                             linspace32(Umin, Umax, pts//2),
                             linspace32(Umax, 0, pts//4)))

    cells = CellArrayCPU(M)

    nfeatures = cells.params.nfeatures

    Umat = np.tile(Usweep[:, np.newaxis, np.newaxis], (M, N))
    Imat = np.empty_like(Umat)
    ymat = empty32((nfeatures, M, N))

    for n in range(N):
        ymat[:, :, n] = cells.y 
        for i in range(pts):
            c = applyVoltage(cells, Umat[i, :, n])
            # no noise (otherwise use Iread)
            Imat[i, :, n] = I(cells, Umat[i, :, n])
            #I[i, :, n] = Iread(c, Umat[i,1], 8, -200f-6, 200f-6, 1f9)
    
    from matplotlib import pyplot as plt

    sqM = int(np.floor(sqrt(M)))
    sqM = min(sqM, 10)
    fig, axs = plt.subplots(sqM, sqM, sharex=True, sharey=True, layout="tight",
                            gridspec_kw=dict(wspace=0, hspace=0))
    colors = plt.cm.jet(np.linspace(0, 1, max(N, 2)))
    lw = .5
    alpha = .7
    for n in range(N):
        for i in range(sqM):
            for j in range(sqM):
                m = sqM * i + j
                Iplot = 1e6 * Imat[:, m, n]
                Uplot = Umat[:, m, n]
                axs[i,j].plot(Uplot, Iplot, lw=lw, alpha=alpha, color=colors[n])
                if j == 0:
                    axs[i,j].set_ylabel("I [μA]")
                if i == sqM - 1:
                    axs[i,j].set_xlabel("U [V]")

    axs[0,0].set_xlim(-1.5, 1.5)
    axs[0,0].set_ylim(-195, 195)

    # Scatterplot of the generated features
    fig, axs = plt.subplots(nrows=nfeatures, sharex=True, figsize=(12, 4))
    colors = [f'C{i}' for i in range(nfeatures)]
    for i in range(nfeatures):
        axs[i].scatter(np.arange(N*M), ymat[i,:,:].T, c=colors[i], s=3, edgecolor="none")
        axs[i].set_ylabel(f"Feature {i}")
        for m in range(M):
            axs[i].axvline(m*N, color="black")
    axs[0].set_yscale("log")
    axs[1].set_xlim(0, N*M)
    axs[-1].set_xlabel("Cycle/Device")
    fig.align_ylabels()

    return Umat, Imat, ymat