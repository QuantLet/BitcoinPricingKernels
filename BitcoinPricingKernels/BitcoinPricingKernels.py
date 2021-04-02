# Set US locale for datetime.datetime.strptime (Conflict with MAER/MAR)
import platform
import locale
plat = platform.system()
if plat == 'Darwin':
    print('Using Macos Locale')
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
else:
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

import pandas as pd
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np 
import time
import sys
import pdb
import pickle
import gc
import csv
import os

from statsmodels.nonparametric.kernel_regression import KernelReg as nadarajawatson
from scipy import stats
from scipy.integrate import simps
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from src.brc import BRC
from src.tee import Tee
from src.strategies import IronCondor
from rpy2 import robjects
from src.helpers import save_dict, load_dict
import rpy2.robjects.packages as rpackages


"""

@ Todo:
1) Often, for higher maturities, finding an instrument in the strategy_range will fail because the spd is not interpolated!!
2) Probability for realization of profit.

From last session:
make a film out of the hockey graphics, maybe also with moneyness on x axis
boxplot for the multiple scatter plots (especially on put side as in example)
convert .tex file to keynote
correct epsilon, insert 'small' adj before infinitesimally

Generally:
- Implied Binomial Trees
- Historical SPD
- Skewness / Kurtosis Trades
    
# Can only read profit using pickle!!

"""

def gausskernel(x):
    return (1/np.sqrt(2 * np.pi)) * np.exp(- 0.5 * x**2)

def create_kernelmatrix(regressors):
    return np.diag(gausskernel(regressors))
    
def decompose_instrument_name(_instrument_names, tradedate, round_tau_digits = 4):
    """
    Input:
        instrument names, as e.g. pandas column / series
        in this format: 'BTC-6MAR20-8750-C'
    Output:
        Pandas df consisting of:
            Decomposes name of an instrument into
            Strike K
            Maturity Date T
            Type (Call | Put)
    """
    try:
        _split = _instrument_names.str.split('-', expand = True)
        _split.columns = ['base', 'maturity', 'strike', 'is_call'] 
        
        # call == 1, put == 0 in is_call
        _split['is_call'] = _split['is_call'].replace('C', 1)
        _split['is_call'] = _split['is_call'].replace('P', 0)

        # Calculate Tau; being time to maturity
        #Error here: time data '27MAR20' does not match format '%d%b%y'
        _split['maturitystr'] = _split['maturity'].astype(str)
        # Funny Error: datetime does recognize MAR with German AE instead of A
        maturitydate        = list(map(lambda x: datetime.datetime.strptime(x, '%d%b%y') + datetime.timedelta(hours = 8), _split['maturitystr'])) # always 8 o clock
        reference_date      = tradedate.dt.date #list(map(lambda x: x.dt.date, tradedate))#tradedate.dt.date # Round to date, else the taus are all unique and the rounding creates different looking maturities
        Tdiff               = pd.Series(maturitydate).dt.date - reference_date #list(map(lambda x: x - reference_date, maturitydate))
        Tdiff               = Tdiff[:len(maturitydate)]
        sec_to_date_factor   = 60*60*24
        _Tau                = list(map(lambda x: (x.days + (x.seconds/sec_to_date_factor)) / 365, Tdiff))#Tdiff/365 #list(map(lambda x: x.days/365, Tdiff)) # else: Tdiff/365
        _split['tau']       = _Tau
        _split['tau']       = round(_split['tau'], round_tau_digits)

        # Strike must be float
        _split['strike'] =    _split['strike'].astype(float)

        # Add maturitydate for trading simulation
        _split['maturitydate_trading'] = maturitydate
        _split['days_to_maturity'] = list(map(lambda x: x.days, Tdiff))

        print('\nExtracted taus: ', _split['tau'].unique(), '\nExtracted Maturities: ',_split['maturity'].unique())

    except Exception as e:
        print('Error in Decomposition: ', e)
    finally:
        return _split

def create_regressors(_newm, _tau, m_0, t_0):
    # SET m_0 and t_0 as index variables for a loop!
    # Create matrix of regressors
    #m_0 = min(_newm)
    #t_0 = min(_tau)
    # as on page 181
    x = pd.DataFrame({  'a': np.ones(_newm.shape[0]),
                        'b':_newm - m_0, 
                        'c':(_newm - m_0)**2,
                        'd': (_tau - t_0), 
                        'e': (_tau - t_0)**2,
                        'f': (_newm - m_0)*(_tau - t_0)})
    return x

def trisurf(x, y, z, xlab, ylab, zlab, filename, blockplots):
    # This is the data we have: vola ~ tau + moneyness
    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x moneyness')
    ax.set_ylabel('y tau')
    ax.set_zlabel('z vola')
    #x, y = np.meshgrid(x, y)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=plt.cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)   
    plt.title('Empirical Vola Smile') 
    plt.savefig(filename)
    plt.draw()
    #plt.show(block = blockplots)

def plot_volasmile(m2, sigma, sigma1, sigma2, mindate, plot_ident, blockplots):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.plot(m2, sigma, label = 'Implied \nVolatility')
    plt.plot(m2, sigma1, label = '1st Derivative')
    plt.plot(m2, sigma2, label = '2nd Derivative')
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Volatility')
    plt.title('Fitted Volatility Smile on 2020-' + str(mindate.month) + '-' + str(mindate.day))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pricingkernel/plots/fitted_vola_smile' + plot_ident)
    plt.draw()
    #plt.show(block = blockplots)

def plot_spd(spd, mindate, tau, plot_ident):
    fig2 = plt.figure()
    plt.scatter(spd['x'], spd['y'])
    plt.title('SPD on 2020-' + str(mindate.month) + '-' + str(mindate.day) + ' and tau: ' + str(np.unique(tau)))
    plt.savefig('pricingkernel/plots/SPD-' + plot_ident)
    plt.draw()
    #plt.show(block = blockplots)

def gaussian_kernel(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return stats.norm.cdf(u_m) * stats.norm.cdf(u_t)

def epanechnikov(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t
    return (3/4) * (1-u_m)**2 * (3/4) * (1-u_t)**2

def k(X, d):
    out = []
    n = len(X)
    for x in X:
        if abs(x) < d:
            o = (1/(2*d)) * (1 + (1/(2*n))) * (1- (x-d)**(2*n))
        else:
            o = 0
        out.append(o)
    return out

def epanechnikov2(M, m, h_m, T, t, h_t):
    u_m = (M-m)/h_m
    u_t = (T-t)/h_t

    l = k(u_m, 0.1)
    r = k(u_t, 0.1)

    return [a*b for a,b in zip(l, r)]
    

def new_epanechnikov(X):
    out = []
    for x in X:
        if abs(x) <= 1:
            k = (3/4) * (1-x**2)
        else:
            k = 0
        out.append(k)
    return out

def smoothing_rookley(df, m, t, h_m, h_t, kernel=gaussian_kernel, extend = False):
    # M = np.array(df.M)
    # Before
    M = np.array(df.moneyness)
    y = np.array(df.mark_iv)

    # After Poly extension
    if extend:
        print('Extending Moneyness and IV in smoothing technique!')
        M, y = extend_polynomial(M, y)
    T = [df.tau.values[0]] * len(M) #np.array(df.tau)
    n = len(M)

    X1 = np.ones(n)
    X2 = M - m
    X3 = (M-m)**2
    X4 = T-t
    X5 = (T-t)**2
    X6 = X2*X4
    X = np.array([X1, X2, X3, X4, X5, X6]).T

    # the kernel lays on M_j - m, T_j - t
    #ker = new_epanechnikov(X[:,5])
    ker = kernel(M, m, h_m, T, t, h_t)
    #test = gausskernel(X[:,5])
    W = np.diag(ker)

    # Compare Kernels
    # This kernel gives too much weight on far-away deviations
    #plt.scatter(M, ker, color = 'green')
    #plt.scatter(M, X[:,5], color = 'red')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.show()

    XTW = np.dot(X.T, W)

    beta = np.linalg.pinv(np.dot(XTW, X)).dot(XTW).dot(y)

    # This is our estimated vs real iv 

    #iv_est = np.dot(X, beta)
    #plt.scatter(df.moneyness, df.mark_iv, color = 'red')
    #plt.scatter(df.moneyness, iv_est, color = 'green')
    #plt.vlines(m, ymin = 0, ymax = 1)
    #plt.title('est vs real iv and current location')
    #plt.show()
    

    return beta[0], beta[1], 2*beta[2]

def extend_polynomial(x, y):
    """
    Extend Smile, first and second derivative so that spd exists completely for large tau
    x = M_std
    y = first
    """
    polynomial_coeff=np.polyfit(x,y,2)
    xnew=np.linspace(0.6,1.4,100)
    ynew=np.poly1d(polynomial_coeff)
    #plt.plot(xnew,ynew(xnew),x,y,'o')
    #plt.title('interpolated smile')
    #plt.show()
    return xnew, ynew(xnew)

def rookley_unique_tau(df, h_m, h_t=0.01, gridsize=149, kernel='epak'):
    # gridsize is len of estimated smile

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    #tau = df.tau.iloc[0]
    M_min, M_max = min(df.moneyness), max(df.moneyness)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.moneyness), max(df.moneyness)
    M_std = np.linspace(M_std_min, M_std_max, num=num)

    # if all taus are the same
    tau_min, tau_max = min(df.tau[(df.tau > 0)]), max(df.tau) # empty sequence for tau precision = 3
    tau = np.linspace(tau_min, tau_max, gridsize)

    x = zip(M_std, tau)
    sig = np.zeros((num, 3)) # fill

    # TODO: speed up with tensor instead of loop
    for i, (m, t) in enumerate(x):
        sig[i] = smoothing_rookley(df, m, t, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1] #/ np.std(df.moneyness)
    second = sig[:, 2] #/ np.std(df.moneyness)

    #plt.plot(df.moneyness, df.mark_iv, 'ro', ms=3, alpha=0.3, color = 'green')
    #plt.title('moneyness vs iv: base for interpolation')
    #plt.show()

    #plt.plot(M_std, smile, 'ro', ms=3, alpha=0.3, color = 'red')
    #plt.plot(M_std, first)
    #plt.plot(M_std, second)
    #plt.title('no interpolation')
    #plt.show()

    #plt.plot(M_std, smile)
    #M_std, smile = extend_polynomial(M_std, smile)
    #M_std, first = extend_polynomial(M_std, first)
    #M_std, second = extend_polynomial(M_std, second)

    S_min, S_max = min(df.index_price), max(df.index_price)
    K_min, K_max = min(df.strike), max(df.strike)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, M_std, tau    


def rookley(df, h_m, h_t=0.01, gridsize=149, kernel='epak'):
    # gridsize is len of estimated smile

    if kernel=='epak':
        kernel = epanechnikov
    elif kernel=='gauss':
        kernel = gaussian_kernel
    else:
        print('kernel not know, use epanechnikov')
        kernel = epanechnikov

    num = gridsize
    #tau = df.tau.iloc[0]
    M_min, M_max = min(df.moneyness), max(df.moneyness)
    M = np.linspace(M_min, M_max, gridsize)
    M_std_min, M_std_max = min(df.moneyness), max(df.moneyness)
    M_std = np.linspace(M_std_min, M_std_max, num=num)

    # if all taus are the same
    tau_min, tau_max = min(df.tau[(df.tau > 0)]), max(df.tau)
    tau = np.linspace(tau_min, tau_max, gridsize)

    x = zip(M_std, tau)
    sig = np.zeros((num, 3)) # fill

    # TODO: speed up with tensor instead of loop
    for i, (m, t) in enumerate(x):
        sig[i] = smoothing_rookley(df, m, t, h_m, h_t, kernel)

    smile = sig[:, 0]
    first = sig[:, 1] #/ np.std(df.moneyness)
    second = sig[:, 2] #/ np.std(df.moneyness)

    #plt.plot(df.moneyness, df.mark_iv, 'ro', ms=3, alpha=0.3, color = 'green')
    #plt.plot(df.moneyness, smile, 'ro', ms=3, alpha=0.3, color = 'red')
    #plt.plot(df.moneyness, first)
    #plt.plot(df.moneyness, second)
    #plt.show()

    S_min, S_max = min(df.index_price), max(df.index_price)
    K_min, K_max = min(df.strike), max(df.strike)
    S = np.linspace(S_min, S_max, gridsize)
    K = np.linspace(K_min, K_max, gridsize)

    return smile, first, second, M, S, K, M_std, tau

def compute_spd(sigma, sigma1, sigma2, tau, s, m2, r_const, mindate, plot_ident):

    # SPDBL
    # Scalars
    #tau = np.mean(tau)
    #s = np.mean(s)
    r = r_const * len(s)
    #r = np.mean(r)
    #r = 0

    # now start spdbl estimation
    st = np.sqrt(tau)
    ert = np.exp(r * tau)
    rt = r * tau
    # error should be here in the length of m
    d1 = (np.log(m2) + tau * (r + 0.5 * (sigma ** 2))) / (sigma * st)
    d2 = d1 - (sigma * st)
    f = stats.norm.cdf(d1, 0, 1) - (stats.norm.cdf(d2, 0, 1)/(ert * m2))

    # First derivative of d1
    d11 = (1/(m2*sigma*st)) - (1/(st*(sigma**2))) * ((np.log(m2) + tau * r) * sigma1) + 0.5 * st * sigma1
    
    # First derivative of d2 term
    d21 = d11 - (st * sigma1)
    
    # Second derivative of d1 term
    d12 = -(1/(st * (m2**2) * sigma)) - sigma1/(st * m2 * (sigma**2)) + sigma2 * (0.5 * st - (np.log(m2) + rt)) / (st * sigma**2) + sigma1 * (2 * sigma1 * (np.log(m2) + rt)) / (st * sigma**3) - 1/(st * m2 * sigma**2)

    # Second derivative of d2 term
    d22 = d12 - (st * sigma2)

    f1 = (stats.norm.pdf(d1, 0, 1) * d11) + (1/ert) * ((-stats.norm.pdf(d2, 0, 1) * d21)/m2 + stats.norm.cdf(d2, 0, 1) / m2**2)
    
    # f2 = dnorm(d1, mean = 0, sd = 1) * d12 - d1 * dnorm(d1, mean = 0, sd = 1) * (d11^2) - (1/(ert * m) * dnorm(d2, mean = 0, sd = 1) * d22) + ((dnorm(d2, mean = 0, sd = 1) * d21)/(ert * m^2)) + (1/(ert * m) * d2 * dnorm(d2, mean = 0, sd = 1) * (d21^2)) - (2 * pnorm(d2, mean = 0, sd = 1)/(ert * (m^3))) + (1/(ert * (m^2)) * dnorm(d2, mean = 0, sd = 1) * d21)
    f2 = stats.norm.pdf(d1, 0, 1) * d12 - d1 * stats.norm.pdf(d1, 0, 1) * d11**2            - (1/(ert * m2) * stats.norm.pdf(d2, 0, 1) * d22) + ((stats.norm.pdf(d2, 0, 1)*d21)/(ert * m2**2))      + (1/(ert * m2) * d2 * stats.norm.pdf(d2, 0, 1)) * d21**2 -(2 * stats.norm.cdf(d2, 0, 1)/(ert * m2**3))     + (1/(ert * m2**2)) * stats.norm.pdf(d2, 0, 1) *d21
   
    # recover strike price
    x = s/m2
    c1 = -(m2**2) * f1
    c2 = s * (1/x**2) * ((m2**2) * f2 + 2 * m2 * f1)

    # Calculate the quantities of interest
    cdf = ert * c1 + 1
    fstar = ert * c2
    delta = f + s + f1/x
    gamma = 2 * f1 / x + s * f2 / (x**2)
    #print('\ndelta: ', delta, '\ngamma:', gamma)

    #plt.plot(fstar)
    #plt.show()

    spd = pd.DataFrame({'x': np.mean(s)/m2, 
                        'y': fstar,
                        'm': m2}) # Moneyness

    #plot_spd(spd, mindate, tau, plot_ident)

    return spd  


def spdbl(_df, mindate, maxdate, tau, r_const, blockplots):
    """
    computes spd according to Breeden and Litzenberger 1978
    """
    plot_ident = '2020-' + str(mindate.month) + '-' + str(mindate.day) + '-' + str(tau) + '.png'
    
    # Subset
    # Only calls, tau in [0, 0.25] and fix one day (bc looking at intra day here)
    sub = _df[(_df['is_call'] == 1) & 
         (_df['date'] > mindate) & (_df['date'] < maxdate) &
         (_df['moneyness'] >= 0.7) & (_df['moneyness'] < 1.3) &
         (_df['mark_iv'] > 0) & (_df['tau'] == tau)]# &
         #(_df['maturity'] == mat)]#(_df['tau'] > 0) & (_df['tau'] < 0.03)]
    # @Todo: tau should always be > 0, else check!

    if sub.shape[0] == 0:
        raise(ValueError('Sub is empty'))

    del sub['date']
    #sub['tau'] = round(sub['tau'], 2)
    sub['moneyness'] = round(sub['moneyness'], 3)
    sub['index_price'] = round(sub['index_price'], 2)

    sub = sub.drop_duplicates()
    #print('Only unique transactions!')
    
    print(sub.describe())

    if sub.shape[1] > 45000:
        raise(ValueError('Sub too large'))

    # Isolate vars
    sub['mark_iv'] = sub['mark_iv'] / 100
    sub['mark_iv'][(sub['mark_iv'] < 0.01)] = 0
    vola = sub['mark_iv'].astype(float)/100
    tau = sub['tau'].astype(float)
    #m = float(sub['index_price']/sub['strike'] # Spot price corrected for div divided by k ((s-d)/k); moneyness of an option; div always 0 here
    r = sub['interest_rate'].astype(float)
    s = sub['index_price'].astype(float)
    k = sub['strike'].astype(float)
    m = s / k
    div = 0
    
    # Forward price
    F = s * np.exp((r-div) * tau)
    K = m * F # k capital
    newm =(s * np.exp(-div*tau))/K

    sigma = []
    sigma1 = []
    sigma2 = []

    h = sub.shape[0] ** (-1 / 9)
    print('Choosing Bandwidth h: ', h)
    sigma, sigma1, sigma2, M, S, K, M_std, newtau = rookley_unique_tau(sub, h)#rookley(sub, h) 

    # Empirical Vola Smile
    # @Todo: Conflict with new tau being just zeros
    #trisurf(newm, tau, vola, 'moneyness', 'tau', 'vola', 'pricingkernel/plots/empirical_vola_smile_' + plot_ident, blockplots)

    # Projected Vola Smile
    plot_volasmile(M, sigma, sigma1, sigma2, mindate, plot_ident, blockplots)  

    # tau is too long here!!
    spd = compute_spd(sigma, sigma1, sigma2, newtau,  S, M, r_const, mindate, plot_ident)

    return spd, sub

def classify_options(dat):
    """
    Classify Options to prepare range trading
    """
    dat['option_type'] = ''
    dat['option_type'][(dat['moneyness'] < 0.9)]                                = 'FOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.9) & (dat['moneyness'] < 0.95)]   = 'NOTM Put'
    dat['option_type'][(dat['moneyness'] >= 0.95) & (dat['moneyness'] < 1)]     = 'ATM Put'
    dat['option_type'][(dat['moneyness'] >= 1) & (dat['moneyness'] < 1.05)]     = 'ATM Call'
    dat['option_type'][(dat['moneyness'] >= 1.05) & (dat['moneyness'] < 1.1)]   = 'NOTM Call'
    dat['option_type'][(dat['moneyness'] >= 1.1)]                               = 'FOTM Call'
    return dat

def verify_density(s):
    # Check if integral is 1
    rev = s.iloc[::-1]
    area = simps(y = rev['y'], x = rev['x'])
    print('Area under SPD: ', area)
    return area

def option_pnl(underlying_at_maturity, strikes, premium_per_strike, call = True, long = True):

    premia = sum(premium_per_strike)
    profit = []
    if call:
        for strike in strikes:
            profit.append(max(underlying_at_maturity - strike, 0))
    else:
        for strike in strikes:
            profit.append(max(strike - underlying_at_maturity, 0))
    
    p = sum(profit)
    if isinstance(p, np.ndarray):
        p = p.astype(float)[0]
    if long:
        return p, -1 * premia
    else:
        return -1 * p, premia

def hist_iv(df):
    sub = df[df.mark_iv > 0]
    sub = sub.sort_values('maturitydate_char')
    #o = sub.groupby(['maturitydate_char', 'instrument_name', 'date']).mean()['mark_iv']
    o = sub.groupby(['maturitydate_char', 'instrument_name']).mean()['mark_iv']
    return o.to_frame()

def plot_atm_iv_term_structure(df,mindate):

    # Set names
    fname = 'pricingkernel/plots/term_structure_atm_options_on_2020-' + str(mindate.month) + '-' + str(mindate.day) + '.png'
    titlename = 'IV Term Structure of ATM' + ' Options on 2020-' + str(mindate.month) + '-' + str(mindate.day)

    df.mark_iv = df.mark_iv / 100 # Rescale 
    call_sub = df[(df.mark_iv > 0) & (df.is_call == 1) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    put_sub = df[(df.mark_iv > 0) & (df.is_call == 0) & (df.moneyness <= 1.1) & (df.moneyness >= 0.9)]
    
    # nimm einfach mal n tau ueber das man die iv plotten kann
    fig = plt.figure()
    ax = plt.subplot(111) #plt.axis()
    call_sub.groupby(["tau"])['mark_iv'].mean().plot(label = 'Call IV')
    put_sub.groupby(["tau"])['mark_iv'].mean().plot(label = 'Put IV')

    ax.set_ylabel('Implied Volatility')
    ax.set_xlabel('Tau')
    plt.title(titlename)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #  Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(fname)
    #plt.show(block = False)
    plt.draw()

    return None

def real_vola(df):
    o = df.groupby(['maturitydate_char']).std()['index_price'] * np.sqrt(252)
    return o.to_frame()

def iv_vs_real_vola(realized_vola, historical_iv):
    plt.plot(realized_vola)
    plt.plot(hist_iv)

if __name__ == '__main__':

    # Redirect Output to log file
    f = open('out/out.txt', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    profit = {}
    errors = []
    realized_vola = []
    historical_iv = []

    # Collect Trading Summary
    trading = pd.DataFrame()

    # Initiate R Objects
    base = rpackages.importr("base") 
    r = robjects.r
    r.source('physicaldensity.R')
    #a, b = r.gen_physical_density(10, 0.01, 10000)

    # Read final prices
    tdat = pd.read_csv('data/BTC_USD_Quandl.csv')#pd.read_csv('data/BTCUSDT.csv')
    tdat = tdat.sort_values('Date', ascending = True) # ascending

    # Initiate BRC instance to query data. First and Last day are stored.
    brc = BRC()
    curr_day = brc.first_day #datetime.datetime(2020, 3, 21, 0, 0, 0)#brc.first_day #datetime.datetime(2020, 9, 6, 0, 0, 0)#
    print('overwriting start date')

    # Simulation Methods
    simmethods = ['SVCJ', 'Milstein'] # ['brownian', 'svcj']

    # Tau-maturitydate combination dict
    tau_maturitydate = {}

    while curr_day < brc.last_day:
        
        try:
            # make sure days are properly set
            curr_day_starttime = curr_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
            curr_day_endtime = curr_day.replace(hour = 23, minute = 59, second = 59, microsecond = 0)

            # Debug
            #curr_day_starttime = datetime.datetime(2020, 4, 5, 0, 0, 0)
            #curr_day_endtime = datetime.datetime(2020, 4, 5, 23, 59, 59)

            print('Starting Simulation from ', curr_day_starttime, ' to ', curr_day_endtime)
            
            dat, int_rate = brc._run(starttime = curr_day_starttime,
                                        endtime   = curr_day_endtime, 
                                        download_interest_rates = True,
                                        download_historical_iv = False)
            dat = pd.DataFrame(dat)

            assert(dat.shape[0] != 0)
    
            # Convert dates, utc
            dat['date'] = list(map(lambda x: datetime.datetime.fromtimestamp(x/1000), dat['timestamp']))
            dat_params  = decompose_instrument_name(dat['instrument_name'], dat['date'])
            dat         = dat.join(dat_params)
            
            dat['interest_rate'] = 0 # assumption here!
            dat['index_price']   = dat['index_price'].astype(float)

            # To check Results after trading 
            dates                       = dat['date']
            dat['strdates']             = dates.dt.strftime('%Y-%m-%d') 
            maturitydates               = dat['maturitydate_trading']
            dat['maturitydate_char']    = maturitydates.dt.strftime('%Y-%m-%d')

            # Calculate mean instrument price
            bid_instrument_price = dat['best_bid_price'] * dat['underlying_price'] 
            ask_instrument_price = dat['best_ask_price'] * dat['underlying_price']
            dat['instrument_price'] = (bid_instrument_price + ask_instrument_price) / 2

            # Prepare for moneyness domain restriction (0.8 < m < 1.2)
            dat['moneyness']    = dat['index_price'] / dat['strike']
            df                  = dat[['index_price', 'strike', 'interest_rate', 'maturity', 'is_call', 'tau', 'mark_iv', 'date', 'moneyness', 'instrument_name', 'days_to_maturity', 'maturitydate_char', 'timestamp', 'underlying_price', 'instrument_price']]    
            
            # Select Tau and Maturity (Tau is rounded, prevent mix up!)
            unique_taus = df['tau'].unique()
            unique_maturities = df['maturity'].unique()
            
            # Save Tau-Maturitydate combination
            #tau_maturitydate[curr_day.strftime('%Y-%m-%d')] = (unique_taus,)
            
            unique_taus.sort()
            unique_taus = unique_taus[(unique_taus > 0) & (unique_taus < 0.25)]
            print('\nunique taus: ', unique_taus,
                    '\nunique maturities: ', unique_maturities)

        except Exception as e:
            print('Download or Processing failed!\n')
            print(e)  

        try:
            # As taus are ascending, once we do not find one instrument for a specific taus it is unlikely to find one for the following
            # as the SPDs degenerate with higher taus.
            for tau in unique_taus:

                s, sub  = spdbl(df, curr_day_starttime, curr_day_endtime, tau, int_rate, blockplots = True)
                area = verify_density(s)

                # Only continue if spd fulfills conditions of a density
                if abs(area) - 1 < 0.05:

                    # @Todo: Check if startprice is ok!!
                    observation_price = sub['index_price'].tail(1) # last one on day which we observed
                    
                    # need at least one day for the physical density, which is fixed in there!
                    time_to_maturity_in_days = sub.days_to_maturity.unique()[0]
                    rdate = base.as_Date(curr_day.strftime('%Y-%m-%d'))
                    rdate_f = base.format(rdate, '%Y-%m-%d')

                    # Todo: Compare svcj results to old hd results
                    for simmethod in simmethods:
                        
                        try:
                            gdom, gstar = r.gen_physical_density(robjects.FloatVector([tau]), 
                                                                robjects.FloatVector([int_rate]), 
                                                                robjects.FloatVector([observation_price]),
                                                                robjects.FloatVector([time_to_maturity_in_days]),
                                                                rdate_f,
                                                                simmethod)
                            
                            strategy = IronCondor(s, gdom, gstar, tau, curr_day.strftime('%Y-%m-%d'), df, tdat, time_to_maturity_in_days, simmethod)   
                            strategy.run()   
                            
                            profitkey = str(curr_day.strftime('%Y-%m-%d')) + str('-') + str(tau)   
                            profit_fname = "out/profit" +  str(simmethod) + ".txt"
                            if os.path.exists(profit_fname):
                                pf = 'a' # append if already exists
                            else:
                                pf = 'wb' # make a new file if not
                            with open(profit_fname, pf) as fp:   #Pickling
                                profit[profitkey] = strategy.overall_payoff
                                pickle.dump(profit, fp)
                        except:
                            print('Sim failed, proceeding with the next one... - ', simmethod)
                else:
                    print('SPD is not a valid density, proceeding with the next one')

        except Exception as e:
            print('error: ', e)
            errors.append(e)
            with open("out/errors.txt", "wb") as fp:   #Pickling
                pickle.dump(errors, fp)

        finally:
            curr_day += datetime.timedelta(1)
