# Python class used to group all the simulations for the temperature modeling

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from jitcode import y as yo, jitcode
# from jitcdde import y as yd, jitcdde
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

R = 8.314472 # gas constant
K0 = 273.15

# function that is used by all the models
def getperiodamplitude(tvals, xvals, threshold=1e-3):
    """gives the period and amplitude of the oscillations. Returns zero if no oscillations are seen.
       Oscillations with amplitude less than threshold are set to zero.
       We start from the last value. """

    def findnextmin(startindex):
        j=startindex
        while xvals[j] >= xvals[j+1]:
            if j+2>=len(xvals):
                #reach the end of the array
                return -1
            j +=1
        return j
    def findnextmax(startindex):
        i = startindex
        while xvals[i] <= xvals[i+1]:
            if i+2>=len(xvals):
                #reach the end of the array
                return -1
            i +=1
        return i
    xvals = xvals[-1::-1] #reverse

    message = '' #say why the period was zero
    if xvals[0] < xvals[1]: #start on an increasing piece
        nma = findnextmax(0)
        nmi = findnextmin(nma+1)
        nma2 = findnextmax(nmi+1)

        per = tvals[nma2] - tvals[nma]
        amp = xvals[nma] - xvals[nmi]
        if nma==-1 or nmi==-1 or nma2==-1:
            per, amp=0,0
            message ='hit an end'

        if abs(xvals[nma]-xvals[nma2])/amp>threshold:
            per,amp=0,0
            message ='maxima {} and {} not close enough'.format(xvals[nma], xvals[nma2])
    else:
        nmi = findnextmin(0)
        nma = findnextmax(nmi+1)
        nmi2 = findnextmin(nma+1)
        per = tvals[nmi2] - tvals[nmi]
        amp = xvals[nma] - xvals[nmi]
        if nma==-1 or nmi==-1 or nmi2==-1:
            per, amp=0,0
            message ='hit an end'
        if abs(xvals[nmi]-xvals[nmi2])/amp>threshold:
            per,amp=0,0
            message ='minima {} and {} not close enough'.format(xvals[nmi], xvals[nmi2])
    per = abs(per)
    amp = abs(amp)
    return per,amp,message

def getperiodamplitude_robust(tvals, xvals, threshold=1e-3, thrper = 0.1, number=1):
    # also based on extrema, but exclude extrema which are too close to one another
    # too close is defined by thrper
    # new 29/10/2020: returns the last X periods and amplitudes where X is given by the number argument
    # number=1 gives old behavior of the function

    xvals = xvals[-1::-1] #reverse
    message=''
    mm = []
    i = 1
    # iterate over the array. Stop if we found 2*number+1 'real' extrema
    # note: we go until 2n+2 extrema, because if we stop at 2n+1, it is possible that
    # the last one is the first of a couple of 'spurious' extrema.
    # we then use the first 2n+1 of this list to obtain periods and amplitudes
    while i < len(tvals)-2 and len(mm) < 2*number+2:
        if xvals[i] < xvals[i-1] and xvals[i] < xvals[i+1]:
            c = (tvals[i],xvals[i], 'min') # candidate
        elif xvals[i] > xvals[i-1] and xvals[i] > xvals[i+1]:
            c = (tvals[i],xvals[i], 'max')
        else:
            c = 0
        if c != 0: # there is a candidate
            if len(mm) == 0:
                mm.append(c) # add anyway because it is the first extremum
            elif abs(c[0]- mm[-1][0]) > thrper:
                mm.append(c)
            else:
                # remove previous one as well as not adding this one
                del mm[-1]
        i+=1
    # at this point, check if we have at least three real extrema
    if len(mm) < 4: # means not enough extrema found to obtain at least one period/amplitude
        message = 'not enough extrema found'
        per,amp=0,0
    else:
        if number==1: # do as before: return one period and amplitude but check for threshold
            per = mm[2][0]-mm[0][0]
            amp = abs(mm[1][1] - mm[0][1])
            if abs(mm[2][1]-mm[0][1])/amp>threshold:
                per,amp=0,0
                message ='{} {} and {} {} not close enough'.format(mm[0][2],mm[0][1],mm[2][2],mm[2][1])
                message += '\n m = ' + str(mm)

        else: # return array of periods and amplitudes (differences)
            # if we have 2*number+2 entries in the list mm, we can
            # extract 2*number - 1 periods and 2*number amplitudes

            #note first check if length of mm is even, if not, remove last entry
            if len(mm) %2 == 1:
                del mm[-1]
            # effective periods we have:
            N = len(mm)//2-1
            # extract the numbers
            per = [mm[i+2][0]-mm[i][0] for i in range(2*N-1)]
            amp = [abs(mm[i+1][1]-mm[i][1]) for i in range(2*N)]
    return per,amp,message
    
def getperiodamplitude_scipyfindpeaks(tvals, xvals, threshold=1e-3):
    # use scipy find peaks method here
    message = ''
    max_inds = find_peaks(xvals)[0]
    min_inds = find_peaks(-xvals)[0]
    if len(max_inds) + len(min_inds) < 3:
        # not enough to define a period
        per, amp = 0,0
        message='not enough extrema found'
    else:
        amp = xvals[max_inds[-1]] - xvals[min_inds[-1]]
        if max_inds[-1]>min_inds[-1]:
            # end in maximum
            per = tvals[max_inds[-1]] - tvals[max_inds[-2]]
            if abs(xvals[max_inds[-1]]-xvals[max_inds[-2]])/amp > threshold:
                per, amp = 0,0
                message = 'minima not close enough'
        else:
            # end in minimum
            per = tvals[min_inds[-1]] - tvals[min_inds[-2]]
            if abs(xvals[min_inds[-1]]-xvals[min_inds[-2]])/amp > threshold:
                per, amp = 0,0
                message = 'minima not close enough'
    return per, amp, message

def classifyosc(periods, amplitudes, thr=0.01):
    """ given a list of periods and amplitudes, decide if the system oscillates """
    periods = np.array(periods)
    amplitudes =np.array(amplitudes)

    try:
        ddd =  iter(periods)
    except:
        # periods is not iterable, means it is zero
        # means detection gave nothing (no extrema or only three and large difference)
        return ('ss0', 0.)
    if len(periods)==1:
        if abs(amplitudes[1] - amplitudes[0])/amplitudes[0] < thr and np.min(amplitudes) > thr:
            return ('osc1', periods[0])
        else:
            return ('ss1', 0.)
    else:
        cvp = np.std(periods)/np.mean(periods)
        cva = np.std(amplitudes) / np.mean(amplitudes)

        if cvp < thr and cva < thr and np.min(amplitudes) > thr:
            return ('osc' + str(len(periods)), np.mean(periods))
        else:
            if all(np.diff(amplitudes)>0):
                # this means decreasing amplitude because we start from the back
                return ('dampedosc' + str(len(periods)), np.mean(periods))
            else:
                return ('irregular' + str(len(periods)), np.mean(periods))



# function for obtaining the parameters for the double exponential function
# for rates
# given E1 and k0 at T0, and E2 and the temperature at which the extremum happens
# E1 >0 E2<0
# temperatures in Kelvin
def getb1b2(E1,E2,k0,T0,Tm):
    D = E1*np.exp(1./R*(E1/Tm+E2/T0)) - E2*np.exp(1./R*(E2/Tm+E1/T0))
    eb1 = 1./D/k0*(-E2)*np.exp(E2/R/Tm)
    eb2 = 1./D/k0*E1*np.exp(E1/R/Tm)
    b1 = np.log(eb1)
    b2 = np.log(eb2)
    return b1,b2

class DelayModel(object):
    """ Class to simulate and analyze the delay model. We use the variable v
    for Cdk1."""

    def __init__(self, **params):
        # set the parameters
        self.ks = params['ks']
        self.bdeg = params['bdeg']
        self.K = params['K']
        self.tau = params['tau']
        self.m = params['m'] # Hill exponent, this can be infinity

    def f(self, v):
        # hill function, unless m is infinity.
        if self.m=='inf':
            return 0.5*(np.sign(v-self.K) +1)
        else:
            return v**self.m / (self.K**self.m + v**self.m)

    def solve(self, T, dt, v0=0, usejac=False):
        # solve the system. Use Euler method. Note we should probably do something better at some point
        # usejac is dummy
        steps = int(T/dt)
        tauind = int(self.tau/dt) # how many indices for one tau
        vv = np.zeros(tauind+steps)
        vv[:tauind] = v0 # use constant history
        for i in range(tauind,tauind+steps):
            vd = vv[i-tauind]
            dv = self.ks - self.bdeg*vv[i-1]*self.f(vd)
            vn = vv[i-1] + dv*dt
            vv[i] = vn

        # keep variables
        self.T = T
        self.dt=dt
        self.tv=np.linspace(-self.tau, T, len(vv))
        self.vv =vv

        # also keep APC/C activity, useful for plotting
        self.uv = self.f(np.roll(vv,tauind))

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr,thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a


class BistableModel(object):
    """ Class to simulate and analyze the bistable model. Note, we need a function chi for the bistability.
    The entire function is an argument, instead of determining the form in this class.
    The variable v denotes Cdk1, the variable u (fast variable) represents APC/C activity"""

    def __init__(self, **params):
        # set the parameters
        self.ks = params['ks']
        self.bdeg = params['bdeg']
        self.K = params['K']
        self.chi = params['chi'] # this is a function on the interval [0,1]
        self.m = params['m'] # Hill exponent
        self.epsilon=params['epsilon'] # speed of APC/C activation, normally small

        # exponential decrease or not? (testing purposes)
        if 'exp' in params.keys():
            self.exp=params['exp']
        else:
            self.exp=1
            
        # modifier for the Hill function ie to model a non-horizontal upper part?
        if 'modif' in params.keys():
            self.modif = params['modif'] # this is a function
        else:
            self.modif = lambda x: 1

    def g(self, u, v):
        # modified Hill function
        return v**self.m / ( (self.K*self.chi(u))**self.m + v**self.m) * self.modif(v)

    def solve(self, T, dt, y0=(0,0), usejac=False):
        # solve the system. Use scipy's built-in solver here.
        # NOTE Jacobian not implemented
        def dxdt(t,y):
            u=y[0] # apc/c
            v=y[1] # cdk1
            return [self.epsilon**(-1)*(self.g(u,v) - u), self.ks-self.bdeg*u*v**self.exp]
        tv = np.linspace(0, T, int(T/dt))
        sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

        # keep variables
        self.T = T
        self.dt=dt
        self.tv=tv
        self.uv = sol.y[0,:]
        self.vv = sol.y[1,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr, thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a

class BistableModelNoHill(object):
    """ Class to simulate and analyze the bistable model without the Hill function as backbone for the bistable curve.
    Provide function chi which defines the shape of the bistable curve.
    v is slow variable, u fast.
    Also need to provide a function 'apc', which says how apc activity depends on cdk. 
    APC activity is used in the cyclin equation (degradation)"""

    def __init__(self, **params):
        # set the parameters
        self.ks = params['ks']
        self.bdeg = params['bdeg']
        self.chi = params['chi'] # this is a function on [0, infty] here
        self.apc = params['apc']
        self.m = params['m'] # Exponent
        self.epsilon=params['epsilon'] # speed of APC/C activation, normally small


    def g(self, u, v):
        # defines the bistable part
        return (v/self.chi(u))**self.m

    def solve(self, T, dt, y0=(0,0), usejac=False):
        # solve the system. Use scipy's built-in solver here.
        # NOTE Jacobian not implemented
        def dxdt(t,y):
            u=y[0] 
            v=y[1]
            return [self.epsilon**(-1)*(self.g(u,v) - u), self.ks-self.bdeg*v*self.apc(u)]
        tv = np.linspace(0, T, int(T/dt))
        sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

        # keep variables
        self.T = T
        self.dt=dt
        self.tv=tv
        self.uv = sol.y[0,:]
        self.vv = sol.y[1,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr, thrper, number)
        elif method=='scipy':
            p,a,message = getperiodamplitude_scipyfindpeaks(self.tv, self.vv, thr)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a
        
    def getfinalperiodtimeseries(self):
        """call this after getperiod. Saves the time series of the last period (between last maxima of v).
        Note that this uses find_peaks from scipy rather than our own peak detection. """
        if self.period == 0:
            self.t0v = 0
            self.u0v = 0
            self.v0v = 0 
        else:
            peak_inds = find_peaks(self.vv, prominence=self.amplitude/2)[0]
            if len(peak_inds)<2:
                # no two big peaks -- means probably something wrong
                self.period=0
                self.amplitude=0
                self.t0v = 0
                self.u0v = 0
                self.v0v = 0
            else:
                self.t0v = self.tv[peak_inds[-2]:peak_inds[-1]+1].copy()
                self.t0v -= self.t0v[0] # set time zero at start
                self.u0v = self.uv[peak_inds[-2]:peak_inds[-1]+1].copy()
                self.v0v = self.vv[peak_inds[-2]:peak_inds[-1]+1].copy()
        
    def getfouriercoeff(self,N):
        """ determine the complex Fourier coefficients of both variables. 
        Call this only after having solved and detected period.
        N is the max harmonics. It means we compute 2N+1 coefficients.
        ie u(t) = sum_{-N}^N c_i exp(i 2 pi / T t)
        """
        if self.period == 0:
            #print('Not oscillatory, setting only c0')
            self.u_coeffs = np.zeros(2*N+1)
            self.v_coeffs = np.zeros(2*N+1)
            self.u_coeffs[N] = self.uv[-1]
            self.v_coeffs[N] = self.vv[-1]
        else:
            u0f = interp1d(self.t0v, self.u0v, kind='linear')
            v0f = interp1d(self.t0v, self.v0v, kind='linear')

            # sample at 2N+1 points and compute fft
            uhat = fftshift(fft(u0f(np.linspace(0,self.t0v[-1],2*N+1,endpoint=False))))/(2*N+1)
            vhat = fftshift(fft(v0f(np.linspace(0,self.t0v[-1],2*N+1,endpoint=False))))/(2*N+1)
            self.u_coeffs = uhat
            self.v_coeffs = vhat
            
            # also keep the function that reconstructs
            def u_rec(t):
                res = np.zeros_like(t, dtype=complex)
                for i in range(-N, N+1):
                    res += self.u_coeffs[i+N]*np.exp(2*np.pi*1j*i/self.period * t)
                return res
            def v_rec(t):
                res = np.zeros_like(t, dtype=complex)
                for i in range(-N, N+1):
                    res += self.v_coeffs[i+N]*np.exp(2*np.pi*1j*i/self.period * t)
                return res
                
            self.fourier_rec_u = u_rec
            self.fourier_rec_v = v_rec
        
class BistableModelFWBW(object):
    """ Bistable model with production and degradation, but the bistability here is not introduced explicitly.
    We use a protein with activation and inactivation as in the project on dynamic bistability"""

    def __init__(self, **params):
        self.ks = params['ks']
        self.bdeg = params['bdeg']
        self.a = params['a']
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']
        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

    def solve(self, T, dt, y0=(0,0), usejac=False):
        # NOTE Jacobian not implemented
        # solve the system. Use scipy's built-in solver here.
        def dxdt(t,y):
            u=y[0]
            v=y[1]
            return [(self.a + self.b * u**self.n / (self.K**self.n + u**self.n))*(v-u) - (self.ap + self.bp*self.Kp**self.m / (self.Kp**self.m + u**self.m))*u, self.ks-self.bdeg*u*v]
        tv = np.linspace(0, T, int(T/dt))
        sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

        # keep variables
        self.T = T
        self.dt=dt
        self.tv=tv
        self.uv = sol.y[0,:]
        self.vv = sol.y[1,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr, thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a

class BistableModelYF(object):
    """ Bistable model with production and degradation, and activation/inactivation. The difference with the FWBW bistable model is
    in the degradation term: there is a Hill function of u here, as in Yang and Ferrell's model. """

    def __init__(self, **params):
        self.ks = params['ks']
        self.bdeg = params['bdeg']
        self.a = params['a']
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']
        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.app = params['app']
        self.bpp = params['bpp']
        self.Kpp = params['Kpp']
        self.l = params['l']

        # extra timescale separation parameter
        if 'epsilon' in params.keys():
            self.epsilon = params['epsilon']
        else:
            self.epsilon = 1.

    def jaco(self,t,y):
        # jacobian matrix, used for the Radau solver to make it more accurate
        u = y[0]
        v = y[1]
        f=self.a + self.b * u**self.n / (self.K**self.n + u**self.n)
        fp = self.b*self.n*self.K**self.n*u**(self.n-1)/(self.K**self.n+u**self.n)**2
        g = self.ap + self.bp*self.Kp**self.m / (self.Kp**self.m + u**self.m)
        gp = -self.bp*self.Kp**self.m*self.m*u**(self.m-1)/(self.Kp**self.m + u**self.m)**2
        h = self.app + self.bpp * u**self.l / (self.Kpp**self.l + u**self.l)
        hp = self.bpp*self.l*self.Kpp**self.l*u**(self.l-1)/(self.Kpp**self.l+u**self.l)**2
        J11 = fp*(v-u) -f- gp*u-g
        J12 = f
        J21 = -self.bdeg*v*hp
        J22 = -self.bdeg*h

        return np.array([[J11/self.epsilon, J12/self.epsilon], [J21, J22]])

    def solve(self, T, dt, y0=(0,0),udeg=0,usejac=False):
        # solve the system. Use scipy's built-in solver here.
        # udeg determines whether APC/C degradation also acts on the u variable (it does in the original YF paper)
        def dxdt(t,y):
            u=y[0]
            v=y[1]
            if udeg:
                udterm = self.bdeg*(self.app + self.bpp * u**self.l / (self.Kpp**self.l + u**self.l))*u
            else:
                udterm = 0
            return [self.epsilon**(-1.)*((self.a + self.b * u**self.n / (self.K**self.n + u**self.n))*(v-u) - (self.ap + self.bp*self.Kp**self.m / (self.Kp**self.m + u**self.m))*u) - udterm, \
            self.ks-self.bdeg*(self.app + self.bpp * u**self.l / (self.Kpp**self.l + u**self.l))*v]
        tv = np.linspace(0, T, int(T/dt))
        if usejac:
            sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF', jac=self.jaco)
        else:
            sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

        # keep variables
        self.T = T
        self.dt=dt
        self.tv=tv
        self.uv = sol.y[0,:]
        self.vv = sol.y[1,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        # if number > 1 a list of periods and amplitudes is returned
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr, thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a
        
    def getfinalperiodtimeseries(self):
        """call this after getperiod. Saves the time series of the last period (between last maxima of v).
        Note that this uses find_peaks from scipy rather than our own peak detection. """
        if self.period == 0:
            self.t0v = 0
            self.u0v = 0
            self.v0v = 0 
        else:
            peak_inds = find_peaks(self.vv, prominence=self.amplitude/2)[0]
            if len(peak_inds)<2:
                # no two big peaks -- means probably something wrong
                self.period=0
                self.amplitude=0
                self.t0v = 0
                self.u0v = 0
                self.v0v = 0
            else:
                self.t0v = self.tv[peak_inds[-2]:peak_inds[-1]+1].copy()
                self.t0v -= self.t0v[0] # set time zero at start
                self.u0v = self.uv[peak_inds[-2]:peak_inds[-1]+1].copy()
                self.v0v = self.vv[peak_inds[-2]:peak_inds[-1]+1].copy()
                
    def getfouriercoeff(self,N):
        """ determine the complex Fourier coefficients of both variables. 
        Call this only after having solved and detected period.
        N is the max harmonics. It means we compute 2N+1 coefficients.
        ie u(t) = sum_{-N}^N c_i exp(i 2 pi / T t)
        """
        if self.period == 0:
            #print('Not oscillatory, setting only c0')
            self.u_coeffs = np.zeros(2*N+1)
            self.v_coeffs = np.zeros(2*N+1)
            self.u_coeffs[N] = self.uv[-1]
            self.v_coeffs[N] = self.vv[-1]
        else:
            u0f = interp1d(self.t0v, self.u0v, kind='linear')
            v0f = interp1d(self.t0v, self.v0v, kind='linear')

            # sample at 2N+1 points and compute fft
            uhat = fftshift(fft(u0f(np.linspace(0,self.t0v[-1],2*N+1,endpoint=False))))/(2*N+1)
            vhat = fftshift(fft(v0f(np.linspace(0,self.t0v[-1],2*N+1,endpoint=False))))/(2*N+1)
            self.u_coeffs = uhat
            self.v_coeffs = vhat
            
            # also keep the function that reconstructs
            def u_rec(t):
                res = np.zeros_like(t, dtype=complex)
                for i in range(-N, N+1):
                    res += self.u_coeffs[i+N]*np.exp(2*np.pi*1j*i/self.period * t)
                return res
            def v_rec(t):
                res = np.zeros_like(t, dtype=complex)
                for i in range(-N, N+1):
                    res += self.v_coeffs[i+N]*np.exp(2*np.pi*1j*i/self.period * t)
                return res
                
            self.fourier_rec_u = u_rec
            self.fourier_rec_v = v_rec

class MassActionModel(object):
    """ Class to simulate the mass-action model which also incorporates the pp2a ensa pathway"""

    def __init__(self, **params):
        # reaction rates
        self.ks = params['ks']
        self.bdeg = params['bdeg'] # cyclin production and degradation
        self.kpa = params['kpa']
        self.kda = params['kda'] # APC/C phosphorylation/deph
        self.kpg = params['kpg']
        self.kdg = params['kdg'] # Greatwall phosphorylation/deph
        self.kpe = params['kpe'] # ensa phosph
        self.kass = params['kass']
        self.kdiss = params['kdiss'] # complex association and dissociation
        self.kcat = params['kcat'] # complex to dephosphorylated ensa + pp2a

        # total amounts
        self.at = params['at'] # APC/C
        self.gt = params['gt'] # Greatwall
        self.et = params['et'] # ENSA
        self.pt = params['pt'] # PP2A

    def jaco(self, t, yy):
        u,v,w,x,y = yy
        J1 = [-self.kpa*v-self.kda*(self.pt-y), self.kpa*(self.at-u), 0,0,self.kda*u]
        J2 = [-self.bdeg*v, -self.bdeg*u, 0, 0, 0]
        J3 = [0, self.kpg*(self.gt-w), -self.kpg*v - self.kdg*(self.pt-y), 0, self.kdg*w]
        J4 = [0,0,-self.kpe*x, -self.kpe*w, self.kcat]
        J5 = [0,0,0, -self.kass*(self.pt-y), self.kass*(-self.et+x-self.pt+2*y)-self.kdiss-self.kcat]
        return np.array([J1, J2, J3, J4, J5])

    def solve(self, T, dt, y0=(0,0,0,0,0), usejac=False, usejitcode=False):
        # solve the system. Use scipy's built-in solver here.
        # order of the variables is u,v,w,x,y which corresponsd to
        # phosph apc, cdk1, phosph gwl, free dephosph ensa, complex

        if usejitcode:
            u,v,w,x,y=yo(0),yo(1),yo(2),yo(3),yo(4)
            du = self.kpa*(self.at - u)*v - self.kda*u*(self.pt-y)
            dv = self.ks - self.bdeg*v*u
            dw = self.kpg*(self.gt - w)*v - self.kdg*(self.pt - y)*w
            dx = -self.kpe*x*w + self.kcat*y
            dy = self.kass*(self.et-x-y)*(self.pt-y)-self.kdiss*y-self.kcat*y
            dxdt = [du,dv,dw,dx,dy]

            ODE = jitcode(dxdt)
            ODE.set_integrator('BDF')
            ODE.set_initial_value(y0, 0.)
            times = np.arange(0,T,dt)
            data = np.zeros((5, len(times)))
            for i,tt in enumerate(times):
                data[:,i] = ODE.integrate(tt)

            self.T = T
            self.dt=dt
            self.tv=times
            self.uv = data[0,:]
            self.vv = data[1,:]
            self.wv = data[2,:]
            self.xv = data[3,:]
            self.yv = data[4,:]
            self.ODE = ODE

        else:

            def dxdt(t,Y):
                u,v,w,x,y = Y
                du = self.kpa*(self.at - u)*v - self.kda*u*(self.pt-y)
                dv = self.ks - self.bdeg*v*u
                dw = self.kpg*(self.gt - w)*v - self.kdg*(self.pt - y)*w
                dx = -self.kpe*x*w + self.kcat*y
                dy = self.kass*(self.et-x-y)*(self.pt-y)-self.kdiss*y-self.kcat*y
                return [du,dv,dw,dx,dy]
            tv = np.linspace(0, T, int(T/dt))
            if usejac:
                sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF', jac=self.jaco)
            else:
                sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

            # keep variables
            self.T = T
            self.dt=dt
            self.tv=tv
            self.uv = sol.y[0,:]
            self.vv = sol.y[1,:]
            self.wv = sol.y[2,:]
            self.xv = sol.y[3,:]
            self.yv = sol.y[4,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        #use cdk1 amplitude
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr,thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a


class TemperatureIterator(object):
    """ Iterates over a given temperature range for model of choice.
    Note: all parameters need to be given, also those without scaling such as m, K etc., give Ea=0 there.
    All arguments to functions must be in Kelvin"""

    R = 8.314472 # gas constant
    def __init__(self, model, p0, T0, Ea, Tv):
        self.model = model # String 'delay', 'bistable', 'massaction'
        self.p0 = p0 # dictionary {parameter: value},  parameters at T0
        self.T0 = T0 # reference temperature in Kelvin
        self.Eav = Ea # activation energies (dictionary parameter:Ea). Only params that have an activation energy

        #NEW: Eav can also be functions, that give the parameter value as function of temperature in Kelvin
        self.Tv = np.array(Tv) # temperatures over which to iterate in Kelvin

    def getperiods(self, Time=200, dt=0.01, thr=1e-3, thrper=1, method='robust', number=1, usejac=0, usejitcode=0):
        # standard is robust method, but argument included for testing purposes
        periods = []
        oscitypes = [] # type of oscillation
        for T in self.Tv:
            p = self.p0.copy() # copy the original
            for nn in self.Eav.keys():
                # for the parameters for which an activation energy was given, multiply
                if callable(self.Eav[nn]):
                    p[nn] = self.Eav[nn](T)
                else: #a number was given, so use the standard Arrhenius scaling
                    p[nn] *= np.exp(-self.Eav[nn]/self.R*(1./T - 1./self.T0))

            if self.model=='delay':
                M = DelayModel(**p)
            elif self.model =='bistable':
                M = BistableModel(**p)
            elif self.model == 'bistableFB':
                M = BistableModelFWBW(**p)
            elif self.model == 'bistableYF':
                M = BistableModelYF(**p)
            elif self.model == 'MA':
                M = MassActionModel(**p)
            else:
                print('Model unknown')
                return 0

            if self.model=='MA':
                M.solve(T=Time, dt=dt, usejac=usejac, usejitcode=usejitcode)
            else:
                M.solve(T=Time, dt=dt, usejac=usejac)
            M.getperiod(thr, thrper, method=method, number=number)

            # new 29 october: period and amplitude can be lists now if number > 1
            if number>1:
                otype, per = classifyosc(M.period, M.amplitude, thr)

                periods.append(per)
                oscitypes.append(otype)
            else:
                periods.append(M.period)
                if M.period > 0:
                    oscitypes.append('osc')
                else:
                    oscitypes.append('ss')
            self.periods = np.array(periods)
            self.oscitypes = np.array(oscitypes)

    def cleanperiods(self):
        # to be run after getperiods has been run.
        # sets to zero all the periods where the type is not oscillatory
        ind=[]
        for j in range(len(self.oscitypes)):
            if self.oscitypes[j][:3] != 'osc':
                ind.append(j)

        self.periodsc = self.periods.copy()
        self.periodsc[ind] = 0.

    def simulateT(self, T, Time=100, dt=0.01, usejac=0):
        # simulate at specific temperature (Kelvin)
        p = self.p0.copy() # copy the original
        for nn in self.Eav.keys():
            # for the parameters for which an activation energy was given, multiply
            if callable(self.Eav[nn]):
                p[nn] = self.Eav[nn](T)
            else: #a number was given, so use the standard Arrhenius scaling
                p[nn] *= np.exp(-self.Eav[nn]/self.R*(1./T - 1./self.T0))

        if self.model=='delay':
            M = DelayModel(**p)
        elif self.model =='bistable':
            M = BistableModel(**p)
        elif self.model == 'bistableFB':
            M = BistableModelFWBW(**p)
        elif self.model == 'bistableYF':
            M = BistableModelYF(**p)
        elif self.model == 'MA':
            M = MassActionModel(**p)
        else:
            print('Model unknown')
            return 0
        M.solve(T=Time, dt=dt, usejac=usejac)
        M.getperiod()
        return M.tv, M.vv, M.period

    def fitarrhenius_begasse(self, Tstart=293, thr=0.1):
        # method from the paper by Begasse et al.
        # Tstart is starting temperature in Kelvin

        # use the oscitypes to only select the 'real' oscillations
        ind = []
        for j in range(len(self.oscitypes)):
            if self.oscitypes[j][:3] != 'osc':
                ind.append(j)

        periods_osc = self.periods.copy()
        periods_osc[ind] = 0.

        logP = np.log(periods_osc)

        # find temperature closest to Tstart
        i=0
        while self.Tv[i]<Tstart: i+=1
        # i is now the index of the middle temperature point

        # if there is no period at Tstart, skip the whole business
        if not (np.isfinite(logP[i]) and np.isfinite(logP[i-1]) and np.isfinite(logP[i+1])):
            #happens if P is zero, then log is minus infty. Check the three initial values used for the fit.
            self.Tarrh = []
            self.Parrh = []
            self.Ea = np.nan
            self.A = np.nan

        else:
            #start by fitting an E0
            a,b = np.polyfit(1./self.Tv[i-1:i+2], logP[i-1:i+2],1)
            Ea0 = -self.R*a
            # compute Ea when adding higher temperatures
            Eah = []
            for j in range(i+2,len(self.Tv)):
                a,b = np.polyfit(1./self.Tv[i-1:j+1], logP[i-1:j+1],1)
                Eah.append(-self.R*a)
            # for lower temperatures
            Eal = []
            for j in range(i-2,-1,-1):
                a,b = np.polyfit(1./self.Tv[j:i+2], logP[j:i+2],1)
                Eal.append(-self.R*a)
            # calculate the relative differences in Ea
            Eahr=[abs((Ea-Ea0)/Ea0) for Ea in Eah]
            Ealr=[abs((Ea-Ea0)/Ea0) for Ea in Eal]

            # check high temperatures where Ea is too different
            j=0
            while j<len(Eahr) and Eahr[j] < thr:
                j+=1
            ih = i+2+j
            # check low temperatures
            j=0
            while j<len(Ealr) and Ealr[j] < thr :
                j+=1
            il = i-1-j
            # final fit and saving of the interval
            Tarrh = self.Tv[il:ih]
            logParrh = logP[il:ih]
            a,b = np.polyfit(1./Tarrh, logParrh,1)

            self.Tarrh = Tarrh
            self.Parrh = np.exp(logParrh)
            self.Ea = -self.R*a
            self.A = np.exp(b)

    def optimaltstart_begasse(self, thr=0.1):
        # iterates over the starting temperature for the Begasse algorithm
        lengths = []
        for Ts in self.Tv[1:-1]: # not first and last
            self.fitarrhenius_begasse(Ts, thr)
            lengths.append(len(self.Tarrh))
        idmax = np.argmax(lengths)+1 # need the offset because we did not start from first temperature above
        return self.Tv[idmax]



### May 2021, to explore temperature forcing


class ForcedBistableYF(object):
    def __init__(self, p0, Ea, T0, Tfunction):
        # p0 is the base parameter set at temperature T0
        # Ea is the activation energies. Keys are param names
        # params not included in Ea are supposed to be temperature independent
        # Tfunction is temperature (Kelvin) as function of time
        self.p0 = p0
        self.Ea = Ea
        self.Tfunction = Tfunction
        self.T0 = T0

    def jaco(self,t,y):
        # jacobian matrix, used for the Radau solver to make it more accurate
        u = y[0]
        v = y[1]
        p = {k: self.p0[k] for k in self.p0.keys()} # copy of the base parameters
        T = self.Tfunction(t) # current temperature
        # modify temperature-dependent rates
        for ppp in self.Ea.keys():
            p[ppp] *= np.exp(-self.Ea[ppp]/R*(1/T - 1/self.T0))

        f=  p['a'] + p['b'] * u**p['n'] / (p['K']*p['n'] + u**p['n'])
        fp = p['b']*p['n']*p['K']**p['n']*u**(p['n']-1)/(p['K']**p['n']+u**p['n'])**2
        g = p['ap'] + p['bp']*p['Kp']**p['m'] / (p['Kp']**p['m'] + u**p['m'])
        gp = -p['bp']*p['Kp']**p['m']*p['m']*u**(p['m']-1)/(p['Kp']**p['m'] + u**p['m'])**2
        h = p['app'] + p['bpp'] * u**p['l'] / (p['Kpp']**p['l'] + u**p['l'])
        hp = p['bpp']*p['l']*p['Kpp']**p['l']*u**(p['l']-1)/(p['Kpp']**p['l']+u**p['l'])**2
        J11 = fp*(v-u) -f- gp*u-g
        J12 = f
        J21 = -p['bdeg']*v*hp
        J22 = -p['bdeg']*h

        return np.array([[J11/p['epsilon'], J12/p['epsilon']], [J21, J22]])

    def solve(self, Time, dt, y0=(0,0),usejac=False):
        # solve the system. Use scipy's built-in solver here.
        def dxdt(t,y):
            u=y[0]
            v=y[1]
            # calculate current parameters
            p = {k: self.p0[k] for k in self.p0.keys()} # copy of the base parameters
            T = self.Tfunction(t) # current temperature
            # modify temperature-dependent rates
            for ppp in self.Ea.keys():
                p[ppp] *= np.exp(-self.Ea[ppp]/R*(1/T - 1/self.T0))

            return [p['epsilon']**(-1.)*((p['a']+ p['b'] * u**p['n'] / (p['K']**p['n'] + u**p['n']))*(v-u) \
            - (p['ap'] + p['bp']*p['Kp']**p['m'] / (p['Kp']**p['m'] + u**p['m']))*u), \
            p['ks']-p['bdeg']*(p['app'] + p['bpp'] * u**p['l'] / (p['Kpp']**p['l'] + u**p['l']))*v]
        tv = np.linspace(0, Time, int(Time/dt))
        if usejac:
            sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF', jac=self.jaco)
        else:
            sol = solve_ivp(dxdt, (tv[0], tv[-1]), y0, t_eval=tv, method='BDF')

        # keep variables
        self.Time = Time
        self.dt=dt
        self.tv=tv
        self.uv = sol.y[0,:]
        self.vv = sol.y[1,:]

    def getperiod(self, thr=1e-3, thrper=1, method='robust', number=1):
        # if number > 1 a list of periods and amplitudes is returned
        if method=='robust':
            p,a,message = getperiodamplitude_robust(self.tv, self.vv, thr, thrper, number)
        else:
            p,a,message = getperiodamplitude(self.tv, self.vv, thr)
        self.period = p
        self.amplitude = a
