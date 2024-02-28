import os
import pandas as pd
import numpy as np
import scipy
import ultranest
import glob
from astropy import units as u
from astropy import constants as c

from .tools import bin_by_time,extract_rvfile,assemble_all_rvs,radial_velocity,weighted_quantile,weighted_avg_and_std

def semi_major_axis(P, Mtotal):
    """Semi-major axis - from radvel but vectorised

    Kepler's third law

    Args:
        P (float): Orbital period [days]
        Mtotal (float): Mass [Msun]

    Returns:
        float or array: semi-major axis in AU
    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Mtotal = np.array(Mtotal)

    Mtotal = Mtotal*c.M_sun.value
    P = (P * u.d).to(u.second).value
    G = c.G.value
    
    a = ((P**2)*G*Mtotal/(4*(np.pi)**2))**(1/3.)
    a = a/c.au.value

    return a

def kepler_vect(Marr, eccarr):
    """Solve Kepler's Equation  - from radvel but vectorised
    Args:
        Marr (array): input Mean anomaly
        eccarr (array): eccentricity
    Returns:
        array: eccentric anomaly
    """

    conv = 1.0e-12  # convergence criterion
    k = 0.85

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
    # fiarr should go to zero when converges
    fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)
    con_ix = np.abs(fiarr) > conv  # which indices have not converged
    nd = np.sum(con_ix)  # number of unconverged elements
    count = 0

    while nd > 0:  # while unconverged elements exist
        count += 1

        #M = Marr[convd]  # just the unconverged elements ...
        #ecc = eccarr[convd]
        #E = Earr[convd]

        #fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
        fip = 1 - eccarr[con_ix] * np.cos(Earr[con_ix])  # d/dE(fi) ;i.e.,  fi^(prime)
        fipp = eccarr[con_ix] * np.sin(Earr[con_ix])  # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        # first, second, and third order corrections to E
        d1 = -fiarr[con_ix] / fip
        d2 = -fiarr[con_ix] / (fip + d1 * fipp / 2.0)
        d3 = -fiarr[con_ix] / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)
        Earr[con_ix]+=d3
        fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
        con_ix = np.abs(fiarr) > conv  # test for convergence
        nd = np.sum(con_ix)
    return Earr

def true_anomaly_vect(t, tp, per, e):
    # f in Murray and Dermott p. 27
    if type(per) in [float,np.float64]:
        m = 2 * np.pi * (((t[None,:] - tp[:,None]) / per) - np.floor((t[None,:] - tp[:,None]) / per))
    else:
        m = 2 * np.pi * (((t[None,:] - tp[:,None]) / per[:,None]) - np.floor((t[None,:] - tp[:,None]) / per[:,None]))
    eccarr = np.zeros_like(m)+e[:,None]
    e1 = kepler_vect(m, eccarr)
    n1 = 1.0 + e
    n2 = 1.0 - e
    nu = 2.0 * np.arctan((n1[:,None] / n2[:,None])**0.5 * np.tan(e1 / 2.0))
    return nu

def timetrans_to_timeperi_vect(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage - from radvel but vectorised

    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        float: time of periastron passage

    """
    tp = np.zeros_like(ecc)
    ok_ix=(ecc < 1)&(ecc>0)
    f = np.pi/2 - omega
    ee = 2 * np.arctan(np.tan(f[ok_ix]/2) * np.sqrt((1-ecc[ok_ix])/(1+ecc[ok_ix])))  # eccentric anomaly

    if type(tc) in [float,np.float64]:
        tp[~ok_ix]=tc
        tp[ok_ix] = tc - per/(2*np.pi) * (ee - ecc[ok_ix]*np.sin(ee))
    elif tc.shape==ecc.shape:
        tp[~ok_ix]=tc[~ok_ix]
        tp[ok_ix] = tc[ok_ix] - per[ok_ix]/(2*np.pi) * (ee - ecc[ok_ix]*np.sin(ee))      # time of periastron

    return tp
    
def radial_velocity_ecc_vect(t, P, tc, e, w, K):
    # Calculate the radial velocity at time t - from radvel but vectorised
    # P: Orbital period
    # tc: Transit time
    # e: Eccentricity
    # w: Argument of periapsis
    # K: Semi-amplitude of radial velocity

    # Solve Kepler's equation for eccentric anomaly
    tp = timetrans_to_timeperi_vect(tc, P, e, w)
    # Calculate true anomaly
    f = true_anomaly_vect(t, tp, P, e)
    # Calculate radial velocity
    return K[:,None] * (np.cos(w[:,None] + f) + e[:,None] * np.cos(w[:,None]))

def radial_velocity_circ_vect(t, P, tc, K):
    # Calculate the radial velocity at time t - from radvel but vectorised
    # P: Orbital period
    # tc: Transit time
    # e: Eccentricity
    # w: Argument of periapsis
    # K: Semi-amplitude of radial velocity

    if type(tc)==float and type(P)==float:
        return -K[:,None] * np.sin(2*np.pi*(t[None,:]-tc)/P)
    else:
        return (-K[:,None] * np.sin(2*np.pi*(t[None,:]-tc[:,None])/P[:,None]))


def TransitPrior_vect(tc, p, ecc, w, a, Rs, Rp_Rs, tdur, bmu=None, bsigma=None):
    """Prior for transiting planets which has constrained transit duration (and non-graxing impact parameter), 
    or even better constrained impact parameter ()
    """
    time_peri = timetrans_to_timeperi_vect(tc,p,ecc,w)
    #print(time_peri)
    true_anom = true_anomaly_vect(np.array([tc]), time_peri, p, ecc)[:,0]
    #print(true_anom)
    # Calculating implied impact param from these paramaters...
    b2 = (1+Rp_Rs)**2 - ((tdur * np.pi * a*(1+ecc*np.cos(true_anom)))/(p*Rs*np.sqrt(1-ecc**2)))**2
    # Only accepting non-grazing positive b.
    maxb=(1-Rp_Rs)**2

    prior=np.zeros_like(b2)
    prior[~np.isfinite(b2)]=-1e10
    negb=np.isfinite(b2)&(b2<=0)
    prior[negb] = -9e9-np.clip(abs(b2[negb]),0,10)*1e8
    grazing=np.isfinite(b2)&(b2>=1)
    prior[grazing] = -9e9-np.clip(b2[grazing]-1,0,10)*1e8
    okb=np.isfinite(b2)&(b2>0.)&(b2<maxb)
    prior[okb] = -0.5 * ((np.sqrt(b2[okb]) - bmu) / bsigma)**2 if bmu is not None and not np.isnan(bmu) and bsigma is not None and not np.isnan(bsigma) else 0
    return prior


def TwoPlanetStabilityPrior_vect(ecclist, smalist, Klist, Rs, Ms):
    #Check that max and minimum distances for each planet does not overlap with either max/minimum distances of the others NOR 2Rs
    sorts=np.argsort(smalist,axis=1)
    Mps = Klist/np.sqrt(6.67e-11/(Ms*1.96e30*smalist*695500000*(1-ecclist**2)))/1.96e30
    unstable=(1-ecclist[sorts==0]-(Mps[sorts==0]/(3*Ms))**(1/3))*smalist[sorts==0]/Rs < (1+ecclist[sorts==1]+(Mps[sorts==1]/(3*Ms))**(1/3))*smalist[sorts==1]/Rs
    return np.where(unstable,-1e5,0)

def ThreePlanetStabilityPrior_vect(ecclist, smalist, Klist, Rs, Ms):
    #Check that max and minimum distances for each planet does not overlap with either max/minimum distances of the others NOR 2Rs
    min_dists=np.zeros_like(Rs);max_dists=np.zeros_like(Rs)
    sorts=np.argsort(smalist,axis=1)
    Mps = Klist/np.sqrt(6.67e-11/(Ms*1.96e30*smalist*695500000*(1-ecclist**2)))/1.96e30
    innerunstable=(1-ecclist[sorts==0]-(Mps[sorts==0]/(3*Ms))**(1/3))*smalist[sorts==0]/Rs < (1+ecclist[sorts==1]+(Mps[sorts==1]/(3*Ms))**(1/3))*smalist[sorts==1]/Rs
    outerunstable=(1-ecclist[sorts==1]-(Mps[sorts==1]/(3*Ms))**(1/3))*smalist[sorts==1]/Rs < (1+ecclist[sorts==2]+(Mps[sorts==2]/(3*Ms))**(1/3))*smalist[sorts==2]/Rs
    return np.where(innerunstable|outerunstable,-1e5,0)

class VectorPriorTransform():
    def __init__(self, pars):
        self.pars = pars
        #print(pars)
        
    def __call__(self, u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest."""
        
        x = np.array(u)  # copy u
        for n,p in enumerate(self.pars):
            if self.pars[p][0]=='uniform':
                x[:,n]=self.pars[p][1] + (self.pars[p][2]-self.pars[p][1]) * u[:,n]
            elif self.pars[p][0]=='norm':
                x[:,n]=scipy.stats.norm.ppf(u[:,n],self.pars[p][1],self.pars[p][2])
            elif self.pars[p][0]=='halfnorm':
                x[:,n]=scipy.stats.halfnorm.ppf(u[:,n],loc=self.pars[p][1],scale=self.pars[p][2])
            elif self.pars[p][0]=='truncnorm':
                a=self.pars[p][3];b=self.pars[p][4]
                ar, br = (a - self.pars[p][1]) / self.pars[p][2], (b - self.pars[p][1]) / self.pars[p][2]
                x[:,n]=scipy.stats.truncnorm.ppf(u[:,n], ar, br, loc=self.pars[p][1], scale=self.pars[p][2])
            elif self.pars[p][0]=='rayleigh':
                x[:,n]=scipy.stats.rayleigh.ppf(u[:,n],self.pars[p][1])
            elif self.pars[p][0]=='beta':
                x[:,n]=scipy.stats.beta.ppf(u[:,n],self.pars[p][1],self.pars[p][2])
        return x

class Vector_FlatLikelihood():
    def __init__(self, name, data, varnames, jitprior='linear', fit_curv=True, refepoch=None, outfileloc=None):
        self.name=name
        self.fit_curv=fit_curv
        self.data = data
        self.jitprior=jitprior
        self.x_index={}
        self.insts = np.unique(self.data['instr'].values).astype(str)
        self.inst_ix_list={i:self.data['instr'].values.astype(str)==i for i in self.insts}
        self.refepoch = np.average(self.data['time'].values) if refepoch is None else refepoch
        self.outfileloc=os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]),name.replace(' ','')+"_noplanets") if outfileloc is None else outfileloc
        #Creating index dictionary 'param' -> number index in x
        for n_i,i in enumerate(self.insts):
            if self.jitprior=='log':
                self.x_index['logjitter_'+str(i)]=varnames.index('logjitter_'+str(i))
            elif self.jitprior=='linear':
                self.x_index['jitter_'+str(i)]=varnames.index('jitter_'+str(i))
            self.x_index['offset_'+str(i)]=varnames.index('offset_'+str(i))
        if self.fit_curv:
            self.x_index['polycurv']=varnames.index('polycurv')
        self.x_index['polytrend']=varnames.index('polytrend')

    def __call__(self, x):
        """The log-likelihood function."""
        #varpars format = [k,secosw,sesinw],[gamma,jitter],dvdt,curv
        logprior=0

        sigma2=np.tile(self.data['errvel'].values**2,(np.shape(x)[0],1))
        instr_offsets=np.zeros((np.shape(x)[0],len(self.data)))
        for n_i,i in enumerate(self.insts):
            if self.jitprior=='log':
                sigma2[:,self.inst_ix_list[i]]+=np.exp(x[:,self.x_index['logjitter_'+str(i)]])[:,None]**2    #jitter
            elif self.jitprior=='linear':
                sigma2[:,self.inst_ix_list[i]]+=x[:,self.x_index['jitter_'+str(i)]][:,None]**2    #jitter
            instr_offsets[:,self.inst_ix_list[i]]+=x[:,self.x_index['offset_'+str(i)]][:,None]          #gamma
        if self.fit_curv:
            trend=np.polyval(np.vstack([x[:,self.x_index['polycurv']][None,:],x[:,self.x_index['polytrend']][None,:],np.zeros(x.shape[0])[None,:]]),
                             self.data['time'].values[:,None] - self.refepoch).T
        else:
            trend=x[:,self.x_index['polytrend']][None,:]*(self.data['time'].values[:,None]-self.refepoch).T

        full_rv_model = instr_offsets+trend
        #print(i,np.sum(idx_planet1),np.sum(np.array(results['positions'][2])==-2),times,ttimes[n][:,0],sigma2)
        
        llk = -0.5 * np.sum((self.data['mnvel'].values[None,:] - full_rv_model) ** 2 / sigma2 + np.log(sigma2),axis=1)
        return llk+logprior
    
    def summarise_results(self,result):
        prcnts={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868,'+3sig':99.74}
        summary_df={}
        for n_v, var in enumerate(result['paramnames']):
            av,sd=weighted_avg_and_std(result['weighted_samples']['points'][:,n_v],errs=None,weights=result['weighted_samples']['weights'])
            summary_df[var]={'mean':av,'sd':sd}
            summary_df[var].update({ipc:weighted_quantile(result['weighted_samples']['points'][:,n_v], 0.01*prcnts[ipc], sample_weight=result['weighted_samples']['weights']) for ipc in prcnts})
        summary_df = pd.DataFrame(summary_df).T
        summary_df=pd.concat([summary_df,pd.DataFrame({'mean':result['logz'],'sd':result['logzerr']},index=['logEvidence'])])
        summary_df.to_csv(os.path.join(self.outfileloc,self.name+"_noplanets_results.csv"))
        return summary_df


class Vector_PlanetLikelihood():
    def __init__(self, name, kprior='linear', fit_curv=True, refepoch=None, jitterprior=None, ecc_prior=None, fit_circ=None,outfileloc=None):
        """Initialise the Planet Log Likelihood object
        Args:
            name (str): name of star (for saving)
            Ms (list): Stellar mass in Msun
            Rs (float): Stellar radius in Rsun
            plinfo (dict): Planetary properties dictionary with an entry for each planet. Including:
                - per (float): orbital period in days
                - tc (float): transit time
                - Rp_Rs (float): Radius ratio
                - tdur (flaot): transit duration
                - bmu (float, optional): median of the impact parameter
                - bsigma (float, optional): sigma of the impact parameter
                - type (str, optional): type of planet. Either "transit" or "rv"
            data (pd.DataFrame): Radial velocity data, with columns:
                - time
                - mnvel
                - errvel
            varnames (list): the list of variables used to initialise the Ultranest object
            kprior (str, optional): The type of prior for semi-amplitude ('log', or the default, 'linear')
            fit_curv (bool, optional): Whether to include second-order curvature term in the RV fit
            refepoch (float, optional): The time in BJD to use as a reference epoch, otherwise the RV time median is used
            fit_circ (bool, optional): Whether to fit a purely circular or eccentric orbit

        """
        self.initialised=False
        self.name = name
        self.refepoch = refepoch
        self.pls = []#len([col for col in plinfo if 'per_' in col])
        
        self.kprior=kprior
        self.jitterprior=self.kprior if jitterprior is None else jitterprior
        self.fit_curv=fit_curv
        self.fit_circ={}
        self.ecc_prior=ecc_prior
        self.outfileloc=os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]),starname.replace(' ','')+"_"+self.kprior+"k") if outfileloc is None else outfileloc
        if not os.path.isdir(self.outfileloc):
            os.mkdir(self.outfileloc)

        #adding sma
        self.plinfo={'x_index':{}}
        self.data = pd.DataFrame()
        self.insts = []
        
    def add_data(self,time,rvs,rv_errs,instr):
        """Add RV data
        
        Args:
            time (np.array): Times
            rvs (np.array): RVs (in m/s)
            rv_errs (np.array): RV errors (in m/s)
            instr (str): Unique instrument name
        """
        if type(instr) in [str,int,float]:
            assert instr not in self.insts
        self.data=pd.concat([self.data,
                             pd.DataFrame({'time':time,'mnvel':rvs,'errvel':rv_errs,'instr':np.tile(str(instr),len(time)) if type(instr) in [str,int,float] else instr})])
        self.insts=np.unique(self.data['instr'].values).astype(str)

    def add_starpars(self,Rs,Ms):
        """Add stellar parameters
        
        Args:
            Rs (float): Stellar radius
            Ms (float): Stellar mass
        """
        self.Rs=Rs
        self.Ms=Ms

    def add_rv_planet(self,plname,per,tcen,model_params,k=None,fix_params=[],e_per=None,e_tcen=None,fit_circ=False):
        """Add RV planet.
        
        Args:
            plname (str): Planet name
            per (float): Period
            tcen (float): Transit time
            model_params (list): [per,tcen,k/logk]
            fix_params (list, optional): [per,tcen]
            e_per (float, optional): Uncertainty on the period
            e_tcen (float): Uncertainty on the time of perihelion
            """
        assert not np.any(np.in1d(fix_params,model_params)), "Cannot have any parameters shared between fixed ("+fix_params+") and model ("+model_params+")"
        assert plname not in self.pls, plname+" Already in the planets to be modelled ("+",".join(self.pls)+")"
        assert not (k is None and 'k' not in model_params and 'logk' not in model_params), "If k or logk not in model_params ("+",".join(model_params)+"), you must specify a value to k to fix"
        self.pls+=[plname]
        self.plinfo.update({'per_'+str(plname):per,
                'tc_'+str(plname):tcen,
                'Rp_Rs_'+str(plname):np.nan,
                'tdur_'+str(plname):np.nan,
                'model_params_'+str(plname):model_params,
                'fix_params'+str(plname):fix_params,
                'type_'+str(plname):'rv'})
        if k is not None:
            self.plinfo['k_'+str(plname)]=k
        self.plinfo['e_tc_'+str(plname)]=e_tcen if e_tcen is not None else 0.55*per
        self.plinfo['e_per_'+str(plname)]=e_per if e_per is not None else 0.1*per
        self.fit_circ[plname]=fit_circ

    def add_transiting_planet(self,plname,tcen,per,Rp_Rs,tdur,model_params,k=None,fix_params=[],e_per=None,e_tcen=None,bmu=None,bsigma=None,fit_circ=False):
        """Add transiting planet.
        
        Args:
            plname (str): Planet name
            per (float): Period
            tcen (float): Transit time
            Rp_Rs
            tdur
            model_params (list): [k/logk]
            fix_params (list, optional): [per,tcen]
            e_per (float, optional): Uncertainty on the period
            e_tcen (float): Uncertainty on the time of perihelion
            """
        assert plname not in self.pls, plname+" Already in the planets to be modelled ("+",".join(self.pls)+")"
        self.pls+=[plname]
        self.plinfo.update({'per_'+str(plname):per,
                'tc_'+str(plname):tcen,
                'Rp_Rs_'+str(plname):np.nan,
                'tdur_'+str(plname):tdur,
                'bmu_'+str(plname):np.nan,
                'bsigma_'+str(plname):np.nan,
                'model_params_'+str(plname):model_params,
                'fix_params_'+str(plname):fix_params,
                'type_'+str(plname):'transiting'})
        if k is not None:
            self.plinfo['k_'+str(plname)]=k
        self.plinfo['e_tc_'+str(plname):e_tcen] if e_tcen is not None else 0.55*per
        self.plinfo['e_per_'+str(plname):e_per] if e_per is not None else 0.1*per
        self.fit_circ[plname]=fit_circ

    def init_model(self):
        """Initialising RVultra model"""

        #Initialising some data
        self.data['med_shift_rvs']=self.data['mnvel'].values[:]
        for scope in np.unique(self.data['instr'].values):
            self.data.loc[self.data['instr']==scope,'med_shift_rvs']-=np.nanmedian(self.data.loc[self.data['instr']==scope,'mnvel'])
        remanom_ptp=abs(np.sort(self.data['med_shift_rvs'])[-2]-np.sort(self.data['med_shift_rvs'])[1])
        self.data=self.data.sort_values('time')

        #Setting up some parameters:
        self.n_pls = len(self.pls)
        if self.ecc_prior is None:
            #Setting eccentricity prior according to multiplicity
            self.ecc_prior = 'vaneylen' if self.n_pls>1 else 'kipping'
        self.refepoch = np.average(self.data['time']) if self.refepoch is None else self.refepoch
        minmaxp=[np.min([self.plinfo['per_'+str(n)] for n in self.pls]),np.max([self.plinfo['per_'+str(n)] for n in self.pls])]
        self.finet = np.arange(np.min(self.data['time'])-0.5*minmaxp[1],np.max(self.data['time'])+0.5*minmaxp[1],minmaxp[0]/33)

        self.varnames=[]
        self.varpars={}
        #Creating index dictionary 'param' -> number index in x
        for i_p in self.pls:
            #some initial deriving/filling in missing params
            if 'a_'+str(i_p) not in self.plinfo:
                self.plinfo['a_'+i_p]=semi_major_axis(self.plinfo['per_'+i_p], self.Ms)*215.03215567054764
            if self.plinfo['type_'+str(i_p)]=='transiting' and 'bmu_'+i_p not in self.plinfo:
                self.plinfo['bmu_'+i_p]=np.nan
                self.plinfo['bsigma_'+i_p]=np.nan
            if self.fit_circ[i_p] is None:
                #Defaulting to False for fitting circular orbits
                self.fit_circ[i_p]=False

            #Looping through all varied parameters to set them in the "official" parameter index:
            if 'per' in self.plinfo['model_params_'+str(i_p)]:
                self.varnames+=['per_'+str(i_p)]
                self.plinfo['x_index']['per_'+str(i_p)]=self.varnames.index('per_'+str(i_p))
                self.varpars.update({'per_'+str(i_p):['uniform',self.plinfo['per_'+i_p]-self.plinfo['e_per_'+i_p],self.plinfo['per_'+i_p]+self.plinfo['e_per_'+i_p]]})
            if 'tc' in self.plinfo['model_params_'+str(i_p)]:
                self.varnames+=['tc_'+str(i_p)]
                self.plinfo['x_index']['tc_'+str(i_p)]=self.varnames.index('tc_'+str(i_p))
                self.varpars.update({'tc_'+str(i_p):['uniform',self.plinfo['tc_'+i_p]-self.plinfo['e_tc_'+i_p],self.plinfo['tc_'+i_p]+self.plinfo['e_tc_'+i_p]]})
            if self.kprior=='linear' and 'k' in self.plinfo['model_params_'+str(i_p)]:
                self.varnames+=['k_'+str(i_p)]
                self.varpars.update({'k_'+str(i_p):['uniform',0.1*np.nanstd(self.data['med_shift_rvs']),remanom_ptp]})
                self.plinfo['x_index']['k_'+str(i_p)]=self.varnames.index('k_'+str(i_p))
            elif self.kprior=='log' and 'logk' in self.plinfo['model_params_'+str(i_p)]:
                self.varnames+=['logk_'+str(i_p)]
                self.varpars.update({'logk_'+str(i_p):['uniform',np.log(0.1*np.nanstd(self.data['med_shift_rvs'])),np.log(remanom_ptp)]})
                self.plinfo['x_index']['logk_'+str(i_p)]=self.varnames.index('logk_'+str(i_p))
            if not self.fit_circ[i_p]:
                self.varnames+=['e_'+str(i_p)]
                if self.ecc_prior=='vaneylen':
                    self.varpars.update({'e_'+str(i_p):['halfnorm',0.0,0.083]})
                elif self.ecc_prior=='kipping':
                    self.varpars.update({'e_'+str(i_p):['beta',0.867,3.03]})
                self.plinfo['x_index']['e_'+str(i_p)]=self.varnames.index('e_'+str(i_p))
            
                self.varnames+=['w_'+str(i_p)]
                self.varpars.update({'w_'+str(i_p):['uniform',-np.pi,np.pi]})
                self.plinfo['x_index']['w_'+str(i_p)]=self.varnames.index('w_'+str(i_p))

        #Instrumental offsets and jitters:
        self.inst_ix_list={i:self.data['instr'].values.astype(str)==i for i in self.insts}
        for n_i,i in enumerate(self.insts):
            if self.jitterprior=='log':
                self.varnames+=['logjitter_'+str(i)]
                self.varpars.update({'logjitter_'+i:['norm', np.log(np.nanmedian(self.data.loc[self.inst_ix_list[i],'errvel'])),0.5]})
                self.plinfo['x_index']['logjitter_'+str(i)]=self.varnames.index('logjitter_'+str(i))
            elif self.jitterprior=='linear':
                self.varnames+=['jitter_'+str(i)]
                self.varpars.update({'jitter_'+i:['norm', np.nanmedian(self.data.loc[self.inst_ix_list[i],'errvel']),0.5]})
                self.plinfo['x_index']['jitter_'+str(i)]=self.varnames.index('jitter_'+str(i))
            self.varnames+=['offset_'+str(i)]
            self.varpars.update({'offset_'+i:['norm',np.nanmedian(self.data.loc[self.inst_ix_list[i],'mnvel'].values),np.nanstd(self.data.loc[self.inst_ix_list[i],'mnvel'].values)]})
            self.plinfo['x_index']['offset_'+str(i)]=self.varnames.index('offset_'+str(i))
        
        #RV trends:
        self.varpars.update({'polytrend':['norm',np.polyfit(self.data['time']-self.refepoch,self.data['mnvel'].values,1)[0],
                                          np.nanstd(self.data['mnvel'])/np.ptp(self.data['time'].values)]})
        self.varnames+=['polytrend']
        self.plinfo['x_index']['polytrend']=self.varnames.index('polytrend')
        if self.fit_curv:
            self.varpars.update({'polycurv':['norm',0.0,(np.nanstd(self.data['mnvel'].values)/(0.2*np.ptp(self.data['time'].values)))**2]})
            self.varnames+=['polycurv']
            self.plinfo['x_index']['polycurv']=self.varnames.index('polycurv')
        self.initialised=True


    def __call__(self, x):
        """Custom Ultranest log-likelihood function."""
        #varpars format = [k,secosw,sesinw],[gamma,jitter],dvdt,curv
        assert self.initialised, "Must have run `model.init_model` before you can start the nested sampling..."

        rv_orbit=np.zeros((np.shape(x)[0],len(self.data)))
        logprior = np.zeros(x.shape[0])

        #Getting the RV model for each planet
        for n in self.pls:
            if self.plinfo['type_'+str(n)]!='rv':
                per= self.plinfo['per_'+str(n)]
                t0 = self.plinfo['tc_'+str(n)]
            else:
                per= x[:,self.plinfo['x_index']['per_'+str(n)]]
                t0 = x[:,self.plinfo['x_index']['tc_'+str(n)]]
            if self.kprior=='log' and 'logk_'+str(n) in self.plinfo['x_index']:
                k=np.exp(x[:,self.plinfo['x_index']['logk_'+str(n)]])
            elif self.kprior=='linear' and 'k_'+str(n) in self.plinfo['x_index']:
                k=x[:,self.plinfo['x_index']['k_'+str(n)]]
            else:
                k=self.plinfo['k_'+str(n)]
            if self.fit_circ[n]:
                e=0;w=0
                rv_orbit += radial_velocity_circ_vect(self.data['time'].values,per, t0, k)
            else:
                e=x[:,self.plinfo['x_index']['e_'+str(n)]]
                w=x[:,self.plinfo['x_index']['w_'+str(n)]]
                rv_orbit += radial_velocity_ecc_vect(self.data['time'].values,per, t0, e, w, k)
            if self.plinfo['type_'+str(n)]!='rv' and not self.fit_circ[n]:
                logprior += TransitPrior_vect(t0,per,e,w,self.plinfo['a_'+str(n)],
                                                  self.Rs,self.plinfo['Rp_Rs_'+str(n)],self.plinfo['tdur_'+str(n)],
                                                  bmu=self.plinfo['bmu_'+str(n)],bsigma=self.plinfo['bsigma_'+str(n)])
                
        #Doing instrumental stuff:
        sigma2=np.tile(self.data['errvel'].values**2,(np.shape(x)[0],1))
        instr_offsets=np.zeros((np.shape(x)[0],len(self.data)))
        for n_i,i in enumerate(self.insts):
            if self.kprior=='log':
                sigma2[:,self.inst_ix_list[i]]+=np.exp(x[:,self.plinfo['x_index']['logjitter_'+str(i)]])[:,None]**2    #jitter
            elif self.kprior=='linear':
                sigma2[:,self.inst_ix_list[i]]+=x[:,self.plinfo['x_index']['jitter_'+str(i)]][:,None]**2    #jitter
            instr_offsets[:,self.inst_ix_list[i]]+=x[:,self.plinfo['x_index']['offset_'+str(i)]][:,None]          #gamma
        if self.fit_curv:
            trend=np.polyval(np.vstack([x[:,self.plinfo['x_index']['polycurv']][None,:],x[:,self.plinfo['x_index']['polytrend']][None,:],np.zeros(x.shape[0])[None,:]]), 
                             self.data['time'].values[:,None] - self.refepoch).T
        else:
            trend=x[:,self.plinfo['x_index']['polytrend']][None,:]*(self.data['time'].values[:,None]-self.refepoch).T
        full_rv_model = rv_orbit+instr_offsets+trend
        llk = -0.5 * np.sum((self.data['mnvel'].values[None,:] - full_rv_model) ** 2 / sigma2 + np.log(sigma2),axis=1)
        return llk
    
    def return_tmodels(self,x):
        """Given a vector x, generate the various RV models for the observations"""
        tmodels={}
        for n in self.pls:
            if self.plinfo['type_'+str(n)]!='rv':
                per= self.plinfo['per_'+str(n)]
                t0 = self.plinfo['tc_'+str(n)]
            else:
                per= x[self.plinfo['x_index']['per_'+str(n)]]
                t0 = x[self.plinfo['x_index']['tc_'+str(n)]]
            if self.kprior=='log':
                k=np.exp(x[self.plinfo['x_index']['logk_'+str(n)]])
            else:
                k=x[self.plinfo['x_index']['k_'+str(n)]]
            if self.fit_circ[n]:
                e=0
                w=0
            else:
                e=x[self.plinfo['x_index']['e_'+str(n)]]
                w=x[self.plinfo['x_index']['w_'+str(n)]]
            tmodels["rv_"+str(n)] = radial_velocity(self.data['time'].values,per, t0, e, w, k)

        tmodels['instr_offsets']=np.zeros(len(self.data))
        for n_i,i in enumerate(self.insts):
            tmodels['instr_offsets'][self.inst_ix_list[i]]+=x[self.plinfo['x_index']['offset_'+i]]          #gamma
        if self.fit_curv:
            tmodels['trend']=np.polyval([x[self.plinfo['x_index']['polycurv']],x[self.plinfo['x_index']['polytrend']],0],self.data['time'].values-self.refepoch)
        else:
            tmodels['trend']=x[self.plinfo['x_index']['polytrend']]*(self.data['time'].values-self.refepoch)
        #tmodels['full'] = np.column_stack([m for m in tmodels])
        #print(i,np.sum(idx_planet1),np.sum(np.array(results['positions'][2])==-2),times,ttimes[n][:,0],sigma2)
        return tmodels
        
    def return_finetmodels(self,x):
        """Given a vector x, generate the various RV models for a fine grid of times"""
        finetmodels={}
        for n in self.pls:
            if self.plinfo['type_'+str(n)]!='rv':
                per= self.plinfo['per_'+str(n)]
                t0 = self.plinfo['tc_'+str(n)]
            else:
                per= x[self.plinfo['x_index']['per_'+str(n)]]
                t0 = x[self.plinfo['x_index']['tc_'+str(n)]]
            if self.kprior=='log':
                k=np.exp(x[self.plinfo['x_index']['logk_'+str(n)]])
            else:
                k=x[self.plinfo['x_index']['k_'+str(n)]]
            if self.fit_circ[n]:
                e=0
                w=0
            else:
                e=x[self.plinfo['x_index']['e_'+str(n)]]
                w=x[self.plinfo['x_index']['w_'+str(n)]]
            finetmodels["rv_"+str(n)] = radial_velocity(self.finet, per, t0, e, w, k)
        if self.fit_curv:
            finetmodels['trend']=np.polyval([x[self.plinfo['x_index']['polycurv']],x[self.plinfo['x_index']['polytrend']],0],self.finet-self.refepoch)
        else:
            finetmodels['trend']=x[self.plinfo['x_index']['polytrend']]*(self.finet-self.refepoch)
        #xmodels['full'] = np.column_stack([m for m in models])
        #print(i,np.sum(idx_planet1),np.sum(np.array(results['positions'][2])==-2),times,ttimes[n][:,0],sigma2)
        return finetmodels
    
    def sample(self,resume=True):
        self.fixsampler = ultranest.ReactiveNestedSampler(self.varnames, self, VectorPriorTransform(self.varpars), 
                                                          log_dir=self.outfileloc, resume=resume, vectorized=True,
                                                          wrapped_params=['w_' in par for par in self.varnames])
        self.result = self.fixsampler.run()
        self.fixsampler.print_results()

    def sample_plots(self):
        assert hasattr(self,'result'), "Must have run `sample` before plotting"
        self.fixsampler.plot_run()
        self.fixsampler.plot_trace()
        self.fixsampler.plot_corner()

    def rv_plot(self,savetype='pdf'):
        #Plotting regions as split panel plot: One panel for each planet, plus the timeseries.
        points=self.result['weighted_samples']['points']
        weights=self.result['weighted_samples']['weights']
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        #Finding number of large jumps in RVs, defined as >20% of ptp missing inbetween
        diffjumps = np.hstack([-1,np.where(np.diff(np.sort(self.data['time']))>np.clip(0.2*np.ptp(self.data['time']),100,1e5))[0],len(self.data['time'])-1])
        timesort=np.sort(self.data['time'])
        rv_islands = {dj:[timesort[diffjumps[dj]+1],timesort[diffjumps[dj+1]]] for dj in range(len(diffjumps)-1)}
        #Basic unit = 30days
        plot_sizes={r:int(np.ceil((rv_islands[r][1]-rv_islands[r][0])/30)) for r in rv_islands}
        
        # Need to get highest probability model, as well as a random selection (weighted by logprob) to generate model percentiles
        ok_points=weights>0
        t_models=[self.return_tmodels(points[ok_points][ok_pt,:]) for ok_pt in range(np.sum(ok_points))]
        finet_models=[self.return_finetmodels(points[ok_points][ok_pt,:]) for ok_pt in range(np.sum(ok_points))]
        
        allt_models={}
        allfinet_models={}
        percentiles={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868}
        percentile_names=list(percentiles.keys())
        
        tmods=self.data.loc[:,:]#Also taking scopes, RVs, etc.
        allt_models['all']=np.zeros((len(t_models),len(t_models[0]['trend'])))
        for col in t_models[0]:
            allt_models[col]=np.vstack([t_models[n][col] for n in range(np.sum(ok_points))])
            allt_models['all']+=allt_models[col] #Summing all models - fine t array
            for n_pc in range(5):
                tmods[col+"_"+percentile_names[n_pc]]=np.hstack([weighted_quantile(allt_models[col][:,n],percentiles[percentile_names[n_pc]]/100,sample_weight=weights[ok_points]) for n in range(len(self.data['time']))])
        for n_pc in range(5):
            tmods["all_"+percentile_names[n_pc]]=np.hstack([weighted_quantile(allt_models['all'][:,n],percentiles[percentile_names[n_pc]]/100.,sample_weight=weights[ok_points]) for n in range(len(self.data['time']))])

        allfinet_models['all']=np.zeros((len(finet_models),len(finet_models[0]['trend'])))
        finemods=pd.DataFrame({'time':self.finet})
        for col in finet_models[0]:
            allfinet_models[col]=np.vstack([finet_models[n][col] for n in range(np.sum(ok_points))])
            allfinet_models['all']+=allfinet_models[col] #Summing all models - fine t array
            for n_pc in range(5):
                finemods[col+"_"+percentile_names[n_pc]]=np.hstack([weighted_quantile(allfinet_models[col][:,n],percentiles[percentile_names[n_pc]]/100.,
                                                                                      sample_weight=weights[ok_points]) for n in range(len(self.finet))])
        for n_pc in range(5):
            finemods["all_"+percentile_names[n_pc]]=np.hstack([weighted_quantile(allfinet_models['all'][:,n],percentiles[percentile_names[n_pc]]/100.,sample_weight=weights[ok_points]) for n in range(len(self.finet))])
        
        tmods.to_csv(os.path.join(self.outfileloc,self.name+"_"+self.kprior+"_rv_models.csv"))
        finemods.to_csv(os.path.join(self.outfileloc,self.name+"_"+self.kprior+"_finet_rv_models.csv"))

        fig = plt.figure()
        gs = gridspec.GridSpec(ncols=np.sum(list(plot_sizes.values())), nrows=1+self.n_pls, figure=fig)
        colcum=0
        rvsubplot={};plsubplots={}

        for r in rv_islands:
            rvsubplot[r]=fig.add_subplot(gs[0,colcum:colcum+plot_sizes[r]])
            colcum+=plot_sizes[r]
            for n_in, instr in enumerate(self.insts):
                ix_rv=(self.data['time']>=rv_islands[r][0])&(self.data['time']<=rv_islands[r][1])&(self.data['instr']==instr)
                if np.sum(ix_rv)>0:
                    if r==np.argmax(list(plot_sizes.values())):
                        lab=instr
                    else:
                        lab=None    
                    rvsubplot[r].errorbar(self.data.loc[ix_rv,'time']-2457000,self.data.loc[ix_rv,'mnvel']-tmods['instr_offsets_med'][ix_rv],
                                          yerr=self.data.loc[ix_rv,'errvel'],fmt='.',color='C'+str(n_in),label=lab)
            rvsubplot[r].fill_between(self.finet-2457000,finemods['all_-2sig'],finemods['all_+2sig'],color='C6',alpha=0.15)
            rvsubplot[r].fill_between(self.finet-2457000,finemods['all_-1sig'],finemods['all_+1sig'],color='C6',alpha=0.15)
            rvsubplot[r].plot(self.finet-2457000,finemods['all_med'],'--',color='C6',alpha=0.75)

            if r==0:
                rvsubplot[r].set_ylabel("RV [m/s]")
            else:
                rvsubplot[r].set_yticks([])
                rvsubplot[r].set_yticklabels([])
            
            rvsubplot[r].set_ylim(np.min(np.hstack([finemods['all_-1sig'],np.sort(self.data['mnvel']-tmods['instr_offsets_med'])[1:-1]])-1),
                                  np.max(np.hstack([finemods['all_+1sig'],np.sort(self.data['mnvel']-tmods['instr_offsets_med'])[1:-1]])+1))
            
            rvsubplot[r].set_xlim(rv_islands[r][0]-5-2457000,rv_islands[r][1]+5-2457000)
            rvsubplot[r].xaxis.tick_top()
            if r==np.argmax(list(plot_sizes.values())):
                #Putting axis label only on biggest axis
                rvsubplot[r].set_xlabel("Time [BJD-2457000]")
                rvsubplot[r].xaxis.set_label_position('top')
                if len(self.insts)==2 or len(self.insts)>3:
                    rvsubplot[r].legend(fontsize=8,ncols=2)
                else:
                    rvsubplot[r].legend(fontsize=8)
            #Catching very small plot size labels
            if plot_sizes[r]<0.06*np.sum(list(plot_sizes.values())):
                #Doing a single tick for tiny axes
                label=int(50*np.round((np.average(rv_islands[r])-2457000)/50))
                rvsubplot[r].set_xticks([label])
                rvsubplot[r].set_xticklabels([str(label)])
            
        for n_pl,pl in enumerate(self.pls):
            #Plotting phase-folded planet
            plsubplots[pl]=fig.add_subplot(gs[1+n_pl,:])
            if self.plinfo['type_'+str(pl)]!='rv':
                phase=(self.data['time'].values-0.5*self.plinfo['per_'+str(pl)]-self.plinfo['tc_'+str(pl)])%self.plinfo['per_'+str(pl)]-0.5*self.plinfo['per_'+str(pl)]
            else:
                ix = weights>1e-4
                per,eper = weighted_avg_and_std(points[ix,self.plinfo['x_index']['per_'+pl]],weights=weights[ix])
                t0,et0   = weighted_avg_and_std(points[ix,self.plinfo['x_index']['tc_'+pl]], weights=weights[ix])
                phase=(self.data['time'].values-0.5*per-t0)%per-0.5*per
            for n_in,instr in enumerate(self.insts):
                ix_rv=self.data['instr']==instr
                plt.errorbar(phase[ix_rv],self.data.loc[ix_rv,'mnvel']-(tmods.loc[ix_rv,'all_med']-tmods.loc[ix_rv,'rv_'+str(pl)+'_med']),
                             yerr=self.data.loc[ix_rv,'errvel'],fmt='.',color='C'+str(n_in))
            plt.errorbar(np.hstack([phase-self.plinfo['per_'+str(pl)],phase+self.plinfo['per_'+str(pl)]]),
                         np.hstack([self.data.loc[:,'mnvel']-(tmods['all_med']-tmods['rv_'+str(pl)+'_med']),
                                    self.data.loc[:,'mnvel']-(tmods['all_med']-tmods['rv_'+str(pl)+'_med'])]),
                         yerr=np.hstack([self.data.loc[:,'errvel'],self.data.loc[:,'errvel']]),
                         fmt='.',color='gray',alpha=0.7)
            bins=np.linspace(-0.5*self.plinfo['per_'+str(pl)],0.5*self.plinfo['per_'+str(pl)],11)
            if len(self.data)>15:
                labelled=False
                for nbin in range(10):
                    ix=(phase>bins[nbin])&(phase<=bins[nbin+1])
                    if np.sum(ix)>1:
                        #Incorporate additional jitter here?
                        out=weighted_avg_and_std(self.data.loc[ix,'mnvel']-(tmods['all_med'][ix]-tmods['rv_'+str(pl)+'_med'][ix]),errs=self.data.loc[ix,'errvel'])
                        if not labelled:
                            plt.errorbar(0.5*(bins[nbin]+bins[nbin+1]),out[0],yerr=out[1],fmt='o',color='k',
                                     lw=1,markeredgecolor='k',markerfacecolor='w',alpha=0.9,label="binned")
                        else:
                            plt.errorbar(0.5*(bins[nbin]+bins[nbin+1]),out[0],yerr=out[1],fmt='o',color='k',
                                     lw=1,markeredgecolor='k',markerfacecolor='w',alpha=0.9)
            if self.plinfo['type_'+str(pl)]!='rv':
                finetphase=(self.finet-0.5*self.plinfo['per_'+str(pl)]-self.plinfo['tc_'+str(pl)])%self.plinfo['per_'+str(pl)]-0.5*self.plinfo['per_'+str(pl)]
            else:
                finetphase=(self.finet-0.5*per-t0)%per-0.5*per
            plsubplots[pl].fill_between(np.sort(finetphase),finemods['rv_'+str(pl)+"_-2sig"][np.argsort(finetphase)],finemods['rv_'+str(pl)+"_+2sig"][np.argsort(finetphase)],color='C6',alpha=0.15)
            plsubplots[pl].fill_between(np.sort(finetphase),finemods['rv_'+str(pl)+"_-1sig"][np.argsort(finetphase)],finemods['rv_'+str(pl)+"_+1sig"][np.argsort(finetphase)],color='C6',alpha=0.15)
            plsubplots[pl].plot(np.hstack([np.sort(finetphase)-self.plinfo['per_'+str(pl)],np.sort(finetphase),np.sort(finetphase)+self.plinfo['per_'+str(pl)]]),
                                np.hstack([finemods['rv_'+str(pl)+"_med"][np.argsort(finetphase)],finemods['rv_'+str(pl)+"_med"][np.argsort(finetphase)],finemods['rv_'+str(pl)+"_med"][np.argsort(finetphase)]]),c='C6')
            plsubplots[pl].set_ylim(np.min(np.hstack([np.sort(self.data.loc[:,'mnvel']-tmods['instr_offsets_med']-tmods['trend_med'])[1:-1]*1.1]+[finemods['rv_'+str(i_pl)+"_-2sig"]*1.75 for i_pl in self.pls])),
                                    np.max(np.hstack([np.sort(self.data.loc[:,'mnvel']-tmods['instr_offsets_med']-tmods['trend_med'])[1:-1]*1.1]+[finemods['rv_'+str(i_pl)+"_+2sig"]*1.75 for i_pl in self.pls])))
            plsubplots[pl].set_xlim(-0.7*self.plinfo['per_'+str(pl)],0.7*self.plinfo['per_'+str(pl)])
            if pl==self.pls[-1]:
                plsubplots[pl].set_xlabel("Time from transit [d]")
                plsubplots[pl].set_ylabel("RV [m/s]")
            else:
                plsubplots[pl].set_xticks([])
                plsubplots[pl].set_xticklabels([])
            plsubplots[pl].set_ylabel("RV [m/s]")
        fig.subplots_adjust(wspace=0.02*colcum)
        fig.subplots_adjust(hspace=0.03*(1+self.n_pls))
        if not os.path.isdir(os.path.join(self.outfileloc,"plots")):
            os.mkdir(os.path.join(self.outfileloc,"plots"))
        fig.savefig(os.path.join(self.outfileloc,"plots"+self.name+"fit_rvmodel_plot"+savetype))
    
    def summarise_results(self):

        prcnts={'-2sig':2.2750132, '-1sig':15.8655254, 'med':50., '+1sig':84.1344746, '+2sig':97.7249868,'+3sig':99.74}
        summary_df={}
        for n_v, var in enumerate(self.result['paramnames']):
            av,sd=weighted_avg_and_std(self.result['weighted_samples']['points'][:,n_v],errs=None,weights=self.result['weighted_samples']['weights'])
            summary_df[var]={'mean':av,'sd':sd}
            summary_df[var].update({ipc:weighted_quantile(self.result['weighted_samples']['points'][:,n_v], 0.01*prcnts[ipc], sample_weight=self.result['weighted_samples']['weights']) for ipc in prcnts})
        for pl in self.pls:
            if self.kprior=='log':
                k=np.exp(self.result['weighted_samples']['points'][:,self.plinfo['x_index']['logk_'+pl]])
            if 'per_'+pl in self.result['paramnames']:
                per=self.result['weighted_samples']['points'][:,self.plinfo['x_index']['per_'+pl]]
            else:
                per=self.plinfo['per_'+pl]

            if 'e_'+pl in self.result['paramnames']:
                e=self.result['weighted_samples']['points'][:,self.plinfo['x_index']['e_'+pl]]
            elif self.fit_circ:
                e=0

            mpls=((per*86400/(2*np.pi*6.6e-11))**(1/3)*k*(self.Ms*1.96e30)**(2/3)*np.sqrt(1-e**2))/5.96e24
            av,sd=weighted_avg_and_std(mpls, errs=None,weights=self.result['weighted_samples']['weights'])
            summary_df['mpsini_'+pl]={'mean':av, 'sd':sd}
            summary_df['mpsini_'+pl].update({ipc:weighted_quantile(mpls, 0.01*prcnts[ipc], sample_weight=self.result['weighted_samples']['weights']) for ipc in prcnts})
        
        summary_df = pd.DataFrame(summary_df).T
        summary_df=pd.concat([summary_df,pd.DataFrame({'mean':self.result['logz'],'sd':self.result['logzerr']},index=['logEvidence'])])
        summary_df.to_csv(os.path.join(self.outfileloc,self.name+"_"+self.kprior+"k_self.results.csv"))
        #[K*np.sqrt(self.result['logzerr']**2+self.result['logzerr']**2)]},index=['BayesFact'])
        return summary_df


# def run(tic,kprior='log', fit_curv=True, fit_circ=False, resume=True, outfileloc="/home/hosborn/home1/python/RVultra/NewUltranestOutputs"):
    
#     stars=pd.read_excel("/home/hosborn/home1/python/RVultra/AllTargetInfo.xlsx", sheet_name='stars')
#     stars=stars.loc[(stars["In paper?"]==1)&(stars["Needs validation"]>0)]
#     star=stars.loc[stars['TIC'].values.astype(int)==int(tic)]
#     if type(star)==pd.DataFrame:
#         star=star.iloc[0]#(stars["In paper?"]==1)&(stars["Needs validation"]>0)&(stars['TIC']==128703021)].iloc[0]
#     pls=pd.read_excel("/home/hosborn/home1/python/RVultra/AllTargetInfo.xlsx", sheet_name='planets')
#     pl=pls.loc[pls['TIC']==tic]
    
#     # Define global planetary system and dataset parameters
#     starname = star['Paper name'].replace(" ","")
#     print(outfileloc,starname+"_"+kprior+"K")
#     print(os.path.join(outfileloc,starname+"_"+kprior+"K"))

#     #print(star['mass'],star['radius'],type(star['radius']),type(star['mass']))
#     resume=True if (resume=="True" or resume=="1") else "overwrite"
#     lik=Vector_PlanetLikelihood(name=starname,kprior=kprior, fit_curv=fit_curv)
#     lik.add_starpars(Ms=star['mass'], Rs=star['radius'])
    
#     data=assemble_all_rvs("/home/hosborn/home1/python/RVultra/allrvs/"+starname)
#     lik.add_data(time=data['time'].values,rvs=data['mnvel'].values,rv_errs=data['errvel'].values,instr=data['instr'].values)
#     for i_p,ipl in enumerate(pl.iterrows()):
#         if ipl[1]['type']=='rv':
#             lik.add_rv_planet(per=ipl[1]['period'],tcen=2457000+ipl[1]['tcen1'],fit_circ=True)
#         else:
#             iper=ipl[1]['period'] if (~pd.isnull(ipl[1]['period']))&(ipl[1]['period']!='per') else ipl[1]['true period']
#             lik.add_transiting_planet(plname=(ipl[1]['Paper name']+ipl[1]['name']).replace(' ',''),
#                                       per=iper,tcen=np.nanmax(ipl[1][['tcen1','tcen2','tcen3']].values),
#                                       Rp_Rs=ipl[1]['rprs'],tdur=ipl[1]['duration'],
#                                       bmu=ipl[1]['bmu'], bsigma=ipl[1]['bsigma'],
#                                       model_params=['logk'],fix_params=['per','tc'],fit_circ=False)

#     lik.init_model()
#     #plinfo=plinfo, data=data, varnames=list(varpars.keys()), 
                                
#     #print(star['mass'], star['radius'], refepoch, pl.shape[0], plinfo, kprior, fit_curv,resume)
#     fixsampler = ultranest.ReactiveNestedSampler(lik.varnames, lik, VectorPriorTransform(lik.varpars), 
#                                                 log_dir=os.path.join(outfileloc,starname+"_"+kprior+"K"),
#                                                 resume=resume,vectorized=True,
#                                                 wrapped_params=['w_' in par for par in lik.varnames])
#     result = fixsampler.run()
#     fixsampler.print_results()

#     fixsampler.plot_run()
#     fixsampler.plot_trace()
#     fixsampler.plot_corner()

#     lik.plot(fixsampler.results['weighted_samples']['points'], fixsampler.results['weighted_samples']['weights'], os.path.join(outfileloc,starname+"_"+kprior+"K","plots"))
    
#     flatvarpars={vpk:varpars[vpk] for vpk in varpars if vpk[:4] in ['poly','jitt','logj','offs']}
#     flatlik=Vector_FlatLikelihood(data,varnames=list(flatvarpars.keys()), jitprior=kprior, fit_curv=fit_curv)
#     flatsampler = ultranest.ReactiveNestedSampler(list(flatvarpars.keys()), flatlik, VectorPriorTransform(flatvarpars), 
#                                                 log_dir=os.path.join(outfileloc,starname+"_"+kprior+"nopl"),
#                                                 resume=resume, vectorized=True)
#     flatresult = flatsampler.run()
#     K = np.exp(result['logz'] - flatresult['logz'])
#     print("K = %.2f" % K)
#     print("The planetary model is %.2f times more probable than the no-planet model" % K)
#     idf1 = pd.DataFrame(data=result['samples'], columns=[c+"_plmodel" for c in result['paramnames']])
#     idf2 = pd.DataFrame(data=flatresult['samples'], columns=[c+"_noplmodel" for c in flatresult['paramnames']])
#     df = pd.concat([pd.DataFrame({'mean':[K],'std':[K*np.sqrt(result['logzerr']**2+result['logzerr']**2)]},index=['BayesFact']),idf1.describe().T,idf2.describe().T])
#     df.to_csv(os.path.join(outfileloc,starname+"_"+kprior+"K","results.csv"))

#     return lik, fixsampler

# if __name__=="__main__":
#     import argparse
    
    
#     # Initialize parser
#     parser = argparse.ArgumentParser()
    
#     # Adding optional argument
#     parser.add_argument("-t", "--TIC", help = "TESS Input Catalog number")
#     parser.add_argument("-kp", "--Kprior", help = "Prior on semi-amplitude K; either linear or log")
#     parser.add_argument("-c", "--curv", help = "Whether to include curvature in the background fit or not")
#     parser.add_argument("-r", "--resume", help = "Whether to resume or not")
#     parser.add_argument("-o", "--outfileloc", help = "Output file location")

#     args = parser.parse_args()
#     _=run(int(args.TIC), kprior=args.Kprior, fit_curv=bool(args.curv),resume=args.resume,
#           outfileloc=args.outfileloc)
# result = fixsampler.run(max_ncalls=25000)
# fixsampler.print_results()

# fixsampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=(1+(len(varpars)))*2)
# stepresult = fixsampler.run(max_ncalls=1e9,dlogz=0.5,region_class=ultranest.mlfriends.RobustEllipsoidRegion)
#                             #viz_callback=False,show_status=False