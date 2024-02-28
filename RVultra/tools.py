import os
import pandas as pd
import numpy as np
import radvel
import scipy
import ultranest
import glob

def bin_by_time(rv,bin_days=1):
    rv['grouped_dates'] = np.floor(rv['time']/bin_days).astype(int)
    rv_mean = rv.loc[:,['grouped_dates','time','mnvel','errvel']].groupby('grouped_dates',as_index=False).mean()
    rv_n = np.array([np.sum(rv['grouped_dates']==jdd) for jdd in rv_mean['grouped_dates'].values])
    rv_mean['errvel'] = rv_mean['errvel'] / np.array(np.sqrt(rv_n))
    rv_mean['grouped_dates']*=bin_days
    return rv_mean

def extract_rvfile(rvfile):
    if '.rdb' in rvfile:
        scopename=""
        if "harpsn" in rvfile.lower() or "harps-n" in rvfile.lower() or "harpn" in rvfile.lower():
            scopename="HARPS-N"
        elif "harps" in rvfile.lower():
            scopename="HARPS"
        elif "cor" in rvfile.lower():
            scopename="COR"
        else:
            print(rvfile,"'Scope not recognised")
        rvdat=pd.read_table(rvfile,delim_whitespace=True).iloc[1:]
        for col in rvdat.columns:
            try:
                rvdat[col]=pd.to_numeric(rvdat[col].values)
            except:
                print(col, " - column in "+rvfile.split("/")[-1]+" cannot be converted numerically")
        rvdat['time']=rvdat['rjd']
        rvdat['mnvel']=rvdat['vrad']
        rvdat['errvel']=rvdat['svrad']
    elif '.pfs.txt' in rvfile:
        rvdat=pd.read_fwf(rvfile,names=['jdb','vrad','svrad','a','b','c','d'])
        rvdat['time']=rvdat['jdb']
        rvdat['mnvel']=rvdat['vrad']
        rvdat['errvel']=rvdat['svrad']
        scopename="PFS"
    elif 'CCF_data.txt' in rvfile:
        #MINERVA
        rvdat=pd.read_fwf(rvfile,names=['jdb','vrad','svrad','a','b','c','d'])
        rvdat['time']=rvdat['jdb']
        # #Subtracting yearly averages as not stable
        # digis=np.digitize(rvdat['time'],np.arange(np.min(rvdat['time']),np.max(rvdat['time']),365))
        # offsets={d:np.average(rvdat.loc[digis==d,'vrad']) for d in np.unique(digis)}
        # offset_array=[offsets[digis[i]] for i in range(len(digis))]
        # rvdat['mnvel']=rvdat['vrad']-offset_array
        rvdat['mnvel']=rvdat['vrad']
        rvdat['errvel']=rvdat['svrad']
        #Binned by 3 days
        rvdat=bin_by_time(rvdat,3)#Binning by 3nights.
        scopename="MINERVA"
    elif '_cleaned2.csv' in rvfile:
        if "harpsn" in rvfile.lower() or "harps-n" in rvfile.lower():
            scopename="HARPS-N"
        elif "harps" in rvfile.lower():
            scopename="HARPS"
        elif "cor" in rvfile.lower():
            scopename="COR"
        else:
            print(rvfile,"'Scope not recognised")
        rvdat=rvdat=pd.read_csv(rvfile)
        rvdat['time']=rvdat['jdb']
        rvdat['mnvel']=rvdat['vrad_cleaned']
        rvdat['errvel']=rvdat['svrad']
    elif '.vels' in rvfile:
        rvdat=pd.read_csv(rvfile,delim_whitespace=True,names=['jdb','vrad','svrad','a','b','c','texp'])
        rvdat['time']=rvdat['jdb']
        rvdat['mnvel']=rvdat['vrad']
        rvdat['errvel']=rvdat['svrad']
        rvdat=bin_by_time(rvdat)
        scopename="PFS"
    elif '.csv' in rvfile:
        rvdat=pd.read_csv(rvfile)
        if 'RV(km/s)' in rvdat.columns:
            rvdat['time']=rvdat['date (BJD)']
            rvdat['mnvel']=rvdat['RV(km/s)']*1000
            rvdat['errvel']=rvdat['sigRV(km/s)']*1000
            scopename="SOPHIE"
        elif 'mjd_obs' in rvdat.columns:
            rvdat['time']=rvdat['mjd_obs']
            rvdat['mnvel']=rvdat['drs_ccf_rvc']
            rvdat['errvel']=rvdat['drs_dvrms']
            scopename="HARPS"
        else:
            rvdat['time']=rvdat['Unnamed: 0']
            rvdat['mnvel']=rvdat['0']*1000
            rvdat['errvel']=np.nanstd(rvdat['0'])*1000
            scopename="SOPHIE"
    else:
        print(rvfile,"Scope not recognised - data could not be read")

    #Fixing time and RV unit errors:
    if np.all(rvdat['time']>50000) and np.all(rvdat['time']<2450000):
        rvdat['time']+=2400000
    elif np.all(rvdat['time']>6000) and np.all(rvdat['time']<50000):
        rvdat['time']+=2450000
    if np.all(rvdat['errvel']<0.3):
        #Probably in kms here, dividing by 1000.
        rvdat['mnvel']/=1000
        rvdat['errvel']/=1000
    
    print(scopename,rvfile)
    return rvdat, scopename

def assemble_all_rvs(folder):
    rvfiles = glob.glob(os.path.join(folder,"*"))
    rvdfs={}
    for irv,rvfile in enumerate(rvfiles):
        irvdat,irvname=extract_rvfile(rvfile)
        if irvname in rvdfs:
            #renaming old one to have a "0" and adding a "1" to the name for the current one
            rvdfs[irvname+"0"]=rvdfs[irvname].loc[:,:]
            rvdfs[irvname+"0"]['instr']=irvname+"0"
            _=rvdfs.pop(irvname)
            irvname=irvname+"1"
            
        elif irvname+"1" in rvdfs:
            #Adding a new number
            irvname=irvname+str(int(np.sum([dfname[:-1] in rvdfs for dfname in rvdfs])))
        rvdfs[irvname]=irvdat.loc[:,:]
        rvdfs[irvname]['file']=rvfile.split('/')[-1]
        rvdfs[irvname]['instr']=irvname
        #'time', 'mnvel', 'errvel', and 'instr'
    
    rvdfs['all']=pd.concat([rvdfs[r].loc[:,['time','mnvel','errvel','instr']] for r in rvdfs])
    return rvdfs['all'].sort_values('time')

def radial_velocity(t, P, tc, e, w, K):
    # Calculate the radial velocity at time t
    # P: Orbital period
    # tc: Transit time
    # e: Eccentricity
    # w: Argument of periapsis
    # K: Semi-amplitude of radial velocity

    # Solve Kepler's equation for eccentric anomaly
    if e>0:
        tp = radvel.orbit.timetrans_to_timeperi(tc, P, e, w)
        # Calculate true anomaly
        f = radvel.orbit.true_anomaly(t, tp, P, e)
        # Calculate radial velocity
        return K * (np.cos(w + f) + e * np.cos(w))
    else:
        # Calculate true anomaly
        f = radvel.orbit.true_anomaly(t, tc, P, 0)
        # Calculate radial velocity
        return K * np.cos(w + f)


def ImpactParamPrior(tc, p, ecc, w, a, Rs, Rp_Rs, tdur, bmu=None, bsigma=None):
    time_peri =  radvel.orbit.timetrans_to_timeperi(tc,p,ecc,w)
    #print(time_peri)
    true_anom = radvel.orbit.true_anomaly(np.array([tc]), time_peri, p, ecc)
    #print(true_anom)
    # Calculating implied impact param from these paramaters...
    b2 = (1+Rp_Rs)**2 - ((tdur * np.pi * a*(1+ecc*np.cos(true_anom)))/(p*Rs*np.sqrt(1-ecc**2)))**2
    # Only accepting non-grazing positive b.
    maxb=(1-Rp_Rs)**2
    #print(b2, np.sqrt(b2),np.isfinite(b2))
    if not np.isfinite(b2):
        return -1e10
    elif b2<=0:
        return -9e9-np.clip(abs(b2),0,10)*1e8
    elif b2>=1:
        return -9e9-np.clip(b2-1,0,10)*1e8
    elif (b2>0.)&(b2<maxb):
        if bmu is not None and not np.isnan(bmu) and bsigma is not None and not np.isnan(bsigma):
            #Normally distributed:
            return - 0.5 * ((np.sqrt(b2) - bmu) / bsigma)**2
        else:
            return 0
    elif (b2>maxb)&(b2<1):
        #Almost correct - sharp linear grade during "grazing" configration
        return -1e3*(b2-maxb)/(1-maxb) 

def StabilityPrior(ecclist, smalist, Klist, Rs, Ms):
    #Check that max and minimum distances for each planet does not overlap with either max/minimum distances of the others NOR 2Rs
    min_dists=[];max_dists=[]
    sortedlist=np.arange(len(ecclist))[np.argsort(smalist)]
    for npl in sortedlist:
        #K = Mp*np.sin(i)*sqrt(G/(M*a*(1-ecc**2)))
        Mp = Klist[npl]/np.sqrt(6.67e-11/(Ms*1.96e30*smalist[npl]*695500000*(1-ecclist[npl]**2)))/1.96e30
        min_dists+=[(1-ecclist[npl]-(Mp/(3*Ms))**(1/3))*smalist[npl]/Rs]#Hill sphere inner edge at perihelion in Rs
        max_dists+=[(1+ecclist[npl]+(Mp/(3*Ms))**(1/3))*smalist[npl]/Rs]#Hill sphere outer edge at aphelion in Rs
    max_dists=np.array(max_dists)
    logprior=0
    for npl in range(len(sortedlist)):
        if not np.isfinite(min_dists[npl]) or not np.isfinite(max_dists[npl]):
            #Infinite/nan values:
            logprior+=-1.01e5
        elif npl==0 and min_dists[npl]<2:
            #Making a steep linear trend for the orbits just inside the Roche limit
            logprior+=-9e4-np.clip(2-min_dists[npl],0,1)*1e4
        elif npl>0 and np.any(min_dists[npl]<max_dists[:npl]):
            #Making a steep linear trend for the orbits just beyond stability.
            logprior+=-9e4+np.clip(min_dists[npl]/np.min(max_dists[:npl])-1,-0.1,0)*1e5
    return logprior

def TwoPlanetStabilityPrior(ecclist, smalist, Klist, Rs, Ms):
    #Check that max and minimum distances for each planet does not overlap with either max/minimum distances of the others NOR 2Rs
    min_dists=[];max_dists=[]
    sortedlist=np.arange(len(ecclist))[np.argsort(smalist)]
    Mps = [Klist[npl]/np.sqrt(6.67e-11/(Ms*1.96e30*smalist[npl]*695500000*(1-ecclist[npl]**2)))/1.96e30 for npl in range(2)]
    if smalist[0]>smalist[1]:
        if (1-ecclist[0]-(Mp/(3*Ms))**(1/3))*smalist[0]/Rs < (1+ecclist[1]+(Mp/(3*Ms))**(1/3))*smalist[1]/Rs:
            return -1e5
        else:
            return 0
    else:
        if (1-ecclist[1]-(Mp/(3*Ms))**(1/3))*smalist[1]/Rs < (1+ecclist[0]+(Mp/(3*Ms))**(1/3))*smalist[0]/Rs:
            return -1e5
        else:
            return 0

def KippingPrior(e):
    if (e>0.001)&(e<0.999):
        #print((e**(0.867-1) * (1-e)**(3.03-1)) / 0.4271856369315835)
        return (e**(0.867-1) * (1-e)**(3.03-1)) / 0.4271856369315835 #(gamma(a) * gamma(b) / gamma(a + b))
    else:
        return -1e20

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def weighted_avg_and_std(values, errs=None, weights=None, masknans=True, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if len(values)>1:
        if weights is None and errs is not None:
            average = np.average(values, weights=1/errs**2,axis=axis)
            # Fast and numerically precise:
            variance = np.average((values-average)**2, weights=1/errs**2,axis=axis)
        elif weights is not None and errs is None:
            average = np.average(values, weights=weights,axis=axis)
            # Fast and numerically precise:
            variance = np.average((values-average)**2, weights=weights,axis=axis)
        else:
            average = np.average(values)
            variance = np.std(values)
        binsize_adj = np.sqrt(len(values)) if axis is None else np.sqrt(values.shape[axis])
        return [average, np.sqrt(variance)/binsize_adj]
    elif len(values)==1:
        return [values[0], errs[0]]
    else:
        return [np.nan, np.nan]

class MyPriorTransform():
    def __init__(self, pars):
        self.pars = pars
        #print(pars)
        
    def __call__(self, u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest."""

        x = np.array(u)  # copy u
        for n,p in enumerate(self.pars):
            if self.pars[p][0]=='uniform':
                x[n]=self.pars[p][1] + (self.pars[p][2]-self.pars[p][1]) * u[n]
            elif self.pars[p][0]=='norm':
                x[n]=scipy.stats.norm.ppf(u[n],self.pars[p][1],self.pars[p][2])
            elif self.pars[p][0]=='halfnorm':
                x[n]=scipy.stats.halfnorm.ppf(u[n],self.pars[p][1],self.pars[p][2])
            elif self.pars[p][0]=='truncnorm':
                a=self.pars[p][3];b=self.pars[p][4]
                ar, br = (a - self.pars[p][1]) / self.pars[p][2], (b - self.pars[p][1]) / self.pars[p][2]
                x[n]=scipy.stats.truncnorm.ppf(u[n], ar, br, loc=self.pars[p][1], scale=self.pars[p][2])
            elif self.pars[p][0]=='rayleigh':
                x[n]=scipy.stats.rayleigh.ppf(u[n],self.pars[p][1])
            elif self.pars[p][0]=='beta':
                x[n]=scipy.stats.beta.ppf(u[n],self.pars[p][1],self.pars[p][2])
        return x
