import numpy as np
import xarray as xr
from scipy.signal import argrelmax,argrelmin
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
def pad_borders(ds,w):
    '''
    Pad the borders along longitudes with data based on the size of the
    window used for the ridge regression
    '''
    ds360 = ds.where(ds.lon+w>360,drop=True)
    ds360 = ds360.assign_coords({'lon':ds360['lon']-360})
    ds000 = ds.where(ds.lon-w<0,drop=True)
    ds000 = ds000.assign_coords({'lon':ds000['lon']+360})
    ds = xr.concat([ds360,ds,ds000],dim='lon')
    return ds
    
def preprocess_5year(pr):
    '''
    The function takes annual mean precipitation data as input
    and provides a smoothed data to be used as priors, resampled 
    data to be used as observations and observational noise
    '''
    year_axis = pr['year'].values
    time_axis = xr.cftime_range(start='1850-01-01',end='2099-12-31',freq='1YS')
    pr = pr.rename({'year':'time'})
    pr = pr.assign_coords({'time':time_axis})
    
    insize = '5YS'
    hist_time = slice('1851-01-01','1950-12-31')
    # Historical Mean precipitation
    histP = pr.sel(time=hist_time).mean('time')
    # Standard deviation of historical precipitation
    histStDevP = pr.sel(time=hist_time).std('time',ddof=1)
    # Five year anomalies relative to historical mean
    FiveYr = (pr.resample(time=insize)\
              .mean('time',skipna=True)-histP)/histStDevP
    FiveYr = FiveYr.sel(time=slice('1870-01-01','2099-12-31'))#.shift(time=1)

    # 50 year moving average
    smoothedMA = (pr.rolling(time=51,min_periods=20,center=True)\
                  .mean('time',skipna=True)-histP)/histStDevP
    smoothedMA = smoothedMA.dropna('time',how='all')
    smoothedMA = smoothedMA.sel(time=slice('1850-01-01','2099-12-31'))
    # 50 year running mean sampled every 5 years. Used as prior for GPR
    smoothedMA5 = smoothedMA.resample(time=insize).interpolate('nearest')\
        .sel(time=slice('1870-01-01','2099-12-31')).transpose('models','time','lon','lat')
    
    # Renaming the coordinates to match input
    smoothedMA5 = smoothedMA5.assign_coords({'time':smoothedMA5['time.year']})
    smoothedMA5 = smoothedMA5.rename({'time':'year'})
    FiveYr = FiveYr.assign_coords({'time':FiveYr['time.year']})
    FiveYr = FiveYr.rename({'time':'year'})    
    return pad_borders(smoothedMA5,60), pad_borders(FiveYr,60)

def get_local_maxs(ds):
    smoothed = uniform_filter1d(ds, size=3, mode='wrap')
    maxs = argrelmax(np.squeeze(smoothed),mode='wrap',order=3)[0]
    if len(maxs)==1:
        return maxs+1,-1
    else:
        # Sort the indices based on the corresponding values in the data array
        sorted_maxima_indices = maxs[np.argsort(smoothed[maxs])][::-1]
        return sorted_maxima_indices[0]+1,sorted_maxima_indices[1]+1

def get_local_mins(ds):
    smoothed = uniform_filter1d(ds, size=3, mode='wrap')
    mins = argrelmin(np.squeeze(smoothed),mode='wrap',order=3)[0]
    if len(mins)==1:
        return mins+1,-1
    else:
        # Sort the indices based on the corresponding values in the data array
        sorted_minima_indices = mins[np.argsort(smoothed[mins])]
        return sorted_minima_indices[0]+1,sorted_minima_indices[1]+1

#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
def validate_posterior(gprModel, uchoice, time_steps):
    InRange_95pCI = []
    InRange_68pCI = []
    rmse = []
    rmse_train = []
    merror = []
    posteriors = []
    for ts in time_steps:
        fitslice = slice(None,ts)
        gprModel.fit(fitslice)
        gprModel.predict()
        predicted = gprModel.predicted
        posteriors.append(predicted)
        if len(gprModel.observations.dims)>1:
            true_value = gprModel.true_values\
                            .sel(lat=gprModel.lat,lon=gprModel.lon)
            oos_gcm_post = gprModel.observations.sel(lat=gprModel.lat,lon=gprModel.lon,\
                                                   year=slice(ts,None))
        else:
            true_value = gprModel.true_values
            oos_gcm_post = gprModel.observations.sel(year=slice(ts,None))
            
        pred_means = gprModel.predicted.drop_vars('std').drop_vars('nstd')
        rmse.append(((pred_means - true_value).sel(year=slice(ts,None))**2)\
                    .mean('year'))
        rmse_train.append(((pred_means - true_value).sel(year=slice(None,ts))**2)\
                    .mean('year'))
        merror.append((pred_means - true_value).sel(year=2075,method='nearest'))
    
        # 95% CI
        upperLimit = (predicted['mu_star'] + 1.96*predicted[uchoice]).sel(year=slice(ts,None))
        lowerLimit = (predicted['mu_star'] - 1.96*predicted[uchoice]).sel(year=slice(ts,None))
        ind = (oos_gcm_post < upperLimit) & (oos_gcm_post > lowerLimit)
        InRange_95pCI.append(np.sum(ind)/len(upperLimit))
        # 68% CI
        upperLimit = (predicted['mu_star'] + predicted[uchoice]).sel(year=slice(ts,None))
        lowerLimit = (predicted['mu_star'] - predicted[uchoice]).sel(year=slice(ts,None))
        ind = (oos_gcm_post < upperLimit) & (oos_gcm_post > lowerLimit)  
        InRange_68pCI.append(np.sum(ind)/len(upperLimit))


    validations = xr.concat(rmse,dim='tobs').assign_coords({'tobs':time_steps})
    validations['mu'] =  xr.concat(rmse_train,dim='tobs')\
                            .assign_coords({'tobs':time_steps})['mu_star']
    validations['i68'] = xr.concat(InRange_68pCI,dim='tobs')\
                                .assign_coords({'tobs':time_steps})
    validations['i95'] = xr.concat(InRange_95pCI,dim='tobs')\
                                .assign_coords({'tobs':time_steps})
    validations['mean_error'] = xr.concat(merror,dim='tobs')\
                                    .assign_coords({'tobs':time_steps})['mu_star']
    posteriors = xr.concat(posteriors,dim='tobs')\
                        .assign_coords({'tobs':time_steps})
    return validations,posteriors
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
def plot_predicted(gprModel, uchoice, plot_gcms):
    plt.rcParams.update({'font.size': 7})
    oos_model = gprModel.observations['models'].item()
    fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
    if gprModel.fitslice:
        yobs = gprModel.fitslice.stop
    else:
        yobs = gprModel.trainslice.stop
    if len(gprModel.observations.dims)>1:
        obs = gprModel.observations.sel(lat=gprModel.lat,lon=gprModel.lon)
        priors = gprModel.priors.sel(lat=gprModel.lat,lon=gprModel.lon)
        true_gcm = gprModel.true_values.sel(lat=gprModel.lat,lon=gprModel.lon)
    else:
        obs = gprModel.observations
        priors = gprModel.priors
        true_gcm = gprModel.true_values

    pads = .2
    ymin = np.floor(obs.min()+pads*obs.min())
    ymax = np.ceil(obs.max()+pads*obs.max())
    texty = obs.min()+(pads-.05)*obs.min()
    textsize = 5
    ax.axhline(y=0,lw=.3,c='k')
    if plot_gcms:
        for m in gprModel.priors.models:
            if m==gprModel.priors.models[0]:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                alpha=.25,lw=.5,label='GCMs')
            else:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                   alpha=.25,lw=.5)
                
    priors.mean('models').plot(ax=ax,c='tab:blue',label='Prior Mean')
    true_gcm.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
    obs.sel(year=slice(None,yobs))\
                    .plot(ax=ax,c='k',alpha=.5,lw=.75,label='Observations')
    
    gprModel.predicted['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
    ax.fill_between(gprModel.predicted['year'].values,\
                    gprModel.predicted['mu_star']+gprModel.predicted[uchoice],\
                    gprModel.predicted['mu_star']-gprModel.predicted[uchoice],\
                    color='tab:red',zorder=2,alpha=.15,lw=.1,\
                    label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
    
    ax.text(obs['year'].sel(year=yobs).item(),texty,yobs,\
               c='w',ha='center',verticalalignment='center',\
               bbox=dict(facecolor='k',pad=.005),fontsize=textsize)
    ax.axvline(x=obs['year'].sel(year=yobs).item(),\
               c='k',lw=.3,linestyle='--')
    
    ax.set_xlim(obs['year'][0].item(),\
                obs['year'][-1].item())
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(r'$\Delta P$')
    ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
    ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
    ax.grid(alpha=.3,linestyle='--',lw=.2)
    plt.tight_layout()

def plot_emergence(gprModel, uchoice, plot_priors=False):
    plt.rcParams.update({'font.size': 7})
    oos_model = gprModel.observations['models'].item()
    fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
    if gprModel.fitslice:
        yobs = gprModel.fitslice.stop
    else:
        yobs = gprModel.trainslice.stop
    if len(gprModel.observations.dims)>1:
        obs = gprModel.observations.sel(lat=gprModel.lat,lon=gprModel.lon)
        priors = gprModel.priors.sel(lat=gprModel.lat,lon=gprModel.lon)
        true_gcm = gprModel.true_values.sel(lat=gprModel.lat,lon=gprModel.lon)
    else:
        obs = gprModel.observations
        priors = gprModel.priors
        true_gcm = gprModel.true_values

    pads = .2
    ymin = np.floor(obs.min()+pads*obs.min())
    ymax = np.ceil(obs.max()+pads*obs.max())
    texty = obs.min()+(pads-.05)*obs.min()
    textsize = 5
    ax.axhline(y=0,lw=.3,c='k')
    if plot_priors:
        for m in priors.models.values:
            if m==priors.models.values[0]:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                alpha=.4,lw=.5,label='GCMs')
            else:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',alpha=.4,lw=.5)
    priors.mean('models').plot(ax=ax,c='tab:blue',label='Prior Mean')
    true_gcm.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
    obs.sel(year=slice(None,yobs))\
                    .plot(ax=ax,c='k',alpha=.5,label='Observations')
    
    gprModel.emerge_on_full['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
    ax.fill_between(priors['year'].values,gprModel.emerge_on_full['mu_star']+gprModel.emerge_on_full[uchoice],\
                    gprModel.emerge_on_full['mu_star']-gprModel.emerge_on_full[uchoice],color='tab:red',zorder=2,\
                    alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
    if ~np.isnan(gprModel.emergence.toc.item()):
        ax.text(priors['year'].sel(year=gprModel.emergence.toc.item()).item(),texty,\
                gprModel.emergence.toc.item(), c='w',ha='center',verticalalignment='center',\
                bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
        ax.axvline(x=priors['year'].sel(year=gprModel.emergence.toc.item()).item(),\
                   c='g',lw=.3,linestyle='--',label='ToC')
    else:
        ax.text(priors['year'].isel(year=10).item(),texty,'No ToC',\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

    if ~np.isnan(gprModel.emergence.toe.item()):
        ax.text(priors['year'].sel(year=gprModel.emergence.toe.item()).item(),texty-texty*.25,\
                gprModel.emergence.toe.item(),c='w',ha='center',verticalalignment='center',\
                bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
        ax.axvline(x=priors['year'].sel(year=gprModel.emergence.toe.item()).item(),\
                   c='r',lw=.3,linestyle='--',label='ToE')
    else:
        ax.text(priors['year'].isel(year=10).item(),texty-texty*.25,'No ToE',\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
    
    ax.set_xlim(priors['year'][0].item(),priors['year'][-1].item())
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(r'$\Delta P$')
    ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
    ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
    ax.grid(alpha=.3,linestyle='--',lw=.2)
    plt.tight_layout()
    
def plot_emergence_on_toc(gprModel, uchoice, plot_priors=False):
    plt.rcParams.update({'font.size': 7})
    oos_model = gprModel.observations['models'].item()
    fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
    yobs = gprModel.emergence.toc.item()
    if len(gprModel.observations.dims)>1:
        obs = gprModel.observations.sel(lat=gprModel.lat,lon=gprModel.lon)
        priors = gprModel.priors.sel(lat=gprModel.lat,lon=gprModel.lon)
        true_gcm = gprModel.true_values.sel(lat=gprModel.lat,lon=gprModel.lon)
    else:
        obs = gprModel.observations
        priors = gprModel.priors
        true_gcm = gprModel.true_values

    pads = .2
    ymin = np.floor(obs.min()+pads*obs.min())
    ymax = np.ceil(obs.max()+pads*obs.max())
    texty = obs.min()+(pads-.05)*obs.min()
    textsize = 5
    ax.axhline(y=0,lw=.3,c='k')
    if plot_priors:
        for m in priors.models.values:
            if m==priors.models.values[0]:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                alpha=.4,lw=.5,label='GCMs')
            else:
                priors.sel(models=m).plot(ax=ax,c='tab:orange',alpha=.4,lw=.5)
    priors.mean('models').plot(ax=ax,c='tab:blue',label='Prior Mean')
    true_gcm.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
    
    obs.sel(year=slice(None,yobs))\
                    .plot(ax=ax,c='k',alpha=.5,label='Observations')

    gprModel.emerge_on_toc['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
    ax.fill_between(priors['year'].values,gprModel.emerge_on_toc['mu_star']+gprModel.emerge_on_toc[uchoice],\
                    gprModel.emerge_on_toc['mu_star']-gprModel.emerge_on_toc[uchoice],color='tab:red',zorder=2,\
                    alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
    
    if ~np.isnan(gprModel.emergence.toc.item()):
        ax.text(priors['year'].sel(year=yobs).item(),texty,gprModel.emergence.toc.item(),\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
        ax.axvline(x=priors['year'].sel(year=yobs).item(),\
                   c='g',lw=.3,linestyle='--',label='ToC')
    else:
        ax.text(priors['year'].isel(year=10).item(),texty,'No ToC',\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

    if ~np.isnan(gprModel.emergence.toeonc.item()):
        ax.text(priors['year'].sel(year=gprModel.emergence.toeonc.item()).item(),texty-.25*texty,\
                gprModel.emergence.toeonc.item(),c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
        ax.axvline(x=priors['year'].sel(year=gprModel.emergence.toeonc.item()).item(),\
                   c='r',lw=.3,linestyle='--',label='ToE')
    else:
        ax.text(priors['year'].isel(year=10).item(),texty-.25*texty,'No ToE',\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)

    ax.set_xlim(priors['year'][0].item(),priors['year'][-1].item())
    ax.set_ylim(ymin,ymax)
    ax.set_ylabel(r'$\Delta P$')
    ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
    ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
    ax.grid(alpha=.3,linestyle='--',lw=.2)
    plt.tight_layout()