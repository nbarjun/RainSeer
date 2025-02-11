import numpy as np
import xarray as xr
import pandas as pd
import dask
from tqdm import tqdm
from scipy import optimize
from scipy.linalg import cho_solve, cholesky
from sklearn.gaussian_process import kernels
import matplotlib.pyplot as plt
GPR_CHOLESKY_LOWER = True

class cmip_retrained_hyparams_mgpr:
    def __init__(self, gpr_kernel, time, oos_gcm, other_gcms, oos_smoothed, lon, lat):
        self.kernel = gpr_kernel
        self.lon = lon
        self.lat = lat
        self.X_full = time
        self.Y_full = oos_gcm
        self.Y_smooth = oos_smoothed
        self.priors = other_gcms
        self.trainslice = False
        self.fitslice = False
        
        noise_time = int(5/self.Y_full['year'].diff('year')[0].item())
        self.mnoise = (self.Y_smooth - self.Y_full).coarsen(year=noise_time,boundary='trim').mean()\
                    .sel(year=slice(1870,1975)).var('year',ddof=1)\
                    .sel(lat=self.lat, lon=self.lon).item()

    def prior_mean(self,X):
        return np.mean(X,axis=-1)

    def prior_covariance(self,X):
        return np.cov(X,ddof=1)

    def log_marginal_likelihood(self,theta,x,y,pmean,pcov,noise):
        '''
        Function to calculate marignal likelihood without 
        added observational noise
        '''
        kernel = self.kernel.clone_with_theta(theta)
        K = kernel(np.atleast_2d(x))
        K += pcov
        K += noise*np.eye(*K.shape)

        y_adj = y-pmean
        try:
            # Perform Cholesky decomposition of the kernel matrix
            L = np.linalg.cholesky(K)  # K = L * L.T
            # Solve for alpha using the Cholesky factorization
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_adj))
            log_likelihood_value = -0.5 * np.dot(y_adj.T, alpha) - np.sum(np.log(np.diagonal(L)))\
                                   -0.5 * len(x) * np.log(2 * np.pi)
            return -log_likelihood_value  # Negative log-likelihood for minimization
        except np.linalg.LinAlgError:
            return np.inf

    def train(self, tslice):
        '''
        Function to optimize the marginal likelihood given 
        initial guess 
        '''
        self.trainslice = tslice
        self.X_train = pd.DataFrame(list(self.X_full\
                    .sel(year=self.trainslice).values),\
                     columns=['year','lon','lat'])
        self.Y_train = self.Y_full.sel(year=self.trainslice)\
                        .stack(spacetime=['year','lon','lat'])
        self.pm = self.prior_mean(self.priors.sel(year=self.trainslice)\
                            .stack(spacetime=['year','lon','lat'])\
                            .transpose('spacetime','models'))
        self.pc = self.prior_covariance(self.priors.sel(year=self.trainslice)\
                            .stack(spacetime=['year','lon','lat'])
                            .transpose('spacetime','models'))
        climate_years = int(51/self.Y_full['year'].diff('year')[0].item())
        self.ob_noise = (self.Y_full.sel(year=self.trainslice)-self.Y_full.sel(year=self.trainslice)\
                    .rolling(year=climate_years,center=True).mean('year'))\
                    .stack(spacetime=['year','lon','lat']).var('spacetime',ddof=1).item()*2
        
        mlhood = optimize.minimize(self.log_marginal_likelihood, self.kernel.theta,
                args=(self.X_train.values,self.Y_train.values,\
                      self.pm.values,self.pc,self.ob_noise),\
                bounds=self.kernel.bounds,\
                method='L-BFGS-B',tol=1e-10)  
        self.kernel_ = self.kernel.clone_with_theta(mlhood.x)
        self.min_mll = mlhood.fun

    def fit(self, tslice):
        self.fitslice = tslice
        models = self.Y_full['models'].values
        self.Y_fit = self.Y_full.sel(year=self.fitslice)\
                            .stack(spacetime=['year','lon','lat'])
        self.X_fit = pd.DataFrame(list(self.X_full.sel(year=self.fitslice).values))
        other_gcms_smoothed = self.priors.sel(year=self.fitslice)\
                            .stack(spacetime=['year','lon','lat'])\
                            .transpose('spacetime','models')
        climate_years = int(51/self.Y_full['year'].diff('year')[0].item())
        self.ob_noise = (self.Y_full.sel(year=self.trainslice)-self.Y_full.sel(year=self.trainslice)\
                    .rolling(year=climate_years,center=True).mean('year'))\
                    .stack(spacetime=['year','lon','lat']).var('spacetime',ddof=1).item()
                
        self.pm = self.prior_mean(other_gcms_smoothed)
        self.pc = self.prior_covariance(other_gcms_smoothed)
        
        
        # Compute kernel matrix with prior covariance
        self.K = self.kernel_(np.atleast_2d(self.X_fit.values)) + self.pc
        # Add noise to the diagonal 
        self.K += self.ob_noise * np.eye(*self.K.shape)

    def predict(self):
        X_pred = pd.DataFrame(list(self.X_full.values))
        X_pred = X_pred[(X_pred[2]==self.lat)&(X_pred[1]==self.lon)]

        self.pm_star = self.prior_mean(self.priors.sel(lon=self.lon,lat=self.lat)\
                                      .transpose('year','models')).values
        self.pc_stacked = self.prior_covariance(np.vstack([self.priors.sel(year=self.fitslice)
                        .stack(spacetime=['year','lon','lat']).transpose('spacetime','models').values,\
                        self.priors.sel(lat=self.lat,lon=self.lon)\
                        .transpose('year','models').values]))
        
        self.K_stacked = self.kernel_(pd.concat([self.X_fit,X_pred],axis=0)) + self.pc_stacked
        self.K_star = self.K_stacked[self.Y_fit.shape[0]:,self.Y_fit.shape[0]:]
        self.K_K_star = self.K_stacked[:self.Y_fit.shape[0],self.Y_fit.shape[0]:]
        y_adj = self.Y_fit - self.pm
        
        # Mean
        f_bar_star = self.pm_star + np.dot(self.K_K_star.T, np.linalg.solve(self.K, y_adj))
        # Covariance
        cov_f_star = self.K_star - np.dot(self.K_K_star.T, \
                            np.linalg.solve(self.K, self.K_K_star))

        predicted = xr.DataArray(f_bar_star,\
            coords={'year':self.Y_full.year}).to_dataset(name='mu_star')
        predicted['mu'] = xr.DataArray(self.pm_star,coords={'year':self.Y_full.year})
        predicted['std'] = xr.DataArray((np.diag(cov_f_star)+self.mnoise)**.5,\
                                        coords={'year':self.Y_full.year})
        predicted['nstd'] = xr.DataArray((np.diag(cov_f_star)+self.ob_noise)**.5,\
                                        coords={'year':self.Y_full.year})
        return predicted

    def calculate_emergence(self, conf, fraction_past_emergence, nchoice, optimize=True):
        self.ToC = np.nan
        self.ToE = np.nan
        self.ToEonToC = np.nan
        self.awt = np.nan
        self.awtToC = np.nan
        self.ToC_sign = 0 
        self.ToE_sign = 0
        self.emerge_on_toc = []
        self.emerge_on_full = []

        years = self.Y_full['year'].values
        # Specify the year from which the ToC search has to begin
        syear = years[np.squeeze(np.where(years>=2000))[0]]

        looper = 0
        confidence = 0
        while confidence == 0:
            # Starting in 1975
            yr =  syear + (years[1]-years[0])*looper
            fitslice = slice(years[0],yr)
            if optimize:
                self.train(slice(years[0],yr))
            self.fit(fitslice)
            predicted = self.predict()
            
            upperLimit = predicted['mu_star'] + conf*predicted[nchoice]
            lowerLimit = predicted['mu_star'] - conf*predicted[nchoice]
            
            emerge_ll = (lowerLimit> 0)  
            emerge_ll_count = emerge_ll.reindex(year=list(reversed(emerge_ll.year))).cumsum(dim='year')
            emerge_ll_frac =  emerge_ll_count/(emerge_ll_count*0+1).cumsum('year')
            emerge_ul = (upperLimit< 0) 
            emerge_ul_count = emerge_ul.reindex(year=list(reversed(emerge_ul.year))).cumsum(dim='year')
            emerge_ul_frac = emerge_ul_count/(emerge_ul_count*0+1).cumsum('year')
            
            emerge_ul_yrs = emerge_ul_frac['year']\
                    .where((emerge_ul_frac>fraction_past_emergence)&(emerge_ul),drop=True)
            emerge_ll_yrs = emerge_ll_frac['year']\
                    .where((emerge_ll_frac>fraction_past_emergence)&(emerge_ll),drop=True)
            
            if (len(emerge_ul_yrs)*len(emerge_ll_yrs))>0:
                print('Error. Multiple Emergences in Different Direction')
            elif len(emerge_ul_yrs)>0:
                self.ToC = yr
                self.ToEonToC = int(emerge_ul_yrs[-1].item())
                self.awtToC = self.ToEonToC-self.ToC
                self.ToC_sign = -1
                self.emerge_on_toc = predicted
                confidence = 1
            elif len(emerge_ll_yrs)>0:
                self.ToC = yr
                self.ToEonToC = int(emerge_ll_yrs[-1].item())
                self.awtToC = self.ToEonToC-self.ToC
                self.ToC_sign = 1
                self.emerge_on_toc = predicted
                confidence = 1
      
            if (confidence == 0) & (yr == years[-2]):
                confidence = 1
                self.emerge_on_toc = predicted
            looper+=1

        fullslice = slice(years[0],years[-1])
        self.fit(fullslice)
        predicted = self.predict()
        upperLimit = predicted['mu_star'] + conf*predicted[nchoice]
        lowerLimit = predicted['mu_star'] - conf*predicted[nchoice]
        
        emerge_ll = (lowerLimit> 0)  
        emerge_ll_count = emerge_ll.reindex(year=list(reversed(emerge_ll.year))).cumsum(dim='year')
        emerge_ll_frac =  emerge_ll_count/(emerge_ll_count*0+1).cumsum('year')
        emerge_ul = (upperLimit< 0) 
        emerge_ul_count = emerge_ul.reindex(year=list(reversed(emerge_ul.year))).cumsum(dim='year')
        emerge_ul_frac = emerge_ul_count/(emerge_ul_count*0+1).cumsum('year')
        
        emerge_ul_yrs = emerge_ul_frac['year']\
                .where((emerge_ul_frac>fraction_past_emergence)&(emerge_ul),drop=True)
        emerge_ll_yrs = emerge_ll_frac['year']\
                .where((emerge_ll_frac>fraction_past_emergence)&(emerge_ll),drop=True)
            
        
        if (len(emerge_ul_yrs)*len(emerge_ll_yrs))>0:
            print('Error. Multiple Emergences in Different Direction')
        elif len(emerge_ul_yrs)>0:
            self.ToE = int(emerge_ul_yrs[-1].item())
            self.ToE_sign = -1
            self.awt = self.ToE-self.ToC
            self.emerge_on_full = predicted

        elif len(emerge_ll_yrs)>0:
            self.ToE = int(emerge_ll_yrs[-1].item())
            self.ToE_sign = 1
            self.awt = self.ToE-self.ToC
            self.emerge_on_full = predicted
        else:
            self.emerge_on_full = predicted

    def validate_posterior(self, time_steps):
        InRange_95pCI = []
        InRange_68pCI = []
        rmse = []
        rmse_train = []
        merror = []
        posteriors = []
        true_value = self.Y_smooth.sel(lat=self.lat,lon=self.lon)
        
        for ts in time_steps:
            fitslice = slice(None,ts)
            self.train(fitslice)
            self.fit(fitslice)
            predicted = self.predict()
            posteriors.append(predicted)
            pred_means = predicted.drop_vars('std').drop_vars('nstd')
            oos_gcm_pr = self.Y_full.sel(year=slice(ts,None),\
                                              lat=self.lat,lon=self.lon)
            
            rmse.append(((pred_means - true_value).sel(year=slice(ts,None))**2)\
                        .mean('year'))
            rmse_train.append(((pred_means - true_value).sel(year=slice(None,ts))**2)\
                        .mean('year'))
            merror.append((pred_means - true_value).sel(year=2075,method='nearest'))
            
            # 95% CI
            upperLimit = (predicted['mu_star'] + 1.96*predicted['nstd']).sel(year=slice(ts,None))
            lowerLimit = (predicted['mu_star'] - 1.96*predicted['nstd']).sel(year=slice(ts,None))
            ind = (oos_gcm_pr < upperLimit) & (oos_gcm_pr > lowerLimit)
            InRange_95pCI.append(np.sum(ind)/len(upperLimit))
            # 68% CI
            upperLimit = (predicted['mu_star'] + predicted['nstd']).sel(year=slice(ts,None))
            lowerLimit = (predicted['mu_star'] - predicted['nstd']).sel(year=slice(ts,None))
            ind = (oos_gcm_pr < upperLimit) & (oos_gcm_pr > lowerLimit)  
            InRange_68pCI.append(np.sum(ind)/len(upperLimit))
    
        self.in68 = xr.concat(InRange_68pCI,dim='tobs')\
                            .assign_coords({'tobs':time_steps})
        self.in95 = xr.concat(InRange_95pCI,dim='tobs')\
                            .assign_coords({'tobs':time_steps})
        self.rmse = xr.concat(rmse,dim='tobs').assign_coords({'tobs':time_steps})
        self.rmse_train = xr.concat(rmse_train,dim='tobs').assign_coords({'tobs':time_steps})
        self.merror = xr.concat(merror,dim='tobs').assign_coords({'tobs':time_steps})
        self.posteriors = xr.concat(posteriors,dim='tobs')\
                            .assign_coords({'tobs':time_steps})
        
    def error_calculations(self, oos_model, time_steps):
        rmse = []
        rmse_train = []
        merror = []
        true_value = self.priors.sel(models=oos_model,lat=self.lat,lon=self.lon)
        for ts in time_steps:
            fitslice = slice(None,ts)
            self.fit(oos_model, fitslice)
            predicted = self.predict(oos_model)
            pred_means = predicted.drop_vars('std').drop_vars('nstd')
            
            rmse.append(((pred_means - true_value).sel(year=slice(ts,None))**2)\
                        .mean('year'))
            rmse_train.append(((pred_means - true_value).sel(year=slice(None,ts))**2)\
                        .mean('year'))
            merror.append((pred_means - true_value).sel(year=2075,method='nearest'))
        self.rmse = xr.concat(rmse,dim='tobs').assign_coords({'tobs':time_steps})
        self.rmse_train = xr.concat(rmse_train,dim='tobs').assign_coords({'tobs':time_steps})
        self.merror = xr.concat(merror,dim='tobs').assign_coords({'tobs':time_steps})
        
    def plot_predicted(self, posterior, oos_model, nchoice, plot_priors=False):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        yobs = self.fitslice.stop
        oos_gcm = self.Y_full.sel(lat=self.lat,lon=self.lon)
        other_gcms_smoothed = self.priors.sel(lat=self.lat,lon=self.lon)\
                                .transpose('year','models')
        
        ymin = np.floor(oos_gcm.min()+.5*oos_gcm.min())
        ymax = np.ceil(oos_gcm.max()+.5*oos_gcm.max())
        texty = oos_gcm.min()+.45*oos_gcm.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')

        if plot_priors:
            for m in other_gcms_smoothed.models.values:
                if m==other_gcms_smoothed.models.values[0]:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.4,lw=.5,label='GCMs')
                else:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',alpha=.4,lw=.5)
        self.prior_mean(other_gcms_smoothed).plot(ax=ax,c='tab:blue',label='Prior Mean')
        self.Y_smooth.sel(lat=self.lat,lon=self.lon)\
                    .plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        oos_gcm.sel(year=self.fitslice)\
                        .plot(ax=ax,c='k',alpha=.5,lw=.75,label='Observations')
        
        posterior['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(posterior['year'].values,posterior['mu_star']+posterior[nchoice],\
                        posterior['mu_star']-posterior[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        
        ax.text(oos_gcm['year'].sel(year=yobs,method='nearest').item(),texty,yobs,\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='k',pad=.005),fontsize=textsize)
        ax.axvline(x=oos_gcm['year'].sel(year=yobs,method='nearest').item(),\
                   c='k',lw=.3,linestyle='--')
        ax.set_xlim(oos_gcm['year'][0].item(),oos_gcm['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()

    def plot_emergence_on_toc(self, oos_model, nchoice, plot_priors=False):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        yobs = self.ToC
        oos_gcm = self.Y_full.sel(lat=self.lat,lon=self.lon)
        other_gcms_smoothed = self.priors.sel(lat=self.lat,lon=self.lon)\
                                .transpose('year','models')
        
        ymin = np.floor(oos_gcm.min()+.5*oos_gcm.min())
        ymax = np.ceil(oos_gcm.max()+.5*oos_gcm.max())
        texty = oos_gcm.min()+.45*oos_gcm.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')

        if plot_priors:
            for m in other_gcms_smoothed.models.values:
                if m==other_gcms_smoothed.models.values[0]:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.4,lw=.5,label='GCMs')
                else:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',alpha=.4,lw=.5)
        self.prior_mean(other_gcms_smoothed).plot(ax=ax,c='tab:blue',label='Prior Mean')
        self.Y_smooth.sel(lat=self.lat,lon=self.lon)\
                    .plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        oos_gcm.sel(year=slice(None,yobs))\
                        .plot(ax=ax,c='k',alpha=.5,label='Observations')
        
        self.emerge_on_toc['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(oos_gcm['year'].values,self.emerge_on_toc['mu_star']+self.emerge_on_toc[nchoice],\
                        self.emerge_on_toc['mu_star']-self.emerge_on_toc[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        
        if ~np.isnan(self.ToC):
            ax.text(oos_gcm['year'].sel(year=yobs).item(),texty,yobs,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
            ax.axvline(x=oos_gcm['year'].sel(year=yobs).item(),\
                       c='g',lw=.3,linestyle='--',label='ToC')
        else:
            ax.text(oos_gcm['year'].isel(year=10).item(),texty,'No ToC',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

        if ~np.isnan(self.ToEonToC):
            ax.text(oos_gcm['year'].sel(year=self.ToEonToC).item(),texty-.25*texty,self.ToEonToC,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
            ax.axvline(x=oos_gcm['year'].sel(year=self.ToEonToC).item(),\
                       c='r',lw=.3,linestyle='--',label='ToE')
        else:
            ax.text(oos_gcm['year'].isel(year=10).item(),texty-.25*texty,'No ToE',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)

        ax.set_xlim(oos_gcm['year'][0].item(),oos_gcm['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()

    def plot_emergence(self, oos_model, nchoice, plot_priors=False):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        yobs = self.Y_full['year'].values[-1]
        oos_gcm = self.Y_full.sel(lat=self.lat,lon=self.lon)
        other_gcms_smoothed = self.priors.sel(lat=self.lat,lon=self.lon)\
                                .transpose('year','models')
        
        ymin = np.floor(oos_gcm.min()+.5*oos_gcm.min())
        ymax = np.ceil(oos_gcm.max()+.5*oos_gcm.max())
        texty = oos_gcm.min()+.45*oos_gcm.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')

        if plot_priors:
            for m in other_gcms_smoothed.models.values:
                if m==other_gcms_smoothed.models.values[0]:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.4,lw=.5,label='GCMs')
                else:
                    other_gcms_smoothed.sel(models=m).plot(ax=ax,c='tab:orange',alpha=.4,lw=.5)
        self.Y_smooth.sel(lat=self.lat,lon=self.lon)\
                    .plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        oos_gcm.sel(year=slice(None,yobs))\
                        .plot(ax=ax,c='k',alpha=.5,label='Observations')
        
        self.emerge_on_full['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(oos_gcm['year'].values,self.emerge_on_full['mu_star']+self.emerge_on_full[nchoice],\
                        self.emerge_on_full['mu_star']-self.emerge_on_full[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        if ~np.isnan(self.ToC):
            ax.text(oos_gcm['year'].sel(year=self.ToC).item(),texty,self.ToC,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
            ax.axvline(x=oos_gcm['year'].sel(year=self.ToC).item(),\
                       c='g',lw=.3,linestyle='--',label='ToC')
        else:
            ax.text(oos_gcm['year'].isel(year=10).item(),texty,'No ToC',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

        if ~np.isnan(self.ToE):
            ax.text(oos_gcm['year'].sel(year=self.ToE).item(),texty-texty*.25,self.ToE,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
            ax.axvline(x=oos_gcm['year'].sel(year=self.ToE).item(),\
                       c='r',lw=.3,linestyle='--',label='ToE')
        else:
            ax.text(oos_gcm['year'].isel(year=10).item(),texty-texty*.25,'No ToE',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
        
        ax.set_xlim(oos_gcm['year'][0].item(),oos_gcm['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()