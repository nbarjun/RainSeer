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

class cmip_retrained_hyparams_ugpr:
    def __init__(self, gpr_kernel, time, oos_gcm, other_gcms, oos_smoothed):
        self.kernel = gpr_kernel
        self.X_full = time
        self.Y_full = oos_gcm
        self.Y_smooth = oos_smoothed
        self.priors = other_gcms
        self.trainslice = False
        self.fitslice = False
        
        noise_time = int(5/self.Y_full['year'].diff('year')[0].item())
        self.mnoise = self.Y_full.coarsen(year=noise_time,boundary='trim').mean()\
                        .sel(year=slice(1870,1975)).var('year',ddof=1).item()

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
        K = kernel(x.reshape(-1,1))
        K += pcov
        K += noise*np.identity(len(y))

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
        self.X_train = self.X_full.sel(year=self.trainslice)['year']
        self.Y_train = self.Y_full.sel(year=self.trainslice)
        self.pm = self.prior_mean(self.priors.sel(year=self.trainslice).transpose('year','models'))
        self.pc = self.prior_covariance(self.priors.sel(year=self.trainslice).transpose('year','models'))
        climate_years = int(51/self.Y_full['year'].diff('year')[0].item())
        self.ob_noise = (self.Y_full.sel(year=self.trainslice)-self.Y_full.sel(year=self.trainslice)\
                    .rolling(year=climate_years,center=True).mean('year')).var('year',ddof=1).item()
        
        mlhood = optimize.minimize(self.log_marginal_likelihood, self.kernel.theta,
                args=(self.X_train.values,self.Y_train.values,\
                      self.pm.values,self.pc,self.ob_noise),\
                bounds=self.kernel.bounds,\
                method='L-BFGS-B',tol=1e-10)  
        self.kernel_ = self.kernel.clone_with_theta(mlhood.x)
        self.min_mll = mlhood.fun
        
    def fit(self, tslice):
        self.fitslice = tslice
        self.Y_fit = self.Y_full.sel(year=self.fitslice).values
        self.X_fit = self.X_full['year'].sel(year=self.fitslice).values
        climate_years = int(51/self.Y_full['year'].diff('year')[0].item())
        self.ob_noise = (self.Y_full.sel(year=self.fitslice)-self.Y_full.sel(year=self.fitslice)\
                    .rolling(year=climate_years,center=True).mean('year')).var('year',ddof=1).item()
        self.pm = self.prior_mean(self.priors.sel(year=self.fitslice).transpose('year','models'))
        self.pc = self.prior_covariance(self.priors.sel(year=self.fitslice).transpose('year','models'))
        
        # Compute kernel matrix with prior covariance
        self.K = self.kernel_(self.X_fit.reshape(-1,1)) + self.pc
        # Add noise to the diagonal 
        self.K += self.ob_noise * np.eye(*self.Y_fit.shape)
        
    def predict(self):
        self.pc_star  = self.prior_covariance(self.priors\
                                .transpose('year','models')) 
        self.pm_star = self.prior_mean(self.priors\
                                 .transpose('year','models'))
        X_pred = self.X_full['year'].values

        self.K_stacked = self.kernel_(np.hstack([self.X_fit,X_pred]))
        self.K_star = self.K_stacked[self.Y_fit.shape[0]:,self.Y_fit.shape[0]:]+ self.pc_star
        self.K_K_star = self.K_stacked[:self.Y_fit.shape[0],self.Y_fit.shape[0]:]\
                            + self.pc_star[:self.Y_fit.shape[0],:]
        
        y_adj = self.Y_fit - self.pm
        
        # Mean
        f_bar_star = self.pm_star + np.dot(self.K_K_star.T, np.linalg.solve(self.K, y_adj))
        # Covariance
        cov_f_star = self.K_star - np.dot(self.K_K_star.T, \
                            np.linalg.solve(self.K, self.K_K_star))
        # print(np.diag(cov_f_star))
        predicted = xr.DataArray(f_bar_star,\
            coords={'year':X_pred}).to_dataset(name='mu_star')
        predicted['mu'] = xr.DataArray(self.pm_star,coords={'year':X_pred})
        predicted['std'] = xr.DataArray((np.abs(np.diag(cov_f_star))+self.mnoise)**.5,\
                                        coords={'year':X_pred})
        predicted['nstd'] = xr.DataArray((np.abs(np.diag(cov_f_star))+self.ob_noise)**.5,\
                                        coords={'year':X_pred})
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
        
        years = self.X_full.values
        # Specify the year from which the ToC search has to begin
        syear = years[np.squeeze(np.where(years>=2000))[0]]
        
        looper = 0
        confidence = 0
        while confidence == 0:
            # Starting in 1975
            yr =  syear + (years[1]-years[0])*looper
            fitslice = slice(years[0],yr)
            if optimize:
                self.train(fitslice)
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
        if optimize:
            self.train(fullslice)
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

    def plot_predicted(self, posterior, oos_model, nchoice, plot_gcms):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        if self.fitslice:
            yobs = self.fitslice.stop
        else:
            yobs = self.trainslice.stop
        ymin = np.floor(self.Y_full.min()+.25*self.Y_full.min())
        ymax = np.ceil(self.Y_full.max()+.25*self.Y_full.max())
        texty = self.Y_full.min()+.25*self.Y_full.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')
        
        if plot_gcms:
            for m in self.priors.models:
                if m==self.priors.models[0]:
                    self.priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.25,lw=.5,label='GCMs')
                else:
                    self.priors.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                       alpha=.25,lw=.5)
                    
        posterior['mu'].plot(ax=ax,c='tab:blue',label='Prior Mean')
        self.Y_smooth.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        self.Y_full.sel(year=slice(None,yobs))\
                        .plot(ax=ax,c='k',alpha=.5,lw=.75,label='Observations')
        
        posterior['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(posterior['year'].values,posterior['mu_star']+posterior[nchoice],\
                        posterior['mu_star']-posterior[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        
        ax.text(self.X_full['year'].sel(year=yobs).item(),texty,yobs,\
                   c='w',ha='center',verticalalignment='center',\
                   bbox=dict(facecolor='k',pad=.005),fontsize=textsize)
        ax.axvline(x=self.X_full['year'].sel(year=yobs).item(),\
                   c='k',lw=.3,linestyle='--')
        
        ax.set_xlim(self.X_full['year'][0].item(),self.X_full['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()

    def plot_emergence_on_toc(self, oos_model, nchoice, plot_gcms):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        yobs = self.ToC
        ymin = np.floor((self.Y_full.min()+.25*self.Y_full.min()).item())
        ymax = np.ceil((self.Y_full.max()+.25*self.Y_full.max()).item())
        texty = self.Y_full.min()+.25*self.Y_full.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')

        if plot_gcms:
            for m in other_gcms.models:
                if m==other_gcms.models[0]:
                    other_gcms.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.25,lw=.5,label='GCMs')
                else:
                    other_gcms.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                  alpha=.25,lw=.5)
        self.emerge_on_toc['mu'].plot(ax=ax,c='tab:blue',label='Prior Mean')
        self.Y_smooth.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        self.Y_full.sel(year=slice(None,yobs))\
                        .plot(ax=ax,c='k',alpha=.5,label='Observations',lw=.75)
        
        self.emerge_on_toc['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(self.X_full['year'].values,self.emerge_on_toc['mu_star']+self.emerge_on_toc[nchoice],\
                        self.emerge_on_toc['mu_star']-self.emerge_on_toc[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        
        if ~np.isnan(self.ToC):
            ax.text(self.X_full['year'].sel(year=yobs).item(),texty,yobs,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
            ax.axvline(x=self.X_full['year'].sel(year=yobs).item(),\
                       c='g',lw=.3,linestyle='--',label='ToC')
        else:
            ax.text(self.X_full['year'].isel(year=10).item(),texty,'No ToC',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

        if ~np.isnan(self.ToEonToC):
            ax.text(self.X_full['year'].sel(year=self.ToEonToC).item(),texty+1,self.ToEonToC,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
            ax.axvline(x=self.X_full['year'].sel(year=self.ToEonToC).item(),\
                       c='r',lw=.3,linestyle='--',label='ToE')
        else:
            ax.text(self.X_full['year'].isel(year=10).item(),texty+1,'No ToE',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)

        ax.set_xlim(self.X_full['year'][0].item(),self.X_full['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()

    def plot_emergence(self, oos_model, nchoice, plot_gcms):
        plt.rcParams.update({'font.size': 7})
        fig,ax = plt.subplots(1,1, figsize=(5,1.5), dpi=300)
        yobs = self.X_full['year'][-1].item()
        ymin = np.floor(self.Y_full.min()+.25*self.Y_full.min())
        ymax = np.ceil(self.Y_full.max()+.25*self.Y_full.max())
        texty = self.Y_full.min()+.25*self.Y_full.min()
        textsize = 5
        ax.axhline(y=0,lw=.3,c='k')

        if plot_gcms:
            for m in other_gcms.models:
                if m==other_gcms.models[0]:
                    other_gcms.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                    alpha=.25,lw=.5,label='GCMs')
                else:
                    other_gcms.sel(models=m).plot(ax=ax,c='tab:orange',\
                                                  alpha=.25,lw=.5)
        self.emerge_on_full['mu'].plot(ax=ax,c='tab:blue',label='Prior Mean')
        self.Y_smooth.plot(ax=ax,c='tab:green',alpha=.85,label=oos_model,lw=.8)
        self.Y_full.sel(year=slice(None,yobs))\
                        .plot(ax=ax,c='k',alpha=.5,label='Observations',lw=.75)
        
        self.emerge_on_full['mu_star'].plot(ax=ax,c='tab:red',lw=1.5,label='GPR',alpha=.75)
        ax.fill_between(self.X_full['year'].values,self.emerge_on_full['mu_star']+self.emerge_on_full[nchoice],\
                        self.emerge_on_full['mu_star']-self.emerge_on_full[nchoice],color='tab:red',zorder=2,\
                        alpha=.15,lw=.1,label=r'Posterior $\pm$1$\sigma (t_{{obs}}=${})'.format(yobs))
        if ~np.isnan(self.ToC):
            ax.text(self.X_full['year'].sel(year=self.ToC).item(),texty,self.ToC,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)
            ax.axvline(x=self.X_full['year'].sel(year=self.ToC).item(),\
                       c='g',lw=.3,linestyle='--',label='ToC')
        else:
            ax.text(self.X_full['year'].isel(year=10).item(),texty,'No ToC',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='g',edgecolor='g', pad=.005),fontsize=textsize)

        if ~np.isnan(self.ToE):
            ax.text(self.X_full['year'].sel(year=self.ToE).item(),texty+1,self.ToE,\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
            ax.axvline(x=self.X_full['year'].sel(year=self.ToE).item(),\
                       c='r',lw=.3,linestyle='--',label='ToE')
        else:
            ax.text(self.X_full['year'].isel(year=10).item(),texty+1,'No ToE',\
                       c='w',ha='center',verticalalignment='center',\
                       bbox=dict(facecolor='r',edgecolor='r', pad=.005),fontsize=textsize)
        
        ax.set_xlim(self.X_full['year'][0].item(),self.X_full['year'][-1].item())
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(r'$\Delta P$')
        ax.set_title('Projection for model '+str(oos_model),y=.95,fontsize=7)
        ax.legend(ncols=10,frameon=False,fontsize=3.5,loc='upper left')
        ax.grid(alpha=.3,linestyle='--',lw=.2)
        plt.tight_layout()

    def validate_posterior(self, time_steps):
        InRange_95pCI = []
        InRange_68pCI = []
        rmse = []
        rmse_train = []
        merror = []
        posteriors = []
        true_value = self.Y_smooth
        for ts in time_steps:
            fitslice = slice(None,ts)
            self.train(fitslice)
            self.fit(fitslice)
            predicted = self.predict()
            posteriors.append(predicted)
            pred_means = predicted.drop_vars('std').drop_vars('nstd')
            rmse.append(((pred_means - true_value).sel(year=slice(ts,None))**2)\
                        .mean('year'))
            rmse_train.append(((pred_means - true_value).sel(year=slice(None,ts))**2)\
                        .mean('year'))
            merror.append((pred_means - true_value).sel(year=2075,method='nearest'))
        
            oos_gcm_pr = self.Y_full.sel(year=slice(ts,None))
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
        true_value = self.priors.sel(models=oos_model)
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