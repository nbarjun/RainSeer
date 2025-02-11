import numpy as np
import xarray as xr
import torch
import gpytorch

#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
class temporalGPR:
    '''
    This class defines the framework to be applied for a GPR just along the time 
    dimension at a gridpoint. 
    The class is defined by providing:\
        1. oos_gcm: The out-of-sample GCM which will be used as observations
        2. smooth_oos_gcm: The smoothed values of the oos_gcm. This is used
            to calculate observational noise and error analysis.
        3. smoothed_gcms: The smoothed data used to calculate the values for 
            prior mean and prior covariances.
        4. rslice: represents the reference period. The reference period is 
            used to calculate the observational noise.
    '''
    def __init__(self, oos_gcm, smooth_oos_gcm, smoothed_gcms, rslice):
        self.observations = oos_gcm
        self.priors = smoothed_gcms
        self.true_values = smooth_oos_gcm
        self.ob_noise = (smooth_oos_gcm - oos_gcm).sel(year=rslice)\
                                            .var('year',ddof=1)
        
    def standardized_coords(self,gcm_coords):
        '''
        The function is used to standardize the x-values which in this case 
        are the value for years. The standardization makes the values for year
        from 0-1.
        This is done for the stability of the GPR.
        The function takes in the coordinate values and applies standardization
        '''
        vmax = self.priors['year'].max().item()
        vmin = self.priors['year'].min().item()
        std_coords = (gcm_coords['year']-vmin)/(vmax-vmin)
        return torch.from_numpy(std_coords.values).float()
    
    def initialize_model(self, gp_model, train_period):
        '''
        This function initializes the values that are going to be fed into 
        gpytorch. The main function is to convert everything to torch Tensors
        so the computation can be done using pytorch.

        The function takes the following parameters
        1. gp_model: The class defining  the GP model as specified by gpytorch.Models.
            The exact architecture of the model is defined in the script.
        2. train_period: The training period for the GPR. The oos_gcm data for 
            this period is used for training the GPR.
        '''
        self.trainslice = train_period
        self.fitslice = None
        # Standardizing year to 1-0
        self.train_x = self.standardized_coords(self.observations\
                            .sel(year=self.trainslice))
        self.train_y = torch.from_numpy(self.observations\
                            .sel(year=self.trainslice).values).float()
        self.nsamples = self.train_y.size()[0]
        # Extracting the prior values for the training period
        self.prior_y = torch.from_numpy(self.priors.sel(year=self.trainslice)\
                            .transpose('models','year').values).float()
        # Extracting the prior values for the prediction period
        self.prior_ys = torch.from_numpy(self.priors\
                            .transpose('models','year').values).float()
        # Extracting the x values for the prediction period
        self.prior_x = self.standardized_coords(self.observations['year'])
        # Defining the likelihood function. Here we used a likelihood with fixed heteroscedastic noise
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(\
                            noise=torch.from_numpy(self.ob_noise.values).float().unsqueeze(dim=0),\
                            learn_additional_noise=True)
        # Putting it all together. Defining the actual GPR model for gpytorch
        self.model = gp_model(self.train_x, self.train_y,\
                            self.prior_x,self.prior_y,self.prior_ys,self.likelihood)

    def train_model(self,training_iter):
        '''
        The function takes in the number of training iterations to be performed and 
        trains the GPR model using Adam optimiser.
        '''
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,\
                                            self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

    def fit(self,fitslice):
        '''
        The function is used to add additional obervations in to the model without retraining.
        It takes fitslice as argument which specifies the time period untill which we use as
        observed data. 

        The rest of the function is used to redefine the parameters in the GP model.
        '''
        self.fitslice = fitslice
        self.fitted_x = self.standardized_coords(self.observations\
                            .sel(year=self.fitslice))
        self.fitted_y = torch.from_numpy(self.observations\
                            .sel(year=self.fitslice).values).float()
        self.fsamples = self.fitted_y.size()[0]
        self.prior_y = torch.from_numpy(self.priors.sel(year=self.fitslice)\
                        .transpose('models','year').values).float()
        self.prior_ys = torch.from_numpy(self.priors\
                        .transpose('models','year').values).float()
        # Assigns new set of training data.
        self.model.set_train_data(self.fitted_x,self.fitted_y,strict=False) 
        # Redifines the prior mean module
        self.model.mean_module.refit(self.prior_y,self.prior_ys)
        # Redifines the prior covariance module
        self.model.covar_module.kernels[-1].refit(self.prior_y,self.prior_ys)

    def predict(self):
        '''
        The function is called without any argument to make predictions for the entire time period
        '''
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
       
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.predicted_dist = self.likelihood(self.model(self.prior_x))
            # The predictions are stored as xarray using coordinates from observations
            self.predicted = (self.observations*0+self.predicted_dist.mean.numpy())\
                                    .to_dataset(name='mu_star')
            self.predicted['std'] = self.observations*0+self.predicted_dist.stddev.numpy()
            self.predicted['nstd'] = self.observations*0+(self.ob_noise.values.item()\
                                            +self.predicted_dist.stddev.numpy()**2)**.5

    def calculate_emergence(self, start, conf, fraction_past_emergence, uchoice):
        '''
        The function calculates the emergence of precipitation signals. The input parameters are;
        
        1. start: The year at which we start looking for emergence
        2. conf: Confidence intervels used to define emergence
        3. fraction_past_emergence: The fraction of the posterior that tells us that the 
            signal has emerged.
        4. uchoice: The choice of uncertainty range to be used. Its either std (without added noise)
            or nstd (with added noise). nstd is always greater than std.
        '''
        ToC = np.nan
        ToE = np.nan
        ToEonToC = np.nan
        ToC_sign = 0 
        ToE_sign = 0

        years = self.priors['year'].values
        # Specify the year from which the ToC search has to begin
        syear = years[np.squeeze(np.where(years>=start))[0]]

        looper = 0
        confidence = 0
        while confidence == 0:
            # Starting in start
            yr =  syear + (years[1]-years[0])*looper
            fitslice = slice(years[0],yr)
            self.fit(fitslice)
            self.predict()
            predicted = self.predicted
            
            upperLimit = predicted['mu_star'] + conf*predicted[uchoice]
            lowerLimit = predicted['mu_star'] - conf*predicted[uchoice]
            
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
                ToC = yr
                ToEonToC = int(emerge_ul_yrs[-1].item())
                ToC_sign = -1
                self.emerge_on_toc = predicted
                confidence = 1
            elif len(emerge_ll_yrs)>0:
                ToC = yr
                ToEonToC = int(emerge_ll_yrs[-1].item())
                ToC_sign = 1
                self.emerge_on_toc = predicted
                confidence = 1
      
            if (confidence == 0) & (yr == years[-1]):
                confidence = 1
                self.emerge_on_toc = predicted
            looper+=1

        fullslice = slice(years[0],years[-1])
        self.fit(fullslice)
        self.predict()
        predicted = self.predicted

        upperLimit = predicted['mu_star'] + conf*predicted[uchoice]
        lowerLimit = predicted['mu_star'] - conf*predicted[uchoice]
        
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
            ToE = int(emerge_ul_yrs[-1].item())
            ToE_sign = -1
            self.emerge_on_full = predicted

        elif len(emerge_ll_yrs)>0:
            ToE = int(emerge_ll_yrs[-1].item())
            ToE_sign = 1
            self.emerge_on_full = predicted
        else:
            self.emerge_on_full = predicted
        
        emergence = xr.DataArray(ToE,coords={'lon':self.observations.lon,\
                            'lat':self.observations.lat}).to_dataset(name='toe')
        emergence['toe_sign'] = ToE_sign
        emergence['toc'] = ToC
        emergence['toc_sign'] = ToC_sign
        emergence['toeonc'] = ToEonToC
        emergence['awt'] = emergence['toe']-emergence['toc']
        self.emergence = emergence
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
class spacetimeGPR:
    '''
    This class defines the framework to be applied for a GPR just along the space and time 
    dimension at a gridpoint. 
    The class is defined by providing:\
        1. oos_gcm: The out-of-sample GCM which will be used as observations
        2. smooth_oos_gcm: The smoothed values of the oos_gcm. This is used
            to calculate observational noise and error analysis.
        3. smoothed_gcms: The smoothed data used to calculate the values for 
            prior mean and prior covariances.
        4. rslice: represents the reference period. The reference period is 
            used to calculate the observational noise.
        5. lon: The longitide at which the GPR is being applied
        6. lat: The latitude at which the GPR is being applied
        7. wize: The size of the 2D box used.
    '''
    def __init__(self, oos_gcm, smooth_oos_gcm, smoothed_gcms, rslice, lon, lat, wsize):
        self.observations = oos_gcm
        self.priors = smoothed_gcms
        self.true_values = smooth_oos_gcm
        self.ob_noise = (smooth_oos_gcm - oos_gcm).sel(year=rslice)\
                            .var('year',ddof=1).transpose('lon','lat')
        self.lon = lon
        self.lat = lat
        self.wsize = wsize
        
        
    def standardized_coords(self,gcm_coords):
        '''
        The function is used to standardize the x-values which in this case 
        are the value for years, longitude, and latitude. The standardization makes the
        to 0-1.
        This is done for the stability of the GPR.
        The function takes in the coordinate values and applies standardization
        '''
        vmax = self.priors['year'].max().item()
        vmin = self.priors['year'].min().item()
        time_coords = ((gcm_coords['year']-vmin)/(vmax-vmin)).values
        vmax = self.priors['lat'].max().item()
        vmin = self.priors['lat'].min().item()
        lat_coords = ((gcm_coords['lat']-vmin)/(vmax-vmin)).values
        vmax = self.priors['lon'].max().item()
        vmin = self.priors['lon'].min().item()
        lon_coords = ((gcm_coords['lon']-vmin)/(vmax-vmin)).values
        time, lon, lat = np.meshgrid(time_coords, lon_coords, lat_coords,\
                                            indexing='ij')
        std_coords = torch.from_numpy(np.vstack([time.flatten(),\
                    lon.flatten(), lat.flatten()]).T).float()
        return std_coords
    
    def initialize_model(self, gp_model, train_period):
        '''
        This function initializes the values that are going to be fed into 
        gpytorch. The main function is to convert everything to torch Tensors
        so the computation can be done using pytorch.

        The function takes the following parameters
        1. gp_model: The class defining  the GP model as specified by gpytorch.Models.
            The exact architecture of the model is defined in the script.
        2. train_period: The training period for the GPR. The oos_gcm data for 
            this period is used for training the GPR.
        '''
        self.trainslice = train_period
        self.fitslice = None
        self.train_x = self.standardized_coords(self.observations\
                            .sel(year=self.trainslice))
        self.train_y = torch.from_numpy(self.observations.sel(year=self.trainslice)\
                        .stack(spacetime=['year','lon','lat']).values).float()
        self.prior_y = torch.from_numpy(self.priors.sel(year=self.trainslice)\
                        .transpose('models','year','lon','lat')\
                        .stack(spacetime=['year','lon','lat']).values).float()
        self.prior_ys = torch.from_numpy(self.priors.sel(lat=self.lat,lon=self.lon)\
                        .transpose('models','year').values).float()
        self.prior_x = self.standardized_coords(self.observations\
                                        .sel(lat=self.lat,lon=self.lon))
        self.nsamples = self.train_y.size()[0]
        self.psize = self.prior_ys.size()[0]
        # Defining the likelihood function. Here we used a likelihood with fixed heteroscedastic noise
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(\
                            noise=torch.from_numpy(self.ob_noise.values.flatten()).float()\
                            .repeat(len(self.observations['year'])),\
                            learn_additional_noise=True)
        # Putting it all together
        self.model = gp_model(self.train_x, self.train_y,self.prior_x,\
                              self.prior_y,self.prior_ys, self.likelihood)

    def train_model(self, training_iter):
        '''
        The function takes in the number of training iterations to be performed and 
        trains the GPR model using Adam optimiser.
        '''
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,\
                                            self.model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     self.model.likelihood.noise.item()
            # ))
            optimizer.step()

    def fit(self,fitslice):
        '''
        The function is used to add additional obervations in to the model without retraining.
        It takes fitslice as argument which specifies the time period untill which we use as
        observed data. 

        The rest of the function is used to redefine the parameters in the GP model.
        '''
        self.fitslice = fitslice
        # New set of training values
        self.fitted_x = self.standardized_coords(self.observations\
                            .sel(year=self.fitslice))
        self.fitted_y = torch.from_numpy(self.observations.sel(year=self.fitslice)\
                        .stack(spacetime=['year','lon','lat']).values).float()
        # New set of prior values
        self.prior_y = torch.from_numpy(self.priors.sel(year=self.fitslice)\
                        .transpose('models','year','lon','lat')\
                        .stack(spacetime=['year','lon','lat']).values).float()
        self.prior_ys = torch.from_numpy(self.priors.sel(lat=self.lat,lon=self.lon)\
                        .transpose('models','year').values).float()
        # Redefines the training values
        self.model.set_train_data(self.fitted_x,self.fitted_y,strict=False) 
        # Redefines the mean module
        self.model.mean_module.refit(self.prior_y,self.prior_ys)
        # Redefines the covariance module
        self.model.covar_module.kernels[-1].refit(self.prior_y,self.prior_ys)
        

    def predict(self):
        '''
        The function is called without any argument to make predictions for the entire time period
        '''
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.predicted_dist = self.likelihood(self.model(self.prior_x))
            self.predicted = (self.observations.sel(lat=self.lat,lon=self.lon)*0\
                                +self.predicted_dist.mean.numpy()).to_dataset(name='mu_star')
            self.predicted['std'] = self.observations.sel(lat=self.lat,lon=self.lon)*0\
                                +self.predicted_dist.stddev.numpy()
            self.predicted['nstd'] = self.observations.sel(lat=self.lat,lon=self.lon)*0\
                                    +(self.ob_noise.sel(lat=self.lat,lon=self.lon).values.item()\
                                        +self.predicted_dist.stddev.numpy()**2)**.5

    def calculate_emergence(self, start, conf, fraction_past_emergence, uchoice):
        '''
        The function calculates the emergence of precipitation signals. The input parameters are;
        
        1. start: The year at which we start looking for emergence
        2. conf: Confidence intervels used to define emergence
        3. fraction_past_emergence: The fraction of the posterior that tells us that the 
            signal has emerged.
        4. uchoice: The choice of uncertainty range to be used. Its either std (without added noise)
            or nstd (with added noise). nstd is always greater than std.
        '''
        ToC = np.nan
        ToE = np.nan
        ToEonToC = np.nan
        ToC_sign = 0 
        ToE_sign = 0

        years = self.priors['year'].values
        # Specify the year from which the ToC search has to begin
        syear = years[np.squeeze(np.where(years>=start))[0]]

        looper = 0
        confidence = 0
        while confidence == 0:
            # Starting in start
            yr =  syear + (years[1]-years[0])*looper
            fitslice = slice(years[0],yr)
            self.fit(fitslice)
            self.predict()
            predicted = self.predicted
            
            upperLimit = predicted['mu_star'] + conf*predicted[uchoice]
            lowerLimit = predicted['mu_star'] - conf*predicted[uchoice]
            
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
                ToC = yr
                ToEonToC = int(emerge_ul_yrs[-1].item())
                ToC_sign = -1
                self.emerge_on_toc = predicted
                confidence = 1
            elif len(emerge_ll_yrs)>0:
                ToC = yr
                ToEonToC = int(emerge_ll_yrs[-1].item())
                ToC_sign = 1
                self.emerge_on_toc = predicted
                confidence = 1
      
            if (confidence == 0) & (yr == years[-1]):
                confidence = 1
                self.emerge_on_toc = predicted
            looper+=1

        fullslice = slice(years[0],years[-1])
        self.fit(fullslice)
        self.predict()
        predicted = self.predicted
        upperLimit = predicted['mu_star'] + conf*predicted[uchoice]
        lowerLimit = predicted['mu_star'] - conf*predicted[uchoice]
        
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
            ToE = int(emerge_ul_yrs[-1].item())
            ToE_sign = -1
            self.emerge_on_full = predicted

        elif len(emerge_ll_yrs)>0:
            ToE = int(emerge_ll_yrs[-1].item())
            ToE_sign = 1
            self.emerge_on_full = predicted
        else:
            self.emerge_on_full = predicted
        
        emergence = xr.DataArray(ToE,coords={'lon':self.lon,\
                            'lat':self.lat}).to_dataset(name='toe')
        emergence['toe_sign'] = ToE_sign
        emergence['toc'] = ToC
        emergence['toc_sign'] = ToC_sign
        emergence['toeonc'] = ToEonToC
        emergence['awt'] = emergence['toe']-emergence['toc']
        self.emergence = emergence