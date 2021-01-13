#wd = '/home/lax/src/repos/spd/'
#setwd(wd)
#tau = 0.1
#interest_rate = 0.01
#initial_price = 10000
#days_to_maturity = 200
#n_simulations = 100
#mil = FALSE
#curr_date = '2020-03-04'
#source(paste0(wd, 'src/sim.R'))
source('src/sim.R')
source('src/svcj_simulate_returns.R')


gen_physical_density = function(tau, interest_rate, initial_price, days_to_maturity, curr_date, simmethod = 'SVCJ', n_simulations = 100, f = 'BTC_USD_QUANDL', cached_svcj_parameters = TRUE){
    
    days_to_maturity = max(days_to_maturity, 1)

    # Read historical Bitcoin Data
    if(f == 'BTCUSDT'){
        # Older, shorter data
        hist_dat = read.csv('data/BTCUSDT.csv', stringsAsFactors = FALSE)
        hist_dat$date = as.Date(hist_dat$Date)
        hist_dat = hist_dat[hist_dat$date <= curr_date,]
        hist_dat$Date = NULL
        tdat = ts(hist_dat$Adj.Close)
        rets = diff(log(tdat))
    }else if(f == 'BTC_USD_QUANDL'){
        
        hist_dat = read.csv('data/BTC_USD_Quandl.csv', stringsAsFactors = FALSE)
        hist_dat$date = as.Date(hist_dat$Date)
        hist_dat = hist_dat[hist_dat$date <= curr_date & hist_dat$date >= '2017-01-01',]
        hist_dat$Date = NULL
        hist_dat = hist_dat[rev(seq_len(nrow(hist_dat))), , drop = FALSE]
        tdat = ts(hist_dat$Adj.Close)
        rets = diff(log(tdat))
    }else{
        stop('no time series')
    }


    if(simmethod != 'SVCJ'){

        if(simmethod == 'garch'){
            print('using garch, delete this later!')

            # Estimate sigma with GARCH model
            # Parameters for calculation/simulation
            numbapprox  	= 2000			# fineness of the grid
            N		= n_simulations	# only run once, because the loop is happening in physicaldensity.R
            # Check return series for ARMA effects, e.g. with the following function
            # auto.arima(dax.retts, max.p=10, max.q=10, max.P=5, max.Q=5, 
            # start.p=1, start.q=1,start.P=1, start.Q=1, stationary=T, seasonal=F)
            p		= 0
            q		= 0
            arma= c(p,q)
            # specify garch order (need to be checked)
            m		= 1
            s		= 1
            garch		= c(m,s)
            garchmodel	= "eGARCH"
            submodel	= "GARCH"
            # underlying distribution (default: "sstd" - skewed stundent t's)
            # (alternatives: "norm" - normal, "ghyp"- generalized hyperbolic)
            udist		= "sstd"
            # set archm=T for ARCH in mean model (archpow specifies the power)
            archm		= F
            archpow		= 1
            # set include.mean = F if you don't want to include a mean in the mean model
            include.mean 	= T  
            spec			= ugarchspec(variance.model = list(model = garchmodel, 
                                garchOrder = garch, submodel = submodel), mean.model = 
                                list(armaOrder = arma, archm=archm,archpow=archpow,
                                include.mean=include.mean), distribution.model = udist)
            #garchfit = rugarch::ugarchfit(data = rets, spec = spec, solver = "hybrid")
            garchsim = ugarchsim(garchfit, n.sim = days_to_maturity, 
                                n.start = 0, m.sim=N, startMethod=("sample"), 
                                mexsimdata=TRUE)
            est_sigma = garchsim@simulation$sigmaSim

            # Florens-Zmirou Estimator
            diffusion = sde::ksdiff(tdat, n = length(tdat)) 
        }

        # Run a bunch of simulations
        simulations = list()
        for(i in 1:n_simulations){
            #cat('\nSimulation #', i)
            if(simmethod == 'Milstein'){
                simulations[[i]] = milstein(tdat, r = interest_rate, startvalue = initial_price)
            }else if(simmethod == 'brownian'){
                simulations[[i]] = simple_brownian(r = interest_rate, startvalue = initial_price, days = days_to_maturity, sigma = est_sigma[,i])#days = round(tau * 365))
            }else{
                stop('Didnt specify simulation method')
            }   
        }

        # Collect Last prices and calculate returns
        simulation_return = lapply(simulations, function(z, startprice = initial_price){
            last_price = tail(z, 1)
            ret = (last_price - startprice)/startprice
            return(ret)
        })

        #hist(unlist(simulation_return))
        #plot(unlist(simulations), main = 'trajectories of physical denisty')

    }else{
        # The loop is executed within svcj_sim
        print('SVCJ!')
            simulation_return = execute_svcj_sim(ndays = days_to_maturity, 
                                                n = 2000, 
                                                iter = 5000, 
                                                startdate = curr_date, 
                                                startvalue = initial_price, 
                                                do_plot = FALSE, 
                                                prices = hist_dat,
                                                cached_svcj_parameters = cached_svcj_parameters)
        
        
    }

    # fix one t for g
    # then iterate over domain!
    g_dom = seq(min(tdat), max(tdat), 1)
    g_star = c()
    #browser()
    # This t is wrong! Thats why the PD values are too high!
    #t = length(tdat) - round(tau * 365) #t = 980 # increasing t means decreasing tau, estimation is stabilizing for small tau :)
    if(initial_price == 0L){
        S_t = tail(tdat, 1)
    }else{
        S_t = initial_price
    }
    
    
    #browser()
    for(i in 1:length(g_dom)){

        #print(i)
        u = log(g_dom[i] / S_t)
        p = gen_p_hat(simulation_return, u)$p_hat
        g_star[i] = p/S_t # find closest value
    }

    #plot(g_dom, g_star, main = 'Physical Density ')

    return(list(g_dom, g_star))

}

# Example
#d = gen_physical_density(tau = 0.1, interest_rate = 0.01, initial_price = 0L, days_to_maturity = 10, curr_date = as.Date('2019-01-01'), n_simulations = 100, simmethod = 'svcj', f = 'BTC_USD_QUANDL', cached_svcj_parameters = FALSE)
