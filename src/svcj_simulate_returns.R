
#setwd('/home/lax/src/repos/spd/')
# -----------------
# Libraries loading
# -----------------

# List of libraries to be used
lib <- list("truncnorm", "MASS", "MCMCpack", "ggplot2", "readr", "Rlab")

# Installing or calling the libraries
invisible(lapply(lib, function(x){
  result <- library(x, logical.return=T, character.only =T)
  if(result == F) install.packages(x)
  library(x, character.only =T)
  #print(paste0(x, " loaded"))
}))
rm(lib)

# Get svcj function
source("src/svcj_model.R")

execute_svcj_sim = function(ndays = 10, n = 2000, iter = 5000, startdate = as.Date('2020-01-01'), startvalue = 0L, do_plot = TRUE, prices = NULL, cached_svcj_parameters = FALSE){

    # @ Todo: this is failing in main proabably because n is too small!
    # so just subset after n days and run for 1000 days anyways :)

    # Load price feed!
    # @Todo Check if daily data is ok
    #tdat = read.csv('data/BTCUSDT.csv', stringsAsFactors = FALSE)
    if(is.null(prices)){
        print('loading btc returns')
        btc_price_time_series_file = 'data/BTCUSDT.csv'
        tdat = read.csv(btc_price_time_series_file, stringsAsFactors = FALSE)
        prices = data.frame(list('Adj.Close' = tdat$Adj.Close,
                            'date' = as.Date(tdat$Date)))
        rm(tdat)
    }

    # Fit SVCJ
    # Some initial set up parameters: 
    #n      = 1000  # Number of observaions in each simulation (i.e number of future days)
    #iter = 5000 # Number of price paths
    #startdate = as.Date('2017-12-01')
    #do_plot = False
    p = prices$Adj.Close[prices$date <= startdate]
    print(summary(p))
    #browser()
    if(cached_svcj_parameters){
        #print('loading svcj params')
        load('svcj_results_R.RDATA')
        assign('btc_svcj', svcj_results)
    }else{
        print('Fitting SVCJ...')
        btc_svcj <- svcj_model(p, N = iter, n = n)
    }
    
    print(btc_svcj$parameters)
    #save(btc_svcj,file="Data/btc_svcj_own.Rda")


    # --------------------------------------
    # Price Simulation using SVCJ parameters
    # --------------------------------------

    #cryptos <- c("btc")
    crypto = 'btc'
    svcj_results = list('btc' = btc_svcj)

    simulated_returns_long = list()
    parameters = svcj_results[[crypto]]$parameters
    mu = parameters[1,2]
    mu_y = parameters[2,2]  
    sigma_y = parameters[3,2]  
    lambda = parameters[4,2]
    alpha = parameters[5,2]
    beta = parameters[6,2]
    rho = parameters[7,2]
    sigma_v = parameters[8,2]
    rho_j = parameters[9,2]
    mu_v = parameters[10,2]
    kappa = -(alpha/beta)
    theta = alpha/kappa
    
    # Create empty vectors to store the simulated values
    V    = matrix(0, nrow = iter, ncol  = n)  # Volatility of log return
    Y    = matrix(0, nrow = iter, ncol  = n)  # Log return
    Jv   = matrix(0, nrow = iter, ncol  = n)  # Jumps in volatility
    Jy   = matrix(0, nrow = iter, ncol  = n)  # Jumps in log return
    V[,1] = mu_v  # Initial value of volatility = mean of volatilty
    
        for (i in 2:n) {
        Z = mvrnorm(n = iter, mu = c(0,0), Sigma = matrix(c(1,rho,rho,1), nrow = 2))  # Standard normal random value
        Z1 = Z[,1]
        Z2 = Z[,2] 
        J = rbern(n = iter, prob = lambda)  # Bernoulli distributed random value with lambda = 0.051 for determining whether a jump exists
        XV       = rexp(n = iter, rate = 1/mu_v)  # Exponential distributed random value with mV = 0.709 for jump size in volatility
        X        = rnorm(n = iter, mean = mu_y + rho_j * XV, sd = sigma_y)  # Jump size of log return
        #V[i]     = alpha + beta*V[i-1] + sigma_v*sqrt(V[i-1])*Z2 + XV*J  # Volatilty
        V[,i]     = kappa * theta + (1 - kappa) * V[,i- 1] + sigma_v*sqrt(V[,i-1])*Z2 + XV*J
        Y[,i]     = mu + sqrt(V[,i-1])*Z1 + X*J  # Log return
        Jv[,i]    = XV*J  # Jumps in volatilty (0 in case of no jump)
        Jy[,i]    = X*J  # Jumps in log return (0 in case of no jump)
        #print(paste0("Simulation ",i," of ", n, " for ", crypto))
        }
    
    simulated_returns_long[[crypto]] <- setNames(list(Y, V), 
                                                c("simulated_returns", "simulated_volatility"))
        
    #}
    

    #rm(parameters, alpha, beta, crypto, i, J, 
    #Jv, Jy, kappa, lambda, mu, mu_v, mu_y, rho, rho_j, sigma_v, sigma_y,
    #theta, V, X, XV, Y, Z1, Z2, Z)

    #save(simulated_returns_long, file = "simulated_returns_long.Rda")
    #load("simulated_returns_long.Rda")

    # Store a smaller list of simulated returns (to save memory space for the app) 
    #simulated_returns <- list()
    #for (i in cryptos) {
    #simulated_returns[[i]] <- simulated_returns_long[[i]]$simulated_returns[,1:361]
    #}


    # Plot some simulated price paths for ETH
    #if(startvalue == 0L){
    #print('using startvalue from btc time series')
    init_price <- prices$Adj.Close[prices$date == startdate] 
    print(paste0('init price in svcj sim: ', init_price))
    #}else{
    #    init_price = startvalue
    #}
    sim_price <- data.frame(matrix(NA, nrow = iter, ncol = n))
    sim_price[,1] <- init_price

    # Revert from returns back to prices
    for (i in 2:n) {
        #print(i)
    sim_price[,i] <- (simulated_returns_long$btc$simulated_returns[,i]/sqrt(250) + 1)*sim_price[,i-1] 
    }
    
    # @Todo need to restrict sim_price to the correct amount of days here
    out = apply(sim_price, 1, function(z, .ndays = ndays){
        raw = na.omit(z)
        f = raw[1] #head(raw, 1)
        l = raw[.ndays] #tail(raw, 1)
        ret = (l-f)/f
        return(ret)
    })
    #browser()
    if(do_plot){

        # Plot density too

        # Plot simulated return for BTC
        ggplot() + 
        geom_line(aes(x = seq(1:1000), y = simulated_returns_long$Adj.Close$simulated_returns[100,]), col = "black") + 
        geom_line(aes(x = seq(1:1000), y = simulated_returns_long$Adj.Close$simulated_returns[1000,]), col = "blue") + 
        labs(ylab("Daily Returns")) +
        labs(xlab("Days")) +
        theme_bw() +
        theme(panel.grid = element_blank()) 

        # Plot some simulated price paths for ETH
        ggplot() + 
        geom_line(aes(x = seq(1:1000), y = unlist(sim_price[500,])), col = "black") + 
        geom_line(aes(x = seq(1:1000), y = unlist(sim_price[1000,])), col = "blue") + 
        geom_line(aes(x = seq(1:1000), y = unlist(sim_price[2501,])), col = "green") +
        geom_line(aes(x = seq(1:1000), y = unlist(sim_price[3500,])), col = "red") +
        geom_line(aes(x = seq(1:1000), y = unlist(sim_price[5000,])), col = "purple") +
        labs(ylab("Daily Returns")) +
        labs(xlab("Days")) +
        theme_bw() +
        theme(panel.grid = element_blank()) 
    }

    # Return list of simulated returns
    return(out)

}

#execute_svcj_sim(ndays = 10, n = 1000, iter = 5000, startdate = as.Date('2020-01-01'), startvalue = 0L, do_plot = FALSE, prices = NULL)