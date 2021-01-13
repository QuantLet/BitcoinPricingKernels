# Install R Libraries

libs = list('truncnorm', 
            'MASS', 
            'MCMCpack', 
            'ggplot2', 
            'readr',
            'Rlab',
            'sde', 
            'stats',
            'rugarch',
	    'sm')

lapply(libs, function(z){
    t = try(install.packages(z, dependencies = TRUE))
    if('try-error' %in% class(t)) cat('failed to install ', z)
})

is_loaded = lapply(libs, require, character.only = TRUE)
cat('is loaded: ', is_loaded)
