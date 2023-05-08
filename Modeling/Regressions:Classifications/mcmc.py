import numpy as np
import pandas as pd

##########################################
# [Markov chain monte carlo simulation] is 
# one form of stochastic process. The 
# Metropolis-Hastings algorithm is one 
# instance where the stationary distribution
# of a markov chain is exploited as it tends 
# towards P distribution of the data.

# typically, there is a burn-in period of
# sampled points from the algorithm, during
# which we throw out the samples as they 
# may not accurately represent the underlying
# distribution.

# Our first use case could be at financial
# time series data.

# Another use case may use a beta distribution
# as the conjugate prior - 
# E[X] = alpha/(beta+alpha).
# V[X] = (alpha*beta)/(alpha+beta)^2(alpha+beta+1)
# We can quantify the uncertainty in our predictions
# a bit better. For instance, we could implement a 
# beta regression.
#################################################

class MCMC:
	def __init__(self, data):
		pass





if __name__ == "__main__":
	pass
