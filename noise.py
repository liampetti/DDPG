# ===========================
#   Noise - Ornstein Uhlenbeck
#   Modified from: http://www.turingfinance.com/random-walks-down-wall-street-stochastic-processes-in-python/
#   Author: Liam Pettigrew
# ===========================
import numpy as np

class Noise(object):

    def __init__(self, delta, sigma, ou_a, ou_mu):
        # Noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu

    def brownian_motion_log_returns(self):
        """
        This method returns a Wiener process. The Wiener process is also called Brownian motion. For more information
        about the Wiener process check out the Wikipedia page: http://en.wikipedia.org/wiki/Wiener_process
        :return: brownian motion log returns
        """
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def ornstein_uhlenbeck_level(self, prev_ou_level):
        """
        This method returns the rate levels of a mean-reverting ornstein uhlenbeck process.
        :return: the Ornstein Uhlenbeck level
        """
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        return prev_ou_level + drift + randomness
