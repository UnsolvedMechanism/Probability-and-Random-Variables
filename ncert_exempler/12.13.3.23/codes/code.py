import numpy as np
import math
from scipy.stats import bernoulli

simlen = 10000.0
draws = 8

p_def = 0.9
ctr = 0.0
for i in range(int(simlen)):
    data_bern = bernoulli.rvs(size=draws,p=p_def)
    err_ind = np.nonzero(data_bern == 1)
    if(np.size(err_ind)<draws):
        ctr += 1
    
prob = ctr/simlen
print("Probability - simulation, actual :", prob," 0.569533")
