import numpy as np
#import pyjion
#pyjion.config(level=2)
#pyjion.enable()

def gibbs_sample_x(lotus_n_papers, p_x, gamma: float, delta: float):
    #calculate prior
    p_x_1 = p_x
    p_x_0 = 1 - p_x_1
    
    #calculate likelihood of data given x
    likelihood_x_0 = likelihood(lotus_n_papers, np.zeros_like(p_x), gamma, delta)
    likelihood_x_1 = likelihood(lotus_n_papers, np.ones(shape=p_x.shape), gamma, delta)
    
    #compute likelihood times prior
    numerator_0 = likelihood_x_0 * p_x_0
    numerator_1 = likelihood_x_1 * p_x_1

    denominator = numerator_0 + numerator_1
    
    conditional_probability = numerator_1/denominator
    return np.random.binomial(n=1, p=conditional_probability)

def likelihood(lotus_n_papers, x, gamma, delta):
    # Calculate the total number of papers published per molecule
    P_m = np.sum(lotus_n_papers, axis=1)
    
    # Calculate the total number of papers published per species
    Q_s = np.sum(lotus_n_papers, axis=0)
    
    # Calculate the total research effort for each (molecule, species) pair
    clipped_exponent = np.clip(-gamma * P_m[:, None] - delta * Q_s[None, :], -50, 50)
    total_research_effort = 1 - np.exp(clipped_exponent, dtype="float32")
    
    likelihood_ = np.zeros_like(total_research_effort)
    
    #add the four conditions of our model
    x_0_L_0 = (lotus_n_papers == 0) & (x==0)
    #x_0_L_1 =((lotus_n_papers != 0) & (x==0)
    x_1_L_0 = (lotus_n_papers == 0) & (x!=0)
    x_1_L_1 = (lotus_n_papers != 0) & (x!=0)
    
    #fullfil the four conditions
    likelihood_[x_0_L_0] = 1
    #likelihood[x_0_L_1] = 0
    likelihood_[x_1_L_0] = 1-total_research_effort[x_1_L_0]
    likelihood_[x_1_L_1] = total_research_effort[x_1_L_1]
    
    return likelihood_

# Define a function that calculates the log likelihood of the LOTUS model
def log_likelihood(lotus_n_papers, x, gamma, delta):
    likelihood_ = likelihood(lotus_n_papers, x, gamma, delta)
    
    # Calculate the likelihood of observing the data
    log_likelihood = np.sum(np.log(likelihood_ + 1e-12))
    return log_likelihood

# Define a function that calculates the log prior of the parameters gamma and delta
def log_prior(gamma, delta, gamma_min=0, gamma_max=2, delta_min=0, delta_max=2):
    # Return 0 if the parameters are within the specified bounds, and -inf otherwise
    if gamma_min <= gamma <= gamma_max and delta_min <= delta <= delta_max:
        return 0
    else:
        return -np.inf

# Define a function that proposes new values for gamma and delta based on the current values
def proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta):
    # Generate new values for gamma and delta by adding Gaussian noise with mean 0 and standard deviation proposal_scale_gamma and proposal_scale_delta, respectively
    gamma_new = gamma + np.random.normal(loc=0, scale=proposal_scale_gamma)
    delta_new = delta + np.random.normal(loc=0, scale=proposal_scale_delta)
    
    # Apply a reflection strategy by taking the absolute value of the new values
    gamma_new = np.abs(gamma_new)
    delta_new = np.abs(delta_new)
    
    return gamma_new, delta_new

# Define a function that determines whether to accept or reject the proposed values of gamma and delta
def metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta_new):
    # Calculate the log likelihood and log prior of the current and proposed values of gamma and delta
    curr_log_likelihood = log_likelihood(lotus_n_papers, x, gamma, delta)
    new_log_likelihood = log_likelihood(lotus_n_papers, x, gamma_new, delta_new)
    curr_log_prior = log_prior(gamma, delta)
    new_log_prior = log_prior(gamma_new, delta_new)
    
    # Check if the log likelihood and log prior are finite, and return False if any of them are not
    if not (np.isfinite(curr_log_likelihood) and np.isfinite(new_log_likelihood) and np.isfinite(curr_log_prior) and np.isfinite(new_log_prior)):
        return False
    
    # Calculate the log ratio of the new and current values of gamma and delta
    log_ratio = (new_log_likelihood + new_log_prior) - (curr_log_likelihood + curr_log_prior)
    
    # Accept the proposal with probability exp(min(0, log_ratio))
    return np.random.uniform() < np.exp(min(0, log_ratio))

# Define a function that runs the Metropolis-Hastings MCMC algorithm
def run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter, gamma_init, delta_init,
                        p_x, target_acceptance_rate=(0.25, 0.35), check_interval=500):
    gamma, delta, x = gamma_init, delta_init, x_init
    
    # Initialize the samples array to store gamma and delta values
    samples = np.zeros((n_iter, 2))
    
    # Initialize a list to store x values at each iteration
    x_samples = []

    accept_gamma = 0
    accept_delta = 0
    proposal_scale_gamma = 0.0005
    proposal_scale_delta = 0.0005
    print_var = n_iter/10

    for i in range(n_iter):
        if i+1 % print_var == 0:
            print("Done :", i/n_iter *100, "% of total iterations")
        # Update gamma
        gamma_new, _ = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma_new, delta):
            gamma = gamma_new
            accept_gamma += 1

        # Update delta
        _, delta_new = proposal(gamma, delta, proposal_scale_gamma, proposal_scale_delta)
        if metropolis_hastings_accept(lotus_n_papers, x, gamma, delta, gamma, delta_new):
            delta = delta_new
            accept_delta += 1
        
        # Update x using Gibbs sampling
        x = gibbs_sample_x(lotus_n_papers, p_x, gamma, delta)
        
        # Store gamma and delta values in the samples array
        samples[i] = [gamma, delta]
        
        # Append the current x value to the x_samples list
        x_samples.append(x)

        # Check acceptance rate and adjust proposal_scale if necessary
        if (i + 1) % check_interval == 0:
            acceptance_rate_gamma = accept_gamma / check_interval
            acceptance_rate_delta = accept_delta / check_interval

            if not (target_acceptance_rate[0] <= acceptance_rate_gamma <= target_acceptance_rate[1] and
                    target_acceptance_rate[0] <= acceptance_rate_delta <= target_acceptance_rate[1]):
                # Adjust proposal_scale_gamma and proposal_scale_delta
                if acceptance_rate_gamma < target_acceptance_rate[0]:
                    proposal_scale_gamma *= 0.8
                elif acceptance_rate_gamma > target_acceptance_rate[1]:
                    proposal_scale_gamma *= 1.2

                if acceptance_rate_delta < target_acceptance_rate[0]:
                    proposal_scale_delta *= 0.8
                elif acceptance_rate_delta > target_acceptance_rate[1]:
                    proposal_scale_delta *= 1.2

            # Reset acceptance counters
            accept_gamma = 0
            accept_delta = 0

    return samples, x_samples, acceptance_rate_gamma, acceptance_rate_delta
