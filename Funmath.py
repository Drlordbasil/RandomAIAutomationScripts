import numpy as np

# Constants
P_0, alpha, beta, gamma, delta, epsilon, zeta, eta = 1, 0.1, 0.1, 0.05, 0.2, 0.15, 0.25, 0.3
G, v, c = 1, 0, 299792458  # Using c for completeness

# Initial static inputs
Q, E, Psy, Env = 0.3, 0.7, 0.5, 0.2  # Initial values for simplicity

# Function to update and calculate P_target based on dynamic Sentiment Index
def update_P_target(SI):
    P_target = P_0 * np.exp(alpha*G + beta*(v**2/c**2) + gamma*Q + delta*E + epsilon*Psy + zeta*Env + eta*SI)
    return P_target

# Simulate dynamic Sentiment Index updates
sentiment_updates = [0.4, 0.6, -0.3, 0.5]  

for SI in sentiment_updates:
    P_target = update_P_target(SI)
    print(f"With Sentiment Index {SI}, Predicted Future Time Perception: {P_target:.2f}")
