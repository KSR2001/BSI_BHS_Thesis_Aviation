import numpy as np
s = np.load("models_lstm/scaler.npz")
mu, sd = s["mu"], s["sd"]
print(mu.shape, sd.shape)
print(mu[:10], sd[:10])