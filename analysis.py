import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
import spectrum

# INFO: Root Mean Square Error
def rmse(org: np.ndarray, forecast: np.ndarray) -> float:
    rmse = math.sqrt((1/len(org)) * np.sum((org - forecast)**2 ))
    return rmse


# INFO: Display coeff signal 
# `original_signal` must be (n,) not (n,1)
# To represent how well the LPC coefficients work
# the LPC coeff are applied as a filter to the original signal.
# Every new sample is created based on the multiplication of the 
# n samples and the n coefficient.
def displaying_lpc(original_signal: np.ndarray, coeff: np.ndarray):
    b = np.hstack([[0], -1 * coeff[1:]])

    y_hat = scipy.signal.lfilter(b, [1], original_signal)

    rmse_ = rmse(original_signal, y_hat)
    print(f"Mean Avergae Percentage Error: {rmse_}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original_signal, label='Original Signal')
    ax.plot(y_hat, linestyle='--', label='LPC Prediction')
    ax.legend()
    ax.set_title(f'LPC Model (Order 12) Prediction')
    plt.tight_layout()
    plt.show()


# INFO: Calculating LPC
# NOTE: Librosa takes in 1 x N
# Output: lpc_coeff array
signal = np.load("start-09.npy")
signal_squeeze = signal.squeeze()
# print(f"shape: {signal.shape}, type: {signal.dtype}")
# print(f"shape: {signal_squeeze.shape}, type: {signal_squeeze.dtype}")
lpc_coeff = librosa.lpc(signal_squeeze, order=12)
# print(f"Shape coeff: {coeff.shape}")
# print(f"Coeff: {coeff}")

# INFO: Calculate Auto-correlation coefficients
# Output: auto_coeff array
auto_coeff = librosa.autocorrelate(signal_squeeze,max_size=12)
# print(f"Auto coeff shape: {auto_coeff.shape}")
# print(f"Auto coeff: {auto_coeff}")


# displaying_lpc(signal_squeeze, coeff)
