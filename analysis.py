import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
import spectrum
from tqdm import tqdm
import os

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
def displaying_lpc(original_signal: np.ndarray, coeff: np.ndarray, show_graphs=False):
    b = np.hstack([[0], -1 * coeff[1:]])

    y_hat = scipy.signal.lfilter(b, [1], original_signal)
    rmse_ = rmse(original_signal, y_hat)
    print(f"Mean Avergae Percentage Error: {rmse_}")

    if show_graphs==True:
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
def get_lpc_and_auto_coefficients(signal: np.ndarray, framerate=16000, print_rsme=True, show_graphs=False):
    nframes = len(signal)
    samples_per_20ms = int(0.02 * framerate)
    samples_per_10ms = int(0.01 * framerate)
    num_frames_in_signal = int((nframes-samples_per_20ms)/samples_per_10ms) + 1

    order = 12
    coefficients = np.zeros((num_frames_in_signal,2,order), dtype=np.float64) 

    # print(f"Num frames in signal: {num_frames_in_signal}")
    for i in range(num_frames_in_signal):
        start = i*samples_per_10ms
        finish = start + samples_per_20ms

        if i == num_frames_in_signal-1:  
            frame_to_evalute = signal[start:-1]
        else:
            frame_to_evalute = signal[start:finish]

        # NOTE: Order 12 gives a (13,) array
        lpc_coeff = librosa.lpc(frame_to_evalute, order=order-1) 
        # print(f"Shape coeff: {lpc_coeff.shape}")
        # print(f"LPC: {lpc_coeff}")

        # INFO: Calculate Auto-correlation coefficients
        # Output: auto_coeff array
        auto_coeff = librosa.autocorrelate(frame_to_evalute,max_size=order)
        # print(f"Auto coeff shape: {auto_coeff.shape}")
        # print(f"Auto coeff: {auto_coeff}")

        coefficients[i][0] = np.copy(lpc_coeff);
        coefficients[i][1] = np.copy(auto_coeff);

        if(print_rsme==True):
            displaying_lpc(frame_to_evalute, lpc_coeff, show_graphs=show_graphs)

    return coefficients 

# INFO: Assuming that every comand has 10 
# files for training
def create_coefficients(commands: list):
    for word in commands:
        path = f"Data/Processed/{word}/"
        output_path = f"Data/Coeff/{word}/"

        if not os.path.exists(output_path):
               os.makedirs(output_path)

        for i in tqdm(range(0,10,1), desc=f"Processing LPC and Auto for: {word}"): 
            name = f"{word}-{i+1:02d}"
            signal = np.load(path+name+".npy")
            signal_squeeze = signal.squeeze()
            # print(f"{i+1}")
            output = get_lpc_and_auto_coefficients(signal_squeeze, print_rsme=False)
            np.save(output_path+name+".npy", output)
            # print(f"\n")

# INFO: Assumes that each file contains the lpc and auto coeff
# of that word. We must read every file. One file will mean only
# one training word
def create_code_vector(commands: list):


if __name__ == "__main__":
    commands = ["start", "finish", "go", "stop"]
    create_coefficients(commands)
    # commands = ["start"]



