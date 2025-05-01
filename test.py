import numpy as np
import math
from scipy.signal import deconvolve
import librosa
import spectrum
import numpy as np

# commands = ["start"]
commands = ["start", "finish", "go", "stop"]
for word in commands:
    lpc_coeff_per_word = 0
    for i in range(0,10,1):
        name = f"{word}-{i+1:02d}"
        path = f"Data/Processed/{word}/"
        output_path = f"Data/Coeff/{word}/"

        coeff = np.load(output_path+name+".npy").astype(np.float64)
        lpc_coeff_per_word += coeff.shape[0]
        print(f"{coeff.shape[0]}", end=" ")
    print(f"LPC coeff: {lpc_coeff_per_word}")


# print(f"Shape: {coeff.shape}")
# for i in range(coeff.shape[0]):
#     print(f"LPC: {coeff[i][0]}")
#     print(f"Auto: {coeff[i][1]}")
#     
#     print(f"{coeff[i][1][-1]:.9f}")
#        





