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


# INFO: Calculating LPC of a signal
# NOTE: Librosa takes in 1 x N
# Output: lpc_coeff array
def get_lpc_and_auto_coefficients(signal: np.ndarray, framerate=16000, print_rsme=True, show_graphs=False):
    nframes = len(signal)
    samples_per_20ms = int(0.02 * framerate)
    samples_per_10ms = int(0.01 * framerate)
    num_frames_in_signal = int((nframes-samples_per_20ms)/samples_per_10ms) + 1

    # NOTE: The last element of the auto_coeff is zero. This is to be able to save the 
    # 13 values of the lpc (counting 1) in the same np.array as lpc_coeff.
    order = 12
    coefficients = np.zeros((num_frames_in_signal,2,order+1), dtype=np.float64) 

    # print(f"Num frames in signal: {num_frames_in_signal}")
    for i in range(num_frames_in_signal):
        start = i*samples_per_10ms
        finish = start + samples_per_20ms

        if i == num_frames_in_signal-1:  
            frame_to_evalute = signal[start:-1]
        else:
            frame_to_evalute = signal[start:finish]

        # NOTE: Order 12 gives a (13,) array
        lpc_coeff = librosa.lpc(frame_to_evalute, order=order) 
        # print(f"Shape coeff: {lpc_coeff.shape}")
        # print(f"LPC: {lpc_coeff}")

        # INFO: Calculate Auto-correlation coefficients
        # Output: auto_coeff array
        auto_coeff = librosa.autocorrelate(frame_to_evalute,max_size=order)
        # print(f"Auto coeff shape: {auto_coeff.shape}")
        # print(f"Auto coeff: {auto_coeff}")

        coefficients[i][0] = np.copy(lpc_coeff)
        coefficients[i][1][:12] = np.copy(auto_coeff) # last element is zero

        if(print_rsme==True):
            displaying_lpc(frame_to_evalute, lpc_coeff, show_graphs=show_graphs)

    return coefficients 

# INFO: Assuming that every comand has 10 
# files for training
def create_coefficients(word: str):
    path = f"Data/Processed/{word}/"
    output_path = f"Data/Coeff/{word}/"

    if not os.path.exists(output_path):
           os.makedirs(output_path)

    lpc_all = np.zeros((1,14))
    auto_all = np.zeros((1,13))
    for i in tqdm(range(0,10,1), desc=f"Processing LPC and Auto for: {word}"): 
        name = f"{word}-{i+1:02d}"
        signal = np.load(path+name+".npy")
        signal_squeeze = signal.squeeze()
        # print(f"{i+1}")
        output = get_lpc_and_auto_coefficients(signal_squeeze, print_rsme=False)
        for i in range(output.shape[0]):
            temp_lpc = output[i][0]
            temp_auto = output[i][1].reshape(1,-1)
            temp_lpc= np.insert(temp_lpc, 0, 0).reshape(1,-1) # insert space to identify centroid
            lpc_all = np.append(lpc_all, temp_lpc, axis=0)
            auto_all = np.append(auto_all, temp_auto, axis=0)

    np.save(output_path+word+"_lpc.npy", lpc_all[1:,:]) # Remove the first array with 0
    np.save(output_path+word+"_auto.npy", auto_all[1:,:])


def print_array(array: np.ndarray):
    array = array.reshape(-1,1);
    for i in range(len(array)):
        print(f"{array[i]}", end=" ")     

    print(f"\n")

def itakura_saito_distance(auto_coeff_raw: np.ndarray, auto_coeff_centroid: np.ndarray):
    auto_coeff_raw = auto_coeff_raw.flatten()
    auto_coeff_centroid = auto_coeff_centroid.flatten()

    auto_coeff_raw_len = len(auto_coeff_raw)
    auto_coeff_centroid_len = len(auto_coeff_centroid)

    if(auto_coeff_raw_len != auto_coeff_centroid_len):
        print(f"Arrays must be same length")

    distance = auto_coeff_raw[0] * auto_coeff_centroid[0]

    mult = np.multiply(auto_coeff_raw[1:], auto_coeff_centroid[1:])

    distance += 2 * np.sum(mult)

    return distance

# INFO: Calculate LSF Centroids based on lpc_coeff[0] id
# the number of centroids is max id in the first column
# of all rows
# Input
# `lpc_coeff` Shape: [order lpc = 12+1, num_centroids]
# Output
# `centroids_lsf` Shape: [dim=12, num_centroids]
def calculate_lsf_centroids(lpc_coeff: np.ndarray) -> np.ndarray:
    num_lpc = lpc_coeff.shape[0]
    dim = lpc_coeff.shape[1]-2 # remove 1 for id and another for dim reduction when poly2lsf 
    num_centroids = int(lpc_coeff[:,0].max())+1
    
    centroids_lsf = np.zeros((dim,num_centroids))
    centroids_lsf_count = np.zeros((1,num_centroids))

    for i in range(num_lpc):
        id = int(lpc_coeff[i,0])
        temp_lsf = np.array(spectrum.poly2lsf(lpc_coeff[i,1:]))
        centroids_lsf[:,id] = np.add(centroids_lsf[:,id],temp_lsf) 
        centroids_lsf_count[0,id] += 1

    for i in range(num_centroids):
        centroids_lsf[:,i] = np.divide(centroids_lsf[:,i], centroids_lsf_count[0,i])

    return centroids_lsf

# INFO: `id_centroid` will store the assignment of the auto_coeff_frames to each centroid
# Input
# `id_centroid` Shape: [num of frames,]
# `auto_coeff_frames`: [num of frames, order of lpc=12 ]
# `centroids_lsf` : [order of lpc=12, num centroids]
# Output
# `global_distance` : float
# `id_centroid` : [num of frames, 1]
def calculate_distances(auto_coeff_frames: np.ndarray, centroids_lsf: np.ndarray, id_centroid: np.ndarray):
    num_centroids_lsf = centroids_lsf.shape[1]  
    dim_centroids_lpc = centroids_lsf.shape[0]+1 # When converting to lpc, one more dim
    dim_auto_coeff_centroids = dim_centroids_lpc - 1 # -1 to so it has same dimension as auto coeff frame

    centroids_lpc = np.zeros((dim_centroids_lpc,num_centroids_lsf))
    auto_coeff_centroids_lpc = np.zeros((dim_auto_coeff_centroids,num_centroids_lsf))
    for i in range(num_centroids_lsf): # Converting to lpc and getting autocorrelation coefficients
        centroids_lpc[:,i] = np.array(spectrum.lsf2poly(centroids_lsf[:,i]))
        auto_coeff_centroids_lpc[:,i] = librosa.autocorrelate(centroids_lpc[:,i], max_size=dim_auto_coeff_centroids) 

    distances = np.zeros((num_centroids_lsf,1)) 
    num_auto_coeff_frames = auto_coeff_frames.shape[0]

    global_distance = 0
    for i in range(num_auto_coeff_frames): # Calculating distances for every frame with every centroid
        for j in range(num_centroids_lsf):
            vector = auto_coeff_centroids_lpc[:,j]
            vector2 = auto_coeff_frames[i,:]

            distances[j,0] = itakura_saito_distance(vector2, vector)  
        
            min_index_distance = distances.argmin()

            id_centroid[i] = min_index_distance
        
        global_distance += distances[min_index_distance,0]
        
    output = {
        "global_distance": global_distance,
        "id_centroid": id_centroid
    }
    return output

# def new_epsilon_centroids(centroids_lsf: np.ndarray):

    
# INFO: Assumes that each file contains the lpc and auto coeff
# of each frame of that word. We must read every file. 
# One file will mean only one training word.
def create_code_vector(word: str, centroids=16, epsilon1=1.001, epsilon2=0.999, global_distance_threshold=1):

    iterations = int(math.log2(centroids))
    ispowerof2 =  2**iterations == centroids 
    if (not ispowerof2):
        print(f"Centroids must be power of 2")
        return -1

    path = f"Data/Coeff/{word}/"
    name_lpc = f"{word}_lpc.npy"
    name_auto = f"{word}_auto.npy"
    lpc_coeff = np.load(path+name_lpc)  
    auto_coeff = np.load(path+name_auto)[:,0:-1]
    number_of_lpc = lpc_coeff.shape[0]
    number_of_auto = auto_coeff.shape[0]

    print(f"LPC Shape: {lpc_coeff.shape}")
    lsf_coeff = np.zeros((12,1))
    top1 = 2
    for i in range(number_of_lpc):
        extracting_lpc = lpc_coeff[i,1:]
        lsf_temp = np.array(spectrum.poly2lsf(extracting_lpc)).reshape(-1,1)
        lsf_coeff = np.add(lsf_coeff, lsf_temp)
        # if i > top1: break

    
    first_centroid_lsf = np.divide(lsf_coeff,number_of_lpc) # Initialize centroid in LSF
    centroids_lsf = np.zeros((12,2))
    centroids_lsf[:,0] = (first_centroid_lsf*epsilon1).flatten() # Obtain 2 new centroid
    centroids_lsf[:,1] = (first_centroid_lsf*epsilon2).flatten()

    global_distance = 0
    prev_global_distance = 0
    diff_global_distance = global_distance_threshold+1 
    top1 = 2
    for j in range(12):
    # j = 0
    # while(diff_global_distance>global_distance_threshold):
        print(f"\nIter: {j}")
        id_centroid = lpc_coeff[:,0]
        output = calculate_distances(auto_coeff, centroids_lsf, id_centroid)
        global_distance = output["global_distance"]
        id_centroid = output["id_centroid"]

        lpc_coeff[:,0] = id_centroid

        diff_global_distance = math.fabs(global_distance - prev_global_distance)
        prev_global_distance = global_distance
        print(f"GD: {global_distance:,}")
        print(f"Diff: {diff_global_distance:,}")

        centroids_lsf = calculate_lsf_centroids(lpc_coeff)
        j+=1


if __name__ == "__main__":
    # commands = ["start", "finish", "go", "stop"]
    # for word in commands:
    #     create_coefficients(word)

    commands = ["start"]
    create_code_vector(commands[0])

    # Testing calculate_lsf_centroids
    # word = commands[0]
    # path = f"Data/Coeff/{word}/"
    # name_lpc = f"{word}_lpc.npy"
    # name_auto = f"{word}_auto.npy"
    # lpc_coeff = np.load(path+name_lpc)  
    # calculate_lsf_centroids(lpc_coeff)





