import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_confusion_matrix(predictions_list, classes, test_words, save_path="Images/confusion_matrix"):
    # Number of classes
    n_classes = len(classes)
    
    # Initialize the confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes))
    
    # For each test word and its predictions
    for i, predictions in enumerate(predictions_list):
        true_class = test_words.index(test_words[i])
        
        # Count predictions for each class
        for j in range(n_classes):
            conf_matrix[true_class, j] = np.sum(predictions[:, j])
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    # plt.tight_layout()
    # plt.show()
    
    # return conf_matrix

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


# INFO: 
# Input:
# `auto_coeff_signal` : [num, dim]
# `auto_coeff_code_vector` : [dim, num]
def calcualte_min_dist_codevector(auto_coeff_signal: np.ndarray, auto_coeff_code_vector: np.ndarray):
    num_frames_signal = auto_coeff_signal.shape[0]
    num_centroids = auto_coeff_code_vector.shape[1]
    global_distance = 0
    min_distance = 0
    distances = np.zeros(num_frames_signal)
    for i in range(num_frames_signal):
        for j in range(num_centroids):
            distance = itakura_saito_distance(auto_coeff_signal[i,:], auto_coeff_code_vector[:,j])
            if j == 0: 
                min_distance = distance
            if distance < min_distance:
                min_distance = distance

        distances[i] = min_distance

    return np.mean(distances) 

def word_recognition(word:str , code_vectors: list[str]):
    path_auto = f"Data/Coeff/{word}/"

    start = 10
    finish = 15
    num_words_test = finish - start
    num_code_vectors = len(code_vectors)
    recognition_dist = np.zeros((num_words_test,num_code_vectors))
    recognition = np.zeros(recognition_dist.shape)
    count = 0
    for i in range(start,finish):
        name = f"{word}-{i+1:02d}"
        # print(f"name: {name}")
        for j in range(num_code_vectors):
            path_code_vector = f"Data/CodeVector/{code_vectors[j]}/"
            auto_coeff_signal = np.load(path_auto+name+"_test_auto.npy")[:,0:-1]
            auto_coeff_code_vector = np.load(path_code_vector+code_vectors[j]+"_auto_coeff_lpc_code_vector.npy")
            recognition_dist[count,j] = calcualte_min_dist_codevector(auto_coeff_signal, auto_coeff_code_vector)
        count+=1

    for i in range(recognition.shape[0]):
        min_index = recognition_dist[i,:].argmin()
        recognition[i,min_index] = 1
        recognition[i,:] = np.eye(1, num_code_vectors, min_index)[0]


    # print(f"Code Vectors: {code_vectors}")
    print(f"{recognition}")
    # print(f"{recognition_dist}")

    return recognition


if __name__ == "__main__":
    # INFO: Code vector [12,16]
    # INFO: Signal coeff data is [0:12]

    code_commands = ["start", "finish", "go", "stop"]
    test_comands = ["start", "finish", "go", "stop"]
    results = []
    for test_word in test_comands:
        print(f"test word: {test_word}")
        results.append(word_recognition(test_word, code_commands))

    # print(f"{results}")

    create_confusion_matrix(results, code_commands, test_comands)
    

