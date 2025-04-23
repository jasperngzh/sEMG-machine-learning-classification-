import numpy as np
import scipy.io
import numpy
from scipy.io import loadmat
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
     
# load dataset
mat_dataset = scipy.io.loadmat('S1_A1_E1.mat')

EMG_data = mat_dataset['emg'][0:8316,:8]
stimulus_data = mat_dataset['restimulus'][0:8316,:8]# find the break time and split accordingly
num_rows , num_columns = EMG_data.shape
plt.figure(figsize=(10, 6 * num_columns))

# variance = []
# EMG_extract = EMG_data[stimulus_data.flatten() == 1, :]
# total_EMG = len(EMG_extract)
# step_size = total_EMG//10
# EMG_extract[::step_size]
# EMG_extract = np.array(EMG_extract)
# variance = np.var(EMG_extract)

# calculate RMS
RMS_val = []
# Plot emg data in according the row spacing where data can be extracted
def calculated_RMS():
   for i in range(0,num_rows,1200):
        extractable_data = EMG_data[i:i+1200,:5]
        calculated_RMS = np.sqrt(np.mean(extractable_data**2))
        RMS_val.append(calculated_RMS)
for i in range (num_columns):
    Data_row = EMG_data[:,i]

# Plot the EMG signal
for i in range(num_columns):
    channel_sig = EMG_data[:,i]
    plt.subplot(num_columns ,1, i+1)
    time = range(len(EMG_data))
    plt.plot(time,channel_sig)
    plt.plot(time , stimulus_data, alpha= 0.3,)
    plt.xlabel('Time')
    plt.ylabel('EMG Amplitude')
    plt.title('EMG Signal')
    plt.ylim(0,0.9)
plt.savefig('indiv semg first 10 ')
plt.show()

# knn
x = EMG_data
y = stimulus_data
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
plt.figure(figsize=(8, 6))
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='o', edgecolors='k', label='Actual Labels')
plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions, marker='x', s=100, linewidth=1, edgecolors='k', label='Predicted Labels')
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
# plt.savefig('confussionm')
plt.show()
# plt.xlabel('EMG')
# plt.ylabel('RESTIMULUS')
# plt.title('KNN Classifier Results')
# plt.legend()
# plt.savefig('fyp knn', dpi=300)
# plt.show()
#
# # print('calculated_RMS=',RMS_val)
