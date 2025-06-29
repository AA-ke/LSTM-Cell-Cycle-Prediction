# LSTM-Cell-Cycle-Prediction
This project uses LSTM (Long Short-Term Memory model) to model the division cycle of Escherichia coli and predict its division moments. First, a Mother Machine is constructed using a multi-channel microfluidic chip, and continuously dividing cells are photographed under bright field and fluorescence modes to create a dataset from different frames' images. The opencv vision library is used for channel segmentation, cell edge detection, and visual feature extraction, storing them as a dataset. LSTM is utilized to model the dataset, predicting the division state of the current frame's cells based on the visual information of cells from different frames, achieving an accuracy of approximately 70% with a prediction error of ±10 frames for future division moments.
![image](https://github.com/user-attachments/assets/dabe8e87-4a2f-40ca-8abd-8456256a7519)
Structure:
~Encoder: LSTM 
~Decoder: MLP
Process:
1.Preprocessing the image data
2.Extracting  features and label the data
3.Training the model 
4.Testing and adjusting
![image](https://github.com/user-attachments/assets/b8cd9da6-8d80-4036-bb4b-8c193be44609)
Brightfield images are too noisy, so use fluorescence field images:
1. Use OpenCV to load and display images in 16-bit format
2. Gussian Blur:reduce the noise
3. Morphological manipulation-swelling, corrosion:Eliminate small noises and subtle boundary interference
4. Extract boundaries information:Convenient for feature extraction
Only the visual features of cells on the top mother machine are extracted
1.Divide and number the channels
2. Cells at the top of each channel were detected and their features extracted: Length, total and mean fluorescence intensity, area
3. Label data：The data of each frame is recorded, and the split moment is recorded with a 30% reduction in length as a split marker. Save all data as a .csv file for later training.
![image](https://github.com/user-attachments/assets/88c70a9a-1d1a-4ef3-825b-16558af20824)
![image](https://github.com/user-attachments/assets/96c1afeb-db9f-47f0-b14e-7d2efd9efdfd)
Extracting  features and label the data
Model: BI-LSTM+MLP
LSTM is used to deal with time series dependence of time series data, while MLP is used to deal with nonlinear relationships of features. 
Bidirectional LSTMs process both future and past information.
1. Format the data into a time series that is suitable for LSTM input.
2. LSTM is used to extract time series features, and MLP is used to predict splitting time.
3. Use the training set for training and the test set for evaluation.
![image](https://github.com/user-attachments/assets/a63f4de6-71fc-498e-b64c-47825ead87b5)
![image](https://github.com/user-attachments/assets/721c06f2-feca-4942-9562-64d30914adf4)
![image](https://github.com/user-attachments/assets/de9efbf1-8b70-464b-be35-2298b787f3e4)
Parameter setting:
30 epochs, batchsize 64
2 layers of LSTMs, 3 layers of MLP
Learning rate=0.0005
LSTM Window size=20
Prediction Error <=10
The method is feasible for predicting the splitting trend, but the accuracy of the splitting time needs to be improved.



