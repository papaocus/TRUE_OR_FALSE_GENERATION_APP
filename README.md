# TRUE_OR_FALSE_GENERATION_APP

True or False App
This is a machine learning project developed using TensorFlow and Streamlit. The purpose of the project is to classify whether a given question is true or false based on a trained model.

Code Explanation:
The necessary libraries are imported, including Streamlit, TensorFlow, and other relevant modules for data preprocessing and model building.

The dataset is loaded from a CSV file using pandas. The file path is specified as "D:\A.I Planet assigment\my_dataframe.csv".

The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The feature columns are assigned to X (transformed_text) and the target column to y (is_correct). The split is done with a test size of 20% and a random state of 42 for reproducibility.

Tokenization is applied to convert the text data into numerical sequences. The Tokenizer class from Keras is used to fit on the training text data (X_train) and then transform both the training and testing sets (X_train and X_test) into sequences.

The sequences are padded using pad_sequences to ensure they have a consistent length. The maxlen parameter is set to 20, meaning sequences longer than 20 tokens will be truncated, and shorter sequences will be padded with zeros.

The model architecture is defined using a sequential model from Keras. It consists of an embedding layer, two bidirectional LSTM layers, dense layers, and a final sigmoid layer for binary classification.

The learning rate schedule is defined using ExponentialDecay from TensorFlow. It starts with an initial learning rate of 0.001 and decays exponentially over 10000 steps with a decay rate of 0.9. The Adam optimizer is used with the learning rate schedule.

The model is compiled with binary cross-entropy loss and accuracy as the evaluation metric.

The model is trained on the training set (X_train and y_train) for 10 epochs. The validation data is provided as the testing set (X_test and y_test).

The model is evaluated on the testing set, and the test loss and accuracy are printed.

The Streamlit app is defined with the title "True or False app".

The user can input a question using the text_input function from Streamlit.

If the user clicks the "Submit" button, the entered question is transformed into a numerical sequence using the tokenizer and padded to match the desired sequence length.

The model predicts the probability of the question being true (greater than 0.5) or false (less than or equal to 0.5).

The prediction is displayed as "True" or "False" using st.write from Streamlit.

Usage:
Make sure you have all the necessary libraries installed, including Streamlit, TensorFlow, and scikit-learn.

Prepare your dataset in CSV format and update the file path in the code (df = pd.read_csv("D:\A.I Planet assigment\my_dataframe.csv")) to point to your dataset file.

Run the code and wait for the model to train.

Once the model is trained, a Streamlit app will be launched.

Enter a question in the input field and click "Submit" to see the predicted classification ("True" or "False") for the question.

Note: The accuracy of the model and the effectiveness of the app depend on the quality and representativeness of the training data. It is important to ensure that the dataset used for training is diverse and balanced to achieve
