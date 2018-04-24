# Importing the libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import os 

# Load and prepare the data 
from pandas import read_csv 
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
%pylab

current_dir = os.getcwd()
dataset_path = os.path.join(os.getcwd(), os.pardir, 'data', 'diamond_prices.csv')
diamonds = read_csv(dataset_path)
diamonds =pd.get_dummies(diamonds)

TARGET = 'price'
X_data = diamonds.iloc[:, 1:].values 
y_data = diamonds[TARGET].values 

## Separating into training and testing 
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, 
        train_size=0.95, random_state=12)

# now, let's add some Gaussian noise.
mu, sigma = 326, 0.1  # about 20% of the min y value 
noise = np.random.normal(mu, sigma, y_train.shape)

y_train = y_train + noise

#######################################   Build the input pipeline 
BATCH_SIZE = 128 

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(BATCH_SIZE)
iterator = train_dataset.make_initializable_iterator()
next_element = iterator.get_next()

#############################   Build a function for delploying Dense Neural Network
n_hidden1 = 128
n_hidden2 = 64
n_hidden3 = 32
n_hidden4 = 16
n_outputs = 1

def DNN(inputs):
    wt_init = tf.contrib.layers.xavier_initializer()
    hidden1 = tf.layers.dense(inputs, units=n_hidden1, activation=tf.nn.relu,
                              kernel_initializer=wt_init)
    hidden2 = tf.layers.dense(hidden1, units=n_hidden2, activation=tf.nn.relu, 
                              kernel_initializer=wt_init)
    hidden3 = tf.layers.dense(hidden2, units=n_hidden3, activation=tf.nn.relu, 
                              kernel_initializer=wt_init)
    hidden4 = tf.layers.dense(hidden3, units=n_hidden4, activation=tf.nn.relu, 
                              kernel_initializer=wt_init)

    y_pred = tf.layers.dense(hidden4, units=n_outputs, activation=None)
    return tf.squeeze(y_pred)

################## Create placeholders to pass values for training and testing 
n_inputs = X_data.shape[1]
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
y = tf.placeholder(tf.float32, name = 'target')

###########################################  Define the loss function 
y_pred = DNN(X)
mse = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
#TensorBoard visualization 
rmse = tf.sqrt(mse)
tf.summary.scalar('RMSE', rmse)
# We will also visualize the error rate distribution 
errors = y_pred - y 
tf.summary.histogram('Errors', errors)
# Summary writer object for TensorBoard
summary_values = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join(current_dir, 'reg_logs', 'train'))
val_writer = tf.summary.FileWriter(os.path.join(current_dir, 'reg_logs', 'validation'))

################################## Define the optimizer and training operation 
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(mse)

#####################################  Function to run training operation 
def train_model(epoch_number): 
    """ This function perfoms the training operation for one epoch, 
    record metrics for TensorBoard visualization"""
    print(epoch_number, end =',')
    iterator.initializer.run()
    while True:
        try: 
            X_values, y_values = sess.run(next_element)
            sess.run(training_op, feed_dict={X: X_values, y:y_values})
        except tf.errors.OutOfRangeError:
            break
    ## Training RMSE (computed only for the last batch of the epoch)
    summaries = sess.run(summary_values, feed_dict={X:X_values, y:y_values})
    train_writer.add_summary(summaries, epoch_number)
    # Validation RMSE (for all the validation set)
    summaries = sess.run(summary_values, feed_dict={X:X_val, y:y_val})
    val_writer.add_summary(summaries, epoch_number)

###################################### Run the computation graph 
N_EPOCHS = 300
with tf.Session() as sess: 
    tf.global_variables_initializer().run() # initialize global variables
    train_writer.add_graph(sess.graph)

    # Training loop
    print( "Epoch: ")
    for epoch in range(1, N_EPOCHS+1):
        train_model(epoch)
    print("\nDone Training!") 

    # Close the file writers
    train_writer.close()
    val_writer.close()

    # Getting the predictions for the validation dataset
    predictions = sess.run(y_pred, feed_dict={X: X_val})


########################### Visualization/analyze the results ##############
x = np.linspace(0,18000)
fig, ax = plt.subplots(figsize=(8,5))
plt.scatter(predictions, y_val, alpha=0.5)
ax.plot(x, x, 'r')
ax.set_xlabel('Predicted prices')
ax.set_ylabel('Observed prices')
ax.set_title('Predictions vs. Observed Values in the validation set')
fig.show();

fig, ax = plt.subplots(figsize=(8,5))
ax.hist(abs(y_val-predictions), bins=100, edgecolor='black')
ax.set_xlim(0, 4e3)
ax.set_xlabel('Observations')
ax.set_ylabel('Prediction Error')
ax.set_title('Error Across Observations')
fig.show();

print("The average MAE is " + str(np.mean(abs(y_val-predictions))))
print("The MAE share of average price is " + str(np.mean(abs(y_val-predictions))/np.mean(y_val) * 100))


