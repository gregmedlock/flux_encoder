import cobra
from cobra.test import create_test_model
from cobra.flux_analysis import sample
import tensorflow as tf
import numpy as np

# load the test model
model = create_test_model("textbook")

# generate 10,000 samples with oxygen available
num_samples = 10000
model.optimize()
print('sampling aerobic e. coli')
s_aerobic = sample(model, num_samples, processes=8)
# remove oxygen exchange as a predictor
s_aerobic = s_aerobic.drop('EX_o2_e',axis=1)
aerobic_input = s_aerobic.as_matrix()

# close oxygen uptake, then simulate again
model.reactions.EX_o2_e.lower_bound = 0
print('sampling anaerobic e. coli')
s_anaerobic = sample(model, num_samples, processes=8)
# remove oxygen exchange from this dataframe as well
s_anaerobic = s_anaerobic.drop('EX_o2_e',axis=1)
anaerobic_input = s_anaerobic.as_matrix()

# combine samples and create labels
all_samples = np.concatenate((s_aerobic,s_anaerobic),axis=0).astype(np.float32)
# labels should be one-hot
aerobic_labels = np.concatenate((np.zeros(len(s_aerobic.index))+1,np.zeros(len(s_aerobic.index))),axis=0)
anaerobic_labels = np.concatenate((np.zeros(len(s_aerobic.index)),np.zeros(len(s_aerobic.index))+1),axis=0)
all_labels = np.vstack((aerobic_labels,anaerobic_labels)).T

# convert to a tensor
#samples_tensor = tf.constant(all_samples, dtype = tf.float32)
#label_tensor = tf.constant(all_labels, dtype=tf.float32)

# convert the sampling data to a tensorflow DataSet


print('beginning tensorflow session-----------')
# declare the tensorflow session to start building the computation graph
sess = tf.InteractiveSession()

# build placeholders
x = tf.placeholder(all_samples.dtype, shape=all_samples.shape)
y_ = tf.placeholder(all_labels.dtype, shape=all_labels.shape)
dataset = tf.contrib.data.Dataset.from_tensor_slices((x,y_))
# dataset = dataset.map()
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
num_epochs = 10
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_initializable_iterator()
next_example, next_label = iterator.get_next()

# build the weights and bias vectors
W = tf.Variable(tf.zeros([len(s_anaerobic.columns),2]))
b = tf.Variable(tf.zeros([2]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(next_example,W) + b
# declare a loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=next_label, logits=y))
# define the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess.run(iterator.initializer, feed_dict={x:all_samples,y_:all_labels})
for i in range(10):
    #v = sess.run(next_example, next_label)
    v = sess.run(train_step)
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(cross_entropy)

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(accuracy.eval(feed_dict={x:all_samples,y_:all_labels}))


# start the session
#sess.run(tf.global_variables_initializer())
# implement the model using the placeholders and variables
#y = tf.matmul(x,W) + b




#sess.run(iterator.initializer, feed_dict={x:all_samples,y_:all_labels}

#with tf.train.MonitoredTrainingSession() as sess:
#    while not sess.should_stop():
#        sess.run(train_step, feed_dict)



#num_epochs = 10
#for i in range(num_epochs):
#    sess.run(iterator.initializer)
#    while True:
#        try:
#            sess.run(next_element)
#        except tf.errors.OutOfRangeError:
#            break
