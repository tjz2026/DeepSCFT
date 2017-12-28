import tensorflow as tf
import numpy as np
import os



train_data = np.random.rand(200,2)
train_label = np.random.rand(200,1)
train_label[:,0] = np.cos(train_data[:,0] + train_data[:,1]**2)**2
print "train_label[0]",train_label[0,0]
learning_rate = 0.02
epochs = 100000
display_step=200


input_data = tf.placeholder(tf.float32, [None, 2], name='input')

#output_data = tf.placeholder(tf.float32, [None, 1], name='Output')
output_data = tf.placeholder(tf.float32, [None,1], name='Output')

layer1_num = 10

weights = {
    'weight1': tf.Variable(tf.random_normal([2, layer1_num])),
    'weight2': tf.Variable(tf.random_normal([layer1_num, 1])),
    #'weight2': tf.Variable(tf.random_normal([32, 64])),
    #'weight3': tf.Variable(tf.random_normal([64, 1])),

}

bias = {
    'bias1': tf.Variable(tf.random_normal([layer1_num])),
    'bias2': tf.Variable(tf.random_normal([1])),
    #'bias2': tf.Variable(tf.random_normal([64])),
    #'bias3': tf.Variable(tf.random_normal([1])),

}


def   model(x1, weights, bias):
        #layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['weight1']), bias['bias1']))
        #layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
        #layer_out = tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3'])

        #layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['weight1']), bias['bias1']))
        #layer1 = tf.nn.relu(tf.add(tf.matmul(x1, weights['weight1']), bias['bias1']))
        layer1 = tf.nn.tanh(tf.add(tf.matmul(x1, weights['weight1']), bias['bias1']))
        layer_out = tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2'])
        return layer_out


pred = model(input_data, weights, bias)

# mean square error
cost = tf.reduce_mean(tf.square(output_data - pred))
# cross entropy
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_data, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.01).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    for each_epoch in range(epochs):
        loss, data, _ = sess.run([cost, pred, optimizer],
                                 feed_dict={ input_data: train_data,
                                           output_data: train_label})

        if each_epoch % display_step == 0:
            print("Iter =", each_epoch, ", Loss= ", loss)
            res = sess.run(pred,feed_dict={input_data:train_data[0].reshape(1,2)})
            print("predicted result:%f"%(res))
            print("acctual result:%f"%(train_label[0,0]))

    model_dir = "regression"
    model_name = "function"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    saver.save(sess, os.path.join(model_dir, model_name))
    print("model saved sucessfully")
