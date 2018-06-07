import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../docs/MNIST_data/", one_hot=True)


#[inputNodes,hl1Nodes,hl2Nodes,hl3Nodes,outputNodes]
# This array is used for a forloop and actully represent the number of nodes in each layer
n_nodes = [784, 500, 500, 500, 10]

batch_size = 100

# height x width
x = tf.placeholder("float",[None, n_nodes[0]])
y = tf.placeholder("float")

def neural_network_model(x):
    #(input * weight) + biases

    # hidden_1_layer = {
    # 'weight': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    # 'biases': tf.Variable(tf.random_normal(n_nodes_hl1))
    # }
    #
    # hidden_2_layer = {
    # 'weight': tf.Variable(tf.random_normal([n_node_hl1, n_nodes_hl2])),
    # 'biases': tf.Variable(tf.random_normal(n_nodes_hl2))
    # }
    #
    # hidden_3_layer = {
    # 'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    # 'biases': tf.Variable(tf.random_normal(n_nodes_hl3))
    # }
    # output_layer = {
    # 'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    # 'biases': tf.Variable(tf.random_normal([n_classes]))
    # }

    hidden_layer = []
    #This forloop replace the sentdex code comented above
    for i in range(0,len(n_nodes)-1):
        i_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes[i], n_nodes[i+1]])),
        'biases': tf.Variable(tf.random_normal([n_nodes[i+1]]))
        }
        hidden_layer.append(i_layer)


    # l1 = tf.add(tf.matmul(data,hidden_1_layer['weigths']) + hidden_1_layer['biases'])
    # l1 = tf.nn.relu(l1)
    #
    # l2 = tf.add(tf.matmul(l1,hidden_2_layer['weigths']) + hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    #
    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weigths']) + hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    #
    # output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    # return output
    input_to_layer = []
    input_to_layer.append(x)
    #This forloop replace the sentdex code comented above
    for i in range(len(n_nodes)-1):
        input_to_layer.append( tf.add( tf.matmul( input_to_layer[i], hidden_layer[i]['weights'] ), hidden_layer[i]['biases'] ) )
        if i < ( len(n_nodes)-2 ):
            input_to_layer[i+1] = tf.nn.relu(input_to_layer[i+1])

    return input_to_layer[-1]

def train_neural_network(model, data):
    prediction = model
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run( tf.initialize_all_variables() )

        #Training data
        for epoch in range(0,hm_epochs):
            epoch_loss = 0
            for _ in range(int(data.train.num_examples/batch_size)):
                epoch_x, epoch_y = data.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss',epoch_loss)

        #Prediction accuracy
        correct = tf.equal( tf.argmax(prediction,1), tf.argmax(y,1) )
        accuracy = tf.reduce_mean(tf.cast(correct,'float') )
        print('Accuracy',accuracy.eval({x:data.test.images, y:data.test.labels}))

our_model = neural_network_model(x)
train_neural_network(model=our_model, data=mnist)
