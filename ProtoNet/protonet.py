from __future__ import print_function
import tensorflow as tf
from sklearn.neighbors import BallTree
import numpy as np
import json

class ProtoNet:
    @staticmethod
    def euclidean_distance(a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = tf.shape(a)[0], tf.shape(a)[1]
        M = tf.shape(b)[0]
        a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
        b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(a - b), axis=2)

    def __init__(self):
        def conv_block(inputs, out_channels, name='conv'):
            with tf.variable_scope(name):
                conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
                conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
                conv = tf.nn.relu(conv)
                conv = tf.contrib.layers.max_pool2d(conv, 2)
                return conv

        def encoder(x, h_dim, z_dim, reuse=False):
            with tf.variable_scope('encoder', reuse=reuse):
                net = conv_block(x, h_dim, name='conv_1')
                net = conv_block(net, h_dim, name='conv_2')
                net = conv_block(net, h_dim, name='conv_3')
                net = conv_block(net, z_dim, name='conv_4')
                net = tf.contrib.layers.flatten(net)
                return net

        im_width, im_height, channels = 84, 84, 3
        h_dim = 120
        z_dim = 120

        x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
        # q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
        # x_shape = tf.shape(x)
        # q_shape = tf.shape(q)
        # num_classes, num_support = x_shape[0], x_shape[1]
        # num_queries = q_shape[1]
        # y = tf.placeholder(tf.int64, [None, None])
        # y_one_hot = tf.one_hot(y, depth=num_classes)
        # emb_in = encoder(tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim, z_dim)
        emb = encoder(tf.reshape(x, [-1, im_height, im_width, channels]), h_dim, z_dim)
        # emb_dim = tf.shape(emb_in)[-1]
        #
        # emb_x = tf.reduce_mean(tf.reshape(emb_in, [num_classes, num_support, emb_dim]), axis=1)
        # emb_q = encoder(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim, z_dim, reuse=True)
        # dists = ProtoNet.euclidean_distance(emb_q, emb_x)
        # log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
        # ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
        # acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))
        #
        # train_op = tf.train.AdamOptimizer().minimize(ce_loss)

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver.restore(sess, 'ProtoNet/model/model_132_200.ckpt')

        # loading sample dots
        self.dots = np.load('ProtoNet/model/emb_x_200.npy').reshape((-1,3000))
        self.balltree = BallTree(self.dots, leaf_size=40)


        json_dir = 'ProtoNet/label2name.json'
        with open(json_dir) as f:
            self.label2name = json.load(f)

        self.sess = sess
        self.x = x
        self.emb = emb
        # test_img = np.zeros([1,1,84,84,3])
        # emx = sess.run(emb,feed_dict={x:test_img})
        # print(emx)
        self.persistence = False

    def encode(self, img):
        img = [[img]]
        emx = self.sess.run(self.emb, feed_dict={self.x: img})
        return emx[0]

    def find(self, img, k=1):
        encode = self.encode(img)
        dis, ind = self.balltree.query(encode.reshape((1, -1)), k)
        name = []
        for i in ind[0]:
            name.append(self.label2name[i])
        return name, dis[0].tolist()

    def add_sample(self, image_list, name):
        emx = self.sess.run(self.emb, feed_dict={self.x: [image_list]})
        cx = emx.mean(axis=0)
        self.dots = np.row_stack((self.dots, cx))
        self.label2name.append(name)
        self.balltree = BallTree(self.dots, leaf_size=40)
        if self.persistence:
            np.save(self.dot_dir, self.dots)
            with open(self.json_dir, 'w') as f:
                json.dump(self.label2name, f)
