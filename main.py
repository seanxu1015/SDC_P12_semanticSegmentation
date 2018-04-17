import os
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    layer3_conv1_1 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, 1, 
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer4_conv1_1 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, 1,
        strides=(1, 1),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer7_conv1_1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, 1,
        strides=(1,1),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer7_upsample = tf.layers.conv2d_transpose(
        layer7_conv1_1, num_classes, 4,
        strides=(2,2),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer4_skip = tf.add(layer7_upsample, layer4_conv1_1)
    layer4_upsample = tf.layers.conv2d_transpose(
        layer4_skip, num_classes, 4,
        strides=(2, 2),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer3_skip = tf.add(layer4_upsample, layer3_conv1_1)
    layer3_upsample = tf.layers.conv2d_transpose(
        layer3_skip, num_classes, 16,
        strides=(8, 8),
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return layer3_upsample

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, [-1, num_classes], name='logits')
    labels = tf.reshape(correct_label, [-1, num_classes])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, 
             train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,
             last_layer=None, num_classes=2):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())
    interval = 5
    for epoch in range(epochs):
        step = 1
        for input_images, labels in get_batches_fn(batch_size):
            feed_dict = {input_image: input_images, 
                         correct_label: labels,
                         keep_prob: 0.6, 
                         learning_rate: 1e-4}
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict=feed_dict)
            if step % interval == 0 and last_layer is not None:
               
                iou_feed_dict = {input_image: input_images, 
                                 correct_label: labels, 
                                 keep_prob: 1.0}
                prediction = tf.argmax(last_layer, axis=3)
                ground_truth = tf.argmax(correct_label, axis=3)
                iou, iou_op = tf.metrics.mean_iou(
                    ground_truth, prediction, num_classes)                                
                sess.run(tf.local_variables_initializer())
                sess.run(iou_op, feed_dict=iou_feed_dict)
                mean_iou = sess.run(iou)
                o_str = "Epoch %3d, Step %3d, Loss: %.4f, MeanIOU: %.4f"
                print(o_str % (epoch, step, loss, mean_iou))
            step += 1
            
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    epochs = 20
    batch_size = 8
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(
            sess, vgg_path)
        last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        correct_label = tf.placeholder(
            tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(
            last_layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, 
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate, 
                 last_layer, num_classes)

        # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(
            runs_dir, data_dir, sess, image_shape, 
            logits, keep_prob, image_input)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()


