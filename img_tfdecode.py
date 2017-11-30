import tensorflow as tf
import numpy as np
#import os
import matplotlib.pyplot as plt
#import skimage.io as io

def read_and_decode(tfrecords_file, batch_size):
	'''read and decode tfrecord file, generate (image, label) batches
	Args:
		tfrecords_file: the directory of tfrecord file
		batch_size: number of images in each batch
	Returns:
		image: 4D tensor - [batch_size, width, height, channel]
		label: 1D tensor - [batch_size]
	'''
	# make an input queue from the tfrecord file
	filename_queue = tf.train.string_input_producer([tfrecords_file])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	img_features = tf.parse_single_example(
		serialized_example,
		features={
			'label': tf.FixedLenFeature([], tf.int64),
			'image_raw': tf.FixedLenFeature([], tf.string),
		})
	image = tf.decode_raw(img_features['image_raw'], tf.uint8)

	##########################################################
	# you can put data augmentation here, I didn't use it
	##########################################################
	# all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.

	image = tf.reshape(image, [40, 40, 3])
	label = tf.cast(img_features['label'], tf.int32)
	image_batch, label_batch = tf.train.batch([image, label],
											  batch_size=batch_size,
											  num_threads=64,
											  capacity=2000)
	return image_batch, tf.reshape(label_batch, [batch_size])

# %% TO test train.tfrecord file

def plot_images(images, labels):
	'''plot one batch size
	'''
	for i in np.arange(0, BATCH_SIZE):
		plt.subplot(5, 5, i + 1)
		plt.axis('off')
		# plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)
		if labels[i] == 0:
			plt.title(u'Cell_0', fontsize=12)
		else:
			plt.title(u'noneCell_1', fontsize=12)

		plt.subplots_adjust(top=1.5)
		plt.imshow(images[i])
	plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BATCH_SIZE = 25
tfrecords_file = './/celltfrecord//cell_2_test.tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
with tf.Session()  as sess:
	i = 0
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	try:
		while not coord.should_stop() and i < 1:
			# just plot one batch size
			image, label = sess.run([image_batch, label_batch])
			plot_images(image, label)
			i += 1
	except tf.errors.OutOfRangeError:
		print('done!')
	finally:
		coord.request_stop()
	coord.join(threads)
