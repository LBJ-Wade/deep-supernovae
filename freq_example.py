from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
from model import SequenceClassifier

max_num_points = 125


# Special Data Iterator - from https://github.com/dannyneil/public_plstm
# ----------------------------------------------------
class SinWaveIterator:
    """
    """
    def flow(self, sample_regularly, sample_res, min_period=1, max_period=100, min_spec_period=5, max_spec_period=6,
             batch_size=32, num_examples=10000, min_duration=15, max_duration=125,
             min_num_points=15, max_num_points=max_num_points):
        # Calculate constants
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        min_log_period, max_log_period = np.log(min_period), np.log(max_period)
        b = 0
        while b < num_batches:
            # Choose curve and sampling parameters
            num_points = np.random.uniform(low=min_num_points, high=max_num_points, size=(batch_size))
            duration = np.random.uniform(low=min_duration, high=max_duration, size=batch_size)
            start = np.random.uniform(low=0, high=max_duration-duration, size=batch_size)
            periods = np.exp(np.random.uniform(low=min_log_period, high=max_log_period, size=(batch_size)))
            shifts = np.random.uniform(low=0, high=duration, size=(batch_size))

            # Ensure always at least half is special class
            periods[:len(periods)/2] = np.random.uniform(low=min_spec_period, high=max_spec_period, size=len(periods)/2)

            # Define arrays of data to fill in
            all_t = []
            all_masks = []
            all_wavs = []
            for idx in range(batch_size):
                if sample_regularly:
                    # Synchronous condition
                    t = np.arange(start[idx], start[idx]+duration[idx], step=sample_res)
                else:
                    # Asynchronous condition
                    t = np.sort(np.random.random(int(num_points[idx])))*duration[idx]+start[idx]
                wavs = np.sin(2*np.pi/periods[idx]*t-shifts[idx])
                mask = np.ones(wavs.shape)
                all_t.append(t)
                all_masks.append(mask)
                all_wavs.append(wavs)

            # Now pack all the data down into masked matrices
            lengths = np.array([len(item) for item in all_masks])
            #max_length = np.max(lengths)
            max_length = max_num_points
            bXt = np.zeros((batch_size, max_length, 1))
            bX = np.zeros((batch_size, max_length, 1))
            # Modifiy to zero last part of sequence
            for idx in range(batch_size):
                bX[idx, 0:lengths[idx], 0] = all_wavs[idx]
                bXt[idx, 0:lengths[idx]:, 0] = all_t[idx]

            bY = np.zeros((batch_size, 2))
            bY[(periods >= min_spec_period)*(periods <= max_spec_period), 0] = 1
            bY[periods < min_spec_period, 1] = 1
            bY[periods > max_spec_period, 1] = 1

            # Yield data - modified to include length of sequences in batch
            yield bXt.astype('float32'), bX.astype('float32'), lengths.astype('int'), bY.astype('int32')
            b += 1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', default='PLSTM', help='LSTM or PLSTM')
    parser.add_argument('--hidden', nargs='+', default=[110], help='Hidden size')
    parser.add_argument('--train', default=5000,   type=int, help='Training size')
    parser.add_argument('--batch', default=32,   type=int, help='Batch size')
    parser.add_argument('--epochs', default=100,  type=int, help='Number of epochs to train for')
    parser.add_argument('--sample_regularly', default=0, type=int, help='Whether to sameple regularly or irregularly')
    parser.add_argument('--sample_res',       default=0.5, type=float, help='Resolution at which to sample')
    parser.add_argument('--log_file', default='log.tsv')
    args = parser.parse_args()
    num_hidden = [int(n) for n in args.hidden]
    batch_size = args.batch
    train_size = args.train
    num_epochs = args.epochs
    cell_type = args.cell

    x = tf.placeholder(tf.float32, (batch_size, max_num_points, 1))
    t = tf.placeholder(tf.float32, (batch_size, max_num_points, 1))

    if cell_type == 'PLSTM':
        inputs = (t, x)
    else:
        inputs = x

    sequence_length = tf.placeholder(tf.int32, [batch_size])
    labels = tf.placeholder(tf.float32, (batch_size, 2))

    model = SequenceClassifier(inputs, labels, num_hidden, cell_type, sequence_length=sequence_length)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    d = SinWaveIterator()

    for epoch in range(num_epochs):
        for data in d.flow(batch_size=batch_size, num_examples=train_size, sample_regularly=args.sample_regularly, sample_res=args.sample_res):
            loss, acc, _ = sess.run([model.loss, model.accuracy, model.optimize], feed_dict={t: data[0], x: data[1], sequence_length: data[2], labels: data[3]})
        test_loss = 0
        test_acc = 0
        test_batches = 0
        for data in d.flow(batch_size=batch_size, num_examples=train_size, sample_regularly=args.sample_regularly, sample_res=args.sample_res):
            loss, acc = sess.run([model.loss, model.accuracy], feed_dict={t: data[0], x: data[1], sequence_length: data[2], labels: data[3]})
            test_loss += loss
            test_acc += acc
            test_batches += 1
        test_loss = test_loss / test_batches
        test_acc = test_acc / test_batches
        print('epoch = {0} | test loss = {1:.3f} | test acc = {2:.3f}'.format(str(epoch).zfill(6), test_loss, test_acc))

if __name__ == '__main__':
    main()
