from __future__ import print_function
import argparse
import time
import tensorflow as tf
from model import SequenceClassifier
from dataset import DataLoader
from dataset import sn1a_keys, type_keys, subtype_keys
from dataset import sn1a_classes, type_classes, subtype_classes
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as cm
import numpy as np
from utils import plot_confusion_matrix


def get_metrics(model, z, min_z=0.0, max_z=100.0):
    indices = tf.where(tf.logical_and(tf.greater(z, min_z), tf.less_equal(z, max_z)))
    actual = tf.gather(model.actual, indices)
    predictions = tf.gather(model.predictions, indices)
    labels = tf.gather(model.labels, indices)
    scores = tf.gather(model.scores, indices)
    accuracy, accuracy_op = tf.metrics.accuracy(actual, predictions)
    TP, TP_op = tf.metrics.true_positives(actual, predictions)
    TN, TN_op = tf.metrics.true_negatives(actual, predictions)
    FP, FP_op = tf.metrics.false_positives(actual, predictions)
    FN, FN_op = tf.metrics.false_negatives(actual, predictions)
    # Precision (purity) is TP/(TP+FP)
    precision, precision_op = tf.metrics.precision(actual, predictions)
    # Recall (completeness, efficiency) is TP/(TP+FN)
    recall, recall_op = tf.metrics.recall(actual, predictions)
    F1, F1_op = 1.0/(tf.cast(TP, tf.float32) + tf.cast(FN, tf.float32))*tf.cast(TP, tf.float32)**2.0/(tf.cast(TP, tf.float32) + 3.0*tf.cast(FP, tf.float32)), tf.group(TP_op, TN_op, FP_op, FN_op)
    AUC, AUC_op = tf.metrics.auc(labels, scores, num_thresholds=200)
    metrics = [accuracy, TP, TN, FP, FN, precision, recall, F1, AUC]
    metric_update_ops = [accuracy_op, TP_op, TN_op, FP_op, FN_op, precision_op, recall_op, F1_op, AUC_op]
    return metrics, metric_update_ops


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', default='PLSTM', help='LSTM or PLSTM')
    parser.add_argument('--hidden', nargs='+', default=[16], help='Hidden size')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train for')
    parser.add_argument('--test_fraction', default=0.5, type=float, help='Test fraction')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')
    parser.add_argument('--nz_bins', default=7, type=int)
    parser.add_argument('--dz_bins', default=0.2, type=float)
    parser.add_argument('-augment', action='store_true', help='Use augmentation')
    parser.add_argument('-nohostz', action='store_true', help='No host z data')
    parser.add_argument('--challenge', default='sn1a', help='sn1a, 123 or full')
    parser.add_argument('--dataset', default='simgen', help='simgen or des')
    parser.add_argument('-nonrep', action='store_true', help='Use non representative data')
    parser.add_argument('--addrep', default=0, type=int, help='Additional representative samples')
    parser.add_argument('--log', default='./logs/')
    parser.add_argument('-nosummary', action='store_true', help='Do not write summary log')
    args = parser.parse_args()
    num_hidden = [int(n) for n in args.hidden]
    dropout = args.dropout
    cell_type = args.cell

    if args.challenge == 'sn1a':
        keys = sn1a_keys
        class_labels = sn1a_classes
    elif args.challenge == '123':
        keys = type_keys
        class_labels = type_classes
    elif args.challenge == 'full':
        keys = subtype_keys
        class_labels = subtype_classes
    else:
        raise ValueError('Challenge {} not implemented.'.format(args.challenge))

    if args.dataset == 'simgen':
        file_root = 'data/SIMGEN_PUBLIC_DES_PROCESS/'
    elif args.dataset == 'des':
        file_root = 'data/DES_BLIND+HOSTZ_PROCESS/'
    else:
        raise ValueError('Dataset {} not implemented.'.format(args.dataset))

    loader = DataLoader(file_root=file_root, test_fraction=args.test_fraction, use_hostz=not args.nohostz, keys=keys,
                        representative=not args.nonrep, additional_representative=args.addrep)
    train_dataset, test_dataset = loader.get_dataset(batch_size=args.batch, augment=args.augment)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    t, flux_values, _, _, _, additional, sequence_length, labels, z, _ = iterator.get_next()
    x = tf.concat([flux_values, additional], 2)

    if cell_type == 'PLSTM':
        inputs = (t, x)
    else:
        inputs = x

    keep_prob = tf.placeholder(tf.float32)
    model = SequenceClassifier(inputs, labels, num_hidden, cell_type, sequence_length=sequence_length, keep_prob=keep_prob)

    metrics_z = []
    with tf.name_scope('metrics'):
        loss, loss_op = tf.metrics.mean(model.loss)
        confusion_matrix = tf.confusion_matrix(model.actual, model.predictions, num_classes=loader.num_classes)
        (accuracy, TP, TN, FP, FN, precision, recall, F1, AUC), (accuracy_op, TP_op, TN_op, FP_op, FN_op, precision_op, recall_op, F1_op, AUC_op) = get_metrics(model, z, min_z=0, max_z=100.0)
        for i in range(0, args.nz_bins):
            metrics_z.append(get_metrics(model, z, min_z=args.dz_bins*i, max_z=args.dz_bins*(i+1)))

    metric_update_ops = [loss_op]
    metric_update_ops += [accuracy_op, TP_op, TN_op, FP_op, FN_op, precision_op, recall_op, F1_op, AUC_op]
    for i in range(0, args.nz_bins):
        metric_update_ops += metrics_z[i][1]

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    log_file = cell_type
    log_file += '-' + '.'.join(str(n) for n in num_hidden)
    log_file += '-' + str(args.test_fraction)
    log_file += '-' + args.dataset
    log_file += '-' + str(args.batch)
    log_file += '-' + str(args.augment)
    log_file += '-' + args.challenge
    log_file += '-' + str(not args.nohostz)
    log_file += '-' + str(not args.nonrep)
    log_file += '-' + str(args.addrep)
    log_file += '-' + str(int(time.time()*1000))

    if not args.nosummary:
        train_writer = tf.summary.FileWriter(args.log + '/' + log_file + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log + '/' + log_file + '/test', sess.graph)

    tf.summary.scalar('loss', loss)

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('TP', TP)
    tf.summary.scalar('TN', TN)
    tf.summary.scalar('FP', FP)
    tf.summary.scalar('FN', FN)
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)
    tf.summary.scalar('F1', F1)
    tf.summary.scalar('AUC', AUC)

    AUC_SK = tf.placeholder(tf.float32)
    tf.summary.scalar('AUC_SK', AUC_SK)

    for i in range(0, args.nz_bins):
        tf.summary.scalar('accuracy_z%s' % i, metrics_z[i][0][0])
        tf.summary.scalar('TP_z%s' % i, metrics_z[i][0][1])
        tf.summary.scalar('TN_z%s' % i, metrics_z[i][0][2])
        tf.summary.scalar('FP_z%s' % i, metrics_z[i][0][3])
        tf.summary.scalar('FN_z%s' % i, metrics_z[i][0][4])
        tf.summary.scalar('precision_z%s' % i, metrics_z[i][0][5])
        tf.summary.scalar('recall_z%s' % i, metrics_z[i][0][6])
        tf.summary.scalar('F1_z%s' % i, metrics_z[i][0][7])
        tf.summary.scalar('AUC_z%s' % i, metrics_z[i][0][8])

    merged = tf.summary.merge_all()

    for epoch in range(args.epochs):
        sess.run(tf.local_variables_initializer())
        sess.run(train_init_op)
        while True:
            try:
                sess.run([model.optimize] + metric_update_ops, feed_dict={keep_prob: 1.0 - dropout})
            except tf.errors.OutOfRangeError:
                break
        train_loss, train_acc, train_AUC, summary = sess.run([loss, accuracy, AUC, merged], feed_dict={AUC_SK: 0})
        if not args.nosummary:
            train_writer.add_summary(summary, epoch)
        sess.run(tf.local_variables_initializer())
        sess.run(test_init_op)
        # TODO find a better way of doing a streaming multi-class AUC and confusion matrix in TF
        labels, scores, actual, predictions = [], [], [], []
        while True:
            try:
                res = sess.run([model.labels, model.scores, model.actual, model.predictions] + metric_update_ops, feed_dict={keep_prob: 1.0})
                labels.append(res[0])
                scores.append(res[1])
                actual.append(res[2])
                predictions.append(res[3])
            except tf.errors.OutOfRangeError:
                break
        labels = np.vstack(labels)
        scores = np.vstack(scores)
        actual = np.concatenate(actual)
        predictions = np.concatenate(predictions)
        average_auc = roc_auc_score(labels, scores, average='macro')
        test_loss, test_acc, summary = sess.run([loss, accuracy, merged], feed_dict={keep_prob: 1.0, AUC_SK: average_auc})
        # Plot normalized confusion matrix
        plot_confusion_matrix(cm(actual, predictions), classes=class_labels, normalize=True, 
                              filename=args.log + '/' + log_file + '/test/confusion_matrix.pdf')

        if not args.nosummary:
            test_writer.add_summary(summary, epoch)
        print('epoch = {0} | train loss = {1:.3f} | train acc = {2:.3f} | test loss = {3:.3f} | test acc = {4:.3f}'.format(str(epoch+1).zfill(5), train_loss, train_acc, test_loss, test_acc))

if __name__ == '__main__':
    main()
