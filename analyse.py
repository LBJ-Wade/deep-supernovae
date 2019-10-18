from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
import argparse


variables = [
    'accuracy',
    'accuracy_1',
    'F1',
    'AUC',
    'AUC_SK',
    'precision',
    'recall',
    'loss'
]
z_suffices = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5']
z = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

colors = ['blue', 'red', 'green']

z_variables = []
for variable in variables:
    for z_suffix in z_suffices:
        z_variables.append(variable + '_' + z_suffix)
variables += z_variables

table_columns = [
    'accuracy_1',
    'precision',
    'recall',
    'F1',
    'AUC'
]

table_scale = [
    100,
    100,
    100,
    1,
    1
]

table_precision = [
    1,
    1,
    1,
    3,
    3
]


def analyse(file_roots=[], last_epochs=10, verbose=False):
    results = {}
    if len(file_roots) == 0:
        for filename in sorted(glob.glob('logs/*/test/*')):
            file_root = '-'.join(filename.split('/')[1].split('-')[:-1])
            if file_root not in file_roots:
                file_roots.append(file_root)
    for file_root in file_roots:
        print(file_root)
        metrics = []
        results[file_root] = {}
        for filename in glob.glob('logs/' + file_root + '*/test/*'):
            values = [[] for _ in range(len(variables))]
            try:
                for e in tf.train.summary_iterator(filename):
                    for v in e.summary.value:
                        if v.tag in variables:
                            index = variables.index(v.tag)
                            values[index].append(v.simple_value)
                metrics.append(values)
            except:
                pass
        table_summary = ['' for _ in range(len(table_columns))]
        for i in range(len(variables)):
            values = []
            for j in range(len(metrics)):
                if len(metrics[j][i]) > 0:
                    values.append(np.mean(metrics[j][i][-last_epochs:]))
            if len(values) > 1:
                results[file_root][variables[i]] = {'mean': np.mean(values), 'error': np.std(values), 'values': values}
                if verbose and not any(z_suffix in variables[i] for z_suffix in z_suffices):
                    print("{0} {1} {2:.3f} {3:.3f}".format(variables[i], values, np.mean(values), np.std(values)))
                if variables[i] in table_columns:
                    idx = table_columns.index(variables[i])
                    scale = table_scale[idx]
                    precision = table_precision[idx]
                    table_summary[idx] = ' $ {0:.{2}f} \pm {1:.{2}f} $'.format(np.mean(values)*scale, np.std(values)*scale, precision)
        print(' & '.join(table_summary))
        print()
    return results


def results_to_array(results, variable):
    mean = []
    error = []
    for z_suffix in z_suffices:
        if variable + '_' + z_suffix in results:
            mean.append(results[variable + '_' + z_suffix]['mean'])
            error.append(results[variable + '_' + z_suffix]['error'])
    return np.array(mean), np.array(error)


def plot(results, file_roots=[]):

    fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(4, 10))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    ax = axes[0]
    i = 0
    for file_root in file_roots:
        if file_root in results:
            mean, error = results_to_array(results[file_root], 'accuracy')
            ax.plot(z, mean * 100, color=colors[i], linewidth=1)
            ax.fill_between(z, mean * 100 - error * 100, mean * 100 + error * 100, alpha=0.25, color=colors[i])
            i += 1
    ax.set_xlim([0.1, 1.1])
    ax.set_ylim([70, 100])
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid()

    ax = axes[1]
    i = 0
    for file_root in file_roots:
        if file_root in results:
            mean, error = results_to_array(results[file_root], 'precision')
            ax.plot(z, mean * 100, color=colors[i], linewidth=1)
            ax.fill_between(z, mean * 100 - error * 100, mean * 100 + error * 100, alpha=0.25, color=colors[i])
            i += 1
    ax.set_xlim([0.1, 1.1])
    ax.set_ylim([50, 100])
    ax.set_ylabel('Precision/purity (%)', fontsize=8)
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid()

    ax = axes[2]
    i = 0
    for file_root in file_roots:
        if file_root in results:
            mean, error = results_to_array(results[file_root], 'recall')
            ax.plot(z, mean * 100, color=colors[i], linewidth=1)
            ax.fill_between(z, mean * 100 - error * 100, mean * 100 + error * 100, alpha=0.25, color=colors[i])
            i += 1
    ax.set_xlim([0.1, 1.1])
    ax.set_ylim([60, 100])
    ax.set_ylabel('Recall/completeness (%)', fontsize=8)
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.set_yticks([70, 80, 90, 100])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid()

    ax = axes[3]
    i = 0
    for file_root in file_roots:
        if file_root in results:
            mean, error = results_to_array(results[file_root], 'F1')
            ax.plot(z, mean, color=colors[i], linewidth=1)
            ax.fill_between(z, mean - error, mean + error, alpha=0.25, color=colors[i])
            i += 1
    ax.set_xlim([0.1, 1.1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('F1', fontsize=8)
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid()

    ax = axes[4]
    i = 0
    for file_root in file_roots:
        if file_root in results:
            mean, error = results_to_array(results[file_root], 'AUC')
            ax.plot(z, mean, color=colors[i], linewidth=1)
            ax.fill_between(z, mean - error, mean + error, alpha=0.25, color=colors[i])
            i += 1
    ax.set_xlim([0.1, 1.1])
    ax.set_ylim([0.85, 1])
    ax.set_xlabel('Redshift', fontsize=8)
    ax.set_ylabel('AUC', fontsize=8)
    ax.set_xticks([0.25, 0.5, 0.75, 1])
    ax.set_yticks([0.9, 0.95, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid()

    plt.savefig('data/plots/metrics.pdf')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', nargs='+', default=[16], help='Hidden size')
    parser.add_argument('--dataset', default='simgen', help='simgen or des')
    parser.add_argument('-nohostz', action='store_true', help='No host z data')
    parser.add_argument('-verbose', action='store_true', help='Verbose')
    args = parser.parse_args()

    hidden = '.'.join([str(n) for n in args.hidden])

    results = analyse(verbose=args.verbose)

    file_roots = [
        'PLSTM-' + hidden + '-0.94827-' + args.dataset + '-32-True-sn1a-' + str(not args.nohostz) + '-True-0',
        'PLSTM-' + hidden + '-0.94827-' + args.dataset + '-32-True-sn1a-' + str(not args.nohostz) + '-False-100',
        'PLSTM-' + hidden + '-0.5-' + args.dataset + '-32-True-sn1a-' + str(not args.nohostz) + '-True-0',
    ]

    plot(results, file_roots=file_roots)
