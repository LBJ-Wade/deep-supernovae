from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
import glob
import os
from itertools import groupby
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


"""
Classes are
Ia 1
II 2, IIn 21, IIP 22, IIL 23
Ibc 3, Ib 32, Ic 33
"""

type_to_name = {1: 'Ia', 2: 'II', 21: 'IIn', 22: 'IIP', 23: 'IIL', 3: 'Ibc', 32: 'Ib', 33: 'Ic'}

# Changed Ia to class 1 as TF converts to bool to get correct metrics for 2 class problem
sn1a_keys = {'Ia': 1, 'II': 0, 'IIn': 0, 'IIP': 0, 'IIL': 0, 'Ibc': 0, 'Ib': 0, 'Ic': 0}
type_keys = {'Ia': 0, 'II': 1, 'IIn': 1, 'IIP': 1, 'IIL': 1,  'Ibc': 2, 'Ib': 2, 'Ic': 2}
subtype_keys = {'Ia': 0, 'II': 1, 'IIn': 2, 'IIP': 3, 'IIL': 4, 'Ibc': 5, 'Ib': 6, 'Ic': 7}

sn1a_classes = ['Non Ia', 'Ia']
type_classes = ['Ia', 'II', 'Ibc']
subtype_classes = ['Ia', 'II', 'IIn', 'IIP', 'IIL', 'Ibc', 'Ib', 'Ic']


class DataLoader:

    def __init__(self, file_root='data/SIMGEN_PUBLIC_DES_PROCESS/', test_fraction=0.94827, use_hostz=True, time_shift=40,
                 gaussian_noise=1.0, keys=sn1a_keys, pattern='*', representative=True, additional_representative=0):
        self.time_shift = time_shift
        self.gaussian_noise = gaussian_noise
        data = load_data(file_root=file_root, test_fraction=test_fraction, use_hostz=use_hostz, keys=keys, pattern=pattern,
                         representative=representative, additional_representative=additional_representative)
        self.train, self.test, (self.length_train, self.length_test, self.max_sequence_len, self.num_classes) = data

    def _random_missing(self, t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days):
        # Imputes missing values with random value between min and max
        new_flux_values = flux_values
        new_flux_values += flux_min + tf.random_uniform(tf.shape(flux_values), minval=0, maxval=1) * (flux_max - flux_min)
        return t, new_flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days

    def _mean_missing(self, t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days):
        # Imputes missing values witn mean value between min and max
        new_flux_values = flux_values
        new_flux_values += flux_min + 0.5*(flux_max - flux_min)
        return t, new_flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days

    def _add_noise(self, t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days):
        # Add Gaussian noise
        new_flux_values = flux_values
        new_flux_values += tf.random_normal(tf.shape(flux_errors), mean=0, stddev=self.gaussian_noise) * flux_errors
        return t, new_flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days

    def _shift_time(self, t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days):
        # Shifts the time by a random offset
        shifted_t = t + tf.random_uniform([1], minval=-self.time_shift, maxval=self.time_shift)
        return shifted_t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days

    def _truncate(self, t, flux_values, flux_errors, flux_min, flux_max, additional, sequence_length, labels, z, truncate_days):
        # Truncates the lightcurve
        new_sequence_length = tf.random_shuffle(truncate_days)[0]
        return t, flux_values, flux_errors, flux_min, flux_max, additional, new_sequence_length, labels, z, truncate_days

    def get_dataset(self, batch_size=32, augment=True, test_as_train=False):

        train_dataset = tf.data.Dataset.from_tensor_slices(self.train)

        if augment:
            train_dataset = train_dataset.map(self._shift_time)
            train_dataset = train_dataset.map(self._random_missing)
            train_dataset = train_dataset.map(self._add_noise)
            train_dataset = train_dataset.map(self._truncate)

        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=100)

        if test_as_train:
            test_dataset = tf.data.Dataset.from_tensor_slices(self.train)
        else:
            test_dataset = tf.data.Dataset.from_tensor_slices(self.test)

        if augment:
            test_dataset = test_dataset.map(self._mean_missing)

        test_dataset = test_dataset.batch(1000)

        return train_dataset, test_dataset


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=-1.):

    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)

        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes), dtype='int32')
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


def index_min(values):
    return min(range(len(values)), key=values.__getitem__)


def time_collector(arr, frac=1):
    bestclustering = True
    while bestclustering:
        a = []
        for key, group in groupby(arr, key=lambda n: n//(1./frac)):
            s = sorted(group)
            a.append(np.sum(s)/len(s)) 
        ind = []
        i = 0
        for key, group in groupby(arr, key=lambda n: n//(1./frac)):
            ind.append([])
            for j in group:
                ind[i].append(index_min(abs(j-np.array(arr))))
            i += 1
        if len([len(i) for i in ind if len(i) > 4]) != 0:
            frac += 0.1
        else:
            bestclustering = False
    return a, ind, frac


def create_colourband_array(ind, arr, err_arr, temp_arr, err_temp_arr):
    temp = [arr[ind[i]] for i in range(len(ind)) if arr[ind[i]] != 0]
    err_temp = [err_arr[ind[i]] for i in range(len(ind)) if err_arr[ind[i]] != 0]
    if len(temp) == 0:
        temp_arr.append(0)
        err_temp_arr.append(0)
        out = True
    elif len(temp) > 1:
        out = False
    else:
        temp_arr.append(temp[0])
        err_temp_arr.append(err_temp[0])
        out = True
    return temp_arr, err_temp_arr, out


def preprocess(grouping=1, file_root='data/SIMGEN_PUBLIC_DES/',
               process_root='data/SIMGEN_PUBLIC_DES_PROCESS/', key_root=None):
    if not os.path.exists(process_root):
        os.makedirs(process_root)
    unblind_ids = []
    unblind_keys = {}
    if key_root is not None:
        with open(key_root, 'rU') as f:
            for line in f:
                s = line.split(':')
                if len(s) > 0 and s[0] == 'SN':
                    o = s[1].split()
                    snid = int(o[0].strip())
                    sn_type = int(o[1].strip())
                    sim_z = float(o[3].strip())
                    unblind_ids.append(snid)
                    unblind_keys[snid] = [type_to_name[sn_type], sim_z]
    print('Processing data %s' % file_root)
    for filename in glob.glob(file_root + 'DES_*.DAT'):
        header = []
        obs = []
        with open(filename, 'rU') as f:
            first_obs = None
            add_header = True
            for line in f:
                s = line.split(':')
                if len(s) > 0:
                    if s[0] == 'SNID':
                        snid = int(s[1].strip())
                if '#' in line:
                    add_header = False
                if add_header:
                    header.append(line)
                elif len(s) > 0 and s[0] == 'OBS':
                    add_header = False
                    g = r = i = z = 0
                    g_error = r_error = i_error = z_error = 0
                    o = s[1].split()
                    if first_obs is None:
                        first_obs = float(o[0])
                    if o[1] == 'g':
                        g = float(o[3])
                        g_error = float(o[4])
                    elif o[1] == 'r':
                        r = float(o[3])
                        r_error = float(o[4])
                    elif o[1] == 'i':
                        i = float(o[3])
                        i_error = float(o[4])
                    elif o[1] == 'z':
                        z = float(o[3])
                        z_error = float(o[4])
                    obs.append([float(o[0])] + [g, r, i, z] + [g_error, r_error, i_error, z_error])
        t_arr = [obs[i][0] for i in range(len(obs))]
        g_arr = [obs[i][1] for i in range(len(obs))]
        g_err_arr = [obs[i][5] for i in range(len(obs))]
        r_arr = [obs[i][2] for i in range(len(obs))]
        r_err_arr = [obs[i][6] for i in range(len(obs))]
        i_arr = [obs[i][3] for i in range(len(obs))]
        i_err_arr = [obs[i][7] for i in range(len(obs))]
        z_arr = [obs[i][4] for i in range(len(obs))]
        z_err_arr = [obs[i][8] for i in range(len(obs))]
        correctplacement = True
        frac = grouping
        while correctplacement:
            t, index, frac = time_collector(t_arr, frac)
            g_temp_arr = []
            g_err_temp_arr = []
            r_temp_arr = []
            r_err_temp_arr = []
            i_temp_arr = []
            i_err_temp_arr = []
            z_temp_arr = []
            z_err_temp_arr = []
            tot = []
            for i in range(len(index)):
                g_temp_arr, g_err_temp_arr, gfail = create_colourband_array(index[i], g_arr, g_err_arr, g_temp_arr, g_err_temp_arr)
                r_temp_arr, r_err_temp_arr, rfail = create_colourband_array(index[i], r_arr, r_err_arr, r_temp_arr, r_err_temp_arr)
                i_temp_arr, i_err_temp_arr, ifail = create_colourband_array(index[i], i_arr, i_err_arr, i_temp_arr, i_err_temp_arr)
                z_temp_arr, z_err_temp_arr, zfail = create_colourband_array(index[i], z_arr, z_err_arr, z_temp_arr, z_err_temp_arr)
                tot.append(gfail*rfail*ifail*zfail)
            if all(tot):
                correctplacement = False
            else:
                frac += 0.1
        with open(process_root + filename.split(file_root)[1], 'w') as f:
            for line in header:
                f.write(line)
            if snid in unblind_ids:
                f.write('SIM_COMMENT: SN Type = %s,\n' % unblind_keys[snid][0])
                f.write('SIM_REDSHIFT: %s \n' % unblind_keys[snid][1])
                f.write('\n\n')
            f.write('VARLIST:  MJD   g FLUX   g FLUXERR   r FLUX   r FLUXERR   i FLUX   i FLUXERR   z FLUX   z FLUXERR\n')
            for i in range(len(t)):
                obs = [t[i], g_temp_arr[i], g_err_temp_arr[i], r_temp_arr[i], r_err_temp_arr[i], i_temp_arr[i], i_err_temp_arr[i], z_temp_arr[i], z_err_temp_arr[i]]
                f.write('OBS:  ' + ' '.join(["{:9.3f}".format(o) for o in obs]))
                f.write('\n')


def load_data(file_root='data/SIMGEN_PUBLIC_DES_PROCESS/', test_fraction=0.94827,
              use_hostz=True, keys=sn1a_keys, pattern='*', representative=True,
              additional_representative=0, max_truncate_days=40):
    labels = []
    times = []
    flux_values = []
    flux_errors = []
    flux_min_interval = []
    flux_max_interval = []
    additional = []
    redshifts = []
    sn_types = []
    max_fluxes = []
    truncate_days = []
    print('Loading data %s' % file_root)
    for filename in glob.glob(file_root + 'DES_SN' + pattern + '.DAT'):
        snid = ra = dec = mwebv = hostz = sim_type = sim_z = None
        sample_times = []
        sample_flux_values = []
        sample_flux_errors = []
        sample_additional = []
        first_obs = None
        max_flux = [0, 0, 0, 0]
        with open(filename, 'rU') as f:
            for line in f:
                s = line.split(':')
                if len(s) > 0:
                    if s[0] == 'SNID':
                        snid = int(s[1].strip())
                    elif s[0] == 'SNTYPE':
                        sn_type = int(s[1].strip())
                    elif s[0] == 'RA':
                        ra = float(s[1].split('deg')[0].strip())
                    elif s[0] == 'DECL':
                        dec = float(s[1].split('deg')[0].strip())
                    elif s[0] == 'MWEBV':
                        mwebv = float(s[1].split('MW')[0].strip())
                    elif s[0] == 'HOST_GALAXY_PHOTO-Z':
                        hostz = float(s[1].split('+-')[0].strip()), float(s[1].split('+-')[1].strip())
                    elif s[0] == 'SIM_COMMENT':
                        sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
                    elif s[0] == 'SIM_REDSHIFT':
                        sim_z = float(s[1])
                    elif s[0] == 'OBS':
                        o = s[1].split()
                        if first_obs is None:
                            first_obs = float(o[0])
                        sample_times.append(float(o[0]) - first_obs)
                        sample_flux_values.append([float(o[1]), float(o[3]), float(o[5]), float(o[7])])
                        sample_flux_errors.append([float(o[2]), float(o[4]), float(o[6]), float(o[8])])
                        if use_hostz:
                            sample_additional.append([mwebv, hostz[0]])
                        else:
                            sample_additional.append([mwebv])
                        if float(o[1]) > max_flux[0]:
                            max_flux[0] = float(o[1])
                        if float(o[3]) > max_flux[1]:
                            max_flux[1] = float(o[3])
                        if float(o[5]) > max_flux[2]:
                            max_flux[2] = float(o[5])
                        if float(o[7]) > max_flux[3]:
                            max_flux[3] = float(o[7])

        if sim_type not in keys:
            continue

        redshifts.append(sim_z)
        sn_types.append(sn_type)
        times.append(sample_times)
        labels.append(keys[sim_type])
        additional.append(sample_additional)
        max_fluxes.append(max_flux)
        flux_values.append(sample_flux_values)
        flux_errors.append(sample_flux_errors)
        # For missing observations obtain previous and next and valid observations
        sample_flux_previous = []
        sample_flux_errors_previous = []
        last = [0, 0, 0, 0]
        last_error = [0, 0, 0, 0]
        for i in range(len(sample_flux_values)):
            flux = []
            for j in range(0, 4):
                if sample_flux_values[i][j] != 0:
                    last[j] = sample_flux_values[i][j]
                if sample_flux_values[i][j] == 0:
                    flux.append(last[j])
                else:
                    flux.append(0)
            sample_flux_previous.append(flux)
            flux_error = []
            for j in range(0, 4):
                if sample_flux_errors[i][j] != 0:
                    last_error[j] = sample_flux_errors[i][j]
                if sample_flux_errors[i][j] == 0:
                    flux_error.append(last_error[j])
                else:
                    flux_error.append(0)
            sample_flux_errors_previous.append(flux_error)
        sample_flux_next = []
        sample_flux_errors_next = []
        last = [0, 0, 0, 0]
        last_error = [0, 0, 0, 0]
        for i in range(len(sample_flux_values)):
            flux = []
            for j in range(0, 4):
                if sample_flux_values[len(sample_flux_values) - i - 1][j] != 0:
                    last[j] = sample_flux_values[len(sample_flux_values) - i - 1][j]
                if sample_flux_values[len(sample_flux_values) - i - 1][j] == 0:
                    flux.append(last[j])
                else:
                    flux.append(0)
            sample_flux_next.append(flux)
            flux_error = []
            for j in range(0, 4):
                if sample_flux_errors[len(sample_flux_errors) - i - 1][j] != 0:
                    last_error[j] = sample_flux_errors[len(sample_flux_errors) - i - 1][j]
                if sample_flux_errors[len(sample_flux_errors) - i - 1][j] == 0:
                    flux_error.append(last_error[j])
                else:
                    flux_error.append(0)
            sample_flux_errors_next.append(flux_error)
        sample_flux_next.reverse()
        sample_flux_errors_next.reverse()
        sample_flux_min_interval = []
        sample_flux_max_interval = []
        for i in range(len(sample_flux_values)):
            flux = []
            for j in range(0, 4):
                flux.append(min(sample_flux_previous[i][j], sample_flux_next[i][j]))
            sample_flux_min_interval.append(flux)
            flux = []
            for j in range(0, 4):
                flux.append(max(sample_flux_previous[i][j], sample_flux_next[i][j]))
            sample_flux_max_interval.append(flux)
        flux_min_interval.append(sample_flux_min_interval)
        flux_max_interval.append(sample_flux_max_interval)

        time_from_end = np.array(sample_times) - max(sample_times)
        if max(sample_times) > 100:
            sample_truncate_days = [max((time_from_end <= -day).sum(), 1) for day in range(0, max_truncate_days + 1)]
        else:
            sample_truncate_days = [len(sample_times) for day in range(0, max_truncate_days + 1)]
        truncate_days.append(sample_truncate_days)

    sequence_length = np.array([len(d) for d in times], dtype='int32')
    length = len(labels)

    max_fluxes = np.array(max_fluxes)

    if representative:
        test_length = int(length*test_fraction)
        indices = np.random.permutation(length)
        train_idx, test_idx = indices[:length-test_length], indices[length-test_length:]
        train_idx -= 1
        test_idx -= 1
    else:
        sn_types = np.array(sn_types)
        train_idx = np.where(sn_types != -9)[0]
        test_idx = np.where(sn_types == -9)[0]
        indices = np.random.permutation(test_idx.shape[0])[0:additional_representative]
        train_idx = np.append(train_idx, test_idx[indices])
        test_idx = np.delete(test_idx, indices)

    length_train = train_idx.shape[0]
    length_test = test_idx.shape[0]
    nb_classes = len(set(keys.values()))

    print('Length train: %s' % (length_train))
    print('Length test: %s' % (length_test))
    print('Num classes: %s' % (nb_classes))

    redshifts = np.array(redshifts)

    times = pad_sequences(times, dtype='float32', padding='post')
    times = np.array(times)
    times = np.expand_dims(times, axis=2)

    labels = np.array(labels, dtype='int32')
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    unique, counts = np.unique(labels_train, return_counts=True)
    print('Train labels:')
    print(np.asarray((unique, counts)).T)
    unique, counts = np.unique(labels_test, return_counts=True)
    print('Test labels:')
    print(np.asarray((unique, counts)).T)
    labels_train = to_categorical(labels_train, nb_classes)
    labels_test = to_categorical(labels_test, nb_classes)

    filename = 'data/training_flux_redshift_%s_%s_%s.txt' % (test_fraction, representative, additional_representative)
    with open(filename, 'w') as f:
        for idx in train_idx:
            f.write('%s %s %s \n' % (redshifts[idx], max_fluxes[idx, 2], labels[idx]))

    flux_values = pad_sequences(flux_values, dtype='float32', padding='post')
    flux_min_interval = pad_sequences(flux_min_interval, dtype='float32', padding='post')
    flux_max_interval = pad_sequences(flux_max_interval, dtype='float32', padding='post')
    flux_errors = pad_sequences(flux_errors, dtype='float32', padding='post')
    length = flux_values.shape[0]
    max_sequence_len = flux_values.shape[1]

    additional = pad_sequences(additional, dtype='float32', padding='post')

    truncate_days = np.array(truncate_days, dtype='int32')

    train = (
        times[train_idx, :, :],
        flux_values[train_idx, :, :],
        flux_errors[train_idx, :, :],
        flux_min_interval[train_idx, :, :],
        flux_max_interval[train_idx, :, :],
        additional[train_idx, :, :],
        sequence_length[train_idx],
        labels_train,
        redshifts[train_idx],
        truncate_days[train_idx]
    )
    test = (
        times[test_idx, :, :],
        flux_values[test_idx, :, :],
        flux_errors[test_idx, :, :],
        flux_min_interval[test_idx, :, :],
        flux_max_interval[test_idx, :, :],
        additional[test_idx, :, :],
        sequence_length[test_idx],
        labels_test,
        redshifts[test_idx],
        truncate_days[test_idx]
    )

    return train, test, (length_train, length_test, max_sequence_len, nb_classes)


def plot_lightcurves(file_root='data/SIMGEN_PUBLIC_DES/', key_root=None):
    labels = []
    g_data, r_data, i_data, z_data = [], [], [], []
    max_fluxes = []
    redshifts = []
    unblind_ids = []
    unblind_keys = {}
    if key_root is not None:
        with open(key_root, 'rU') as f:
            for line in f:
                s = line.split(':')
                if len(s) > 0 and s[0] == 'SN':
                    o = s[1].split()
                    snid = int(o[0].strip())
                    sn_type = int(o[1].strip())
                    sim_z = float(o[3].strip())
                    unblind_ids.append(snid)
                    unblind_keys[snid] = [type_to_name[sn_type], sim_z]
    ids = []
    sn_types = []
    print('Loading data %s' % file_root)
    for filename in glob.glob(file_root + 'DES_*.DAT'):
        snid = None
        g, r, i, z = [], [], [], []
        max_flux = [0, 0, 0, 0]
        first_obs = sim_type = sim_z = None
        with open(filename, 'rU') as f:
            for line in f:
                s = line.split(':')
                if len(s) > 0:
                    if s[0] == 'SNID':
                        snid = int(s[1].strip())
                        ids.append(snid)
                    elif s[0] == 'SNTYPE':
                        sn_type = int(s[1].strip())
                    elif s[0] == 'SIM_COMMENT':
                        sim_type = s[1].split('SN Type =')[1].split(',')[0].strip()
                    elif s[0] == 'SIM_REDSHIFT':
                        sim_z = float(s[1])
                    elif s[0] == 'OBS':
                        o = s[1].split()
                        if first_obs is None:
                            first_obs = float(o[0])
                        obs = [float(o[0]) - first_obs, float(o[3]), float(o[4])]
                        if o[1] == 'g':
                            g.append(obs)
                            if float(o[3]) > max_flux[0]:
                                max_flux[0] = float(o[3])
                        elif o[1] == 'r':
                            r.append(obs)
                            if float(o[3]) > max_flux[1]:
                                    max_flux[1] = float(o[3])
                        elif o[1] == 'i':
                            i.append(obs)
                            if float(o[3]) > max_flux[2]:
                                max_flux[2] = float(o[3])
                        elif o[1] == 'z':
                            z.append(obs)
                            if float(o[3]) > max_flux[3]:
                                max_flux[3] = float(o[3])
        if unblind_ids:
            if snid in unblind_ids:
                labels.append(unblind_keys[snid][0])
                redshifts.append(unblind_keys[snid][1])
            else:
                continue
        else:
            labels.append(sim_type)
            redshifts.append(sim_z)
        sn_types.append(sn_type)
        g_data.append(g)
        r_data.append(r)
        i_data.append(i)
        z_data.append(z)
        max_fluxes.append(max_flux)

    redshifts = np.array(redshifts)
    max_fluxes = np.array(max_fluxes)
    labels = np.array(labels)
    sn_types = np.array(sn_types)

    nrows = 3
    ncolumns = 5
    for ip in range(0, 10):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncolumns, sharex=True, sharey=True, figsize=(7, 4))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        for row in range(0, nrows):
            for column in range(0, ncolumns):
                ax = axes[row, column]
                ax.set_xticks([0, 50, 100])
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_xlabel('Time (days)', fontsize=7)
                if column == 0:
                    ax.set_ylabel('Flux', fontsize=7)
                i = ncolumns*row + column
                data = np.array(g_data[i + 16*ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='green', linewidth=1)
                data = np.array(r_data[i + 16*ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='red', linewidth=1)
                data = np.array(i_data[i + 16*ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='black', linewidth=1)
                data = np.array(z_data[i + 16*ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='blue', linewidth=1)
                ax.text(75.0, 60.0, 'Type: %s' % labels[i + 16*ip], fontsize=7)
                ax.text(75.0, 80.0, 'ID: %s' % str(ids[i + 16*ip]).zfill(6), fontsize=7)
                ax.set_xlim([0, 150])
                ax.set_ylim([-25, 100])
                ax.tick_params(axis='both', which='major', labelsize=7)
        plt.savefig('data/plots/%s.pdf' % ip)


def test_data(pattern='*'):

    loader = DataLoader(pattern=pattern, test_fraction=0.0)
    train_dataset, test_dataset = loader.get_dataset(test_as_train=True)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_values = iterator.get_next()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    nrows = 3
    ncolumns = 5

    fig, axes = plt.subplots(nrows=nrows, ncols=ncolumns, sharex=True, sharey=True, figsize=(7, 4))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # Training data
    for row in range(0, nrows):
        for column in range(0, ncolumns):
            sess.run(train_init_op)
            t, flux_values, _, _, _, additional, sequence_length, labels, z, _ = sess.run(next_values)
            t_plot = t[0, 0:sequence_length[0], 0]
            flux_plot = flux_values[0, 0:sequence_length[0], :]
            ax = axes[row, column]
            ax.set_xlim([0, 150])
            ax.set_ylim([-25, 100])
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlabel('Time (days)', fontsize=7)
            if column == 0:
                ax.set_ylabel('Flux', fontsize=7)
            ax.text(75.0, 60.0, 'Epoch: %s' % (row*ncolumns + column + 1), fontsize=7)
            ax.text(75.0, 80.0, 'ID: %s' % pattern, fontsize=7)
            ax.plot(t_plot, flux_plot[:, 0], color='green')
            ax.plot(t_plot, flux_plot[:, 1], color='red')
            ax.plot(t_plot, flux_plot[:, 2], color='black')
            ax.plot(t_plot, flux_plot[:, 3], color='blue')
            ax.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('data/plots/%s_training.pdf' % pattern)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncolumns, sharex=True, sharey=True)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # Test data
    for row in range(0, nrows):
        for column in range(0, ncolumns):
            sess.run(test_init_op)
            t, flux_values, _, _, _, additional, sequence_length, labels, z, _ = sess.run(next_values)
            t_plot = t[0, 0:sequence_length[0], 0]
            flux_plot = flux_values[0, 0:sequence_length[0], :]
            ax = axes[row, column]
            ax.set_xlim([0, 150])
            ax.set_ylim([-25, 100])
            ax.set_xticks([0, 50, 100])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlabel('Time (days)', fontsize=7)
            if column == 0:
                ax.set_ylabel('Flux', fontsize=7)
            ax.text(75.0, 60.0, 'Epoch: %s' % (row*ncolumns + column + 1), fontsize=7)
            ax.text(75.0, 80.0, 'ID: %s' % pattern, fontsize=7)
            ax.plot(t_plot, flux_plot[:, 0], color='green')
            ax.plot(t_plot, flux_plot[:, 1], color='red')
            ax.plot(t_plot, flux_plot[:, 2], color='black')
            ax.plot(t_plot, flux_plot[:, 3], color='blue')
            ax.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('data/plots/%s_test.pdf' % pattern)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-preprocess', action='store_true')
    parser.add_argument('-lightcurves', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('--pattern', default='005386')
    args = parser.parse_args()

    if args.preprocess:
        preprocess(file_root='data/DES_BLIND+HOSTZ/', process_root='data/DES_BLIND+HOSTZ_PROCESS/', key_root='data/DES_UNBLIND_KEY/DES_UNBLIND+HOSTZ.KEY')
        preprocess(file_root='data/DES_BLINDnoHOSTZ/', process_root='data/DES_BLINDnoHOSTZ_PROCESS/', key_root='data/DES_UNBLIND_KEY/DES_UNBLINDnoHOSTZ.KEY')
        preprocess(file_root='data/SIMGEN_PUBLIC_DES/', process_root='data/SIMGEN_PUBLIC_DES_PROCESS/')

    if args.lightcurves:
        plot_lightcurves(file_root='data/SIMGEN_PUBLIC_DES/')

    if args.test:
        test_data(args.pattern)
