# Importing block #

import csv
import footsim as fs
import io
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import scipy
import operator as op
import time

from collections import defaultdict
from footsim.plotting import plot, figsave
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statannot import add_stat_annotation
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# plt.rc('xtick', labelsize=4)
plt.rcParams.update({'font.size': 20, 'axes.linewidth': 2.5})

# ----------------------------------------- #

# Empirical data manipulation #

# ----------------------------------------- #


def empirical_afferent_locations(filepath, valid_models):

    """ Reads a *.csv file with empirically recorded afferents and generates a dictionary with afferent locations.

           Arguments:

               filepath (str): path to the *.csv file
               valid_models (float): dictionary with the individual afferent models requested, keys need to be
               afferent classes.

           Returns:

               A dictionary with the afferent locations.
           """

    location_mapping = {'GT': 'T1', 'Toes': 'T2_t',
                        'MidMet': 'MMi', 'MedMet': 'MMe', 'LatMet': 'MLa',
                        'MedArch': 'AMe', 'MidArch': 'AMi', 'LatArch': 'ALa',
                        'Heel': 'HR'}

    empirical_location = dict()
    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data

    for afferent_type in valid_models.keys():

        valid = len(valid_models[afferent_type])

        for individual_afferent in range(0, valid):

            if data_file[data_file.Afferent_ID.str.contains(valid_models[afferent_type][individual_afferent])].empty \
                    == False:

                afferent = data_file[data_file.Afferent_ID.str.contains
                (valid_models[afferent_type][individual_afferent])].copy(deep=True)
                afferent_id = str(valid_models[afferent_type][individual_afferent])

                location = afferent.iloc[0]['location_specific']  # Finds the empirical afferent location

                location = location_mapping[location]

                empirical_location[afferent_id] =  location

    return empirical_location


def empirical_data_handling(filepath, freqs):

    """ Groups the empirical thresholds per afferent class

     Arguments:

      filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
      freqs(np.array of float): array containing frequencies of stimulus

    Returns:

       Dictionary with the grouped data.

        """

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file[data_file.Threshold.notnull()].copy(deep=True)  # Gets the thresholds
    output_grouped = dict()

    for aff in range(0, 4):

        afferent_class = grouped_by_threshold[grouped_by_threshold.type.str.contains(afferents[aff])].copy(deep=True)
        output_grouped[afferents[aff]] = list()

        for freq_list in range(0, len(freqs)):

            output_grouped[afferents[aff]].append(np.mean(afferent_class['Amplitude'].where(afferent_class['Frequency'] == freqs[freq_list])))

        output_grouped[afferents[aff]] = np.array(output_grouped[afferents[aff]])

    return output_grouped


def empirical_stimulus_acquisition(filepath, valid_models):

    """ Reads a *.csv file with empirically recorded afferents and generates a dictionary with the stimulus used during
    the original experiment

           Arguments:

               filepath (str): path to the *.csv file
               valid_models (float): dictionary with the individual afferent models requested, keys need to be
               afferent classes.

           Returns:

               A dictionary with the afferent ids as keys and stimulus arrays as values
           """

    location_mapping = {'GT': 'T1', 'Toes': 'T2_t',
                        'MidMet': 'MMi', 'MedMet': 'MMe', 'LatMet': 'MLa',
                        'MedArch': 'AMe', 'MidArch': 'AMi', 'LatArch': 'ALa',
                        'Heel': 'HR'}

    empirical_stimulus = dict()
    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data

    for afferent_type in valid_models.keys():

        valid = len(valid_models[afferent_type])

        for individual_afferent in range(0, valid):

            if data_file[data_file.Afferent_ID.str.contains(valid_models[afferent_type][individual_afferent])].empty == False:

                afferent = data_file[data_file.Afferent_ID.str.contains(valid_models[afferent_type][individual_afferent])].copy(deep=True)
                afferent_id = str(valid_models[afferent_type][individual_afferent])
                freq = list(afferent['Frequency'].where(afferent['AvgInst'] != 0).dropna())  # Gets the frequencies
                amp = list(afferent['Amplitude'].where(afferent['AvgInst'] != 0).dropna())  # Gets the amplitudes
                location = afferent.iloc[0]['location_specific']  # Finds the empirical afferent location

                location = location_mapping[location]

                empirical_stimulus[afferent_id] = dict()
                empirical_stimulus[afferent_id]['Location'] = location
                empirical_stimulus[afferent_id]['Frequency'] = freq
                empirical_stimulus[afferent_id]['Amplitude'] = amp

    return empirical_stimulus


def empirical_stimulus_pairs(empirical_stimulus):

    stimulus_pairs = dict()

    for keys in empirical_stimulus.keys():

        stimulus_pairs[keys] = list()

        for pair in range(0, len(empirical_stimulus[keys]['Frequency'])):

            stimulus_pairs[keys].append((empirical_stimulus[keys]['Frequency'][pair], empirical_stimulus[keys]['Amplitude'][pair]))

    return stimulus_pairs


# ----------------------------------------- #

# Afferent positioning methods #

# ----------------------------------------- #


def empirical_afferent_positioning(filepath, valid_models):  # Positions the afferents as in the empirical recordings

    """ Reads a *.csv file with empirically recorded afferents and generates a simulated afferent popuplation that
    matches it in the same foot sole locations.

        Arguments:

            filepath (str): path to the *.csv file
            valid_models (float): dictionary with the individual afferent models requested, keys need to be
            afferent classes.

        Returns:

            A dictionary with the afferent populations with locations as keys.
        """

    location_mapping = {'GT': 'T1', 'Toes': 'T2_t',
                        'MidMet': 'MMi', 'MedMet': 'MMe', 'LatMet': 'MLa',
                        'MedArch': 'AMe', 'MidArch': 'AMi', 'LatArch': 'ALa',
                        'Heel': 'HR'}

    afferent_populations = dict()  # To be filled with the populations correctly positioned

    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data
    grouped_by_threshold = data_file.copy(deep=True)  # Gets the empirical thresholds if needed

    for i in range(fs.foot_surface.num):

        afferent_populations[fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0]] = fs.AfferentPopulation()

    for afferent_type in valid_models.keys():

        valid = len(valid_models[afferent_type])

        for individual_afferent in range(0, valid):

            idx = individual_afferent  # Specific afferent model

            if grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains
                (valid_models[afferent_type][individual_afferent])].empty == False:

                location_slice = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains
                (valid_models[afferent_type][individual_afferent])].copy(deep=True)

                location = location_slice.iloc[0]['location_specific']  # Finds the empirical afferent location

                location = location_mapping[location]

                # Pins the simulated afferents on the correct foot sole location #

                afferent_populations[location].afferents\
                    .append(fs.Afferent(affclass=afferent_type, idx=int(idx), location=fs.foot_surface.centers[fs.foot_surface.tag2idx(location)[0]]))

    return afferent_populations


def populational_info(populations, valid_models):

    """ Reads a *.csv file with empirically recorded afferents and generates a simulated afferent popuplation that
       matches it in the same foot sole locations.

           Arguments:

               populations(dict): Dictionary of afferent populations with regions as keys.
               valid_models (float): dictionary with the individual afferent models requested, keys need to be
               afferent classes.

           Returns:

               Dictionary with afferent ids, classes and model numbers (idx) of a population.

           """

    afferent_data = dict()

    for location in populations:

        afferent_data[location] = list()

        for afferent in range(0, len(populations[location])):

            affclass = populations[location].afferents[afferent].affclass
            idx = populations[location].afferents[afferent].idx
            afferent_id = valid_models[affclass][idx]

            afferent_data[location].append((afferent_id, affclass, idx))

    return afferent_data


def regional_afferent_positioning():

    """ Creates one afferent population of ALL single models per region

           Returns:

              A dictionary with the afferent populations with locations as keys
           """

    afferent_populations = dict()

    for i in range(fs.foot_surface.num):

        afferent_populations[fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0]] = fs.AfferentPopulation()

    for affclass in fs.Afferent.affclasses:

        for i in range(fs.foot_surface.num):

            for idx in range(fs.constants.affparams[affclass].shape[0]):

                # Pins the afferents in the right locations and fills the regions dictionary #

                afferent_populations[fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0]].afferents.append(
                    fs.Afferent(affclass, idx=idx, location=fs.foot_surface.centers[i]))

    return afferent_populations


def single_afferent_positioning(affclass, idx):

    """ Pins a single model on every surface of the foot sole.

        Arguments:

            affclass (str): afferent model class
            idx (float): afferent model id

        Returns:

           The afferent class, the afferent model and a dictionary with the afferent populations with locations
           as keys
        """

    afferent_populations = dict()

    for i in range(fs.foot_surface.num):

        afferent_populations[fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0]] = fs.AfferentPopulation()

    for i in range(fs.foot_surface.num):

        # Pins the afferents in the right locations and fills the regions dictionary #

        afferent_populations[fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0]].afferents\
            .append(fs.Afferent(affclass, idx=int(idx), location=fs.foot_surface.centers[i]))

    return affclass, idx, afferent_populations


# ----------------------------------------- #

# Investigating afferent responses #

# ----------------------------------------- #

def FR_filtering(filepath, populations, model_firing_rates, valid_models):

    print("Filtering started.")

    empirical_stimulus = empirical_stimulus_acquisition(filepath=filepath, valid_models=valid_models)

    stimulus_pairs = empirical_stimulus_pairs(empirical_stimulus=empirical_stimulus)

    afferent_data = populational_info(populations=populations, valid_models=valid_models)

    for keys in model_firing_rates.keys():  # amp, freq, location

        footsim_pair = (keys[1], keys[0])

        for afferent in range(0, len(afferent_data[keys[2]])):

            afferent_id = afferent_data[keys[2]][afferent][0]  # tuple with afferent_id, affclass, idx

            for stimulus in range(0, len(stimulus_pairs[afferent_id])):

                if footsim_pair == stimulus_pairs[afferent_id][stimulus]:

                    break

                if stimulus == len(stimulus_pairs[afferent_id])-1:

                    model_firing_rates[keys][afferent] = 0 #float('nan')
                    break

    return model_firing_rates


def get_responsive_amplitudes(absolute, amps, filepath, freqs, valid_models, output):

    """ Investigate the responses of an afferent population for a given set of frequencies and amplitudes of stimulus
    computing responses for either absolute of tuning thresholds and grouping the results by afferent class,
    location or model id.

         Arguments:

             absolute (float): Absolute firing threshold in Hz
             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
             populations generation, numbers should match the ones in the *.csv if reproducing experimental data
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus
             output(str): Type of output required, C for class, I for individual models, R for regions


         Returns:

            Dictionary of responsive amplitudes for afferents grouped by by afferent class, location or model id.
         """

    populations = empirical_afferent_positioning(filepath=filepath, valid_models=valid_models)
    #populations = regional_afferent_positioning()
    responsive_afferents = dict()

    absolute = absolute  # Firing rate threshold

    threshold_type = 'A'
    start = time.asctime(time.localtime(time.time()))

    regional_responsive_amplitudes = dict()
    class_responsive_amplitudes = dict()
    response_of_individual_models = dict()

    for affclass in fs.Afferent.affclasses:  # Creating the relevant lists

        for freq_list in range(0, len(freqs)):

            class_responsive_amplitudes[affclass, freqs[freq_list]] = list()  # To be filled with thresholds

            for location in populations:

                regional_responsive_amplitudes[affclass, location, freqs[freq_list]] = list()

                for idx in range(fs.constants.affparams[affclass].shape[0]):

                    response_of_individual_models[affclass, idx, location, freqs[freq_list]] = list()

    rates = dict()  # Firing rates

    rates = model_firing_rates(populations=populations, amps=amps, freqs=freqs)
    rates = FR_filtering(filepath=filepath, populations=populations, model_firing_rates=rates, valid_models=valid_models)

    for keys in rates.keys():

        if threshold_type == "T":

            tuning = 0.8 * keys[1]
            responsive_afferents[keys] = np.where(rates[keys] > tuning)

            responsive_afferents[keys] = responsive_afferents[keys][0]

        if threshold_type == "A":

            responsive_afferents[keys] = np.where(rates[keys] > absolute)

            responsive_afferents[keys] = responsive_afferents[keys][0]

            # For each afferent that responded #

            for t in range(0, responsive_afferents[keys].size):

                afferent_c = int(responsive_afferents[keys][t])
                afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class

                idx = populations[keys[2]][afferent_c].idx

                # Appends the amplitudes where it was responsive #

                regional_responsive_amplitudes[afferent_class, keys[2], keys[1]].append(keys[0])
                class_responsive_amplitudes[afferent_class, keys[1]].append(keys[0])
                response_of_individual_models[afferent_class, idx, keys[2], keys[1]].append(keys[0])

    print("Simulation started at: ", start)
    print("Simulation finished at: ", time.asctime(time.localtime(time.time())))

    userinput = output

    if userinput == "R":

        return regional_responsive_amplitudes

    elif userinput == "C":

        return class_responsive_amplitudes

    elif userinput == "I":

        return response_of_individual_models


def ImpCycle(figpath, filepath, valid_models, amps, freqs):

    populations = empirical_afferent_positioning(filepath=filepath, valid_models=valid_models)

    ImpCycle = model_firing_rates(populations=populations, amps=amps, freqs=freqs)

    for keys in ImpCycle:

        for rate in range(0, len(ImpCycle[keys])):

            ramp_up = keys[1]
            ImpCycle_value = ImpCycle[keys][rate][0]/ramp_up
            ImpCycle[keys][rate][0] = int(ImpCycle_value)

    ImpCycle_csvexport(figpath=figpath, ImpCycle=ImpCycle, population=populations, valid_models=valid_models)

    return ImpCycle


def investigate_all_single_models(figpath, amps, freqs):

    """ Investigate responses of all individual afferents generated by single_afferent_positioning(affclass, idx)

         Arguments:

             figpath(str): Where to save result plots
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus


         Returns:

            Dictionary of minimum responsive amplitudes (thresholds) for afferents grouped by by afferent class
            and model id and plots the thresholds per frequency for each single model in log scale.
         """

    start = time.asctime(time.localtime(time.time()))

    single_threshold_storage = dict()

    for affclass in fs.Afferent.affclasses:

        print("First afferent class is ", affclass, " with ", fs.constants.affparams[affclass].shape[0], "models.")

        for idx in range(fs.constants.affparams[affclass].shape[0]):

            single_min = single_afferent_thresholds(figpath, affclass, idx, amps, freqs)

            single_threshold_storage[affclass, idx] = single_afferent_threshold_grouping(single_min)

    print("Simulation started at: ", start)
    print("Simulation finished at: ", time.asctime(time.localtime(time.time())))

    return single_threshold_storage


def model_firing_rates(populations, amps, freqs):

    """ Computes the firing rates (Hz) of an afferent population for a given set of frequencies and amplitudes of stimulus

         Arguments:

             absolute (float): Absolute firing threshold in Hz
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus

         Returns:

            Dictionary of populational firing rates with amplitude, frequency and location of stimulus as keys
         """

    print("Computing firing rates...")

    s = dict()  # Stimulus
    r = dict()  # Responses
    rates = dict()  # Firing rates

    for freq in range(0, len(freqs)):

        for amp in range(0, len(amps)):

            for location in populations:

                if len(populations[location]) != 0:

                    # Stimulus based on empirical data #

                    s[amps[amp], freqs[freq], location] = \
                        fs.stim_sine(amp=amps[amp] / 2, ramp_type='sin', len=2, pin_radius=0.5,
                                     freq=freqs[freq],
                                     loc=fs.foot_surface.centers[fs.foot_surface.tag2idx(str(location))[0]])

                    # Gathers responses #

                    r[amps[amp], freqs[freq], location] = populations[location].response(
                        s[amps[amp], freqs[freq], location])

                    # Computes firing rates #

                    rates[amps[amp], freqs[freq], location] = r[amps[amp], freqs[freq], location].rate()

    return rates


def single_afferent_responses(affclass, idx, amps, freqs):  #

    """ Gets the response of a single model on the foot sole for a given set of frequencies and amplitudes of stimulus

        Arguments:

            affclass (str): afferent model class
            idx (float): afferent model id
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus

        Returns:

           The afferent class, the afferent model and a dictionary with the afferent population responses
        """

    populations = single_afferent_positioning(affclass, idx)
    affclass = populations[0]  # Afferent class is stored in the first position of the tuple
    idx = populations[1]  # Single model number in the second

    single_afferent_responses = dict()
    responsive_afferents = dict()

    for freq in range(0, len(freqs)):  # To be filled with the responses of the single model per frequency

        for location in populations[2].keys():

            single_afferent_responses[affclass, idx, location, freqs[freq]] = list()

    s = dict()  # Stimulus
    r = dict()  # Responses
    rates = dict()  # Firing rates

    absolute = 1  # Firing rate threshold

    #threshold_type = input("Please input 'A' for absolute thresholds or 'T' for tuning thresholds.")
    threshold_type = "A"

    print(time.asctime(time.localtime(time.time())))

    for freq in range(0, len(freqs)):

        for amp in range(0, len(amps)):

            for location in range(0, len(populations[2].keys())):

                # Stimulus based on empirical data #

                s[amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] = \
                    fs.stim_sine(amp=amps[amp] / 2, ramp_type='sin', len=2, pin_radius=3,
                                 freq=freqs[freq], loc=fs.foot_surface.centers[location])

                print("Investigating location: ", fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]
                      , " for frequency ", freqs[freq], " Hz and ", amps[amp], " mm of amplitude.")

                # Gathers responses #

                r[amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] = \
                    populations[2][fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] \
                        .response(
                        s[amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]])

                rates[amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] = r[
                    amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][
                        0]].rate()  # Computes firing rates

                if threshold_type == "T":  # Tuning threshold

                    tuning = 0.8 * freqs[freq]

                    responsive_afferents = np.where(rates[amps[amp], freqs[freq],
                                                          fs.foot_surface.locate
                                                          (fs.foot_surface.centers[location])[0][0]] > tuning)
                    responsive_afferents = responsive_afferents[0]

                elif threshold_type == "A":  # Absolute threshold

                    responsive_afferents[amps[amp], freqs[freq],
                                         fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] = \
                        np.where(rates[amps[amp], freqs[freq], fs.foot_surface.locate
                        (fs.foot_surface.centers[location])[0][0]] > absolute)

                    responsive_afferents[amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]] = \
                    responsive_afferents[ amps[amp], freqs[freq], fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]][0]

                # For each afferent that responded #

                for t in range(0, responsive_afferents[amps[amp], freqs[freq], fs.foot_surface.locate
                    (fs.foot_surface.centers[location])[0][0]].size):

                    afferent_c = int(responsive_afferents[amps[amp], freqs[freq],
                                                          fs.foot_surface.locate
                                                          (fs.foot_surface.centers[location])[0][0]][t])

                    # Finds the region where the afferent was #

                    loc = fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]

                    # Appends the amplitudes where it was responsive #

                    single_afferent_responses[affclass, idx, fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0], freqs[freq]].append(amps[amp])

    return affclass, idx, single_afferent_responses


# ----------------------------------------- #

# Threshold calculations #

# ----------------------------------------- #


def class_absolute_thresholds(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Find the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus

        Arguments:

            absolute (float): Absolute firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Dictionary of minimum responsive amplitudes per afferent class and plots thresholds per frequency in
           log scale.
        """

    class_min = dict()

    class_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath, valid_models=valid_models,
                                                            amps=amps, freqs=freqs, output="C")

    for key in class_responsive_amplitudes:

        class_min[key] = list()  # To be filled with thresholds

    for key in class_responsive_amplitudes:

        if len(class_responsive_amplitudes[key]) > 0:

            class_min[key] = np.min(class_responsive_amplitudes[key])

        else:

            class_min[key] = float('nan')

    dict_to_file(dict=class_min, filename="class_min_for"+str(absolute)+"Hz", output_path=figpath)

    return class_min


def individual_models_thresholds(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Find the absolute thresholds of all afferent models generated with regional_afferent_positioning()
     for a given set of frequencies and amplitudes of stimulus

        Arguments:

            absolute (float): Firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Dictionary of minimum responsive amplitudes grouped per afferent model.

        """

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'HL', 'HR']

    individual_models_thresholds = get_responsive_amplitudes(absolute=absolute,
                                                             filepath=filepath, valid_models=valid_models,
                                                             amps=amps, freqs=freqs, output="I")
    individual_min = dict()

    for affclass in fs.Afferent.affclasses:

        for freq_list in range(0, len(freqs)):

            for location in range(0, len(foot_locations)):

                for idx in range(fs.constants.affparams[affclass].shape[0]):

                    individual_min[affclass, idx, foot_locations[location], freqs[freq_list]] = list()  # Thresholds

    for keys in individual_models_thresholds:

        if len(individual_models_thresholds[keys]) > 0:

            individual_min[keys] = np.min(individual_models_thresholds[keys])

        else:

            individual_min[keys] = float('nan')

    return individual_min


def multiple_class_absolute_thresholds(filepath, figpath, valid_models, amps, freqs):

    """ Computes the minimum responsive amplitudes of an afferent class for a range of firing thresholds from 1 to 25 Hz

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save the plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Plots of minimum responsive amplitude (log) x frequency (linear)
        """

    for absolute in range(1, 20):

        print("Working ", str(absolute), "Hz threshold definition for afferent classes.")
        class_threshold_visualisation(absolute=absolute, filepath=filepath, figpath=figpath, valid_models=valid_models,
                                      amps=amps, freqs=freqs)


def multiple_individual_absolute_thresholds(filepath, figpath, valid_models, amps, freqs):

    """ Computes the minimum responsive amplitudes of all individual afferent models for a range of afferent firing
    thresholds definitions from 1 to 25 Hz

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save the plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Plots of minimum responsive amplitude (log) x frequency (linear)
        """

    for absolute in range(1, 25):

        print("Working ", str(absolute), "Hz threshold definition for individual models.")
        RMSE_AFTindividualmodels(amps=amps, absolute=absolute, freqs=freqs, valid_models=valid_models,
                                 filepath=filepath, figpath=figpath)


def regional_absolute_thresholds(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Find the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus in
    each region of the foot sole

         Arguments:

             absolute (float): Absolute firing threshold in Hz
             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
             populations generation, numbers should match the ones in the *.csv if reproducing experimental data
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus


         Returns:

            Dictionary of responsive amplitudes for afferents grouped per region, and the plots of thresholds x
            frequency in log scale.
         """

    regional_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath,
                                                               valid_models=valid_models, amps=amps, freqs=freqs,
                                                               output='R')

    afferent_types = ['SA1', 'SA2', 'FA1', 'FA2']
    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'HL', 'HR']

    regional_min = dict()

    for n in range(0, 4):

        for freq_list in range(0, len(freqs)):

            for location in range(0, len(foot_locations)):

                regional_min[afferent_types[n], foot_locations[location], freqs[freq_list]] = list()  # Thresholds

    for keys in regional_responsive_amplitudes:

        if len(regional_responsive_amplitudes[keys]) > 0:

            regional_min[keys] = np.min(regional_responsive_amplitudes[keys])

        else:

            regional_min[keys] = float('nan')

    regional_threshold_visualisation(figpath=figpath, regional_min=regional_min)

    return regional_min


def single_afferent_thresholds(figpath, affclass, idx, amps, freqs):

    """ Computes the thresholds of a single afferent model.

         Arguments:

             affclass (str): afferent model class
             idx (float): afferent model id
             figpath(str): Where to save result plots
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus

         Returns:

            Dictionary of minimum responsive amplitudes (thresholds) for the afferent model and a
             amplitude (threshold) x frequency plot in log scale.
         """

    single_min = dict()
    single_responses = single_afferent_responses(affclass, idx, amps, freqs)

    affclass = single_responses[0]
    idx = single_responses[1]
    single_responses = single_responses[2]

    for freq_list in range(0, len(freqs)):

        for i in range(fs.foot_surface.num):

            single_min[affclass, idx, fs.foot_surface.locate(fs.foot_surface.centers[i])[0][0], freqs[freq_list]] = list()  # To be filled with thresholds

    for key in single_responses:

        if len(single_responses[key]) > 0:

            single_min[key] = np.min(single_responses[key])

    single_afferent_threshold_visualisation(figpath, affclass, idx, single_min)

    return single_min


def single_afferent_threshold_grouping(single_min): # Groups the thresholds per model

    """ Computes the thresholds per single model when using investigate_all_single_models(figpath, amps, freqs) for
    further statistical comparisons with experimental data

         Arguments:

             affclass (str): afferent model class
             idx (float): afferent model id
             figpath(str): Where to save result plots
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus

         Returns:

            Dictionary of minimum responsive amplitudes (thresholds) for the afferent model and a
             amplitude (threshold) x frequency plot in log scale.
         """

    single_model = dict()
    responsive_freqs = dict()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'H']

    for loc in range(0, len(foot_locations)):

        single_model[foot_locations[loc]] = list()
        responsive_freqs[foot_locations[loc]] = list()

    for keys in single_min.keys():

        if type(single_min[keys]) is not list:

            for location in range(0, len(foot_locations)):

                if keys[2] == foot_locations[location]:

                    single_model[keys[2]].append(single_min[keys])
                    responsive_freqs[keys[2]].append(keys[3])

    return single_model

# ----------------------------------------- #

# Data visualisation #

# ----------------------------------------- #


def add_identity(axes, *line_args, **line_kwargs):

    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)

    return axes


def average_AFT_individualmodels(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
       stimulus

       Arguments:

       absolute (float): Absolute firing threshold in Hz
       filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
       afferent population is to mimic some experimental data
       figpath(str): Where to save result plots
       valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
       populations generation, numbers should match the ones in the *.csv if reproducing experimental data
       amps(np.array of float): array containing amplitudes of stimulus
       freqs(np.array of float): array containing frequencies of stimulus

        Returns:

           Plots the average of the AFTs as a proxy for afferent class thresholds, replicating empirical data

        """

    responsive_freqs = dict()

    SA1 = list()
    responsive_freqs['SA1'] = list()

    SA2 = list()
    responsive_freqs['SA2'] = list()

    FA1 = list()
    responsive_freqs['FA1'] = list()

    FA2 = list()
    responsive_freqs['FA2'] = list()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'H']

    individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                                  valid_models=valid_models, amps=amps, freqs=freqs)

    for keys in individual_min:

        if keys[0] == 'SA1':

            for loc in range(0, len(foot_locations)):

                if keys[2] == foot_locations[loc]:

                    for idx in range(fs.constants.affparams['SA1'].shape[0]):

                        if keys[1] == idx:

                            SA1.append(individual_min[keys])
                            responsive_freqs[keys[0]].append(keys[3])

        elif keys[0] == 'FA1':

            for loc in range(0, len(foot_locations)):

                if keys[2] == foot_locations[loc]:

                    for idx in range(fs.constants.affparams['FA1'].shape[0]):

                        if keys[1] == idx:
                            FA1.append(individual_min[keys])
                            responsive_freqs[keys[0]].append(keys[3])

        elif keys[0] == 'SA2':

            for loc in range(0, len(foot_locations)):

                if keys[2] == foot_locations[loc]:

                    for idx in range(fs.constants.affparams['SA2'].shape[0]):

                        if keys[1] == idx:
                            SA2.append(individual_min[keys])
                            responsive_freqs[keys[0]].append(keys[3])

        elif keys[0] == 'FA2':

            for loc in range(0, len(foot_locations)):

                if keys[2] == foot_locations[loc]:

                    for idx in range(fs.constants.affparams['FA2'].shape[0]):

                        if keys[1] == idx:

                            FA2.append(individual_min[keys])
                            responsive_freqs[keys[0]].append(keys[3])

    # ----------------------------------------- #

    # Computing average AFTs #

    # ----------------------------------------- #

    model_output = dict()

    model_output['SA1'] = list()
    model_output['SA2'] = list()
    model_output['FA1'] = list()
    model_output['FA2'] = list()

    tempSA1 = 0
    uniqueSA1 = list(set(responsive_freqs['SA1']))
    uniqueSA1 = sorted(uniqueSA1)

    tempSA2 = 0
    uniqueSA2 = list(set(responsive_freqs['SA2']))
    uniqueSA2 = sorted(uniqueSA2)

    tempFA1 = 0
    uniqueFA1 = list(set(responsive_freqs['FA1']))
    uniqueFA1 = sorted(uniqueFA1)

    tempFA2 = 0
    uniqueFA2= list(set(responsive_freqs['FA2']))
    uniqueFA2 = sorted(uniqueFA2)

    counting = 0

    for frequency in range(0, len(uniqueSA1)):

        for replicates in range(0, len(responsive_freqs['SA1'])):

            if responsive_freqs['SA1'][replicates] == uniqueSA1[frequency]:

                respFreq = responsive_freqs['SA1'][replicates]
                testFreq = uniqueSA1[frequency]

                if math.isnan(SA1[replicates]) == False:

                    tempSA1 = SA1[replicates] + tempSA1
                    counting = counting + 1

            elif responsive_freqs['SA1'][replicates] != uniqueSA1[frequency]:

                tempSA1 = tempSA1/counting
                model_output['SA1'].append(tempSA1)

    counting = 0

    for frequency in range(0, len(uniqueSA2)):

        for replicates in range(0, len(responsive_freqs['SA2'])):

            if responsive_freqs['SA2'][replicates] == uniqueSA2[frequency]:

                if math.isnan(SA2[replicates]) == False:

                    tempSA2 = SA2[replicates] + tempSA2
                    counting = counting + 1

            else:

                tempSA2 = tempSA2 / counting
                model_output['SA2'].append(tempSA2)

    # ----------------------------------------- #

    # Generating Figures #

    # ----------------------------------------- #

    fig = plt.figure(dpi=300)

    ax = sns.lineplot(x=responsive_freqs['FA1'], y=FA1, label='FAI')
    #ax = sns.scatterplot(x=responsive_freqs['FA1'], y=FA1, label='FAI')
    ax = sns.lineplot(x=responsive_freqs['FA2'], y=FA2, label='FAII')
    #ax = sns.scatterplot(x=responsive_freqs['FA2'], y=FA2, label='FAII')
    ax = sns.lineplot(x=responsive_freqs['SA1'], y=SA1, label='SAI')
    #ax = sns.scatterplot(x=responsive_freqs['SA1'], y=SA1, label='SAI')
    ax = sns.lineplot(x=responsive_freqs['SA2'], y=SA2, label='SAII')
    #ax = sns.scatterplot(x=responsive_freqs['SA2'], y=SA2, label='SAII')

    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250],
           title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_averageindividual_AFT_log_noscatter.png", format='png')
    plt.close(fig)

    fig2 = plt.figure(dpi=300)

    ax2 = sns.lineplot(x=responsive_freqs['FA1'], y=FA1, label='FA1')
    ax2 = sns.lineplot(x=responsive_freqs['FA2'], y=FA2, label='FA2')
    ax2 = sns.lineplot(x=responsive_freqs['SA1'], y=SA1, label='SA1')
    ax2 = sns.lineplot(x=responsive_freqs['SA2'], y=SA2, label='SA2')
    ax2.set(ylim=[-0.2, 2], xlim=[0, 250], title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_averageindividual_AFT_LINEAR_.png", format='png')
    plt.close(fig2)

    return model_output


def class_threshold_visualisation(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus

       Arguments:

            absolute (float): Absolute firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus

       Returns:

          Plots thresholds per frequency in log scale.

       """

    SA1 = list()
    FA1 = list()

    SA2 = list()
    FA2 = list()

    origin = freqs

    for runs in range(0, 20):

        class_min = class_absolute_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                              valid_models=valid_models, amps=amps, freqs=origin)

        dict_to_file(dict=class_min, filename="class_min_for" + str(absolute) + "Hz_run_no_"+str(runs), output_path=figpath)

        if runs > 0:

            freqs = np.concatenate((freqs, origin), axis=0)

        for keys in class_min:

            if keys[0] == 'SA1':

                SA1.append(class_min[keys])

            elif keys[0] == 'FA1':

                FA1.append(class_min[keys])

            elif keys[0] == 'SA2':

                SA2.append(class_min[keys])

            elif keys[0] == 'FA2':

                FA2.append(class_min[keys])

    fig = plt.figure(figsize=(10, 10), dpi=300)

    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})

    freqs = np.array(freqs)

    ax = sns.lineplot(x=freqs, y=FA1, label='FAI')
    #ax = sns.scatterplot(x=freqs, y=FA1, label='FAI')
    ax = sns.lineplot(x=freqs, y=FA2, label='FAII')
    #ax = sns.scatterplot(x=freqs, y=FA2, label='FAII')
    ax = sns.lineplot(x=freqs, y=SA1, label='SAI')
    #ax = sns.scatterplot(x=freqs, y=SA1, label='SAI')
    ax = sns.lineplot(x=freqs, y=SA2, label='SAII')
    #ax = sns.scatterplot(x=freqs, y=SA2, label='SAII')

    ax.set(yscale='log', ylim=[10**-3, 10**1], xlim=[0, 250],
           title="Absolute Threshold defined as "+str(absolute)+" Hz")

    plt.savefig(figpath + str(absolute) + "Hz_class_absolute_threshold_log_.png", format='png')
    plt.close(fig)

    fig2 = plt.figure(figsize=(10, 10), dpi=300)

    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})

    ax2 = sns.lineplot(x=freqs, y=FA1, label='FA1')
    ax2 = sns.lineplot(x=freqs, y=FA2, label='FA2')
    ax2 = sns.lineplot(x=freqs, y=SA1, label='SA1')
    ax2 = sns.lineplot(x=freqs, y=SA2, label='SA2')
    ax2.set(ylim=[-0.2, 2], xlim=[0, 250], title="Absolute Threshold defined as "+str(absolute)+" Hz")

    plt.savefig(figpath + str(absolute) + "Hz_class_absolute_threshold_LINEAR_.png", format='png')
    plt.close(fig2)


def class_responsive_amps_visualisation(absolute, filepath, valid_models, amps, figpath, freqs):

    """ Plots the responsive amplitudes of an afferent class for a given set of frequencies and amplitudes of stimulus

           Arguments:

               absolute (float): Absolute firing threshold in Hz
               class_min(dict): Dictionary with the thresholds grouped per afferent class
               figpath(str): Where to save result plots
               freqs(np.array of float): array containing frequencies of stimulus


           Returns:

              Plots thresholds per frequency in log scale.
           """

    class_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath,
                                                            valid_models=valid_models, amps=amps, freqs=freqs)

    SA1 = list()
    FA1 = list()

    SA2 = list()
    FA2 = list()

    SA1_freqs = list()
    FA1_freqs = list()

    SA2_freqs = list()
    FA2_freqs = list()

    for keys in class_responsive_amplitudes:

        if keys[0] == 'SA1':

            for amplitude in class_responsive_amplitudes[keys]:

                SA1.append(amplitude)
                SA1_freqs.append(keys[1])

        elif keys[0] == 'FA1':

            for amplitude in class_responsive_amplitudes[keys]:

                FA1.append(amplitude)
                FA1_freqs.append(keys[1])

        elif keys[0] == 'SA2':

            for amplitude in class_responsive_amplitudes[keys]:

                SA2.append(amplitude)
                SA2_freqs.append(keys[1])

        elif keys[0] == 'FA2':

            for amplitude in class_responsive_amplitudes[keys]:

                FA2.append(amplitude)
                FA2_freqs.append(keys[1])

    fig = plt.figure(dpi=300)

    ax = sns.lineplot(x=FA1_freqs, y=FA1, label='FAI', err_style='bars')
    #ax = sns.scatterplot(x=FA1_freqs, y=FA1)
    ax = sns.lineplot(x=FA2_freqs, y=FA2, label='FAII',err_style='bars')
    #ax = sns.scatterplot(x=FA2_freqs, y=FA2)
    ax = sns.lineplot(x=SA1_freqs, y=SA1, label='SAI',err_style='bars')
    #ax = sns.scatterplot(x=SA1_freqs, y=SA1)
    ax = sns.lineplot(x=SA2_freqs, y=SA2, label='SAII',err_style='bars')
    #ax = sns.scatterplot(x=SA2_freqs, y=SA2)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250],
           title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_class_absolute_threshold_log_BARS.png", format='png')
    plt.close(fig)

    fig2 = plt.figure(dpi=300)

    ax2 = sns.lineplot(x=FA1_freqs, y=FA1, label='FA1')
    ax2 = sns.lineplot(x=FA2_freqs, y=FA2, label='FA2')
    ax2 = sns.lineplot(x=SA1_freqs, y=SA1, label='SA1')
    ax2 = sns.lineplot(x=SA2_freqs, y=SA2, label='SA2')
    ax2.set(ylim=[-0.2, 2], xlim=[0, 250], title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_class_absolute_threshold_LINEAR_.png", format='png')
    plt.close(fig2)


def comparative_scatter_regressions(figpath, comparison, figname, figtitle):

    """ Generate the comparative scatter plots in between footsim outputs and the biological responses for a given set
    of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

         Arguments:

             comparison(dict): Dictionary with both responses
             figpath(str): Where to save result plots
             figname(str): Output file name
             figtitle(str): Output figure suptitle

         Returns:

            Comparative plots
         """

    plotcount = 1
    filename = figname + "r_squared_.txt"
    FAI = dict()
    FAII = dict()
    SAI = dict()
    SAII = dict()

    def reg_angle(coefficient):

        angle = math.atan(regressor.coef_)
        angle = angle * 180 / math.pi

        return (angle)

    #fig = plt.figure(figsize=(25, 12.5), dpi=300)
    fig = plt.figure(figsize=(26, 18), dpi=300)
    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 5,
                                "xtick.labelsize": 15, "ytick.labelsize": 15, "lines.markersize": 10})
    fig.suptitle(figtitle, fontsize=35, y=1)
    file = open(figpath+filename, "w+")

    for keys in comparison:

        if "FAII" in keys:

            FAII[keys] = comparison[keys]

        if "SAII" in keys:

            SAII[keys] = comparison[keys]

    for keys in FAII:

        del comparison[keys]

    for keys in SAII:

        del comparison[keys]

    for keys in comparison:

        if "FAI" in keys:

            FAI[keys] = comparison[keys]

        if "SAI" in keys:

            SAI[keys] = comparison[keys]

    for keys in sorted(FAI):

        if len(FAI[keys][1]) > 0 or len(FAI[keys][0]) > 0:

            plt.subplot(6, 9, plotcount)

            predictors = np.array(FAI[keys][1])
            x = predictors.reshape(-1, 1)

            outcome = np.array(FAI[keys][0])
            y = outcome.reshape(-1, 1)

            # Split 80% of the data to the training set while 20% of the data to test set using below code.

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # Create lists for visualisation

            X_train_list = list()
            X_test_list = list()
            y_train_list = list()
            y_test_list = list()
            y_pred_list = list()

            # Create the model and fit it

            regressor = LinearRegression()

            regressor.fit(X_train, y_train)  # Training the algorithm

            y_pred = regressor.predict(X_test)  # Computing predicted values

            # The coefficients
            print('Coefficient: ', regressor.coef_)

            # The intercept
            print("\nIntercept: ", regressor.intercept_)

            # The mean squared error
            print('\nMean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # The coefficient of determination: 1 is perfect prediction
            print('\nR squared: %.2f' % r2_score(y_test, y_pred))

            lines = keys + " has a R squared of " + str(r2_score(y_test, y_pred))
            file.write(lines)
            file.write('\n')

            for items in range(0, X_test.size):
                X_test_list.append(X_test[items][0])

            for items in range(0, X_train.size):
                X_train_list.append(X_train[items][0])

            for items in range(0, y_test.size):
                y_test_list.append(y_test[items][0])

            for items in range(0, y_train.size):
                y_train_list.append(y_train[items][0])

            for items in range(0, y_pred.size):
                y_pred_list.append(y_pred[items][0])

            ax = sns.scatterplot(x=FAI[keys][1], y=FAI[keys][0], color="blue")  # Plotting the datapoints
            ax = sns.lineplot(x=X_test_list, y=y_pred_list, color="black")  # Adding the regression line
            ax.set_title(keys, pad=10)  # ylim=[0, FAI[keys][0][-1]], xlim=[0, FAI[keys][1][-1]]

            """
            if len(FAI[keys][1]) > len(FAI[keys][0]):

                for firingrates in range(0, len(FAI[keys][1])):

                    print("ei")

            else:

                for firingrates in range(0, len(FAI[keys][0])):

                    print("Oi")


            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if xlim[1] > ylim[1]:

                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, xlim[1]], xlim=[0, xlim[1]])

            else:

                ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, ylim[1]], xlim=[0, ylim[1]])

            """

            plt.xlabel('Actual')
            plt.ylabel('Footsim')
            plotcount = plotcount + 1

    plotcount = plotcount + 3

    for keys in sorted(FAII):

        if len(FAII[keys][1]) > 0 or len(FAII[keys][0]) > 0:

            plt.subplot(6, 9, plotcount)

            predictors = np.array(FAII[keys][1])
            x = predictors.reshape(-1, 1)

            outcome = np.array(FAII[keys][0])
            y = outcome.reshape(-1, 1)

            # Split 80% of the data to the training set while 20% of the data to test set using below code.

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # Create lists for visualisation

            X_train_list = list()
            X_test_list = list()
            y_train_list = list()
            y_test_list = list()
            y_pred_list = list()

            # Create the model and fit it

            regressor = LinearRegression()

            regressor.fit(X_train, y_train)  # Training the algorithm

            y_pred = regressor.predict(X_test)  # Computing predicted values

            # The coefficients
            print('Coefficient: ', regressor.coef_)

            # The intercept
            print("\nIntercept: ", regressor.intercept_)

            # The mean squared error
            print('\nMean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # The coefficient of determination: 1 is perfect prediction
            print('\nR squared: %.2f' % r2_score(y_test, y_pred))

            lines = keys + " has a R squared of " + str(r2_score(y_test, y_pred))
            file.write(lines)
            file.write('\n')

            for items in range(0, X_test.size):
                X_test_list.append(X_test[items][0])

            for items in range(0, X_train.size):
                X_train_list.append(X_train[items][0])

            for items in range(0, y_test.size):
                y_test_list.append(y_test[items][0])

            for items in range(0, y_train.size):
                y_train_list.append(y_train[items][0])

            for items in range(0, y_pred.size):
                y_pred_list.append(y_pred[items][0])

            print(reg_angle(regressor.coef_))

            ax = sns.scatterplot(x=FAII[keys][1], y=FAII[keys][0], color="orange")

            ax = sns.lineplot(x=X_test_list, y=y_pred_list, color="black")
            ax.set_title(keys, pad=10)  # , ylim=[0, FAII[keys][0][-1]], xlim=[0, FAII[keys][1][-1]]

            """

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if xlim[1] > ylim[1]:

                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, xlim[1]], xlim=[0, xlim[1]])

            else:

                ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, ylim[1]], xlim=[0, ylim[1]])

            """

            plt.xlabel('Actual')
            plt.ylabel('Footsim')
            plotcount = plotcount + 1

    plotcount = plotcount + 1

    for keys in sorted(SAI):

        if len(SAI[keys][1]) > 0 or len(SAI[keys][0]) > 0:

            plt.subplot(6, 9, plotcount)

            predictors = np.array(SAI[keys][1])
            x = predictors.reshape(-1, 1)

            outcome = np.array(SAI[keys][0])
            y = outcome.reshape(-1, 1)

            # Split 80% of the data to the training set while 20% of the data to test set using below code.

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # Create lists for visualisation

            X_train_list = list()
            X_test_list = list()
            y_train_list = list()
            y_test_list = list()
            y_pred_list = list()

            # Create the model and fit it

            regressor = LinearRegression()

            regressor.fit(X_train, y_train)  # Training the algorithm

            y_pred = regressor.predict(X_test)  # Computing predicted values

            # The coefficients
            print('Coefficient: ', regressor.coef_)

            # The intercept
            print("\nIntercept: ", regressor.intercept_)

            # The mean squared error
            print('\nMean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # The coefficient of determination: 1 is perfect prediction
            print('\nR squared: %.2f' % r2_score(y_test, y_pred))

            lines = keys + " has a R squared of " + str(r2_score(y_test, y_pred))
            file.write(lines)
            file.write('\n')

            for items in range(0, X_test.size):

                X_test_list.append(X_test[items][0])

            for items in range(0, X_train.size):

                X_train_list.append(X_train[items][0])

            for items in range(0, y_test.size):

                y_test_list.append(y_test[items][0])

            for items in range(0, y_train.size):

                y_train_list.append(y_train[items][0])

            for items in range(0, y_pred.size):

                y_pred_list.append(y_pred[items][0])

            ax = sns.scatterplot(x=SAI[keys][1], y=SAI[keys][0], color="green")

            ax = sns.lineplot(x=X_test_list, y=y_pred_list, color="black")
            ax.set_title(keys, pad=10) # , ylim=[0, SAI[keys][0][-1]], xlim=[0, SAI[keys][1][-1]]

            """

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if xlim[1] > ylim[1]:

                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, xlim[1]], xlim=[0, xlim[1]])

            else:

                ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, ylim[1]], xlim=[0, ylim[1]])

            """
            plt.xlabel('Actual')
            plt.ylabel('Footsim')
            plotcount = plotcount + 1

    plotcount = plotcount + 7

    for keys in sorted(SAII):

        if len(SAII[keys][1]) > 0 or len(SAII[keys][0]) > 0:

            plt.subplot(6, 9, plotcount)

            predictors = np.array(SAII[keys][1])
            x = predictors.reshape(-1, 1)

            outcome = np.array(SAII[keys][0])
            y = outcome.reshape(-1, 1)

            # Split 80% of the data to the training set while 20% of the data to test set using below code.

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # Create lists for visualisation

            X_train_list = list()
            X_test_list = list()
            y_train_list = list()
            y_test_list = list()
            y_pred_list = list()

            # Create the model and fit it

            regressor = LinearRegression()

            regressor.fit(X_train, y_train)  # Training the algorithm

            y_pred = regressor.predict(X_test)  # Computing predicted values

            # The coefficients
            print('Coefficient: ', regressor.coef_)

            # The intercept
            print("\nIntercept: ", regressor.intercept_)

            # The mean squared error
            print('\nMean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # The coefficient of determination: 1 is perfect prediction
            print('\nR squared: %.2f' % r2_score(y_test, y_pred))

            lines = keys + " has a R squared of " + str(r2_score(y_test, y_pred))
            file.write(lines)
            file.write('\n')

            for items in range(0, X_test.size):

                X_test_list.append(X_test[items][0])

            for items in range(0, X_train.size):

                X_train_list.append(X_train[items][0])

            for items in range(0, y_test.size):

                y_test_list.append(y_test[items][0])

            for items in range(0, y_train.size):

                y_train_list.append(y_train[items][0])

            for items in range(0, y_pred.size):

                y_pred_list.append(y_pred[items][0])

            ax = sns.scatterplot(x=SAII[keys][1], y=SAII[keys][0], color="red")

            ax = sns.lineplot(x=X_test_list, y=y_pred_list, color="black")
            ax.set_title(keys, pad=10) # , ylim=[0, SAII[keys][0][-1]], xlim=[0, SAII[keys][1][-1]]

            """

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            if xlim[1] > ylim[1]:

                ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, xlim[1]], xlim=[0, xlim[1]])

            else:

                ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3", color='r')
                ax.set(title=keys, ylim=[0, ylim[1]], xlim=[0, ylim[1]])


            """

            plt.xlabel('Actual')
            plt.ylabel('Footsim')
            plotcount = plotcount + 1

    file.close()

    #fig.subplots_adjust(hspace=1.2)  # wspace=0.5
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='png')
    plt.close(fig)

    FS_classes = dict()

    FS_classes['FAI'] = FAI
    FS_classes['FAII'] = FAII
    FS_classes['SAI'] = SAI
    FS_classes['SAII'] = SAII

    return FS_classes


def plot_duplicates_FAI():

    """ Housekeeping function that plot duplicate FAI models

     Returns:

        Histogram for FAIs.

     """

    uniques, count = np.unique(fs.constants.affidFA1, return_counts=True)
    duplicates_dict = dict(zip(uniques, count))

    plt.bar(*zip(*sorted(duplicates_dict.items())), color='g')
    plt.xlabel('Afferent ID')
    plt.ylabel('Occurrences')
    plt.title('FAIs')
    plt.xticks(rotation=80)
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.savefig('C://Users//pc1rss//Desktop//FAIs.png', format='png')


def FR_vs_hardness(figpath, filepath, valid_models, amps, freqs):

    """ Generate the comparative scatter plots in between footsim firing rate outputs and the skin hardness of the
        regions where they were placed

             Arguments:

             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
             populations generation, numbers should match the ones in the *.csv if reproducing experimental data
             amps(np.array of float): array containing amplitudes of stimulus
             freqs(np.array of float): array containing frequencies of stimulus


             Returns:

                Dictionary of responsive amplitudes for afferents grouped per region, and the plots of thresholds x
                frequency in log scale.
         """


    SA1 = list()
    SA2 = list()
    FA1 = list()
    FA2 = list()

    SA1_hardness = list()
    SA2_hardness = list()
    FA1_hardness = list()
    FA2_hardness = list()

    populations = empirical_afferent_positioning(filepath, valid_models)  # Generates investigated population

    footsim_rates = model_firing_rates(populations=populations, amps=amps, freqs=freqs)  # Gathers firing rates
    footsim_rates = FR_filtering(filepath=filepath, populations=populations, model_firing_rates=footsim_rates,
                                 valid_models=valid_models)  # Filters unused stimulus

    for keys in footsim_rates:

        for t in range(0, footsim_rates[keys].size):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class

            if afferent_class == 'SA1':

                SA1.append(footsim_rates[keys][t][0])
                SA1_hardness.append(fs.additional.hardnessRegion(str(keys[2])))

            elif afferent_class == 'FA1':

                FA1.append(footsim_rates[keys][t][0])
                FA1_hardness.append(fs.additional.hardnessRegion(str(keys[2])))

            elif afferent_class == 'SA2':

                SA2.append(footsim_rates[keys][t][0])
                SA2_hardness.append(fs.additional.hardnessRegion(str(keys[2])))

            elif afferent_class == 'FA2':

                FA2.append(footsim_rates[keys][t][0])
                FA2_hardness.append(fs.additional.hardnessRegion(str(keys[2])))

#    x = np.array(SA1_hardness).reshape((-1, 1))
#    y = np.array(SA1)
#   model = LinearRegression().fit(x, y)
#    y_pred = model.predict(x)

    fig1 = plt.figure(dpi=500)
#    plt.plot(x, y_pred, color='b')
#    plt.xlim([20, 50])
    plt.title("Effects of Hardness on FootSim FRs")
#    plt.xlabel("Hardness [a.u.]")
#    plt.ylabel("Firing rate (Hz)")

    ax = sns.lineplot(y=np.array(FA1), x=np.array(FA1_hardness), label='FA1')
    ax = sns.lineplot(y=np.array(FA2), x=np.array(FA2_hardness), label='FA2')
    ax = sns.lineplot(y=np.array(SA1), x=np.array(SA1_hardness), label='SA1')
    ax = sns.lineplot(y=np.array(SA2), x=np.array(SA2_hardness), label='SA2')
    ax.set(xlim=[20, 50], ylim=[0, 80], ylabel="Firing rate (Hz)", xlabel="Hardness (a.u.)")

    figname = "(Footsim)FR_vs_hardness_lineplot.png"
    plt.savefig(figpath+figname, format='png')


def hard_vs_soft_sole_regions(figpath, absolute, regional_min):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus
        per region compared with regional hardness grouping by hard or soft regions

             Arguments:

                 regional_min(dict): Dictionary with the thresholds grouped per region of interest
                 figpath(str): Where to save result plots


             Returns:

                Plots thresholds per frequency in log scale.
             """

    SA1_thard = list()
    SA2_thard = list()
    FA1_thard = list()
    FA2_thard = list()

    SA1_tsoft = list()
    SA2_tsoft = list()
    FA1_tsoft = list()
    FA2_tsoft = list()

    SA1_hard = list()
    SA2_hard = list()
    FA1_hard = list()
    FA2_hard = list()

    SA1_soft = list()
    SA2_soft = list()
    FA1_soft = list()
    FA2_soft = list()


    for keys in regional_min:

        if keys[0] == 'SA1':

            if fs.additional.hardnessRegion(str(keys[1])) < 31.229:

                SA1_tsoft.append(regional_min[keys])
                SA1_soft.append(fs.additional.hardnessRegion(str(keys[1])))

            else:

                SA1_thard.append(regional_min[keys])
                SA1_hard.append(fs.additional.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA1':

            if fs.additional.hardnessRegion(str(keys[1])) < 31.229:

                FA1_tsoft.append(regional_min[keys])
                FA1_soft.append(fs.additional.hardnessRegion(str(keys[1])))

            else:

                FA1_thard.append(regional_min[keys])
                FA1_hard.append(fs.additional.hardnessRegion(str(keys[1])))

        elif keys[0] == 'SA2':

            if fs.additional.hardnessRegion(str(keys[1])) < 31.229:

                SA2_tsoft.append(regional_min[keys])
                SA2_soft.append(fs.additional.hardnessRegion(str(keys[1])))

            else:

                SA2_thard.append(regional_min[keys])
                SA2_hard.append(fs.additional.hardnessRegion(str(keys[1])))

        if keys[0] == 'FA2':

            if fs.additional.hardnessRegion(str(keys[1])) < 31.229:

                FA2_tsoft.append(regional_min[keys])
                FA2_soft.append(fs.additional.hardnessRegion(str(keys[1])))

            else:

                FA2_thard.append(regional_min[keys])
                FA2_hard.append(fs.additional.hardnessRegion(str(keys[1])))

    # quick stats

    resultFA1 = scipy.stats.ttest_ind(FA1_thard, FA1_tsoft, equal_var=False, nan_policy='omit')

    if resultFA1.pvalue < 0.05:

        print("Significant for FA1")

    resultFA2 = scipy.stats.ttest_ind(FA2_thard, FA2_tsoft, equal_var=False, nan_policy='omit')

    if resultFA2.pvalue < 0.05:

        print("Significant for FA2")

    resultSA1 = scipy.stats.ttest_ind(SA1_thard, SA1_tsoft, equal_var=False, nan_policy='omit')

    if resultSA1.pvalue < 0.05:

        print("Significant for SA1")

    resultSA2 = scipy.stats.ttest_ind(SA2_thard, SA2_tsoft, equal_var=False, nan_policy='omit')

    if resultSA2.pvalue < 0.05:

        print("Significant for SA2")

    return resultFA1, resultFA2, resultSA1, resultSA2

    """
    # generating seaborn plots
    data = {'Hard': FA1_thard, 'Soft': FA1_tsoft}
    sorted_keys, sorted_vals = zip(*sorted(data.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals,  width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="FA1", xlabel="Groups", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "FA1_Hard_vs_Soft_box_.png", format='png')

    data = {'Hard': FA2_thard, 'Soft': FA2_tsoft}
    sorted_keys, sorted_vals = zip(*sorted(data.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="FA2", xlabel="Groups", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "FA2_Hard_vs_Soft_box_.png", format='png')

    data = {'Hard': SA1_thard, 'Soft': SA1_tsoft}
    sorted_keys, sorted_vals = zip(*sorted(data.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="SA1", xlabel="Groups", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "SA1_Hard_vs_Soft_box_.png", format='png')

    data = {'Hard': SA2_thard, 'Soft': SA2_tsoft}
    sorted_keys, sorted_vals = zip(*sorted(data.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="SA2", xlabel="Groups", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "SA2_Hard_vs_Soft_box_.png", format='png')
    
    """


def modelFR_visualisation(figpath, FRs):

    fig = plt.figure(figsize=(10, 5), dpi=300)
    fig.suptitle("Individual Models - Firing Rates", fontsize=12)

    plotcount = 1

    for keys in FRs.keys():

        plt.subplot(5, 9, plotcount)
        plotcount = plotcount + 1

        ax = sns.scatterplot(data=np.array(FRs[keys][0]))
        ax.set(title=str(keys))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath + "Footsim_FRs.png", format='png')


def hardness_vs_thresholds_visualisation(figpath, absolute, regional_min):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus
    per region compared with regional hardness

         Arguments:

             regional_min(dict): Dictionary with the thresholds grouped per region of interest
             figpath(str): Where to save result plots


         Returns:

            Plots thresholds per frequency in log scale.
         """


    SA1 = list()
    SA2 = list()
    FA1 = list()
    FA2 = list()

    SA1_hardness = list()
    SA2_hardness = list()
    FA1_hardness = list()
    FA2_hardness = list()

    for keys in regional_min:

        if keys[0] == 'SA1':

            SA1.append(regional_min[keys])
            SA1_hardness.append(fs.additional.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA1':

            FA1.append(regional_min[keys])
            FA1_hardness.append(fs.additional.hardnessRegion(str(keys[1])))

        elif keys[0] == 'SA2':

            SA2.append(regional_min[keys])
            SA2_hardness.append(fs.additional.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA2':

            FA2.append(regional_min[keys])
            FA2_hardness.append(fs.additional.hardnessRegion(str(keys[1])))

    fig1 = plt.figure(dpi=500)
    #FA1 = np.array(FA1)
    #FA1_hardness = np.array(FA1_hardness)
    ax = sns.lineplot(y=FA1, x=FA1_hardness, label='FA1')
    #    ax1.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[20, 50])
    #    plt.savefig(figpath+"FA1.png", format='png')

    #    fig2 = plt.figure(dpi=150)
    ax = sns.lineplot(y=np.array(FA2), x=np.array(FA2_hardness), label='FA2')
    #    ax2.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[20, 50])
    #    plt.savefig(figpath+"FA2.png", format='png')

    #    fig3 = plt.figure(dpi=150)
    ax = sns.lineplot(y=np.array(SA1), x=np.array(SA1_hardness), label='SA1')
    #    ax3.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[20, 50])
    #    plt.savefig(figpath+"SA1.png", format='png')

    #    fig4 = plt.figure(dpi=150)
    ax = sns.lineplot(y=np.array(SA2), x=np.array(SA2_hardness), label='SA2')
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[20, 50])
    plt.legend()
    plt.savefig(figpath + "(FS)Hardness_threshold_of_" + str(absolute) + "_Hz_.png", format='png')


def ImpCycle_visualisation(ImpPath, figpath):

    data_file = pd.read_excel(ImpPath)

    print('Your data table has the following shape "(lines, columns)":', data_file.shape, "\n")

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    table_affs = ['FA1', 'FA2', 'SA1', 'SA2']

    Imps = dict()

    regions = ['Heel', 'LatArch', 'MedArch', 'MidArch', 'LatMet', 'MidMet', 'MedMet', 'Toes']

    grouped_by_threshold = data_file.copy(deep=True)

    fig = plt.figure(figsize=(10, 10), dpi=300)

    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})

    for aff in range(0, 4):  # Amplitude in Log scale

        afferent_class = grouped_by_threshold[grouped_by_threshold.type.str.contains(table_affs[aff])].copy(deep=True)

        if len(afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1)) > 0:

            Imps[str(table_affs[aff])] = np.array(afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1).dropna())

            ax = sns.lineplot(data=afferent_class, x=afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1),
                              y=afferent_class['Amplitude'].where(afferent_class['ImpCycle'] == 1),
                              label=afferents[aff], ci='sd')

            ax = sns.scatterplot(data=afferent_class,
                                 x=afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1),
                                 y=afferent_class['Amplitude'].where(afferent_class['ImpCycle'] == 1),
                                 label=afferents[aff])

            ax.set(ylim=[10**-3, 10**1], xlim=[0, 250], title="ImpCycle = 1", yscale="log")

    print(Imps)
    figure = figpath + "Impcycle_of_TFits_withscatter_Jul20.png"
    plt.legend(loc='upper right')
    plt.savefig(figure)

    return Imps


def individual_models_threshold_visualisation(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
    stimulus

        Arguments:

        absolute (float): Absolute firing threshold in Hz
        filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
        afferent population is to mimic some experimental data
        figpath(str): Where to save result plots
        valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
        populations generation, numbers should match the ones in the *.csv if reproducing experimental data
        amps(np.array of float): array containing amplitudes of stimulus
        freqs(np.array of float): array containing frequencies of stimulus

         Returns:

            Plots thresholds per frequency in log scale.
         """
    SA1 = dict()
    SA2 = dict()

    FA1 = dict()
    FA2 = dict()

    responsive_freqs = dict()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'H']

    individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                                  valid_models=valid_models, amps=amps, freqs=freqs)

    for keys in individual_min:  # Creating the lists #

        if keys[0] == 'SA1':

            responsive_freqs[keys[0], keys[1], keys[2]] = list()
            SA1[keys[1], keys[2]] = list()

        elif keys[0] == 'FA1':

            responsive_freqs[keys[0], keys[1], keys[2]] = list()
            FA1[keys[1], keys[2]] = list()

        elif keys[0] == 'SA2':

            responsive_freqs[keys[0], keys[1], keys[2]] = list()
            SA2[keys[1], keys[2]] = list()

        elif keys[0] == 'FA2':

            responsive_freqs[keys[0], keys[1], keys[2]] = list()
            FA2[keys[1], keys[2]] = list()

    for runs in range(0, 2):

        individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                                      valid_models=valid_models, amps=amps, freqs=freqs)

        for keys in individual_min:

            if keys[0] == 'SA1':

                for loc in range(0, len(foot_locations)):

                    if keys[2] == foot_locations[loc]:

                        for idx in range(fs.constants.affparams['SA1'].shape[0]):

                            if keys[1] == idx:

                                SA1[idx, foot_locations[loc]].append(individual_min[keys])
                                responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

            elif keys[0] == 'FA1':

                for loc in range(0, len(foot_locations)):

                    if keys[2] == foot_locations[loc]:

                        for idx in range(fs.constants.affparams['FA1'].shape[0]):

                            if keys[1] == idx:

                                FA1[idx, foot_locations[loc]].append(individual_min[keys])
                                responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

            elif keys[0] == 'SA2':

                for loc in range(0, len(foot_locations)):

                    if keys[2] == foot_locations[loc]:

                        for idx in range(fs.constants.affparams['SA2'].shape[0]):

                            if keys[1] == idx:

                                SA2[idx, foot_locations[loc]].append(individual_min[keys])
                                responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

            elif keys[0] == 'FA2':

                for loc in range(0, len(foot_locations)):

                    if keys[2] == foot_locations[loc]:

                        for idx in range(fs.constants.affparams['FA2'].shape[0]):

                            if keys[1] == idx:

                                FA2[idx, foot_locations[loc]].append(individual_min[keys])
                                responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

    # ----------------------------------------- #

    # Generating Figures #

    # ----------------------------------------- #

    fig = plt.figure(figsize=(15, 5), dpi=300)
    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig.suptitle("FootSim", fontsize=22, y=1)

    sns.set_palette("Blues_d")
    plt.subplot(1, 4, 1)

    for keys in FA1:

        ax = sns.lineplot(x=np.array(responsive_freqs['FA1', keys[0], keys[1]]), y=np.array(FA1[keys])) #label="FA1_"+str(keys)
        ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250], title='FA1', xlabel='Frequency', ylabel='Amplitude') # xscale='log'
        ax.tick_params(width=0.3)
        #ax.legend(fontsize=4, loc=1)

    sns.set_palette("Reds")
    plt.subplot(1, 4, 2)


    for keys in FA2:

        ax2 = sns.lineplot(x=np.array(responsive_freqs['FA2', keys[0], keys[1]]), y=np.array(FA2[keys])) #label="FA2_"+str(keys)
        ax2.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250], title='FA2', xlabel='Frequency')
        ax2.tick_params(width=0.3)
        #ax2.legend(fontsize=6, loc=1)

    sns.set_palette("Greens_r")
    plt.subplot(1, 4, 3)

    for keys in SA1:

        ax3 = sns.lineplot(x=np.array(responsive_freqs['SA1', keys[0], keys[1]]), y=np.array(SA1[keys])) #label="SA1_"+str(keys)
        ax3.set(yscale='log', ylim=[10**-3, 10**1], xlim=[0, 250], title='SA1', xlabel='Frequency')
        ax3.tick_params(width=0.3)
        #ax3.legend(fontsize=7.5, loc=1)

    sns.set_palette("Greys_r")
    plt.subplot(1, 4, 4)

    for keys in SA2:

        ax4 = sns.lineplot(x=np.array(responsive_freqs['SA2', keys[0], keys[1]]), y=np.array(SA2[keys])) #label="SA2_"+str(keys)
        ax4.set(yscale='log', ylim=[10**-3, 10**1], xlim=[0, 250], title='SA2',  xlabel='Frequency')
        ax4.tick_params(width=0.3)
        #ax4.legend(fontsize=10, loc=1)

    fig.tight_layout()
    plt.savefig(figpath+"FS_Individual_AFTs_"+str(absolute)+"_Hz.png", format='png')

    grouped_thresholds = dict()  # Groups the results per afferent class
    grouped_thresholds['FA1'] = FA1
    grouped_thresholds['FA2'] = FA2
    grouped_thresholds['SA1'] = SA1
    grouped_thresholds['SA2'] = SA2

    return grouped_thresholds


def plot_responses(figpath, amp, freq, location, response):

    if len(response.spikes) > 0:

        fig = plt.figure(dpi=500)
        r = plot(response)
        figsave(hvobj=r, size=400,
                filename=figpath+"R_for_"+str(location)+"_at_"+str(amp)+"_mm_"+str(freq)+"_Hz")
        plt.close(fig)


def regional_threshold_visualisation(figpath, regional_min):

    """ Plots the absolute thresholds of afferent classes for a given set of frequencies and amplitudes of
        stimulus in the region where they were placed

             Arguments:

                 regional_min(dict): Dictionary with the thresholds grouped per region
                 figpath(str): Where to save result plots

             Returns:

                Plots thresholds per frequency in log scale.
             """

    SA1 = dict()
    SA2 = dict()

    FA1 = dict()
    FA2 = dict()

    responsive_freqs = dict()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'H']

    for loc in range(0, len(foot_locations)):

        responsive_freqs['SA1', foot_locations[loc]] = list()
        SA1[foot_locations[loc]] = list()

        responsive_freqs['SA2', foot_locations[loc]] = list()
        SA2[foot_locations[loc]] = list()

        responsive_freqs['FA1', foot_locations[loc]] = list()
        FA1[foot_locations[loc]] = list()

        responsive_freqs['FA2', foot_locations[loc]] = list()
        FA2[foot_locations[loc]] = list()

    for keys in regional_min:

        if keys[0] == 'SA1':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    SA1[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['SA1', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'FA1':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    FA1[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['FA1', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'SA2':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    SA2[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['SA2', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'FA2':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    FA2[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['FA2', foot_locations[loc]].append(keys[2])

    # ----------------------------------------- #

    # Generating Figures #

    # ----------------------------------------- #

    fig = plt.figure(dpi=150)

    for loc in range(0, len(foot_locations)):

        if foot_locations[loc] in FA1:

            ax = sns.lineplot(x=np.array(responsive_freqs['FA1', foot_locations[loc]]),
                              y=np.array(FA1[foot_locations[loc]]), label='FA1_' + foot_locations[loc])
            ax.set(yscale='log', ylim=[10**-3,10**1], xlim=[0, 250])

    plt.savefig(figpath+"FA1.png", format='png')
    fig2 = plt.figure(dpi=150)

    for loc in range(0, len(foot_locations)):

        if foot_locations[loc] in FA2:

            ax2 = sns.lineplot(x=np.array(responsive_freqs['FA2', foot_locations[loc]]),
                              y=np.array(FA2[foot_locations[loc]]), label='FA2_' + foot_locations[loc])
            ax2.set(yscale='log', ylim=[10**-3,10**1], xlim=[0, 250])

    plt.savefig(figpath+"FA2.png", format='png')
    fig3 = plt.figure(dpi=150)

    for loc in range(0, len(foot_locations)):

        if foot_locations[loc] in SA1:

            ax3 = sns.lineplot(x=np.array(responsive_freqs['SA1', foot_locations[loc]]),
                              y=np.array(SA1[foot_locations[loc]]), label='SA1_' + foot_locations[loc])
            ax3.set(yscale='log',  ylim=[10**-3,10**1], xlim=[0, 250])

    plt.savefig(figpath+"SA1.png", format='png')
    fig4 = plt.figure(dpi=150)

    for loc in range(0, len(foot_locations)):

        if foot_locations[loc] in SA2:

            ax4 = sns.lineplot(x=np.array(responsive_freqs['SA2', foot_locations[loc]]),
                               y=np.array(SA2[foot_locations[loc]]), label='SA2_' + foot_locations[loc])
            ax4.set(yscale='log', ylim=[10**-3,10**1], xlim=[0, 250])

    plt.savefig(figpath+"SA2.png", format='png')


def regional_threshold_boxplots(figpath, regional_min, absolute):

    """ Plots regional box plots of the absolute thresholds of afferent classes for a given set of frequencies and
    amplitudes of stimulus in the region where they were placed

             Arguments:

                 regional_min(dict): Dictionary with the thresholds grouped per region
                 figpath(str): Where to save result plots

             Returns:

                BoxPlots of thresholds per region in log scale.
             """

    SA1 = dict()
    SA2 = dict()

    FA1 = dict()
    FA2 = dict()

    responsive_freqs = dict()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'HL', 'HR']

    for loc in range(0, len(foot_locations)):

        responsive_freqs['SA1', foot_locations[loc]] = list()
        SA1[foot_locations[loc]] = list()

        responsive_freqs['SA2', foot_locations[loc]] = list()
        SA2[foot_locations[loc]] = list()

        responsive_freqs['FA1', foot_locations[loc]] = list()
        FA1[foot_locations[loc]] = list()

        responsive_freqs['FA2', foot_locations[loc]] = list()
        FA2[foot_locations[loc]] = list()

    for keys in regional_min:

        if keys[0] == 'SA1':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    SA1[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['SA1', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'FA1':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:

                    FA1[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['FA1', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'SA2':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    SA2[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['SA2', foot_locations[loc]].append(keys[2])

        elif keys[0] == 'FA2':

            for loc in range(0, len(foot_locations)):

                if keys[1] == foot_locations[loc]:
                    FA2[foot_locations[loc]].append(regional_min[keys])
                    responsive_freqs['FA2', foot_locations[loc]].append(keys[2])

    # generating figures

    sorted_keys, sorted_vals = zip(*sorted(SA1.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="SA1", xlabel="Region", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "SA1_" +str(absolute)+"Hz_regional_boxes_.png", format='png')

    # -- #

    sorted_keys, sorted_vals = zip(*sorted(FA1.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="FA1", xlabel="Region", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "FA1" +str(absolute)+"Hz_regional_boxes_.png", format='png')

    # -- #

    sorted_keys, sorted_vals = zip(*sorted(SA2.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="SA2", xlabel="Region", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "SA2_" +str(absolute)+"Hz_regional_boxes_.png", format='png')

    # -- #

    sorted_keys, sorted_vals = zip(*sorted(FA2.items(), key=op.itemgetter(1)))

    fig = plt.figure(dpi=500)

    ax = sns.boxplot(data=sorted_vals, width=.4)
    sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], title="FA2", xlabel="Region", ylabel="Threshold")
    sns.set(context='notebook', style='whitegrid')

    # category labels
    plt.xticks(plt.xticks()[0], sorted_keys)

    plt.savefig(figpath + "FA2_" +str(absolute)+"Hz_regional_boxes_.png", format='png')


def RMSE_barplot(figpath, all_rms):

    """ Generates a bar plot with RMSE values from the comparison of footsim and empirical data for all afferent classes

         Arguments:

             all_rms(list): List with the root mean square values in the correct order

         Returns:

            Comparative plots
         """
    afferents_footsim = ['FA1', 'FA2', 'SA1', 'SA2']

    sns.barplot(x=afferents_footsim, y=all_rms)
    #plt.bar(afferents_footsim, all_rms, color="m")
    plt.title("RMSE per Afferent Class")
    plt.ylim((0, 2))

    plt.savefig(figpath + "RMSE_TFits.png", format='png')


def RMSE_lineplot(figpath, allclasses):

    """ Generates a line plot with RMSE values from the comparison of footsim and empirical data for all afferent classes

         Arguments:

             allclasses(dict): Dictionary with afferent classes as keys and lists of RMSEs as values

         Returns:

            Comparative plots
         """

    xaxis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    fig = plt.figure(dpi=500)

    ax = sns.lineplot(x=xaxis, y=allclasses['FA1'], label='FA1')
    ax = sns.lineplot(x=xaxis, y=allclasses['FA2'], label='FA2')
    ax = sns.lineplot(x=xaxis, y=allclasses['SA1'], label='SA1')
    ax = sns.lineplot(x=xaxis, y=allclasses['SA2'], label='SA2')

    plt.savefig(figpath + "RMSE_linear.png", format='png')
    plt.close(fig)


def single_afferent_threshold_visualisation(figpath, affclass, idx, single_min):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
        stimulus

             Arguments:

                 individual_min(dict): Dictionary with the thresholds grouped per afferent class
                 figpath(str): Where to save result plots

             Returns:

                Plots thresholds per frequency in log scale.
                         """

    single_model = dict()
    responsive_freqs = dict()

    foot_locations = ['T1', 'T2_t', 'T3_t', 'T4_t', 'MMi', 'MMe', 'T5_t', 'MLa', 'AMe', 'AMi', 'ALa', 'H']

    for loc in range(0, len(foot_locations)):

        single_model[foot_locations[loc]] = list()
        responsive_freqs[foot_locations[loc]] = list()

    for keys in single_min.keys():

        if type(single_min[keys]) is not list:

            for location in range(0, len(foot_locations)):

                if keys[2] == foot_locations[location]:

                    single_model[keys[2]].append(single_min[keys])
                    responsive_freqs[keys[2]].append(keys[3])

        # ----------------------------------------- #

        # Generating Figures #

        # ----------------------------------------- #

    fig = plt.figure(dpi=150)

    for keys in single_model:

        if len(single_model[keys]) > 0:

            ax = sns.lineplot(x=np.array(responsive_freqs[keys]), y=np.array(single_model[keys]), label=str(keys))
            ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250], title=affclass + " model IDX " + str(idx))
            ax.legend(fontsize=4, loc=1)

    figure = figpath + affclass + "_model_IDX_" + str(idx) + "_npmean_.png"

    plt.savefig(figure, format='png')


# ----------------------------------------- #

# Statistical calculations and comparisons #

# ----------------------------------------- #

def delete_model_duplicates():

    """ Temporary function to clear duplicates from the model after the November 2020 update

        Returns:

           Cleaner model.

        """

    FAI_pos_array = np.array([30, 31, 20, 2, 16, 26, 21, 22, 23, 19, 27, 28, 3, 17, 32, 25])  # Idx of the duplicates
    FAI_pos_array = np.sort(FAI_pos_array)  # sorting the positions

    FAII_pos_array = np.array([2, 10, 17, 13, 12, 15, 16, 3, 11, 16])
    FAII_pos_array = np.sort(FAII_pos_array)

    SAI_pos_array = np.array([1, 9, 18, 14, 15, 11, 13, 19, 12])
    SAI_pos_array = np.sort(SAI_pos_array)

    SAII_pos_array = np.array([9, 4, 7])
    SAII_pos_array = np.sort(SAII_pos_array)

    return 0

def FR_model_vs_empirical(figpath, filepath, valid_models, amps, freqs):

    """ Compares Footsim outputs and the original biological responses for a given set of amplitudes and frequencies of
     stimulus.

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents
            figpath(str): Where to save result plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Dictionary of firing rates for each stimulus for both footsim and empirical data

        """

    start = time.asctime(time.localtime(time.time()))

    populations = empirical_afferent_positioning(filepath=filepath, valid_models=valid_models)  # Generates the affpop
    footsim_rates = model_firing_rates(populations=populations, amps=amps, freqs=freqs)  # Investigate fs firing rates

    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data
    grouped_by_threshold = data_file.copy(deep=True)  # Gets the empirical thresholds if needed

    comparison = dict()  # Dictionary that will get all the rates from the model and the empirical dataset

    for keys in footsim_rates:

        for t in range(0, footsim_rates[keys].size):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class
            idx = populations[keys[2]][afferent_c].idx  # Gathers its model number

            comparison[valid_models[afferent_class][idx]] = list()  # Position 0 for Footsim, 1 for Empirical
            comparison[valid_models[afferent_class][idx]].append(list())
            comparison[valid_models[afferent_class][idx]].append(list())


    for keys in footsim_rates:

        for t in range(0, footsim_rates[keys].size):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class
            idx = populations[keys[2]][afferent_c].idx  # Gathers its model number

            # Checking if the spreadsheed of empirical data has said model #

            if grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(valid_models[afferent_class][idx])].empty == False:

                # If yes, isolate that afferent #

                data_slice = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(valid_models[afferent_class][idx])].copy(deep=True)

                if data_slice[data_slice['Frequency'] == keys[1]].empty == True:

                    continue

                else:

                    # Isolate the specific stimulus used for that afferent #

                    empirical_fr = data_slice[data_slice['Frequency'] == keys[1]].copy(deep=True)

                    empirical_fr = empirical_fr[empirical_fr['Amplitude'] == keys[0]].copy(deep=True)

                    if empirical_fr.empty == True:

                        continue

                    else:

                        comparison[valid_models[afferent_class][idx]][0].append(footsim_rates[keys][t][0])

                        comparison[valid_models[afferent_class][idx]][1].append(empirical_fr.iloc[0]['AvgInst'])

    print("Simulation started at ", start)
    print("Simulation finished at ", time.asctime(time.localtime(time.time())))

    FS_classes_comparative = comparative_scatter_regressions(figpath=figpath, comparison=comparison,
                                                             figname="FR_comparison_reg.png",
                                                             figtitle="Comparative Firing Rates")

    residuals = find_residuals(FR_comparative=FS_classes_comparative, figpath=figpath, figname='Residuals_FR_')

    return residuals


def hardness_vs_multiple_thresholds(filepath, figpath, valid_models, amps, freqs):

    for absolute in range(1, 25):

        print("Working threshold definition of ", absolute, " Hz.")

        regional_min = regional_absolute_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                                    valid_models=valid_models,
                                                    amps=amps, freqs=freqs)

        hardness_vs_thresholds_visualisation(figpath=figpath, absolute=absolute, regional_min=regional_min)


def ImpCycle_model_vs_empirical(figpath, filepath, valid_models, amps, freqs):

    start = time.asctime(time.localtime(time.time()))

    populations = empirical_afferent_positioning(filepath, valid_models)  # Generates the fs afferent population
    footsim_ImpCycle = ImpCycle(figpath=figpath, filepath=filepath, valid_models=valid_models, amps=amps, freqs=freqs)

    ImpCycle_csvexport(figpath=figpath, ImpCycle=footsim_ImpCycle, population=populations, valid_models=valid_models)

    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data
    grouped_by_threshold = data_file.copy(deep=True)  # Gets the empirical thresholds if needed

    comparison = dict()  # Dictionary that will get all the rates from the model and the empirical dataset

    for keys in footsim_ImpCycle:

        for ImpCycle_value in range(0, footsim_ImpCycle[keys].size):

            location = keys[2]
            affclass = populations[location].afferents[ImpCycle_value].affclass
            model_idx = populations[location].afferents[ImpCycle_value].idx  # Gathers its model number

            comparison[valid_models[affclass][model_idx]] = list()  # Position 0 for Footsim, 1 for Empirical
            comparison[valid_models[affclass][model_idx]].append(list())
            comparison[valid_models[affclass][model_idx]].append(list())

    for keys in footsim_ImpCycle:

        for t in range(0, len(footsim_ImpCycle[keys])):

            location = keys[2]
            affclass = populations[location].afferents[t].affclass
            model_idx = populations[location].afferents[t].idx  # Gathers its model number

            if grouped_by_threshold[
                grouped_by_threshold.Afferent_ID.str.contains(valid_models[affclass][model_idx])].empty == False:

                data_slice = grouped_by_threshold[
                    grouped_by_threshold.Afferent_ID.str.contains(valid_models[affclass][model_idx])].copy(deep=True)

                empirical_fr = data_slice[data_slice['Frequency'] == keys[1]]  # Gets the stimulus frequency

                if empirical_fr.empty == True:

                    continue

                else:

                    empirical_fr = empirical_fr[empirical_fr['Amplitude'] == keys[0]]  # Gets the stimulus amplitude

                    if empirical_fr.empty == True:

                        continue

                    else:

                        comparison[valid_models[affclass][model_idx]][0].append(footsim_ImpCycle[keys][t][0])
                        comparison[valid_models[affclass][model_idx]][1].append(empirical_fr.iloc[0]['ImpCycle'])

    print("Simulation started at ", start)
    print("Simulation finished at ", time.asctime(time.localtime(time.time())))

    comparative_scatter_regressions(figpath=figpath, comparison=comparison, figname="ImpCycle_comparison_reg.png",
                                    figtitle="Comparative Impulses per cycle")

    return comparison


def list_duplicates(seq):

    """ Housekeeping function that list duplicate afferent models in a list

    Example of use: for dup in sorted(list_duplicates(fs.constants.affidFA1)):
                        print(dup)

         Returns:

            Afferent IDs and indexes of duplicates.

         """

    tally = defaultdict(list)

    for i, item in enumerate(seq):

        tally[item].append(i)

    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def multiple_RMSEs(filepath, figpath, valid_models, amps, freqs):

    """ Computes footsim thresholds with empirical data for a range of firing thresholds from 1 to 25 Hz

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save the plots
            valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Dictionary with afferent classes as keys and RMSE values as entries
           """

    FA1 = list()
    FA2 = list()
    SA1 = list()
    SA2 = list()

    for absolute in range(0, 25):

        all_rms = RMSE_allclasses(absolute, filepath, figpath, valid_models, amps, freqs)

        FA1.append(all_rms[0])
        FA2.append(all_rms[1])
        SA1.append(all_rms[2])
        SA2.append(all_rms[3])

    allclasses = {'FA1': FA1, 'FA2': FA2, 'SA1': SA1, 'SA2': SA2}

    RMSE_lineplot(figpath, allclasses)
    dict_to_file(dict=allclasses, filename="RMSES", output_path=figpath)

    return allclasses


def find_residuals(FR_comparative, figpath, figname):

    """ Calculates the comparative residuals in between footsim outputs and the biological responses for a given set
        of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

             Arguments:

                 FR_comparative(dict): Dictionary with both responses
                 figpath(str): Where to save result plots
                 figname(str): Output file name
                 figtitle(str): Output figure suptitle

             Returns:

                Comparative plots
             """

    # Declaring variables #

    all_rates = 0
    class_residuals = dict()
    class_sum_of_residuals = dict()
    all_classes_dict = dict()


    sum_of_residuals = dict()
    average_residual = dict()

    # Loops through afferent classes #

    for FS_class in FR_comparative:

        class_residuals[FS_class] = list()
        sum_of_residuals[FS_class] = dict()
        class_sum_of_residuals[FS_class] = 0

        # Loops through individual afferents #

        individualaff_averageresidual = list()

        for individual_afferent in FR_comparative[FS_class]:

            sum_of_residuals[FS_class][individual_afferent] = 0

            # Loops through responses of individual afferent to amplitude/frequency pairs #

            for firing_rate in range(0, len(FR_comparative[FS_class][individual_afferent][0])):

                individual_rates = len(FR_comparative[FS_class][individual_afferent][0])

                all_rates = all_rates + individual_rates

                residual = FR_comparative[FS_class][individual_afferent][0][firing_rate] -\
                           FR_comparative[FS_class][individual_afferent][1][firing_rate]

                sum_of_residuals[FS_class][individual_afferent] = sum_of_residuals[FS_class][individual_afferent] + abs(residual)

                class_sum_of_residuals[FS_class] = class_sum_of_residuals[FS_class] + abs(residual)

                class_residuals[FS_class].append(residual)

            individualaff_averageresidual.append(sum_of_residuals[FS_class][individual_afferent] / individual_rates)

        average_residual[FS_class] = class_sum_of_residuals[FS_class] / all_rates

        individualaff_averageresidual = np.array(individualaff_averageresidual)
        all_classes_dict[FS_class] = individualaff_averageresidual

        fig = plt.figure(dpi=200)
        sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "lines.linewidth": 5,
                                    "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.markersize": 5})

        ax = sns.scatterplot(data=individualaff_averageresidual)
        plt.xlabel('Residuals')
        plt.ylabel('Firing Rate (Hz)')
        plt.savefig(figpath + figname + FS_class + '.png', format='png')
        plt.close(fig)

        fig2 = plt.figure(dpi=200)
        sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "lines.linewidth": 2,
                                    "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.markersize": 5})

        RES_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_classes_dict.items()]))

        ax = sns.boxplot(data=RES_df)  # palette='cividis'
        ax = sns.swarmplot(data=RES_df, palette='Greys_r')

        ax.set(title="Residuals - Comparative FRs", ylabel="Average residuals (Hz)")

        plt.savefig(figpath + figname + FS_class + '_boxplot.png', format='png')
        plt.close(fig)

    return class_residuals, sum_of_residuals, average_residual


def RMSE(empirical, footsim):

    """ Compares Footsim thresholds and the original biological responses for a given set of amplitudes and frequencies
    of stimulus.

           Arguments:

               empirical(dict): Dictionary of empirical responses
               footsim(dict): Dictionary of footsim responses
               freqs(np.array of float): array containing frequencies of stimulus


           Returns:

              Root Mean Square Error (RMSE) value.

           """

    differences = 0
    n = 0
    nans = 0

    for nan_check in range(0, footsim.size):

        if np.isnan(footsim[nan_check]) == False:

            nans = 1

    if nans == 1:

        nan_array = np.isnan(footsim)
        not_nan_array = ~ nan_array
        footsim = footsim[not_nan_array]

        if footsim.size <= empirical.size:

            for freq_value in range(0, footsim.size):

                if math.isnan(empirical[freq_value]) is True or math.isnan(footsim[freq_value]) is True:

                    continue

                instant_difference = empirical[freq_value] - footsim[freq_value]  # Differences
                squared_difference = instant_difference ** 2  # Differences squared
                differences = differences + squared_difference  # Sum of
                n = n + 1

        else:

            for freq_value in range(0, empirical.size):

                if math.isnan(empirical[freq_value]) is True or math.isnan(footsim[freq_value]) is True:

                    continue

                instant_difference = empirical[freq_value] - footsim[freq_value]  # Differences
                squared_difference = instant_difference ** 2  # Differences squared
                differences = differences + squared_difference  # Sum of
                n = n + 1

    if n != 0:

        mean_of_differences_squared = differences/n  # the MEAN of ^

        rmse_val = np.sqrt(mean_of_differences_squared)  # ROOT of ^

    else:

        rmse_val = 2

    return rmse_val


def RMSE_AFTindividualmodels(amps, absolute, freqs, valid_models, filepath, figpath):

    # Output files #

    filename = figpath + "Individual_models_RMSE.txt"
    file = open(filename, "w+")
    RMSEs = dict()
    RMSE_allclasses = {"FA1": list(), "FA2": list(), "SA1": list(), "SA2": list()}
    zscores_allclasses = {"SA1": list(), "SA2": list(), "FA1": list(), "FA2": list()}

    # Reading the empirical data #

    data_file = pd.read_excel(filepath)
    grouped_by_threshold = data_file[data_file.Threshold.notnull()].copy(deep=True)  # Gets the empirical thresholds

    # Computing model outputs #

    individual_grouped_thresholds = individual_models_threshold_visualisation(absolute=absolute, filepath=filepath,
                                                                              figpath=figpath,
                                                                              valid_models=valid_models,
                                                                              amps=amps, freqs=freqs)

    # Performing comparisons #

    for affclass in individual_grouped_thresholds.keys():

        for idx in individual_grouped_thresholds[affclass].keys():

            afferent_id = grouped_by_threshold[grouped_by_threshold.Afferent_ID.
                str.contains(str(valid_models[affclass][idx[0]]))].copy(deep=True)

            empty = afferent_id['Amplitude'].empty

            if empty is True:

                continue

            else:

                empirical = np.array(afferent_id['Amplitude'])

                footsim = np.array(individual_grouped_thresholds[affclass][idx])

                key = str(affclass) + "_" + str(idx[0])

                rmse_val = RMSE(empirical=empirical, footsim=footsim)

                if rmse_val != 2:

                    RMSEs[key] = rmse_val/2
                    RMSE_allclasses[affclass].append(rmse_val/2)
                    line = str(key) + ": " + str(RMSEs[key])
                    file.write(line)
                    file.write('\n')


    # Saving output #

    file.close()

    fig = plt.figure(figsize=(10, 10), dpi=300)

    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})

    figname = figpath + "Individual_models_RMSE.png"


    plt.scatter(*zip(*sorted(RMSEs.items())), color='k', s=40)
    plt.xlabel('Afferent IDs')
    plt.xticks(rotation=45)
    plt.title('RMSE of individual models')
    plt.ylabel('RMSE')
    plt.ylim(-0.02, 1)

    plt.savefig(figname, format='png')

    plt.close(fig)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in RMSE_allclasses.items()]))

    pvalues = oneway_ANOVA_plus_TukeyHSD(dataframe=df, file_name="RMSE_ANOVA_TukeyHSD", outputpath=figpath)

    fig2 = plt.figure(dpi=500)

    sns.set("talk", rc={'figure.figsize': (10, 10)})
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, palette='Greys_r')
    ax.set(title="RMSE of all classes", ylabel="Normalised RMSE value", ylim=[0, 1])

    #add_stat_annotation(ax, data=df, text_format='star', loc='inside', verbose=2, linewidth=3,
    #                    box_pairs=[("SA1", "SA2"), ("SA1", "FA1"), ("SA1", "FA2"), ("FA1", "FA2"), ("FA1", "SA2"), ("FA2", "SA2")], pvalues=pvalues,
    #                    perform_stat_test=False, line_offset_to_box=0.15)

    figname_ = figpath + "_RMSE_allclasses_.png"
    plt.savefig(figname_, format='png')

    plt.close(fig2)

    zscores_allclasses['SA1'] = np.abs(stats.zscore(RMSE_allclasses['SA1']))
    zscores_allclasses['SA2'] = np.abs(stats.zscore(RMSE_allclasses['SA2']))
    zscores_allclasses['FA1'] = np.abs(stats.zscore(RMSE_allclasses['FA1']))
    zscores_allclasses['FA2'] = np.abs(stats.zscore(RMSE_allclasses['FA2']))

    fig3 = plt.figure(dpi=500)

    dfz = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in zscores_allclasses.items()]))

    sns.set("talk", rc={"figure.figsize": (10, 10)}, font_scale=1.5)
    ax = sns.barplot(data=dfz, palette='cividis')
    ax = sns.swarmplot(data=dfz, palette='Greys_r')
    ax.set(title="Z-score of RMSEs", ylabel="Z-score", ylim=[0, 3])

    figname_ = figpath + "_zscore_RMSE_.png"
    plt.savefig(figname_, format='png')

    plt.close(fig3)

    return RMSE_allclasses


def RMSE_multipleAFTdefs(filepath, figpath, valid_models, amps, freqs):

    for absolute in range(1, 25):

        RMSE = RMSE_allclasses(absolute=absolute, filepath=filepath, figpath=figpath, valid_models=valid_models,
                               amps=amps,
                               freqs=freqs)

        RMSE_barplot(figpath=figpath, all_rms=RMSE)


def RMSE_AFTsingle_models(amps, freqs, filepath, output):

    """ Compares Footsim thresholds and the original biological responses for a given set of amplitudes and frequencies
       of stimulus for all single models

          Arguments:

              amps(np.array of float): array containing amplitudes of stimulus
              freqs(np.array of float): array containing frequencies of stimulus
              filepath (str): Path of a *.csv file containing empirically recorded afferents
              output(str): Where to save the output file (full path)

          Returns:

             Text file with all the Root Mean Square Error (RMSE) values.

          """

    data_file = pd.read_excel(filepath)  # Reads the empirical data
    grouped_by_threshold = data_file[data_file.Threshold.notnull()].copy(deep=True)  # Gets the empirical thresholds

    footsim_single_models = investigate_all_single_models(filepath, amps,
                                                          freqs)  # Dict with [affclass and idx] as keys

    affidSA1 = ['120508AHH U01SAI',
                '140319DPZ U01SAI',
                '140827DPZ U02SAI',
                '140418GG U02SAI',
                '140627GG U01SAI',
                '120626JAM U01 SAI',
                '120525MEG U02 SAI',
                '140530SMB U01SAI']

    affidSA2 = ['131205AYK U01SAII',
                '130716CKL U01 SAII',
                '140819GG U01SAII',
                '131031LRF U02 SAII']

    affidFA1 = ['130801AHH U01 FAI',
                '140522AHH U01FAI',
                '140312AN U01FAI',
                '130716CKL U02 FAI',
                '131219CKL U01FAI',
                '131219CKL U02FAI',
                '140122DPZ U02FAI',
                '140627GG U02FAI',
                '131031LRF U01 FAI',
                '140403LRF U02FAI',
                '140514LRF U01FAI',
                '120405MEG U02 FAI',
                '120504RLM U01FAI',
                '140530SMB U02FAI']

    affidFA2 = ['120717AHH U03 FAII',
                '140813AHH U02FAII',
                '140424CKL U01FAII',
                '140122DPZ U01FAII',
                '140827DPZ U01FAII',
                '140430LNC U02FAII',
                '140514LRF U02FAII',
                '130925RLM U02 FAII']

    filename = output + ".txt"
    file = open(filename, "w+")

    for keys in footsim_single_models: # Compares empirical data with the footsim output using the RMSE() method

        if keys[0] == 'SA1':

            afferent_id = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(str(affidSA1[keys[1]]))] \
                .copy(deep=True)
            file.write(str(RMSE(afferent_id['Amplitude'], footsim_single_models[keys], freqs)))

        if keys[0] == 'SA2':

            afferent_id = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(str(affidSA2[keys[1]]))] \
                .copy(deep=True)
            file.write(str(RMSE(afferent_id['Amplitude'], footsim_single_models[keys], freqs)))

        if keys[0] == 'FA1':

            afferent_id = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(str(affidFA1[keys[1]]))] \
                .copy(deep=True)
            file.write(str(RMSE(afferent_id['Amplitude'], footsim_single_models[keys], freqs)))

        if keys[0] == 'FA2':

            afferent_id = grouped_by_threshold[grouped_by_threshold.Afferent_ID.str.contains(str(affidFA2[keys[1]]))] \
                .copy(deep=True)
            file.write(str(RMSE(afferent_id['Amplitude'], footsim_single_models[keys], freqs)))

    file.close()

    return footsim_single_models


def RMSE_allclasses(absolute, filepath, figpath, valid_models, amps, freqs):

    """ Compares Footsim thresholds and the original biological responses for a given set of amplitudes and frequencies
           of stimulus for each afferent class

       Arguments:

        absolute (float): Absolute firing threshold in Hz
        filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
        afferent population is to mimic some experimental data
        figpath(str): Where to save result plots
        valid_models(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
        populations generation, numbers should match the ones in the *.csv if reproducing experimental data
        amps(np.array of float): array containing amplitudes of stimulus
        freqs(np.array of float): array containing frequencies of stimulus

      Returns:

         List of RMSE values.

          """

    afferents_empirical = ['FAI', 'FAII', 'SAI', 'SAII']
    afferents_footsim = ['FA1', 'FA2', 'SA1', 'SA2']

    predicted_array = average_AFT_individualmodels(absolute=absolute, filepath=filepath, figpath=figpath, valid_models=valid_models, amps=amps, freqs=freqs)

    empirical_array = empirical_data_handling(filepath=filepath, freqs=freqs)
    #predicted_array = grouping_per_afferent(thresholds)

    all_rmses = list()

    for afferent in range(0, 4):

        rms = RMSE(np.array(empirical_array[afferents_empirical[afferent]]), np.array(predicted_array[afferents_footsim[afferent]]))
        all_rmses.append(rms)

    RMSE_barplot(figpath=figpath, all_rms=all_rmses)

    return all_rmses


def oneway_ANOVA_plus_TukeyHSD(dataframe, file_name, outputpath):

    # Dropping NaNs

    NaNfree = [dataframe[col].dropna() for col in dataframe]

    # Running the ANOVA per se

    fvalue, pvalue = stats.f_oneway(*NaNfree)

    print("F =", fvalue)
    print("p =", pvalue)
    ANOVAresults = "\n" + str(fvalue) + " " + str(pvalue) + "\n\n"

    # Post-hoc tests after stacking the data

    stacked_data = dataframe.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id', 'level_1': 'Class', 0: 'value'})

    MultiComp = MultiComparison(stacked_data['value'].astype('float'), stacked_data['Class'])

    res = MultiComp.tukeyhsd()  # Results of the Tukey HSD

    # Exporting results

    summary = res.summary()
    summary = summary.as_text()  # ANOVA summary

    buffer = io.StringIO()
    dataframe.info(buf=buffer)
    info = buffer.getvalue()  # Dataset info

    mean = str(dataframe.mean(axis=0))
    std = str(dataframe.std(axis=0))
    pvalues = psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total)

    file = open(outputpath + file_name + "_ANOVA.txt", "w+")

    file.write("Means of the dataset: \n")
    file.write(mean)
    file.write("\n \nSTD of the dataset: \n")
    file.write(std)
    file.write("\n \nRelevant info of the dataset: \n")
    file.write(info)
    file.write("\n \nF-value and overall p-value: \n")
    file.write(ANOVAresults)  # ANOVA results
    file.write(summary)  # ANOVA summary
    file.write("\n \np-values: \n")
    file.write(str(pvalues))

    file.close()

    return pvalues

# ----------------------------------------- #

# File export and array handling methods #

# ----------------------------------------- #

def dict_to_file(dict, filename, output_path):  # Please change the filepath accordingly

    """ Writes any dictionary to a text file

           Arguments:

            dict(dict): Dictionary to be written
            filename(str): name of the output file
            output_path(str): Full path of the output file

          Returns:

             Output file

          """

    fullname = output_path + filename + ".txt"
    file = open(fullname, "w+")
    file.write(str(dict))
    file.close()


def grouping_per_afferent(class_min):

    """ Formats the footsim thresholds to be used on the RMSE() calculation

       Arguments:

        class_min(dict): Dictionary with the thresholds grouped per afferent class

      Returns:

         Dictionary with the grouped data.

          """

    grouped = dict()

    SA1 = list()
    FA1 = list()

    SA2 = list()
    FA2 = list()

    for keys in class_min:

        if keys[0] == 'SA1':

            SA1.append(class_min[keys])

        elif keys[0] == 'FA1':

            FA1.append(class_min[keys])

        elif keys[0] == 'SA2':

            SA2.append(class_min[keys])

        elif keys[0] == 'FA2':

            FA2.append(class_min[keys])

    FA1 = np.array(FA1)
    FA2 = np.array(FA2)
    SA1 = np.array(SA1)
    SA2 = np.array(SA2)

    grouped = {"FA1": FA1, "FA2": FA2, "SA1": SA1, "SA2": SA2}

    return grouped


def ImpCycle_csvexport(figpath, ImpCycle, population, valid_models):

    spreadsheet = figpath + 'ImpCycle_43models_27_Nov20.csv'

    with open(spreadsheet, mode='w') as ImpCycle_csv:

        ImpCycle_csv = csv.writer(ImpCycle_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                  lineterminator='\n')

        ImpCycle_csv.writerow(['Afferent_ID', 'type', 'Location', 'Amplitude', 'Frequency',
                               'ImpCycle'])

        for keys in ImpCycle:

            for ImpCycle_value in range(0, len(ImpCycle[keys])):

                amp = keys[0]
                freq = keys[1]
                location = keys[2]
                affclass = population[location].afferents[ImpCycle_value].affclass
                model_idx = population[location].afferents[ImpCycle_value].idx
                affID = valid_models[affclass][model_idx]

                ImpCycle_csv.writerow([str(affID), str(affclass), str(location), str(amp),
                                       str(freq), str(ImpCycle[keys][ImpCycle_value][0])])
