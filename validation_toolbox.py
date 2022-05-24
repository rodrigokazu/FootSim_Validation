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
from math import sqrt

from pylab import *

from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
#from statannot import add_stat_annotation
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# plt.rc('xtick', labelsize=4)
plt.rcParams.update({'font.size': 20, 'axes.linewidth': 2.5})

# ----------------------------------------- #

# Empirical data manipulation and visualisation #

# ----------------------------------------- #


def empirical_afferent_locations(filepath):

    """ Reads a *.csv file with empirically recorded afferents and generates a dictionary with afferent locations.

           Arguments:

               filepath (str): path to the *.csv file
               fs.constants.affid (float): dictionary with the individual afferent models requested, keys need to be
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

    for afferent_type in fs.constants.affid.keys():

        valid = len(fs.constants.affid[afferent_type])

        for individual_afferent in range(0, valid):

            if data_file[data_file['Afferent ID'].str.fullmatch(fs.constants.affid[afferent_type][individual_afferent])].empty \
                    == False:

                afferent = data_file[data_file['Afferent ID'].str.fullmatch
                (fs.constants.affid[afferent_type][individual_afferent])].copy(deep=True)
                afferent_id = str(fs.constants.affid[afferent_type][individual_afferent])

                location = afferent.iloc[0]['locatoin specific ']  # Finds the empirical afferent location

                location = location_mapping[location]

                empirical_location[afferent_id] =  location

    return empirical_location


def empirical_AFTs_perclass(filepath, figpath):

    """ Displays the empirical AFTs per afferent class

        Arguments:

         filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
         freqs(np.array of float): array containing frequencies of stimulus

       Returns:

          Lineplot of the AFTs per afferent class.

           """

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file[data_file['Threshold '].notnull()].copy(deep=True)  # Gets the thresholds

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    for affclass in range(0, 4):

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[affclass])].copy(
            deep=True)

        ax = sns.lineplot(x=afferent_class['Frequency '], y=afferent_class['amplitude '], label=str(afferents[affclass]))

    figure = figpath + "(Empirical)AFTs.png"

    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250], ylabel="Amplitude (mm)", xlabel="Frequency (Hz)")
    plt.title("Absolute Threshold from Microneurography")

    plt.show()
    plt.savefig(figure)


def empirical_AFT_singleafferents(filepath, figpath, modeled_only=True):

    """ Reads the microneurography dataset and plots the AFTs per afferent.

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots

        Returns:

            Empirical AFT lineplots per afferent class.

    """

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file.copy(deep=True)

    grouped_by_threshold = grouped_by_threshold[grouped_by_threshold['Threshold '].notnull()].copy(deep=True)

    if modeled_only:
        valid_models = {'SAI': fs.constants.affidSA1,
                    'SAII': fs.constants.affidSA2,
                    'FAI': fs.constants.affidFA1,
                    'FAII': fs.constants.affidFA2}
    else:
        valid_models = {'SAI': grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch('SAI')]['Afferent ID'].unique(),
                    'SAII': grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch('SAII')]['Afferent ID'].unique(),
                    'FAI': grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch('FAI')]['Afferent ID'].unique(),
                    'FAII': grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch('FAII')]['Afferent ID'].unique()}

    # ----------------------------------------- #

    # Sorting and finding afferent type #

    # ----------------------------------------- #

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    general_location = ['Arch', 'Toes', 'Met', 'Heel']

    palettes = ['Blues', 'Reds', 'Greens', 'Greys']

    fig = plt.figure(figsize=(15, 5), dpi=500)

    sns.set_style("ticks")

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 12,
                                            "ytick.labelsize": 12})
    fig.suptitle("Empirical", fontsize=22, y=1)

    for aff in range(0, 4):  # Amplitude in Log scale

        sns.set_palette(palettes[aff])

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(deep=True)

        plt.subplot(1, 4, aff + 1)

        # ----------------------------------------- #

        # Slicing the data table per afferent #

        # ----------------------------------------- #

        index = 0

        for afferent in valid_models[afferents[aff]]:

            # Captures the individual afferent

            afferentid = afferent_class[
                afferent_class['Afferent ID'].str.fullmatch(valid_models[afferents[aff]][index])].copy(deep=True)

            # ----------------------------------------- #

            # Data visualisation #

            # ----------------------------------------- #

            ax = sns.lineplot(x=afferentid['Frequency '], y=afferentid['amplitude '],
                              err_style="band")  # label=valid_models[afferents[aff]][index]
            index = index + 1

        #ax.set(title=afferents[aff], xlabel='Frequency (Hz)', ylabel='Amplitude (mm)', ylim=[10 ** -3, 10 ** 1],
        #       xlim=[1, 1000])
        ax.set(yscale='log', xscale='log')
        plt.ylim([10 ** -3, 10 ** 1])
        plt.xlim([1, 1000])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # plt.legend(loc=1, fontsize=7.2)
        plt.tight_layout()

        figure = figpath + "Thres_emp.svg"

        plt.savefig(figure, format='svg')


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

    grouped_by_threshold = data_file[data_file['Threshold '].notnull()].copy(deep=True)  # Gets the thresholds
    output_grouped = dict()

    for aff in range(0, 4):

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(deep=True)
        output_grouped[afferents[aff]] = list()

        for freq_list in range(0, len(freqs)):

            output_grouped[afferents[aff]].append(np.mean(afferent_class['amplitude '].where(afferent_class['Frequency '] == freqs[freq_list])))

        output_grouped[afferents[aff]] = np.array(output_grouped[afferents[aff]])

    return output_grouped


def empirical_hardnessperregion(filepath, figpath):

    """ Displays the empirical hardness per region

        Arguments:

         filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
         figpath(str): Complete path of the saving location

       Returns:

          Boxplot of hardness per region;

           """

    location_specific = ['LatArch', 'Toes', 'LatMet', 'Heel', 'MidMet', 'MidArch', 'LatArch', 'MedArch', 'GT', 'MedMet']

    data_file = pd.read_excel(filepath)

    Arch = data_file['RF_hardness'].where(data_file['location_general'] == "Arch")
    Toes = data_file['RF_hardness'].where(data_file['location_general'] == "Toes")
    Met = data_file['RF_hardness'].where(data_file['location_general'] == "Met")
    Heel = data_file['RF_hardness'].where(data_file['location_general'] == "Heel")

    location_general = {"Arch": Arch, "Toes": Toes, "Met": Met, "Heel": Heel}

    location = pd.DataFrame.from_dict(location_general)

    pd.DataFrame.to_csv(location, path_or_buf=figpath+"Hardness.csv")

    oneway_ANOVA_plus_TukeyHSD(dataframe=location, file_name="Hardness_per_region", outputpath=figpath)

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    sns.set_context("talk", rc={"font.size": 25, "axes.titlesize": 16, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 20, "lines.markersize": 5})

    ax = sns.boxplot(x=data_file['location_general'], y=data_file['RF_hardness'])

    sns.set_palette("Greys_r")

    ax = sns.scatterplot(x=data_file['location_general'], y=data_file['RF_hardness'])

    figure = figpath + "(Empirical)Hardness_per_greatregion.png"

    ax.set(ylim=[0, 100], ylabel="Hardness (a.u.)", xlabel="Location")
    plt.title("Hardness spread of the microneurography dataset")

    plt.savefig(figure)


def empirical_AFT_vs_hardness(filepath, figpath):

    # Declarations #

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    palettes = ['Blues', 'Reds', 'Greens', 'Purples']

    # Gathering populational info via footsim's validation toolbox #

    populations = empirical_afferent_positioning(filepath=filepath)

    hard_vs_soft = find_hard_vs_soft_afferents(populations=populations)

    # Applying it to the empirical data #

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file[data_file['Threshold '].notnull()].copy(deep=True)

    # Visualisation of empirical data #

    fig = plt.figure(figsize=(7, 5), dpi=300)

    sns.set(style='whitegrid', font_scale=1.5)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                                            "lines.linewidth": 1, "xtick.labelsize": 10,
                                            "ytick.labelsize": 10})

    #fig.suptitle("Empirical AFTs", fontsize=12, y=1)

    # Plotting the required data for the hard regions #

    plt.subplot(1, 2, 1)

    for aff in range(0, 4):  # Amplitude in Log scale

        sns.set_palette(palettes[aff])

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(deep=True)

        # ----------------------------------------- #

        # Slicing the data table per afferent #

        # ----------------------------------------- #

        for afferent in hard_vs_soft['Hard afferents']:

            # Captures the individual afferent

            afferentid = afferent_class[afferent_class['Afferent ID'].str.fullmatch(afferent)].copy(deep=True)

            # ----------------------------------------- #

            # Data visualisation #

            # ----------------------------------------- #

            ax = sns.lineplot(x=afferentid['Frequency '], y=afferentid['amplitude '], err_style="band", color='r',
                              label=afferent)

        ax.set(title="Hard", xlabel='Frequency (Hz)', ylabel='Amplitude (mm)', ylim=[10 ** -3, 10 ** 1],
               xlim=[0, 250])

        ax.set(yscale='log', xscale='linear')

        plt.legend(loc=1, fontsize=4)

    plt.subplot(1, 2, 2)

    for aff in range(0, 4):  # Amplitude in Log scale

        sns.set_palette(palettes[aff])

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(
            deep=True)

        # ----------------------------------------- #

        # Slicing the data table per afferent #

        # ----------------------------------------- #


        for afferent in hard_vs_soft['Soft afferents']:

            # Captures the individual afferent

            softafferent = afferent_class[
                afferent_class['Afferent ID'].str.fullmatch(afferent)].copy(deep=True)

            # ----------------------------------------- #

            # Data visualisation #

            # ----------------------------------------- #

            ax2 = sns.lineplot(x=softafferent['Frequency '], y=softafferent['amplitude '],
                              err_style="band", color='b', label=afferent)

        ax2.set(title="Soft", xlabel='Frequency (Hz)', ylabel='Amplitude (mm)', ylim=[10 ** -3, 10 ** 1],
               xlim=[0, 250])

        ax2.set(yscale='log', xscale='linear')

        figure = figpath + "Empirical_afts_vs_hardness.png"

        plt.legend(loc=1, fontsize=4)
        plt.tight_layout()
        plt.savefig(figure)

    return 0


def empirical_FAsAFT_vs_hardness(filepath, figpath):

    # Declarations #

    afferents = ['FAI', 'FAII']

    palettes = ['Blues', 'Reds', 'Greens', 'Purples']

    # Gathering populational info via footsim's validation toolbox #

    populations = empirical_afferent_positioning(filepath=filepath)

    hard_vs_soft = find_hard_vs_soft_afferents(populations=populations)

    # Applying it to the empirical data #

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file[data_file['Threshold '].notnull()].copy(deep=True)

    # Visualisation of empirical data #

    fig = plt.figure(figsize=(7, 5), dpi=300)

    sns.set(style='whitegrid', font_scale=1.5)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                                            "lines.linewidth": 1, "xtick.labelsize": 10,
                                            "ytick.labelsize": 10})

    fig.suptitle("Empirical AFTs - Only FAs", fontsize=12, y=1)

    # Plotting the required data for the hard regions #

    plt.subplot(1, 2, 1)

    for aff in range(0, 1):  # Amplitude in Log scale

        sns.set_palette(palettes[aff])

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(deep=True)

        # ----------------------------------------- #

        # Slicing the data table per afferent #

        # ----------------------------------------- #

        for afferent in hard_vs_soft['Hard afferents']:

            # Captures the individual afferent

            afferentid = afferent_class[afferent_class['Afferent ID'].str.fullmatch(afferent)].copy(deep=True)

            # ----------------------------------------- #

            # Data visualisation #

            # ----------------------------------------- #

            ax = sns.lineplot(x=afferentid['Frequency '], y=afferentid['amplitude '], err_style="band", color='r',
                              label=afferent)

        ax.set(title="Hard", xlabel='Frequency (Hz)', ylabel='Amplitude (mm)', ylim=[10 ** -3, 10 ** 1],
               xlim=[0, 250])

        ax.set(yscale='log', xscale='linear')

        plt.legend(loc=1, fontsize=4)

    plt.subplot(1, 2, 2)

    for aff in range(0, 1):  # Amplitude in Log scale

        sns.set_palette(palettes[aff])

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(
            deep=True)

        # ----------------------------------------- #

        # Slicing the data table per afferent #

        # ----------------------------------------- #


        for afferent in hard_vs_soft['Soft afferents']:

            # Captures the individual afferent

            softafferent = afferent_class[
                afferent_class['Afferent ID'].str.fullmatch(afferent)].copy(deep=True)

            # ----------------------------------------- #

            # Data visualisation #

            # ----------------------------------------- #

            ax2 = sns.lineplot(x=softafferent['Frequency '], y=softafferent['amplitude '],
                              err_style="band", color='b', label=afferent)

        ax2.set(title="Soft", xlabel='Frequency (Hz)', ylabel='Amplitude (mm)', ylim=[10 ** -3, 10 ** 1],
               xlim=[0, 250])

        ax2.set(yscale='log', xscale='linear')

        figure = figpath + "Empirical_FAafts_vs_hardness.png"

        plt.legend(loc=1, fontsize=4)
        plt.tight_layout()
        plt.savefig(figure)

    return 0


def empirical_impcycles(filepath, figpath):

    """ Reads the microneurography dataset and plots the ImpCycles per afferent.

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots

        Returns:

            Empirical ImpCycle lineplots per afferent class.

    """

    data_file = pd.read_excel(filepath)

    grouped_by_threshold = data_file.copy(deep=True)

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 10, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 16, "ytick.labelsize": 20, "lines.markersize": 5})

    for aff in range(0, 4):  # Amplitude in Log scale

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(afferents[aff])].copy(deep=True)

        if len(afferent_class['Frequency '].where(afferent_class['ImpCycle'] == 1)) > 0:
            ax = sns.lineplot(data=afferent_class, x=afferent_class['Frequency '].where(afferent_class['ImpCycle'] == 1),
                              y=afferent_class['amplitude '].where(afferent_class['ImpCycle'] == 1),
                              label=afferents[aff])

            # ax = sns.scatterplot(data=afferent_class, x=afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1),
            # y = afferent_class['Amplitude'].where(afferent_class['ImpCycle'] == 1),  label=afferents[aff])

            ax.set(ylim=[10 ** -3, 10 ** 1], xlim=[0, 250], yscale="log")

    fig.suptitle("Microneurography entrainment", fontsize=35, y=0.95)
    figure = figpath + "Empirical_ImpCycle.png"

    plt.savefig(figure)


def empirical_stimulus_acquisition(filepath):

    """ Reads a *.csv file with empirically recorded afferents and generates a dictionary with the stimulus used during
    the original experiment

           Arguments:

               filepath (str): path to the *.csv file
               fs.constants.affid (float): dictionary with the individual afferent models requested, keys need to be
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

    for afferent_type in fs.constants.affid.keys():

        valid = len(fs.constants.affid[afferent_type])

        for individual_afferent in range(0, valid):

            afferent = data_file[data_file['Afferent ID'].str.fullmatch(fs.constants.affid[afferent_type][individual_afferent])].copy(deep=True)
            afferent_id = str(fs.constants.affid[afferent_type][individual_afferent])
            freq = list(afferent['Frequency '].dropna())  # Gets the frequencies .where(afferent['AvgInst'] != 0)
            amp = list(afferent['amplitude '].dropna())  # Gets the amplitudes .where(afferent['AvgInst'] != 0)
            location = afferent.iloc[0]['locatoin specific ']  # Finds the empirical afferent location

            location = location_mapping[location]

            empirical_stimulus[afferent_id] = dict()
            empirical_stimulus[afferent_id]['Location'] = location
            empirical_stimulus[afferent_id]['Frequency '] = freq
            empirical_stimulus[afferent_id]['amplitude '] = amp

    return empirical_stimulus


def empirical_stimulus_pairs(empirical_stimulus):

    """
    Reads a empirical stimulation dictionary generated with empirical_stimulus_acquisition and returns them paired.

        Arguments:

            empirical_stimulus(dict): Dictionary containing the empirical stimulation

        Returns:

            Dictionary with stimuli paired.

    """

    stimulus_pairs = dict()

    for keys in empirical_stimulus.keys():

        stimulus_pairs[keys] = list()

        for pair in range(0, len(empirical_stimulus[keys]['Frequency '])):

            stimulus_pairs[keys].append((empirical_stimulus[keys]['Frequency '][pair], empirical_stimulus[keys]['amplitude '][pair]))

    return stimulus_pairs

# ----------------------------------------- #

# Afferent positioning methods #

# ----------------------------------------- #


def empirical_afferent_positioning(filepath):  # Positions the afferents as in the empirical recordings

    """ Reads a *.csv file with empirically recorded afferents and generates a simulated afferent popuplation that
    matches it in the same foot sole locations.

        Arguments:

            filepath (str): path to the *.csv file

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

    for afferent_type in fs.constants.affid.keys():

        valid = len(fs.constants.affid[afferent_type])

        for individual_afferent in range(0, valid):

            idx = individual_afferent  # Specific afferent model

            if grouped_by_threshold[grouped_by_threshold['Afferent ID'].str.fullmatch
                (fs.constants.affid[afferent_type][individual_afferent])].empty == False:

                location_slice = grouped_by_threshold[grouped_by_threshold['Afferent ID'].str.fullmatch
                (fs.constants.affid[afferent_type][individual_afferent])].copy(deep=True)

                location = location_slice.iloc[0]['locatoin specific ']  # Finds the empirical afferent location

                location = location_mapping[location]

                # Pins the simulated afferents on the correct foot sole location #

                afferent_populations[location].afferents\
                    .append(fs.Afferent(affclass=afferent_type, idx=int(idx), location=fs.foot_surface.centers[fs.foot_surface.tag2idx(location)[0]]))

    return afferent_populations


def populational_info(populations):

    """ Reads a dictionary with locations as keys and afferent populations as values and exports relevant info.

           Arguments:

               populations(dict): Dictionary of afferent populations with regions as keys.
               fs.constants.affid (float): dictionary with the individual afferent models requested, keys need to be
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
            afferent_id = fs.constants.affid[affclass][idx]

            afferent_data[location].append((afferent_id, affclass, idx))

    return afferent_data


def find_hard_vs_soft_afferents(populations):

    """ Reads a dictionary with locations as keys and afferent populations as values and sorts afferents in between hard
    and soft regions.

       Arguments:

           populations(dict): Dictionary of afferent populations with regions as keys.

       Returns:

           Dictionary with afferent ids, sorted in Hard and Soft lists.

       """

    soft_afferents = list()
    hard_afferents = list()

    hard_regions = ['T1', 'T2_t', 'HR']

    for location in populations:

        if location in hard_regions:

            for afferent in range(0, len(populations[location])):

                affclass = populations[location].afferents[afferent].affclass
                idx = populations[location].afferents[afferent].idx

                afferent_id = fs.constants.affid[affclass][idx]

                hard_afferents.append(afferent_id)

        if location not in hard_regions:

            for afferent in range(0, len(populations[location])):

                affclass = populations[location].afferents[afferent].affclass
                idx = populations[location].afferents[afferent].idx

                afferent_id = fs.constants.affid[affclass][idx]

                soft_afferents.append(afferent_id)

    hard_vs_soft = {"Hard afferents": hard_afferents, "Soft afferents": soft_afferents}

    return hard_vs_soft

# ----------------------------------------- #

# Investigating afferent responses #

# ----------------------------------------- #


def get_responsive_amplitudes(absolute, filepath, output, matched=True, threshold_type='A'):

    """ Investigate the responses of an afferent population for a given set of frequencies and amplitudes of stimulus
    computing responses for either absolute of tuning thresholds and grouping the results by afferent class,
    location or model id.


         Arguments:

             absolute (float): Absolute firing threshold in Hz
             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             output(str): Type of output required, C for class, I for individual models, R for regions

         Returns:

            Dictionary of responsive amplitudes for afferents grouped by by afferent class, location or model id.

                # Keys of the returned dictionary, if output == "I":

                    # 0 = Afferent class
                    # 1 = Model number (idx)
                    # 2 = Location on the foot sole
                    # 3 = Frequency where that response happened
         """

    start = time.asctime(time.localtime(time.time()))

    populations = empirical_afferent_positioning(filepath=filepath)
    responsive_afferents = dict()
    regional_responsive_amplitudes = dict()
    class_responsive_amplitudes = dict()
    response_of_individual_models = dict()

    rates = model_firing_rates(populations=populations, filepath=filepath, matched=matched)

    for keys in rates.keys():

        rates[keys] = correct_FRoutputs(rates[keys])

    for keys in rates.keys():

        if threshold_type == "T":

            tuning = 0.8 * keys[1]
            responsive_afferents[keys] = np.where(rates[keys] > tuning)

            responsive_afferents[keys] = responsive_afferents[keys][0]

        if threshold_type == "A":

            responsive_afferents[keys] = np.where(np.array(rates[keys], dtype=object) > absolute)

            if len(responsive_afferents[keys]) > 0:

                responsive_afferents[keys] = responsive_afferents[keys][0]

            else:

                responsive_afferents[keys] = []

            # For each afferent that responded #

            for t in range(0, len(responsive_afferents[keys])):

                afferent_c = int(responsive_afferents[keys][t])

                afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class

                idx = populations[keys[2]][afferent_c].idx

                # Appends the amplitudes where it was responsive #

                try:
                    regional_responsive_amplitudes[afferent_class, keys[2], keys[1]].append(keys[0])

                except:
                    regional_responsive_amplitudes[afferent_class, keys[2], keys[1]] = [keys[0]]

                try:
                    class_responsive_amplitudes[afferent_class, keys[1]].append(keys[0])

                except:
                    class_responsive_amplitudes[afferent_class, keys[1]] = [keys[0]]

                try:
                    response_of_individual_models[afferent_class, idx, keys[2], keys[1]].append(keys[0])

                except:
                    response_of_individual_models[afferent_class, idx, keys[2], keys[1]] = [keys[0]]

    print("Simulation started at: ", start)
    print("Simulation finished at: ", time.asctime(time.localtime(time.time())))

    userinput = output

    if userinput == "R":

        return regional_responsive_amplitudes

    elif userinput == "C":

        return class_responsive_amplitudes

    elif userinput == "I":

        return response_of_individual_models


def ImpCycle(figpath, filepath):

    populations = empirical_afferent_positioning(filepath=filepath)

    ImpCycle = model_firing_rates(populations=populations, filepath=filepath)

    for keys in ImpCycle:

        for rate in range(0, len(ImpCycle[keys])):

            if not math.isnan(ImpCycle[keys][rate]):

                ramp_up = keys[1]
                ImpCycle_value = ImpCycle[keys][rate][0][0]/ramp_up
                ImpCycle[keys][rate][0] = ImpCycle_value

    ImpCycle_csvexport(figpath=figpath, ImpCycle=ImpCycle, population=populations)

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


def model_firing_rates(filepath, populations, matched=True):

    """ Computes the firing rates (Hz) of an afferent population for a given set of frequencies and amplitudes of
    stimulus

         Arguments:

             populations (dict): Dictionary with foot locations as keys and afferent populations as values
             filepath (str): path to the excel file

         Returns:

            Dictionary of populational firing rates with amplitude, frequency and location of stimulus as keys

            # Keys of the output dictionary:

                # 0 = Amplitude of stimulus
                # 1 = Frequency of stimulus
                # 2 = Location where the afferent was tested

         """

    emp = empirical_stimulus_acquisition(filepath=filepath)
    pairs = empirical_stimulus_pairs(empirical_stimulus=emp)
    if not matched:
        stim_all = list(set().union(*list(pairs.values())))
        pairs_all = dict()
        for k in pairs.keys():
            pairs_all[k] = stim_all
        pairs = pairs_all

    s = dict()  # Stimulus
    r = dict()  # Responses
    rates = dict()  # Firing rates
    populational_stimulus = dict()

    for location in populations:

        populational_stimulus[location] = list()

        if len(populations[location]) != 0:

            for afferent in range(0, len(populations[location])):

                afferent_id = fs.constants.affid[populations[location]
                    .afferents[afferent].affclass][populations[location].afferents[afferent].idx]

                for stimulation in range(0, len(pairs[afferent_id])):

                    rates[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location] = list()

                    for afferent_t in range(0, len(populations[location])):

                        rates[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location]\
                            .append(np.nan)

    print("Computing firing rates...")

    for location in populations:

        if len(populations[location]) != 0:

            for afferent in range(0, len(populations[location])):

                afferent_id = fs.constants.affid[populations[location]
                    .afferents[afferent].affclass][populations[location].afferents[afferent].idx]

                populational_stimulus[location].extend(pairs[afferent_id])  # Grouping populational stimulus given

                for stimulation in range(0, len(pairs[afferent_id])):

                    # Stimulus based on empirical data - Pairs are frequency and amplitude#

                    s[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location] = \
                        fs.stim_sine(amp=pairs[afferent_id][stimulation][1] / 2,
                                     freq=pairs[afferent_id][stimulation][0], pin_radius=3,
                                     loc=fs.foot_surface.centers[fs.foot_surface.tag2idx(str(location))[0]],
                                     ramp_type='sin', len=2)

                    # Gathers responses #

                    r[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location] = \
                        populations[location].afferents[afferent].response(s[pairs[afferent_id][stimulation][1],
                                                         pairs[afferent_id][stimulation][0], location])

                    # Computes firing rates #

                    rates[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location][afferent] =\
                        r[pairs[afferent_id][stimulation][1], pairs[afferent_id][stimulation][0], location].rate()

            for afferent in range(0, len(populations[location])):

                afferent_id = fs.constants.affid[populations[location]
                    .afferents[afferent].affclass][populations[location].afferents[afferent].idx]

                for every_stimulus in populational_stimulus[location]:

                    if every_stimulus not in pairs[afferent_id]:

                        rates[every_stimulus[1],every_stimulus[0], location][afferent] = np.nan

    return rates


def single_afferent_responses(affclass, idx, amps, freqs, region_matched=False, threshold_type="A"):

    """ Gets the response of a single model on the foot sole for a given set of frequencies and amplitudes of stimulus

        Arguments:

            affclass (str): afferent model class
            idx (float): afferent model id
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus

        Returns:

           The afferent class, the afferent model and a dictionary with the afferent population responses
        """

    if region_matched:
        reg_tag = fs.constants.affreg[affclass][idx]
        populations = (affclass,idx,{reg_tag:fs.Afferent(affclass,idx=idx,location=fs.foot_surface.centers[fs.foot_surface.tag2idx(reg_tag)])})
    else:
        populations = fs.allregions_affpop_single_models(affclass=affclass, idx=idx)

    single_afferent_responses = dict()

    s = dict()  # Stimulus
    r = dict()  # Responses

    absolute = 5  # Firing rate threshold

    print(time.asctime(time.localtime(time.time())))

    for location in populations[2].keys():
        single_afferent_responses[affclass, idx, location] = np.zeros((len(freqs,)))

        for freq in range(0, len(freqs)):

            rates = np.zeros((len(amps),))  # Firing rates
            for amp in range(0, len(amps)):

                s[amps[amp], freqs[freq], location] = \
                    fs.stim_sine(amp=amps[amp] / 2, ramp_type='sin', len=2, pin_radius=3,
                                 freq=freqs[freq], loc=fs.foot_surface.centers[fs.foot_surface.tag2idx(location)])

                #print("Investigating location: ", fs.foot_surface.locate(fs.foot_surface.centers[location])[0][0]
                #      , " for frequency ", freqs[freq], " Hz and ", amps[amp], " mm of amplitude.")

                # Gathers responses #

                r[amps[amp], freqs[freq], location] = \
                    populations[2][location].response(s[amps[amp], freqs[freq], location])

                rates[amp] = r[amps[amp], freqs[freq], location].rate()  # Computes firing rates

            if threshold_type == "T":  # Tuning threshold
                thres = 0.8 * freqs[freq]
            elif threshold_type == "A":  # Absolute threshold
                thres = absolute

            try:
                thres_id = np.argwhere(rates > thres)[0][0]
                single_afferent_responses[affclass, idx, location][freq] = amps[thres_id]
            except:
                single_afferent_responses[affclass, idx, location][freq] = np.nan

    return single_afferent_responses, rates, r


# ----------------------------------------- #

# AFTs calculations #

# ----------------------------------------- #


def class_absolute_thresholds(absolute, filepath, figpath, freqs):

    """ Find the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus

        Arguments:

            absolute (float): Absolute firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots
            freqs(list): Array with the frequencies of stimulation

        Returns:

           Dictionary of minimum responsive amplitudes per afferent class and plots thresholds per frequency in
           log scale.
        """

    class_min = dict()

    class_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath, output="C")

    for key in class_responsive_amplitudes:

        class_min[key] = list()  # To be filled with thresholds

    for key in class_responsive_amplitudes:

        if len(class_responsive_amplitudes[key]) > 0:

            class_min[key] = np.min(class_responsive_amplitudes[key])

        else:

            class_min[key] = np.nan

    dict_to_file(dict=class_min, filename="class_min_for"+str(absolute)+"Hz", output_path=figpath)

    return class_min


def individual_models_thresholds(absolute, filepath, matched=True):

    """ Find the absolute thresholds of all afferent models generated with regional_afferent_positioning()
     for a given set of frequencies and amplitudes of stimulus

     # Keys of the individual_min (AFT) and of the individual_models_thresholds dictionary:

        # 0 = Afferent class
        # 1 = Model number (idx)
        # 2 = Location on the foot sole
        # 3 = Frequency where that response happened

        Arguments:

            absolute (float): Firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data


        Returns:

           Dictionary of minimum responsive amplitudes grouped per afferent model.

        """

    individual_models_thresholds = get_responsive_amplitudes(absolute=absolute, filepath=filepath, output="I", matched=matched)

    individual_min = dict()

    for keys in individual_models_thresholds:

        if len(individual_models_thresholds[keys]) > 0:

            individual_min[keys] = np.min(individual_models_thresholds[keys])

        else:

            individual_min[keys] = np.nan

    return individual_min


def multiple_class_AFTs(amps, filepath, figpath, freqs):

    """ Computes the minimum responsive amplitudes of an afferent class for a range of firing thresholds from 1 to 25 Hz

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save the plots
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Plots of minimum responsive amplitude (log) x frequency (linear)
        """

    for absolute in range(1, 11):

        print("Working the ", str(absolute), "Hz threshold definition for all afferent classes.")

        #average_AFT_individualmodels(absolute=absolute, filepath=filepath, figpath=figpath, fs.constants.affid=fs.constants.affid,
                                     #amps=amps, freqs=freqs)

        RMSE_AFTindividualmodels(amps=amps, absolute=absolute, freqs=freqs, filepath=filepath, figpath=figpath)


def multiple_individual_absolute_thresholds(filepath, figpath, amps, freqs):

    """ Computes the minimum responsive amplitudes of all individual afferent models for a range of afferent firing
    thresholds definitions from 1 to 25 Hz

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save the plots
            fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Plots of minimum responsive amplitude (log) x frequency (linear)
        """

    for absolute in range(1, 25):

        print("Working ", str(absolute), "Hz threshold definition for individual models.")
        RMSE_AFTindividualmodels(amps=amps, absolute=absolute, freqs=freqs, filepath=filepath, figpath=figpath)


def regional_absolute_thresholds(absolute, filepath, figpath, freqs):

    """ Find the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus in
    each region of the foot sole

         Arguments:

             absolute (float): Absolute firing threshold in Hz
             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             freqs(np.array of float): array containing frequencies of stimulus


         Returns:

            Dictionary of responsive amplitudes for afferents grouped per region, and the plots of thresholds x
            frequency in log scale.
         """

    regional_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath, output='R')

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

            regional_min[keys] = np.nan

    regional_threshold_visualisation(figpath=figpath, regional_min=regional_min)

    return regional_min


def single_afferent_thresholds(figpath, affclass, idx, amps, freqs, region_matched=False):

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
    single_responses = single_afferent_responses(affclass, idx, amps, freqs, region_matched=region_matched)

    rates = single_responses[1]
    r = single_responses[2]
    single_responses = single_responses[0]

    single_afferent_threshold_visualisation(figpath, freqs, single_responses)

    return single_responses


def single_afferent_threshold_grouping(single_min):  # Groups the thresholds per model

    """ Computes the thresholds per single model when using investigate_all_single_models(figpath, amps, freqs) for
    further statistical comparisons with experimental data

         Arguments:

             single_min(dict): Dictionary of thresholds
         Returns:

            Dictionary of minimum responsive amplitudes (thresholds) for the afferent model and a
             amplitude (threshold) x frequency plot in log scale.
         """

    single_model = dict()
    responsive_freqs = dict()

    for loc in range(0, len(fs.constants.foot_tags)):

        single_model[fs.constants.foot_tags[loc]] = list()
        responsive_freqs[fs.constants.foot_tags[loc]] = list()

    for keys in single_min.keys():

        if type(single_min[keys]) is not list:

            for location in range(0, len(fs.constants.foot_tags)):

                if keys[2] == fs.constants.foot_tags[location]:

                    single_model[keys[2]].append(single_min[keys])
                    responsive_freqs[keys[2]].append(keys[3])

    return single_model

# ----------------------------------------- #

# Data visualisation #

# ----------------------------------------- #


def apply_ramp(filepath, output_path, **args):
    """ Apply ramps to the centre of each region of the foot sole and plot the responses per region per afferent class

    Args:
        filepath (str): path to 'microneurography_nocomments.xlsx'
        output_path (str): path to location to store output files - ideally a path to a folder
        **args:
            amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
            pin_radius (float): radius of the stimulus pin used (mm)
            ramp_length (float): length of time the ramp is applied to the foot (sec)
            foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                    all regions will be investigated
            afferent_classes (list): list containing the names of afferent classes to investigate

    Returns:
        .png files with plots of the response of each afferent type per region.

    """
    amplitude = args.get('amplitude', 1.) # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5) # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.) # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index', 'all') # list containing the indexes of the regions to be stimulated
    afferent_classes = args.get('afferent_classes', ['FA1','FA2','SA1','SA2']) # list of afferent classes to investigate

    # generate afferent population
    afferent_populations = empirical_afferent_positioning(filepath=filepath)
    print(afferent_populations)

    afferent_data = populational_info(afferent_populations)
    sorted_afferent_data = sort_afferent_data(afferent_data)

    # generate full list of region indexes
    if foot_region_index == 'all':
        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface
    centres = fs.foot_surface.centers

    # generate list of region names
    regions = list(afferent_populations.keys())

    # loop through region indexes
    for index in foot_region_index:

        # check whethere there are no afferents in the region
        if len(afferent_populations[regions[index]].afferents) == 0:
            continue

        else:
            # get name of foot region
            foot_region = regions[index]

            # generate ramp to centre of foot region
            s = fs.generators.stim_ramp(amp=amplitude,ramp_type='lin', loc=centres[index], len=ramp_length, pin_radius=pin_radius)

            # generate response
            r = afferent_populations[foot_region].response(s)

            # loop through afferent classes
            for affClass in afferent_classes:

                # plot response of afferent class to ramp
                plt.figure(figsize=(25, 15))
                plt.suptitle(affClass + ' responses to ramp at centre of ' + str(foot_region) + ': amplitude = ' + str(amplitude) + \
                             'mm, time = ' + str(ramp_length) + 'seconds, pin size = ' + str(pin_radius) + 'mm', fontsize=25)

                plt.subplot(2, 3, 1)
                plt.plot(s.trace[0])
                plt.ylabel('Trace indentation (mm)')
                plt.ylim(0, 2)
                plt.xticks([])

                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:
                    continue

                else:
                    for i in range(r[afferent_populations[foot_region][affClass]].psth(1).T.shape[1]):
                        plt.subplot(2, 3, i + 2)
                        plt.plot(r[afferent_populations[foot_region][affClass]].psth(1)[i].T, color=fs.constants.affcol[affClass])
                        #plt.title(afferent_data[foot_region][i][0], fontsize=15)
                        plt.title(sorted_afferent_data[foot_region][affClass][i][0], fontsize=15)
                        plt.ylabel('Spike', fontsize=15)
                        plt.xlabel('Time (ms)', fontsize=15)
                        plt.subplots_adjust(hspace=0.4, wspace=0.3)
                    plt.savefig(output_path + foot_region + ' ' + affClass + ' Amp - ' + str(amplitude) + ', Rad - ' + str(pin_radius) + ', length - ' + \
                                str(ramp_length) + '.png')


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


def axes_ticks(FR_list):

    """ Generate ticks for the comparative scatter plots in between footsim outputs and the biological responses
    for a given set of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or
    ImpCycle_model_vs_empirical()

         Arguments:

             FR_list(list): List of firing rates


         Returns:

            List of ticks.

         """

    limit = len(FR_list)
    ticks = list()

    for tick in range(0, limit):

        ticks.append(tick)

    return ticks


def average_AFT_individualmodels(absolute, filepath, figpath):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
       stimulus

       Arguments:

       absolute (float): Absolute firing threshold in Hz
       filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
       afferent population is to mimic some experimental data
       figpath(str): Where to save result plots
       fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
       populations generation, numbers should match the ones in the *.csv if reproducing experimental data

        Returns:

           Plots the average of the AFTs as a proxy for afferent class thresholds, replicating empirical data

        """

    AFTs = dict()  # Dictionary of responsive frequencies and AFTs with lists, 0 for freqs 1 for AFTs

    responsive_freqs = dict()  # Dict with only responsive frequencies

    SA1 = list()
    responsive_freqs['SA1'] = list()

    SA2 = list()
    responsive_freqs['SA2'] = list()

    FA1 = list()
    responsive_freqs['FA1'] = list()

    FA2 = list()
    responsive_freqs['FA2'] = list()

    for runs in range(1, 4):

        print("Run of the AFTs number:", runs)

        individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath)

        for keys in individual_min:

            if keys[0] == 'SA1':

                for loc in range(0, len(fs.constants.foot_tags)):

                    if keys[2] == fs.constants.foot_tags[loc]:

                        for idx in range(fs.constants.affparams['SA1'].shape[0]):

                            if keys[1] == idx:

                                SA1.append(individual_min[keys])
                                responsive_freqs[keys[0]].append(keys[3])

            elif keys[0] == 'FA1':

                for loc in range(0, len(fs.constants.foot_tags)):

                    if keys[2] == fs.constants.foot_tags[loc]:

                        for idx in range(fs.constants.affparams['FA1'].shape[0]):

                            if keys[1] == idx:
                                FA1.append(individual_min[keys])
                                responsive_freqs[keys[0]].append(keys[3])

            elif keys[0] == 'SA2':

                for loc in range(0, len(fs.constants.foot_tags)):

                    if keys[2] == fs.constants.foot_tags[loc]:

                        for idx in range(fs.constants.affparams['SA2'].shape[0]):

                            if keys[1] == idx:

                                SA2.append(individual_min[keys])
                                responsive_freqs[keys[0]].append(keys[3])

            elif keys[0] == 'FA2':

                for loc in range(0, len(fs.constants.foot_tags)):

                    if keys[2] == fs.constants.foot_tags[loc]:

                        for idx in range(fs.constants.affparams['FA2'].shape[0]):

                            if keys[1] == idx:

                                FA2.append(individual_min[keys])
                                responsive_freqs[keys[0]].append(keys[3])

    AFTs['SA1 Frequencies'] = responsive_freqs['SA1']
    AFTs['SA1 AFTs'] = SA1

    AFTs['SA2 Frequencies'] = responsive_freqs['SA2']
    AFTs['SA2 AFTs'] = SA2

    AFTs['FA1 Frequencies'] = responsive_freqs['FA1']
    AFTs['FA1 AFTs'] = FA1

    AFTs['FA2 Frequencies'] = responsive_freqs['FA2']
    AFTs['FA2 AFTs'] = FA2

    AFT_dataframe = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in AFTs.items()]))  # Dataframe from Dictionary

    # Computed AFT for optimal experimental frequencies #

    FA1_T = AFT_dataframe['FA1 AFTs'].where(AFT_dataframe['FA1 Frequencies'] == 60)
    FA1_AFT = FA1_T.mean()

    FA2_T = AFT_dataframe['FA2 AFTs'].where(AFT_dataframe['FA2 Frequencies'] == 100)
    FA2_AFT = FA2_T.mean()

    SA2_T = AFT_dataframe['SA2 AFTs'].where(AFT_dataframe['SA2 Frequencies'] == 5)
    SA2_AFT = SA2_T.mean()

    SA1_T = AFT_dataframe['SA1 AFTs'].where(AFT_dataframe['SA1 Frequencies'] == 5)
    SA1_AFT = SA1_T.mean()

    #print("FA1 AFT:", FA1_AFT, "FA2 AFT:", FA2_AFT, "SA1 AFT:", SA1_AFT, "SA2 AFT:", SA2_AFT)

    # ----------------------------------------- #

    # Generating Figures #

    # ----------------------------------------- #

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    ax = sns.lineplot(x=responsive_freqs['FA1'], y=FA1, label='FAI')
    #ax = sns.scatterplot(x=responsive_freqs['FA1'], y=FA1, label='FAI')
    ax = sns.lineplot(x=responsive_freqs['FA2'], y=FA2, label='FAII')
    #ax = sns.scatterplot(x=responsive_freqs['FA2'], y=FA2, label='FAII')
    ax = sns.lineplot(x=responsive_freqs['SA1'], y=SA1, label='SAI')
    #ax = sns.scatterplot(x=responsive_freqs['SA1'], y=SA1, label='SAI')
    ax = sns.lineplot(x=responsive_freqs['SA2'], y=SA2, label='SAII')
    #ax = sns.scatterplot(x=responsive_freqs['SA2'], y=SA2, label='SAII')

    ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xlim=[0, 250],  ylabel="Amplitude (mm)", xlabel="Frequency (Hz)",
           title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_averageindividual_AFT_log_noscatter.png", format='png')
    #plt.close(fig)

    fig2 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    ax2 = sns.lineplot(x=responsive_freqs['FA1'], y=FA1, label='FA1')
    ax2 = sns.lineplot(x=responsive_freqs['FA2'], y=FA2, label='FA2')
    ax2 = sns.lineplot(x=responsive_freqs['SA1'], y=SA1, label='SA1')
    ax2 = sns.lineplot(x=responsive_freqs['SA2'], y=SA2, label='SA2')
    ax2.set(ylim=[-0.2, 2], xlim=[0, 250],  ylabel="Amplitude (mm)", xlabel="Frequency (Hz)",
            title="Absolute Threshold defined as " + str(absolute) + " Hz")

    plt.savefig(figpath + str(absolute) + "Hz_averageindividual_AFT_LINEAR_.png", format='png')
    plt.close(fig2)

    return AFT_dataframe


def class_threshold_visualisation(absolute, filepath, figpath, amps, freqs):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus

       Arguments:

            absolute (float): Absolute firing threshold in Hz
            filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
            afferent population is to mimic some experimental data
            figpath(str): Where to save result plots
            fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
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

    for runs in range(0, 1):

        class_min = class_absolute_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
                                              amps=amps, freqs=origin)

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

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10,
                                            "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
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

    fig2 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10,
                                            "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                            "xtick.labelsize": 12, "ytick.labelsize": 12})

    sns.set_context("talk", rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 12, "ytick.labelsize": 12})

    ax2 = sns.lineplot(x=freqs, y=FA1, label='FA1')
    ax2 = sns.lineplot(x=freqs, y=FA2, label='FA2')
    ax2 = sns.lineplot(x=freqs, y=SA1, label='SA1')
    ax2 = sns.lineplot(x=freqs, y=SA2, label='SA2')
    ax2.set(ylim=[-0.2, 2], xlim=[0, 250], title="Absolute Threshold defined as "+str(absolute)+" Hz")

    plt.savefig(figpath + str(absolute) + "Hz_class_absolute_threshold_LINEAR_.png", format='png')
    plt.close(fig2)


def class_responsive_amps_visualisation(absolute, filepath, figpath, freqs):

    """ Plots the responsive amplitudes of an afferent class for a given set of frequencies and amplitudes of stimulus

           Arguments:

               absolute (float): Absolute firing threshold in Hz
               filepath(str): Path of the excel file with the afferent recorded using microneurography
               figpath(str): Where to save result plots


           Returns:

              Plots thresholds per frequency in log scale.
           """

    class_responsive_amplitudes = get_responsive_amplitudes(absolute=absolute, filepath=filepath, output=figpath)

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


def class_firingrates(figpath, comparison, figname, figtitle):

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

    FS_classes = classgrouping_comparativeFRs(comparison=comparison)  # Grouping the FR per affclass

    FAI = FS_classes['FAI']
    FAII = FS_classes['FAII']
    SAI = FS_classes['SAI']
    SAII = FS_classes['SAII']

    fig = plt.figure(figsize=(20, 10), dpi=500)

    sns.set(style='white', font_scale=5)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 15, "ytick.labelsize": 15,
                                            "lines.markersize": 10})


    fig.suptitle(figtitle, fontsize=35, y=1)

    for keys in sorted(FAI):

        plt.subplot(1, 4, plotcount)

        axmax = 0

        if len(FAI[keys][1]) > 0 or len(FAI[keys][0]) > 0:

            # Provision to plot against stimulus pairs
            # ticks = axes_ticks(FAI[keys][2])

            # Plotting the FS datapoints
            ax = sns.scatterplot(x=np.array(FAI[keys][1]), y=np.array(FAI[keys][0]), color="blue")  # label="Footsim"

            # Plotting the Empirical datapoints
            # ax = sns.scatterplot(x=np.array(FAIfreq), y=np.array(FAI[keys][1]), color="black", label="Empirical")

    ax.set_title("FAI", pad=10)
    ax.set(xlim=[0, 200], ylim=[0, 200], xlabel="Empirical", ylabel="FootSim")

    plotcount = plotcount + 1


    for keys in sorted(FAII):

        plt.subplot(1, 4, plotcount)

        if len(FAII[keys][1]) > 0 or len(FAII[keys][0]) > 0:

            # Provision to plot against stimulus pairs
            # ticks = axes_ticks(FAII[keys][2])

            # Plotting the FS datapoints
            ax = sns.scatterplot(x=np.array(FAII[keys][1]), y=np.array(FAII[keys][0]),
                                 color="orange", )  # label="Footsim"

            # Plotting the Empirical datapoints
            # ax = sns.scatterplot(x=np.array(FAIIfreqs), y=np.array(FAII[keys][1]), color="black", label="Empirical")

    ax.set_title("FAII", pad=10)
    ax.set(ylim=[0, 250], xlim=[0, 250], xlabel="Empirical", ylabel="FootSim")

    plotcount = plotcount + 1

    for keys in sorted(SAI):

        plt.subplot(1, 4, plotcount)

        if len(SAI[keys][1]) > 0 or len(SAI[keys][0]) > 0:

            # Provision to plot against stimulus pairs
            # ticks = axes_ticks(SAI[keys][2])

            # Plotting the FS datapoints
            ax = sns.scatterplot(x=np.array(SAI[keys][1]), y=np.array(SAI[keys][0]), color="green")  # label="Footsim"

            # Plotting the Empirical datapoints
            # ax = sns.scatterplot(x=np.array(SAIfreqs), y=np.array(SAI[keys][1]), color="black", label="Empirical")

    ax.set_title("SAI", pad=10)
    ax.set(ylim=[0, 80], xlim=[0, 80], xlabel="Empirical", ylabel="FootSim")

    plotcount = plotcount + 1

    for keys in sorted(SAII):

        plt.subplot(1, 4, plotcount)

        if len(SAII[keys][1]) > 0 or len(SAII[keys][0]) > 0:

            # Provision to plot against stimulus pairs
            # ticks = axes_ticks(SAII[keys][2])

            # Plotting the FS datapoints
            ax = sns.scatterplot(x=np.array(SAII[keys][1]), y=np.array(SAII[keys][0]), color="red")  # label="Footsim"

            # Plotting the Empirical datapoints
            # ax = sns.scatterplot(x=np.array(SAIIfreqs), y=np.array(SAII[keys][1]), color="black", label="Empirical")

    ax.set_title("SAII", pad=10)
    ax.set(ylim=[0, 60], xlim=[0, 60], xlabel="Empirical", ylabel="FootSim")

    plotcount = plotcount + 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath + figname, format='png')
    # plt.close(fig)

    return 0


def comparative_scatters(figpath, comparison, figname):

    """ Generate the comparative scatter plots in between footsim outputs and the biological responses for a given set
    of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

         Arguments:

             comparison(dict): Dictionary with both responses
             figpath(str): Where to save result plots
             figname(str): Output file name

         Returns:

            Comparative plots
         """

    plotcount = 1

    FS_classes = classgrouping_comparativeFRs(comparison=comparison) # Grouping the FR per affclass

    classes = ['FAI', 'FAII', 'SAI', 'SAII']

    fig = plt.figure(figsize=(28, 22), dpi=200)

    sns.set(style='white', font_scale=2)

    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 5,
                                "xtick.labelsize": 15, "ytick.labelsize": 15, "lines.markersize": 10})

    for c in classes:

        for keys in sorted(FS_classes[c]):

            if len(FS_classes[c][keys][1]) > 0 or len(FS_classes[c][keys][0]) > 0:

                plt.subplot(6, 9, plotcount)

                ax = sns.scatterplot(x=np.array(FS_classes[c][keys][1]), y=np.array(FS_classes[c][keys][0]), color=fs.constants.affcol[fs.constants.affclass_mapping[c]])
                ax.set_title(keys, pad=10)

                if max(FS_classes[c][keys][0]) > max(FS_classes[c][keys][1]):
                    ax.set(xlim=[0, max(FS_classes[c][keys][0])], ylim=[0, max(FS_classes[c][keys][0])], xlabel="Empirical", ylabel="FootSim")
                    plt.plot([0, max(FS_classes[c][keys][0])], [0, max(FS_classes[c][keys][0])], color='k')

                else:
                    ax.set(xlim=[0, max(FS_classes[c][keys][1])], ylim=[0, max(FS_classes[c][keys][1])], xlabel="Empirical", ylabel="FootSim")
                    plt.plot([0, max(FS_classes[c][keys][1])], [0, max(FS_classes[c][keys][1])], color='k')

                plotcount = plotcount + 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='png')

    return FS_classes


def comparative_scatters_col(figpath, comparison, figname):

    """ Generate the comparative rate intensity plots in between footsim outputs and the biological responses for a given set
    of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

         Arguments:

             comparison(dict): Dictionary with both responses
             figpath(str): Where to save result plots
             figname(str): Output file name

         Returns:

            Comparative plots
         """

    plotcount = 1

    FS_classes = classgrouping_comparativeFRs(comparison=comparison) # Grouping the FR per affclass

    classes = ['FAI', 'FAII', 'SAI', 'SAII']
    lim ={'FAI':320, 'FAII':320, 'SAI':100, 'SAII':100}

    fig = plt.figure(figsize=(28, 22), dpi=200)

    all_freqs = np.array([3, 5, 8, 10, 20, 30, 60, 100, 150, 250])
    idx = np.linspace(0,1,all_freqs.size)
    col = dict()
    for i,f in enumerate(all_freqs):
        col[f] = plt.cm.plasma(idx[i])

    for c in classes:

        ax = plt.subplot(6, 9, plotcount)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_aspect('equal')
        sns.set_style("ticks")

        plt.plot([0,lim[c]],[0,lim[c]],color='black')

        for keys in sorted(FS_classes[c]):

            r_mod = np.array(FS_classes[c][keys][0])
            r_emp = np.array(FS_classes[c][keys][1])
            stim = np.array(FS_classes[c][keys][2])
            freqs = np.unique(stim[:,1])

            for f in freqs:
                idx = stim[:,1]==f
                sns.scatterplot(x=r_emp[idx], y=r_mod[idx], color=col[f],zorder=f)

        plotcount = plotcount + 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='svg')


def comparative_rate_intensity_functions(figpath, comparison, figname):

    """ Generate the comparative rate intensity plots in between footsim outputs and the biological responses for a given set
    of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

         Arguments:

             comparison(dict): Dictionary with both responses
             figpath(str): Where to save result plots
             figname(str): Output file name

         Returns:

            Comparative plots
         """

    plotcount = 1

    FS_classes = classgrouping_comparativeFRs(comparison=comparison) # Grouping the FR per affclass

    classes = ['FAI', 'FAII', 'SAI', 'SAII']

    fig = plt.figure(figsize=(28, 22), dpi=200)

    #sns.set(style='white', font_scale=2)

    #sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 5})#,
                                #"xtick.labelsize": 15, "ytick.labelsize": 15, "lines.markersize": 10})

    all_freqs = np.array([3, 5, 8, 10, 20, 30, 60, 100, 150, 250])
    idx = np.linspace(0,1,all_freqs.size)
    col = dict()
    for i,f in enumerate(all_freqs):
        col[f] = plt.cm.plasma(idx[i])

    for c in classes:

        for keys in sorted(FS_classes[c]):

            r_mod = np.array(FS_classes[c][keys][0])
            r_emp = np.array(FS_classes[c][keys][1])
            stim = np.array(FS_classes[c][keys][2])
            freqs = np.unique(stim[:,1])

            ax = plt.subplot(6, 9, plotcount)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(keys, pad=10)
            plt.xscale('log')
            plt.xlim(0.01,2)

            hd = []
            for f in freqs:
                idx = stim[:,1]==f
                ss = stim[idx,0]
                rm = r_mod[idx]
                re = r_emp[idx]
                ix = np.argsort(ss)
                base_line, = plt.plot(ss[ix],rm[ix],linewidth=2,color=col[f],alpha=0.5,label=f)
                hd.append(base_line)
                plt.plot(ss[ix],re[ix],linewidth=2,linestyle='dashed',color=base_line.get_color())

            #ax.legend(handles=hd)

            plotcount = plotcount + 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='svg')


def combined_visualisation_ramps_and_stimulus(filepath, figpath):

    """ Function for the generation of the combined ramp figure for the FootSim paper

         Returns:

            Line plot of ramp and hold stimulus and raster plots for the combined responses for all afferent classes

         """

    visualise_ramps(figpath=figpath)

    ramps_classrasters(filepath=filepath, output_path=figpath)


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


def FR_vs_hardness(figpath, filepath):

    """ Generate the comparative scatter plots in between footsim firing rate outputs and the skin hardness of the
        regions where they were placed

             Arguments:

             filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
             afferent population is to mimic some experimental data
             figpath(str): Where to save result plots
             fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
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

    populations = empirical_afferent_positioning(filepath)  # Generates investigated population

    footsim_rates = model_firing_rates(populations=populations, filepath=filepath)  # Gathers firing rates

    for keys in footsim_rates:

        for t in range(0, footsim_rates[keys].size):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class

            if afferent_class == 'SA1':

                SA1.append(footsim_rates[keys][t][0])
                SA1_hardness.append(fs.constants.hardnessRegion(str(keys[2])))

            elif afferent_class == 'FA1':

                FA1.append(footsim_rates[keys][t][0])
                FA1_hardness.append(fs.constants.hardnessRegion(str(keys[2])))

            elif afferent_class == 'SA2':

                SA2.append(footsim_rates[keys][t][0])
                SA2_hardness.append(fs.constants.hardnessRegion(str(keys[2])))

            elif afferent_class == 'FA2':

                FA2.append(footsim_rates[keys][t][0])
                FA2_hardness.append(fs.constants.hardnessRegion(str(keys[2])))

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


def hard_vs_soft_sole_regions(regional_min):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus
        per region compared with regional hardness grouping by hard or soft regions

         Arguments:

             regional_min(dict): Dictionary with the thresholds grouped per region of interest

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

            if fs.constants.hardnessRegion(str(keys[1])) < 31.229:

                SA1_tsoft.append(regional_min[keys])
                SA1_soft.append(fs.constants.hardnessRegion(str(keys[1])))

            else:

                SA1_thard.append(regional_min[keys])
                SA1_hard.append(fs.constants.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA1':

            if fs.constants.hardnessRegion(str(keys[1])) < 31.229:

                FA1_tsoft.append(regional_min[keys])
                FA1_soft.append(fs.constants.hardnessRegion(str(keys[1])))

            else:

                FA1_thard.append(regional_min[keys])
                FA1_hard.append(fs.constants.hardnessRegion(str(keys[1])))

        elif keys[0] == 'SA2':

            if fs.constants.hardnessRegion(str(keys[1])) < 31.229:

                SA2_tsoft.append(regional_min[keys])
                SA2_soft.append(fs.constants.hardnessRegion(str(keys[1])))

            else:

                SA2_thard.append(regional_min[keys])
                SA2_hard.append(fs.constants.hardnessRegion(str(keys[1])))

        if keys[0] == 'FA2':

            if fs.constants.hardnessRegion(str(keys[1])) < 31.229:

                FA2_tsoft.append(regional_min[keys])
                FA2_soft.append(fs.constants.hardnessRegion(str(keys[1])))

            else:

                FA2_thard.append(regional_min[keys])
                FA2_hard.append(fs.constants.hardnessRegion(str(keys[1])))

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


def modelFR_visualisation(figpath, FRs, figname):

    """ Generate scatter plots of footsim outputs for a given set of amplitudes and frequencies of stimulus

           Arguments:

               FRs(dict): Dictionary with responses
               figpath(str): Where to save result plots
               figname(str): Output file name


           Returns:

              Scatter plots.
           """

    fig = plt.figure(figsize=(32, 18), dpi=300)

    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 5,
                                "xtick.labelsize": 15, "ytick.labelsize": 15, "lines.markersize": 10})

    fig.suptitle("Debug of Firing Rates", fontsize=35, y=1)

    plotcount = 1

    for keys in FRs.keys():

        plt.subplot(6, 9, plotcount)
        plotcount = plotcount + 1

        ax = sns.scatterplot(data=np.array(FRs[keys][0]))
        ax.set(title=str(keys))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath + figname + "Footsim_FRs.png", format='png')


def hardness_vs_thresholds_visualisation(figpath, absolute, regional_min):

    """ Plots the absolute thresholds of an afferent class for a given set of frequencies and amplitudes of stimulus
    per region compared with regional hardness

         Arguments:

             absolute(int): Threshold definition in Hz
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
            SA1_hardness.append(fs.constants.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA1':

            FA1.append(regional_min[keys])
            FA1_hardness.append(fs.constants.hardnessRegion(str(keys[1])))

        elif keys[0] == 'SA2':

            SA2.append(regional_min[keys])
            SA2_hardness.append(fs.constants.hardnessRegion(str(keys[1])))

        elif keys[0] == 'FA2':

            FA2.append(regional_min[keys])
            FA2_hardness.append(fs.constants.hardnessRegion(str(keys[1])))

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


def individual_models_threshold_visualisation(absolute, filepath, figpath, matched=True):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
    stimulus

        Arguments:

        absolute (float): Absolute firing threshold in Hz
        filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
        afferent population is to mimic some experimental data
        figpath(str): Where to save result plots
        fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
        populations generation, numbers should match the ones in the *.csv if reproducing experimental data

         Returns:

            Plots thresholds per frequency in log scale.
         """

    SA1 = dict()
    SA2 = dict()

    FA1 = dict()
    FA2 = dict()

    stimulus_pairs = dict()
    responsive_freqs = dict()

    # Get the population for looping purposes
    populations = empirical_afferent_positioning(filepath=filepath)

    individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath, matched=matched)

    # Keys of the individual_min (AFT) dictionary:

    # 0 = Afferent class
    # 1 = Model number (idx)
    # 2 = Location on the foot sole
    # 3 = Frequency where that response happened

    for keys in individual_min:  # Creating the lists #

        if keys[0] == 'SA1' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            stimulus_pairs[keys[0], keys[1], keys[2]] = list()
                            SA1[keys[1], keys[2]] = list()
                            responsive_freqs[keys[0], keys[1], keys[2]] = list()

        elif keys[0] == 'FA1' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            stimulus_pairs[keys[0], keys[1], keys[2]] = list()
                            FA1[keys[1], keys[2]] = list()
                            responsive_freqs[keys[0], keys[1], keys[2]] = list()

        elif keys[0] == 'SA2' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            stimulus_pairs[keys[0], keys[1], keys[2]] = list()
                            SA2[keys[1], keys[2]] = list()
                            responsive_freqs[keys[0], keys[1], keys[2]] = list()

        elif keys[0] == 'FA2' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            stimulus_pairs[keys[0], keys[1], keys[2]] = list()
                            FA2[keys[1], keys[2]] = list()
                            responsive_freqs[keys[0], keys[1], keys[2]] = list()

    # for runs in range(1, 4):

    # print("Run of the individual models AFTs number:", runs)

    # individual_min = individual_models_thresholds(absolute=absolute, filepath=filepath, figpath=figpath,
    # fs.constants.affid=fs.constants.affid, amps=amps, freqs=freqs)

    for keys in individual_min:  # Grouping the responsive amplitudes #

        if keys[0] == 'SA1' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:

                            SA1[populations[keys[2]].afferents[afferent].idx, loc].append(individual_min[keys])
                            stimulus_pairs[keys[0], keys[1], keys[2]].append((keys[3], individual_min[keys]))
                            responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

        elif keys[0] == 'FA1' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            FA1[populations[keys[2]].afferents[afferent].idx, loc].append(individual_min[keys])
                            stimulus_pairs[keys[0], keys[1], keys[2]].append((keys[3], individual_min[keys]))
                            responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

        elif keys[0] == 'SA2' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            SA2[populations[keys[2]].afferents[afferent].idx, loc].append(individual_min[keys])
                            stimulus_pairs[keys[0], keys[1], keys[2]].append((keys[3], individual_min[keys]))
                            responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

        elif keys[0] == 'FA2' and keys[2] != "T3_t" and keys[2] != "T4_t" and keys[2] != "T5_t":

            for loc in fs.constants.foot_tags:

                if keys[2] == loc:

                    for afferent in range(len(populations[keys[2]])):  # Loops through afferents in that location

                        if fs.foot_surface.locate(populations[keys[2]].afferents[afferent].location)[0][0] == loc \
                                and populations[keys[2]].afferents[afferent].affclass == keys[0] \
                                and keys[1] == populations[keys[2]].afferents[afferent].idx:
                            FA2[populations[keys[2]].afferents[afferent].idx, loc].append(individual_min[keys])
                            stimulus_pairs[keys[0], keys[1], keys[2]].append((keys[3], individual_min[keys]))
                            responsive_freqs[keys[0], keys[1], keys[2]].append(keys[3])

    grouped_thresholds = dict()  # Groups the results per afferent class
    grouped_thresholds['FA1'] = FA1
    grouped_thresholds['FA2'] = FA2
    grouped_thresholds['SA1'] = SA1
    grouped_thresholds['SA2'] = SA2

    # ----------------------------------------- #

    # Generating Figures #

    # ----------------------------------------- #

    fig = plt.figure(figsize=(15, 5), dpi=500)

    sns.set_style("ticks")

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 12,
                                            "ytick.labelsize": 12})
    fig.suptitle("FootSim", fontsize=22, y=1)

    sns.set_palette("Blues")
    ax = plt.subplot(1, 4, 1)

    for keys in FA1:

        idx = np.argsort(np.array(responsive_freqs['FA1', keys[0], keys[1]]))
        plt.plot(np.array(responsive_freqs['FA1', keys[0], keys[1]])[idx], np.array(FA1[keys])[idx])

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([10 ** -3, 10 ** 1])
    plt.xlim([1, 1000])
    ax.set(xlabel='Frequency', ylabel='Amplitude')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_palette("Reds")
    ax = plt.subplot(1, 4, 2)

    for keys in FA2:

        idx = np.argsort(np.array(responsive_freqs['FA2', keys[0], keys[1]]))
        plt.plot(np.array(responsive_freqs['FA2', keys[0], keys[1]])[idx], np.array(FA2[keys])[idx])

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([10 ** -3, 10 ** 1])
    plt.xlim([1, 1000])
    ax.set(xlabel='Frequency', ylabel='Amplitude')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_palette("Greens")
    ax = plt.subplot(1, 4, 3)

    for keys in SA1:

        idx = np.argsort(np.array(responsive_freqs['SA1', keys[0], keys[1]]))
        plt.plot(np.array(responsive_freqs['SA1', keys[0], keys[1]])[idx], np.array(SA1[keys])[idx])

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([10 ** -3, 10 ** 1])
    plt.xlim([1, 1000])
    ax.set(xlabel='Frequency', ylabel='Amplitude')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_palette("Greys")
    ax = plt.subplot(1, 4, 4)

    for keys in SA2:

        idx = np.argsort(np.array(responsive_freqs['SA2', keys[0], keys[1]]))
        plt.plot(np.array(responsive_freqs['SA2', keys[0], keys[1]])[idx], np.array(SA2[keys])[idx])

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([10 ** -3, 10 ** 1])
    plt.xlim([1, 1000])
    ax.set(xlabel='Frequency', ylabel='Amplitude')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig(figpath + "Thres_mod_" + str(absolute) + "_Hz.svg", format='svg')

    return grouped_thresholds, stimulus_pairs


def impcycle_visualisation(ImpPath, figpath):

    data_file = pd.read_excel(ImpPath)

    afferents = ['FAI', 'FAII', 'SAI', 'SAII']

    table_affs = ['FA1', 'FA2', 'SA1', 'SA2']

    Imps = dict()

    grouped_by_threshold = data_file.copy(deep=True)

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(context='notebook', style='whitegrid', font_scale=2)

    sns.set_context("talk", rc={"font.size": 50, "axes.titlesize": 10, "axes.labelsize": 15, "lines.linewidth": 2,
                                "xtick.labelsize": 16, "ytick.labelsize": 20, "lines.markersize": 5})

    for aff in range(0, 4):  # Amplitude in Log scale

        afferent_class = grouped_by_threshold[grouped_by_threshold['class'].str.fullmatch(table_affs[aff])].copy(deep=True)

        if len(afferent_class['Frequency '].where(afferent_class['ImpCycle'] > 1)) > 0:

            Imps[str(table_affs[aff])] = np.array(afferent_class['Frequency '].where(afferent_class['ImpCycle'] > 1).dropna())

            ax = sns.lineplot(data=afferent_class, x=afferent_class['Frequency '].where(afferent_class['ImpCycle'] > 1),
                              y=afferent_class['amplitude '].where(afferent_class['ImpCycle'] > 1),
                              label=afferents[aff], ci='sd')

            #ax = sns.scatterplot(data=afferent_class,
                                 #x=afferent_class['Frequency'].where(afferent_class['ImpCycle'] == 1),
                                 #y=afferent_class['Amplitude'].where(afferent_class['ImpCycle'] == 1),
                                 #label=afferents[aff])

            ax.set(ylim=[10**-3, 10**1], xlim=[0, 250], yscale="log")

    figure = figpath + "Impcycles.png"
    fig.suptitle("ImpCycle > 1", fontsize=35, y=0.95)
    plt.legend(loc='upper right')
    plt.savefig(figure)

    return Imps


def plot_responses(figpath, amp, freq, location, response):

    if len(response.spikes) > 0:

        fig = plt.figure(dpi=500)
        r = plot(response)
        figsave(hvobj=r, size=400,
                filename=figpath+"R_for_"+str(location)+"_at_"+str(amp)+"_mm_"+str(freq)+"_Hz")
        plt.close(fig)


def ramp_response_rasters(filepath, output_path, **args):
    """ Visualise the responses of the afferents to ramps in a raster plot

    Args:
        filepath: path to 'microneurography_nocomments.xlsx'
        output_path (str): path to location to store output files - ideally a path to a folder
        **args:
            amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
            pin_radius (float): radius of the stimulus pin used (mm)
            ramp_length (float): length of time the ramp is applied to the foot (sec)
            foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                    all regions will be investigated
            afferent_classes (list): list containing the names of afferent classes to investigate

    Returns:
        .png files with plots of the response of each afferent type per region in raster plot style

    """

    amplitude = args.get('amplitude', 1.) # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5) # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.) # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index', 'all') # list containing the indexes of the regions to be stimulated
    afferent_classes = args.get('afferent_classes', ['FA1','FA2','SA1','SA2']) # list of afferent classes to investigate

    # generate afferent population
    afferent_populations = empirical_afferent_positioning(filepath=filepath)

    # generate full list of region indexes
    if foot_region_index == 'all':
        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface
    centres = fs.foot_surface.centers

    # generate list of region names
    regions = list(afferent_populations.keys())

    # loop through region indexes
    for index in foot_region_index:

        # check whethere there are no afferents in the region
        if len(afferent_populations[regions[index]].afferents) == 0:
            continue

        else:
            # get name of foot region
            foot_region = regions[index]

            # generate ramp to centre of foot region
            s = fs.generators.stim_ramp(amp=amplitude, ramp_type='lin', loc=centres[index], len=ramp_length, pin_radius=pin_radius)

            # generate response
            r = afferent_populations[foot_region].response(s)

            # loop through afferent classes
            for affClass in afferent_classes:

                # check whether there is are afferents of this class
                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:
                    continue

                else:
                    # get response of the afferents
                    data = r[afferent_populations[foot_region][affClass]].psth(1)

                    # get spike times of responses to ramp
                    spike_times = np.zeros((data.shape[0], data.shape[1]))
                    for i in range(data.shape[0]):
                        spike_times_index = np.array(np.where(data[i] != 0))
                        for j in range(len(spike_times_index[0])):
                            spike_times[i][j] = spike_times_index[0][j]

                    # plot raster plot
                    plt.figure(figsize=(15, 10))
                    plt.eventplot(spike_times, linelengths=0.5, color=fs.constants.affcol[affClass])
                    plt.ylabel('Afferent')
                    plt.yticks([])
                    plt.xlabel('Time (ms)')
                    plt.title(affClass + ' responses to ramp at centre of ' + str(foot_region) + ': amplitude = ' + str(amplitude) + \
                              'mm, time = ' + str(ramp_length) + 'seconds, pin size = ' + str(pin_radius) + 'mm', fontsize=15)
                    plt.savefig(output_path + foot_region + ' ' + affClass + ' raster.png')


def ramps_classrasters(filepath, output_path, **args):

    """

        Args:
            filepath (str): path to location of microneurography_nocomments.xlsx file
            output_path (str) = path to location to store output files - ideally a path to a folder
            **args:
                amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
                pin_radius (float): radius of the stimulus pin used (mm)
                ramp_length (float): length of time the ramp is applied to the foot (sec)
                foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                        all regions will be investigated

        Returns:

        """

    amplitude = args.get('amplitude', 1.)  # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5)  # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.)  # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index',
                                 'all')  # list containing the indexes of the regions to be stimulated
    afferent_classes = ['FA1', 'FA2', 'SA1', 'SA2']  # list of afferent classes to investigate

    # generate afferent population

    afferent_populations = empirical_afferent_positioning(filepath=filepath)

    # generate full list of region indexes

    if foot_region_index == 'all':

        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface

    centres = fs.foot_surface.centers

    # generate list of region names

    regions = list(afferent_populations.keys())

    number_of_models = {'FA1': 0, 'FA2': 0, 'SA1': 0, 'SA2': 0}

    affclass_spikes = {'FA1': np.zeros((1, 2000)), 'FA2': np.zeros((1, 2000)), 'SA1': np.zeros((1, 2000)), 'SA2': np.zeros((1, 2000))}

    for index in foot_region_index:

       # spikes = np.zeros((1, 2000))  # time window

        # get name of foot region
        foot_region = regions[index]

        # generate ramp to centre of foot region
        s = fs.generators.stim_ramp(amp=amplitude, loc=centres[index], ramp_type='lin', len=ramp_length,
                                    ramp_len=0.2,
                                    pin_radius=pin_radius)

        # check to see if there are any afferents in this region

        if len(afferent_populations[regions[index]].afferents) == 0:

            affClass_colours = []

            continue

        else:

            # generate response
            r = afferent_populations[foot_region].response(s)


            for affClass in afferent_classes:

                # check to see if there are models of the specific afferent class in this region

                # PSTH = Peri-stimulus time histogram #

                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:

                    number_of_models[affClass] = 0

                else:

                    number_of_models[affClass] = r[afferent_populations[foot_region][affClass]].psth(1).shape[0]

                    data = r[afferent_populations[foot_region][affClass]].psth(1)

                    # initialise array to store spike times

                    spike_times = np.zeros((data.shape[0], data.shape[1]))

                    # get spike times

                    for i in range(data.shape[0]):

                        spike_times_index = np.array(np.where(data[i] != 0))

                        for j in range(len(spike_times_index[0])):

                            spike_times[i][j] = spike_times_index[0][j]

                    affclass_spikes[affClass] = np.vstack((spike_times, affclass_spikes[affClass]))

    # ---- Plotting loop ---- #

    for affClass in affclass_spikes:

        flat = affclass_spikes[affClass].flatten()

        fig = plt.figure(figsize=(14, 7), dpi=500)
        plt.suptitle('Responses to ramp and hold stimuli', fontsize=15)
        plt.title(str(affClass), fontsize=20)
        plt.eventplot(flat, linelengths=0.3, color=fs.constants.affcol[affClass])
        plt.xlim([0, 2000])
        plt.savefig(output_path + str(affClass) + "_ramps.png", dpi=300)


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


def RMSE_boxplot(figpath, RMSE_allclasses):

    """ Generates a bar plot with RMSE values from the comparison of footsim and empirical data for all afferent classes

     Arguments:

         RMSE_allclasses(dict): Dictionary with RMSEs for individual models and afferent classes as keys

     Returns:

        Comparative boxplot.

     """

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in RMSE_allclasses.items()]))

    fig2 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 12, "ytick.labelsize": 12})
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, palette='Greys_r')
    ax.set(title="RMSE of Firing Rates", ylabel="RMSE value", xlabel="Afferent class", ylim=[0, 75])

    figname_ = figpath + "_RMSE_allclasses_.png"

    plt.savefig(figname_, format='png')
    # plt.close(fig2)


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


def single_afferent_threshold_visualisation(figpath, freqs, single_min):

    """ Plots the absolute thresholds of individual afferent models for a given set of frequencies and amplitudes of
        stimulus

         Arguments:

             single_min(dict): Dictionary with the thresholds grouped per afferent class
             figpath(str): Where to save result plots
             idx(int): Afferent model ID
             affclass(str): Afferent model class

         Returns:

            Plots thresholds per frequency in log scale.
         """

    fig = plt.figure(dpi=150)

    for keys in single_min:

        if len(single_min[keys]) > 0:

            ax = sns.lineplot(x=np.array(freqs), y=np.array(single_min[keys]), label=str(keys))
            ax.set(yscale='log', ylim=[10 ** -3, 10 ** 1], xscale='log', xlim=[1, 1000])
            ax.legend(fontsize=4, loc=1)

    figure = figpath + "_thres_.png"

    plt.savefig(figure, format='png')


def visualise_ramps(figpath, **args):

    """ Visualise the ramp stimuli given for a ramp and hold experiment

       Args:
           filepath: path to 'microneurography_nocomments.xlsx'
           figpath (str): path to location to store output files - ideally a path to a folder
           **args:
               amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
               pin_radius (float): radius of the stimulus pin used (mm)
               ramp_length (float): length of time the ramp is applied to the foot (sec)
               foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                       all regions will be investigated
               afferent_classes (list): list containing the names of afferent classes to investigate

       Returns:
           .png files with plots of the response of each afferent type per region in raster plot style

       """

    amplitude = args.get('amplitude', 1.)  # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5)  # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.)  # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index', 'all')  # list containing the indexes of the regions to be stimulated

    # generate afferent population

    s = fs.generators.stim_ramp(amp=amplitude, ramp_type='lin', len=ramp_length, ramp_len=0.2, pin_radius=pin_radius)

    fig = plt.figure(figsize=(14, 7), dpi=500)

    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 50, "axes.labelsize": 15, "lines.linewidth": 10,
                                "xtick.labelsize": 16, "ytick.labelsize": 20, "lines.markersize": 50})

    samples = list(range(0, 10000))

    trace = s.trace

    trace = trace[0]

    ax = sns.lineplot(x=samples, y=trace, color="black")
    ax.set(ylabel="Milimetres", xlabel="Seconds")

    plt.savefig(figpath + "strace.png", format='png')
    plt.close(fig)


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


def FR_model_vs_empirical(figpath, filepath, scatter=True):

    """ Compares Footsim outputs and the original biological responses for a given set of amplitudes and frequencies of
     stimulus.

        Arguments:

            filepath (str): Path of a *.csv file containing empirically recorded afferents
            figpath(str): Where to save result plots
            fs.constants.affid(dict): Dictionary with afferent classes as keys containing afferent numbers to be used in the
            populations generation, numbers should match the ones in the *.csv if reproducing experimental data
            amps(np.array of float): array containing amplitudes of stimulus
            freqs(np.array of float): array containing frequencies of stimulus


        Returns:

           Dictionary of firing rates for each stimulus for both footsim and empirical data

        """

    start = time.asctime(time.localtime(time.time()))

    populations = empirical_afferent_positioning(filepath=filepath)  # Generates the affpop
    footsim_rates = model_firing_rates(populations=populations, filepath=filepath)  # Investigate fs firing rates

    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data
    grouped_by_threshold = data_file.copy(deep=True)  # Gets the empirical thresholds if needed

    comparison = dict()  # Dictionary that will get all the rates from the model and the empirical dataset

    for keys in footsim_rates:

       for t in range(0, len(footsim_rates[keys])):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class
            idx = populations[keys[2]][afferent_c].idx  # Gathers its model number

            comparison[fs.constants.affid[afferent_class][idx]] = list()  # Position 0 for Footsim, 1 for Empirical,
            comparison[fs.constants.affid[afferent_class][idx]].append(list())
            comparison[fs.constants.affid[afferent_class][idx]].append(list())

            comparison[fs.constants.affid[afferent_class][idx]].append(list())  # Stimulus pairs

    for keys in footsim_rates:

        for t in range(0, len(footsim_rates[keys])):

            afferent_c = t  # Enquires with afferent
            afferent_class = populations[keys[2]][afferent_c].affclass  # Gathers its class
            idx = populations[keys[2]][afferent_c].idx  # Gathers its model number

            # Checking if the spreadsheed of empirical data has said model #

            if grouped_by_threshold[grouped_by_threshold['Afferent ID'].
                    str.fullmatch(fs.constants.affid[afferent_class][idx])].empty == False:

                # If yes, isolate that afferent #

                data_slice = grouped_by_threshold[grouped_by_threshold['Afferent ID'].
                    str.fullmatch(fs.constants.affid[afferent_class][idx])].copy(deep=True)

                if data_slice[data_slice['Frequency '] == keys[1]].empty == True:

                    continue

                else:

                    # Isolate the specific stimulus used for that afferent #

                    empirical_fr = data_slice[data_slice['Frequency '] == keys[1]].copy(deep=True)

                    empirical_fr = empirical_fr[empirical_fr['amplitude '] == keys[0]].copy(deep=True)

                    if empirical_fr.empty == False: #and empirical_fr.iloc[0]['AvgInst'] != 0:

                        comparison[fs.constants.affid[afferent_class][idx]][1].append(empirical_fr.iloc[0]['AvgInst'])

                        comparison[fs.constants.affid[afferent_class][idx]][2].append((keys[0], keys[1]))

                        if isinstance(footsim_rates[keys][t], np.ndarray) == True:

                            comparison[fs.constants.affid[afferent_class][idx]][0].append(footsim_rates[keys][t][0][0])

                        else:

                            comparison[fs.constants.affid[afferent_class][idx]][0].append(footsim_rates[keys][t])

    print("Simulation started at ", start)
    print("Simulation finished at ", time.asctime(time.localtime(time.time())))

    file = open(figpath + "FR_model_x_empirical.txt", "w+")

    for keys in comparison:

        before = str(keys) + " FS: " + str(comparison[keys][0])
        file.write(before)

        file.write('\n')
        file.write('\n')

        after = str(keys) + " empirical: " + str(comparison[keys][1])
        file.write(after)
        file.write('\n')
        file.write('\n')

        after = str(keys) + " Stimulus pairs: " + str(comparison[keys][2])
        file.write(after)
        file.write('\n')
        file.write('\n')
        file.write('\n')


    #grouped_classesFR = class_firingrates(figpath=figpath, comparison=comparison, figname="Group_attempt.png",
                                          #figtitle="Class FR Comparison")


    if scatter:
        FS_classes_comparative = comparative_scatters(figpath=figpath, comparison=comparison.copy(), figname="FR_comparison.png")
    else:
        FS_classes_comparative = comparative_rate_intensity_functions(figpath=figpath, comparison=comparison.copy(), figname="FR_comparison_ri.png")

    #residuals = find_residuals(FR_comparative=FS_classes_comparative, figpath=figpath, figname='Residuals_FR_')

    return comparison


def hardness_vs_multiple_thresholds(filepath, figpath, freqs):

    for absolute in range(1, 25):

        print("Working threshold definition of ", absolute, " Hz.")

        regional_min = regional_absolute_thresholds(absolute=absolute, filepath=filepath, figpath=figpath, freqs=freqs)

        hardness_vs_thresholds_visualisation(figpath=figpath, absolute=absolute, regional_min=regional_min)


def ImpCycle_model_vs_empirical(figpath, filepath):

    start = time.asctime(time.localtime(time.time()))

    populations = empirical_afferent_positioning(filepath)  # Generates the fs afferent population
    footsim_ImpCycle = ImpCycle(figpath=figpath, filepath=filepath)

    data_file = pd.read_excel(filepath)  # Reads the file with the empirical data
    grouped_by_threshold = data_file.copy(deep=True)  # Gets the empirical thresholds if needed

    comparison = dict()  # Dictionary that will get all the rates from the model and the empirical dataset

    for keys in footsim_ImpCycle:

        for ImpCycle_value in range(0, len(footsim_ImpCycle[keys])):

            location = keys[2]
            affclass = populations[location].afferents[ImpCycle_value].affclass
            model_idx = populations[location].afferents[ImpCycle_value].idx  # Gathers its model number

            comparison[fs.constants.affid[affclass][model_idx]] = list()  # Position 0 for Footsim, 1 for Empirical
            comparison[fs.constants.affid[affclass][model_idx]].append(list())
            comparison[fs.constants.affid[affclass][model_idx]].append(list())
            comparison[fs.constants.affid[affclass][model_idx]].append(list())

    for keys in footsim_ImpCycle:

        for t in range(0, len(footsim_ImpCycle[keys])):

            location = keys[2]
            affclass = populations[location].afferents[t].affclass
            model_idx = populations[location].afferents[t].idx  # Gathers its model number

            if grouped_by_threshold[
                grouped_by_threshold['Afferent ID'].str.fullmatch(fs.constants.affid[affclass][model_idx])].empty == False:

                data_slice = grouped_by_threshold[
                    grouped_by_threshold['Afferent ID'].str.fullmatch(fs.constants.affid[affclass][model_idx])].copy(deep=True)

                empirical_fr = data_slice[data_slice['Frequency '] == keys[1]]  # Gets the stimulus frequency

                if empirical_fr.empty == False:

                    comparison[fs.constants.affid[affclass][model_idx]][1].append(empirical_fr.iloc[0]['ImpCycle'])

                    comparison[fs.constants.affid[affclass][model_idx]][2].append((keys[0], keys[1]))

                    if not math.isnan(footsim_ImpCycle[keys][t]):

                        comparison[fs.constants.affid[affclass][model_idx]][0].append(footsim_ImpCycle[keys][t][0][0])

                    else:

                        comparison[fs.constants.affid[affclass][model_idx]][0].append(0)

    print("Simulation started at ", start)
    print("Simulation finished at ", time.asctime(time.localtime(time.time())))

    comparative_scatters(figpath=figpath, comparison=comparison, figname="ImpCycle_comparison_reg.png",
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

    all_classes_dict = dict()
    average_residual = dict()
    class_residuals = dict()
    class_sum_of_residuals = dict()
    individual_residuals = dict()
    sum_of_residuals = dict()

    # Loops through afferent classes #

    for FS_class in FR_comparative:

        sum_of_residuals[FS_class] = dict()
        class_residuals[FS_class] = list()

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
            individual_residuals[str(individual_afferent)] = sum_of_residuals[FS_class][individual_afferent] / individual_rates

        average_residual[FS_class] = class_sum_of_residuals[FS_class] / all_rates

        individualaff_averageresidual = np.array(individualaff_averageresidual)
        all_classes_dict[FS_class] = individualaff_averageresidual


    # Converting into dataframes and plotting #

    indivRES_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in individual_residuals.items()]))

    fig = plt.figure(figsize=(28, 15), dpi=500)

    sns.set_context("talk", rc={"font.size": 20, "axes.titlesize": 10, "axes.labelsize": 15, "lines.linewidth": 10,
                                "xtick.labelsize": 16, "ytick.labelsize": 20, "lines.markersize": 50})

    plt.scatter(*zip(*sorted(individual_residuals.items())), color='r', s=120)
    plt.xlabel('Afferent IDs')
    plt.xticks(rotation=80)
    plt.ylim([0, 20])
    plt.tight_layout()

    plt.ylabel('Average Residual (Hz)')
    plt.savefig(figpath + figname + 'FR_allmodels.png', format='png')
    plt.close(fig)

    fig2 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 12, "ytick.labelsize": 12})

    RES_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_classes_dict.items()]))

    ax = sns.boxplot(data=RES_df)  # palette='cividis'
    ax = sns.swarmplot(data=RES_df, palette='Greys_r')

    ax.set(title="Residuals - Comparative FRs", ylabel="Average residuals (Hz)", ylim=[0, 50])

    plt.savefig(figpath + figname +'FRresiduals_boxplot.png', format='png')
    #plt.close(fig)

    return class_residuals, sum_of_residuals, average_residual


def FR_dict_comparison(figpath, raw_rates, filtered_rates):

    """ Compares firing rates generated with model_firing_rates() against filtered firing rates after running
    FR_filtering()

               Arguments:

                   raw_rates(dict): Dictionary with initial responses
                   filtered_rates(dict): Dictionary with filtered responses
                   figpath(str): Output file path


               Returns:

                  Comparative text file with both dictionaries
               """

    file = open(figpath + "FR_array_sizecomparison.txt", "w+")

    beforesize = 0
    aftersize = 0

    for keys in raw_rates:

        beforesize = beforesize + raw_rates[keys].size
        aftersize = aftersize + filtered_rates[keys].size

        if filtered_rates[keys].size > 0:

            before = str(keys) + "before array: " + str(raw_rates[keys].size)
            file.write(before)

            file.write('\n')

            after = str(keys) + "after array: " + str(filtered_rates[keys].size)
            file.write(after)
            file.write('\n')

    file.close()

    print("Before:", beforesize, " and after: ", aftersize)


def empirical_hardness_statistics(filepath, figpath):

    """ Runs an ANOVA to compare empirical hardness in all specific locations

       Arguments:

        filepath (str): Path of a *.csv file containing empirically recorded afferents if the simulated
        figpath(str): Complete path of the saving location

      Returns:

         Test results as a *.csv spreadsheet.

          """

    # Relevant location lists #

    location_specific = ['LatArch', 'Toes', 'LatMet', 'Heel', 'MidMet', 'MidArch', 'MedArch', 'GT', 'MedMet']
    arch_specific = ['LatArch', 'MidArch', 'MedArch']
    met_specific = ['LatMet', 'MidMet', 'MedMet']

    data_file = pd.read_excel(filepath)

    # First we test general locations #

    Archseries = data_file['RF_hardness'].where(data_file['location_general'] == "Arch")
    Toes = data_file['RF_hardness'].where(data_file['location_general'] == "Toes")
    Metseries = data_file['RF_hardness'].where(data_file['location_general'] == "Met")
    Heel = data_file['RF_hardness'].where(data_file['location_general'] == "Heel")

    location_general = {"Arch": Archseries, "Toes": Toes, "Met": Metseries, "Heel": Heel} # Acquires general locations

    location = pd.DataFrame.from_dict(location_general)  # Converts it to a dataframe

    pd.DataFrame.to_csv(location, path_or_buf=figpath + "Hardness.csv")  # Saves it as *.csv

    oneway_ANOVA_plus_TukeyHSD(dataframe=location, file_name="Hardness_per_region", outputpath=figpath) # ANOVA test

    arch_dict = dict()
    met_dict = dict()

    # And then finds the specific locations and adds them to a dictionary #

    for arch in arch_specific:

        arch_dict[arch] = data_file['RF_hardness'].where(data_file['locatoin specific '] == arch)

    for met in met_specific:

        met_dict[met] = data_file['RF_hardness'].where(data_file['locatoin specific '] == met)

    # Converts them to a dataframe

    arch_df = pd.DataFrame.from_dict(arch_dict)
    met_df = pd.DataFrame.from_dict(met_dict)

    # Saves them as *.csv

    pd.DataFrame.to_csv(arch_df, path_or_buf=figpath + "Arch_specific_hardness.csv")
    pd.DataFrame.to_csv(met_df, path_or_buf=figpath + "Met_specific_hardness.csv")

    # One-way ANOVA tests with Tukey HSD

    oneway_ANOVA_plus_TukeyHSD(dataframe=arch_df, file_name="Hardness_in_Arch", outputpath=figpath)
    oneway_ANOVA_plus_TukeyHSD(dataframe=met_df, file_name="Hardness_in_Met", outputpath=figpath)

    return 0


def oneway_ANOVA_plus_TukeyHSD(dataframe, file_name, outputpath):

    # Dropping NaNs

    NaNfree = [dataframe[col].dropna() for col in dataframe]

    # Running the ANOVA per se

    fvalue, pvalue = stats.f_oneway(*NaNfree)

    #print("F =", fvalue)
    #print("p =", pvalue)
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


def regressions(figname, figpath):

    filename = figname + "r_squared_.txt"

    RMSE_allclasses = {"FA1": list(), "FA2": list(), "SA1": list(), "SA2": list()}

    #def reg_angle(coefficient):

        #angle = math.atan(regressor.coef_)
        #angle = angle * 180 / math.pi

        #return (angle)

    file = open(figpath + filename, "w+")

    # predictors = np.array(FAI[keys][1])
    # x = predictors.reshape(-1, 1)

    # outcome = np.array(FAI[keys][0])
    # y = outcome.reshape(-1, 1)

    # Split 80% of the data to the training set while 20% of the data to test set using below code.

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Create lists for visualisation

    # X_train_list = list()
    # X_test_list = list()
    # y_train_list = list()
    # y_test_list = list()
    # y_pred_list = list()

    # Create the model and fit it

    # regressor = LinearRegression()

    # if len(X_train) > 1 and len(y_train) > 1:

    # regressor.fit(X_train, y_train)  # Training the algorithm

    # y_pred = regressor.predict(X_test)  # Computing predicted values

    # The coefficients
    # print('Coefficient: ', regressor.coef_)

    # The intercept
    # print("\nIntercept: ", regressor.intercept_)

    # The mean squared error
    # print('\nMean squared error: %.2f' % mean_squared_error(y_test, y_pred))

    # The coefficient of determination: 1 is perfect prediction
    # print('\nR squared: %.2f' % r2_score(y_test, y_pred))

    # lines = keys + " has a R squared of " + str(r2_score(y_test, y_pred))
    # RMSE = keys + " has a RMSE of " + str(RMSE_optimised(FAI[keys][0], FAI[keys][1]))
    # RMSE_allclasses['FA1'].append(RMSE_optimised(FAI[keys][0], FAI[keys][1]))
    # file.write(RMSE)
    # file.write('\n')
    # file.write(lines)
    # file.write('\n')

    # for items in range(0, X_test.size):
    # X_test_list.append(X_test[items][0])

    # for items in range(0, X_train.size):
    # X_train_list.append(X_train[items][0])

    # for items in range(0, y_test.size):
    # y_test_list.append(y_test[items][0])

    # for items in range(0, y_train.size):
    # y_train_list.append(y_train[items][0])

    # for items in range(0, y_pred.size):
    # y_pred_list.append(y_pred[items][0])

    # ax = sns.lineplot(x=X_test_list, y=y_pred_list, color="black")  # Adding the regression line

    file.close()

    return 0


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

        if np.isnan(footsim[nan_check]) == True:

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


def RMSE_AFTindividualmodels(absolute, filepath, figpath):

    # Output files #

    filename = figpath + "Individual_models_RMSE_" + str(absolute) + "Hz.txt"
    file = open(filename, "w+")
    RMSEs = dict()
    RMSE_allclasses = {"FA1": list(), "FA2": list(), "SA1": list(), "SA2": list()}
    zscores_allclasses = {"SA1": list(), "SA2": list(), "FA1": list(), "FA2": list()}

    # Reading the empirical data #

    data_file = pd.read_excel(filepath)
    grouped_by_threshold = data_file[data_file['Threshold '].notnull()].copy(deep=True)  # Gets the empirical thresholds

    # Computing model outputs #

    individual_grouped_thresholds = individual_models_threshold_visualisation(absolute=absolute, filepath=filepath,
                                                                              figpath=figpath)

    model_freqs = individual_grouped_thresholds[1]

    individual_grouped_thresholds = individual_grouped_thresholds[0]


     # Stimulation pairs that footsim responds to

    # Performing comparisons #

    for affclass in individual_grouped_thresholds.keys():

        for idx in individual_grouped_thresholds[affclass].keys():

            afferent_id = grouped_by_threshold[grouped_by_threshold['Afferent ID'].
                str.fullmatch(str(fs.constants.affid[affclass][idx[0]]))].copy(deep=True)

            empty = afferent_id['amplitude '].empty

            if empty is True:

                continue

            else:

                empirical = np.array(afferent_id['amplitude '])

                empiricalfreqs = np.array(afferent_id['Frequency '])

                footsim = np.array(individual_grouped_thresholds[affclass][idx])

                key = str(affclass) + "_" + str(idx[0])

                affID = str(fs.constants.affid[affclass][idx[0]])

                footsim_pairs = model_freqs[affclass, idx[0], idx[1]]

                rmse_val = RMSE_optimised(affID=affID, empirical=empirical, empiricalfreqs=empiricalfreqs, filepath=filepath, footsim=footsim, footsim_pairs=footsim_pairs)

                if rmse_val != 2 and rmse_val != 0:

                    RMSEs[key] = rmse_val/2
                    RMSE_allclasses[affclass].append(rmse_val/2)
                    line = str(key) + ": " + str(RMSEs[key])
                    file.write(line)
                    file.write('\n')

    # Saving output #

    file.close()

    fig = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10,
                                            "axes.titlesize": 20, "axes.labelsize": 15, "lines.linewidth": 2,
                                            "xtick.labelsize": 12, "ytick.labelsize": 12})

    figname = figpath + "Individual_models_RMSE" + "_" + str(absolute) + "Hz.png"

    plt.scatter(*zip(*sorted(RMSEs.items())), color='k', s=40)
    plt.xlabel('Afferent IDs')
    plt.xticks(rotation=45)
    plt.title('RMSE of individual models')
    plt.ylabel('RMSE')
    plt.ylim(-0.02, 1)

    plt.savefig(figname, format='png')

    plt.close(fig)

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in RMSE_allclasses.items()]))

    pvalues = oneway_ANOVA_plus_TukeyHSD(dataframe=df, file_name="RMSE_ANOVA_TukeyHSD_" + str(absolute) + "Hz",
                                         outputpath=figpath)

    fig2 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook',  rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                             "lines.linewidth": 2, "xtick.labelsize": 12, "ytick.labelsize": 12})
    ax = sns.boxplot(data=df)
    ax = sns.swarmplot(data=df, palette='Greys_r')
    ax.set(title="RMSE of all classes", ylabel="Normalised RMSE value", ylim=[0, 1])

    #add_stat_annotation(ax, data=df, text_format='star', loc='inside', verbose=2, linewidth=3,
    #                    box_pairs=[("SA1", "SA2"), ("SA1", "FA1"), ("SA1", "FA2"), ("FA1", "FA2"), ("FA1", "SA2"), ("FA2", "SA2")], pvalues=pvalues,
    #                    perform_stat_test=False, line_offset_to_box=0.15)

    figname_ = figpath + "_RMSE_allclasses_" + "_" + str(absolute) + "Hz.png"

    plt.savefig(figname_, format='png')


    zscores_allclasses['SA1'] = np.abs(stats.zscore(RMSE_allclasses['SA1']))
    zscores_allclasses['SA2'] = np.abs(stats.zscore(RMSE_allclasses['SA2']))
    zscores_allclasses['FA1'] = np.abs(stats.zscore(RMSE_allclasses['FA1']))
    zscores_allclasses['FA2'] = np.abs(stats.zscore(RMSE_allclasses['FA2']))

    fig3 = plt.figure(figsize=(13, 10), dpi=500)

    sns.set(style='whitegrid', font_scale=2)

    sns.set_context(context='notebook', rc={"font.size": 10, "axes.titlesize": 20, "axes.labelsize": 15,
                                            "lines.linewidth": 2, "xtick.labelsize": 12, "ytick.labelsize": 12})

    dfz = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in zscores_allclasses.items()]))

    ax = sns.barplot(data=dfz)
    ax = sns.swarmplot(data=dfz, palette='Greys_r')
    ax.set(title="Z-score of RMSEs", ylabel="Z-score", ylim=[0, 3])

    figname_ = figpath + "_zscore_RMSE_" + "_" + str(absolute) + "Hz.png"
    plt.savefig(figname_, format='png')

    plt.close(fig3)

    return RMSE_allclasses


def RMSE_multipleAFTdefs(filepath, figpath, amps, freqs):

    for absolute in range(1, 25):

        RMSE_AFTindividualmodels(absolute=absolute, filepath=filepath, figpath=figpath)


def RMSE_optimised(affID, filepath, empirical, empiricalfreqs, footsim, footsim_pairs):

    """ Compares Footsim thresholds and the original biological responses for a given set of amplitudes and frequencies
       of stimulus.

          Arguments:

              empirical(dict): Dictionary of empirical responses
              footsim(dict): Dictionary of footsim responses


          Returns:

             Root Mean Square Error (RMSE) value.

          """

    # Removing NaNs #

    nan_array = np.isnan(footsim)
    not_nan_array = ~ nan_array
    footsim_minus_nans = footsim[not_nan_array]

    comp_fs = list()  # Creating the arrays for the RMSE comparison
    comp_emp = list()

    for emp in range(0, empirical.size):  # Matching the arrays using responses to matching frequencies

            for fs in range(0, footsim_minus_nans.size):

                footsimfreq = footsim_pairs[fs][0]

                if footsimfreq == empiricalfreqs[emp]:

                    comp_fs.append(footsim_minus_nans[fs])
                    comp_emp.append(empirical[emp])

    if len(comp_fs) == len(comp_emp):

        rmse = mean_squared_error(np.array(comp_emp), np.array(comp_fs), squared=False)  # Sklearn

        return rmse

    else:

        return 0


# ----------------------------------------- #

# File export and array handling methods #

# ----------------------------------------- #

def classgrouping_comparativeFRs(comparison):

    """ Groups the comparative dictionaries in between footsim outputs and the biological responses for a given set
       of amplitudes and frequencies of stimulus generated with FR_model_vs_empirical() or ImpCycle_model_vs_empirical()

            Arguments:

                comparison(dict): Dictionary with both responses
                figpath(str): Where to save result plots
                figname(str): Output file name
                figtitle(str): Output figure suptitle

            Returns:

               Four dictionaries, one per class
            """


    FAI = dict()
    FAII = dict()
    SAI = dict()
    SAII = dict()

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

    FS_classes = dict()

    FS_classes['FAI'] = FAI
    FS_classes['FAII'] = FAII
    FS_classes['SAI'] = SAI
    FS_classes['SAII'] = SAII

    return FS_classes


def correct_FRoutputs(FRs):

    clean_array = list()

    for items in FRs:

        if type(items) == np.ndarray:

            clean_array.append(items[0][0])

        else:

            clean_array.append(np.nan)

    #clean_array = np.array(clean_array)

    return clean_array


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


def ImpCycle_csvexport(figpath, ImpCycle, population):

    spreadsheet = figpath + 'ImpCycle_19_Feb21.csv'

    with open(spreadsheet, mode='w') as ImpCycle_csv:

        ImpCycle_csv = csv.writer(ImpCycle_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                  lineterminator='\n')

        ImpCycle_csv.writerow(['Afferent_ID', 'type', 'Location', 'Amplitude', 'Frequency',
                               'ImpCycle'])

        for keys in ImpCycle:

            for ImpCycle_value in range(0, len(ImpCycle[keys])):

                if not math.isnan(ImpCycle[keys][ImpCycle_value]):

                    amp = keys[0]
                    freq = keys[1]
                    location = keys[2]
                    affclass = population[location].afferents[ImpCycle_value].affclass
                    model_idx = population[location].afferents[ImpCycle_value].idx
                    affID = fs.constants.affid[affclass][model_idx]

                    ImpCycle_csv.writerow([str(affID), str(affclass), str(location), str(amp),
                                           str(freq), str(ImpCycle[keys][ImpCycle_value][0][0])])


def sort_afferent_data(afferent_data):
    """ Sorts afferent_data dictionary so that classes are grouped within regions

    Args:
        afferent_data:

    Returns:
        sorted_afferent_data (dict): Region: Class: (Model ID, model index)

    """

    # initialise dictionary
    sorted_afferent_data = dict()

    # afferent classes
    classes = ['FA1','FA2','SA1','SA2']

    # loop through regions
    for region in afferent_data:

        sorted_afferent_data[region] = dict()

        # loop through afferent classes
        for affClass in classes:

            sorted_afferent_data[region][affClass] = list()

            for i in range(len(afferent_data[region])):

                if afferent_data[region][i][1] == affClass:

                    sorted_afferent_data[region][affClass].append([afferent_data[region][i][0], afferent_data[region][i][2]])

                else:
                    continue

    return sorted_afferent_data


def ramps_regional(filepath, output_path, **args):
    """ Generates raster plots for responses by empirically positioned afferents to a ramp and hold stimulus

    Args:
        filepath (str): path to location of microneurography_nocomments.xlsx file
        output_path (str) = path to location to store output files - ideally a path to a folder
        **args:
            amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
            pin_radius (float): radius of the stimulus pin used (mm)
            ramp_length (float): length of time the ramp is applied to the foot (sec)
            foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                    all regions will be investigated

    Returns:

    """

    amplitude = args.get('amplitude', 1.)  # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5)  # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.)  # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index', 'all')  # list containing the indexes of the regions to be stimulated
    afferent_classes = ['FA1', 'FA2', 'SA1', 'SA2']  # list of afferent classes to investigate

    # generate afferent population
    afferent_populations = empirical_afferent_positioning(filepath=filepath)

    # generate full list of region indexes
    if foot_region_index == 'all':
        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface
    centres = fs.foot_surface.centers

    # generate list of region names
    regions = list(afferent_populations.keys())

    number_of_models = {'FA1': 0, 'FA2': 0, 'SA1' : 0, 'SA2' : 0}

    for index in foot_region_index:

        spikes = np.zeros((1, 2000))

        # get name of foot region
        foot_region = regions[index]

        # generate ramp to centre of foot region
        s = fs.generators.stim_ramp(amp=amplitude, loc=centres[index], ramp_type='lin', len=ramp_length, ramp_len=0.2,
                                    pin_radius=pin_radius)

        # check to see if there are any afferents in this region
        if len(afferent_populations[regions[index]].afferents) == 0:

            continue

        else:

            # generate response
            r = afferent_populations[foot_region].response(s)

            for affClass in afferent_classes:

                # check to see if there are models of the specific afferent class in this region
                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:

                    number_of_models[affClass] = 0

                else:
                    number_of_models[affClass] = r[afferent_populations[foot_region][affClass]].psth(1).shape[0]

                    data = r[afferent_populations[foot_region][affClass]].psth(1)

                    # initialise array to store spike times
                    spike_times = np.zeros((data.shape[0], data.shape[1]))

                    # get spike times
                    for i in range(data.shape[0]):

                        spike_times_index = np.array(np.where(data[i] != 0))

                        for j in range(len(spike_times_index[0])):

                            spike_times[i][j] = spike_times_index[0][j]

                    spikes = np.vstack((spike_times,spikes))


        spikes = np.delete(spikes, -1, 0)

        affClass_colours = [[0,0,0]]

        for affClass in number_of_models:

            class_colours = [fs.constants.affcol[affClass]] * number_of_models[affClass]

            affClass_colours = class_colours + affClass_colours

        affClass_colours = affClass_colours[:-1]

        # plot raster plot
        plt.figure(figsize=(15, 10))
        plt.suptitle('Responses  of all classes to ramp at centre of ' + str(foot_region) + ': amplitude = ' + str(amplitude) + \
                  'mm, time = ' + str(ramp_length) + 'seconds, pin size = ' + str(pin_radius) + 'mm', fontsize=15)

        plt.subplot(2,1,1)
        plt.plot(s.trace[0])
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('Ramp stimulus')

        plt.subplot(2,1,2)
        plt.eventplot(spikes, linelengths=0.5, color=affClass_colours)
        plt.ylabel('Afferent')
        plt.yticks([])
        plt.xticks([])
        plt.xlabel('Time (ms)')
        plt.savefig(output_path + foot_region +  ' all afferents raster.png')


def ramps_regional_singlefigure_multipanel(filepath, output_path, **args):
    """

    Args:
        filepath (str): path to location of microneurography_nocomments.xlsx file
        output_path (str) = path to location to store output files - ideally a path to a folder
        **args:
            amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
            pin_radius (float): radius of the stimulus pin used (mm)
            ramp_length (float): length of time the ramp is applied to the foot (sec)
            foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                    all regions will be investigated

    Returns:

    """

    amplitude = args.get('amplitude', 1.)  # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5)  # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.)  # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index', 'all')  # list containing the indexes of the regions to be stimulated
    afferent_classes = ['FA1', 'FA2', 'SA1', 'SA2']  # list of afferent classes to investigate

    # generate afferent population
    afferent_populations = empirical_afferent_positioning(filepath=filepath)

    # generate full list of region indexes
    if foot_region_index == 'all':
        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface
    centres = fs.foot_surface.centers

    # generate list of region names
    regions = list(afferent_populations.keys())

    number_of_models = {'FA1': 0, 'FA2': 0, 'SA1': 0, 'SA2' : 0}

    plt.figure(figsize=(15, 15), dpi=600)
    plt.suptitle('Responses to ramp and hold stimuli', fontsize=15)

    q = 0

    for index in foot_region_index:

        spikes = np.zeros((1, 2000))

        # get name of foot region
        foot_region = regions[index]

        # generate ramp to centre of foot region
        s = fs.generators.stim_ramp(amp=amplitude, loc=centres[index], ramp_type='lin', len=ramp_length, ramp_len=0.2,
                                    pin_radius=pin_radius)

        # check to see if there are any afferents in this region
        if len(afferent_populations[regions[index]].afferents) == 0:

            affClass_colours = []

            continue

        else:

            # generate response
            r = afferent_populations[foot_region].response(s)

            for affClass in afferent_classes:

                # check to see if there are models of the specific afferent class in this region
                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:

                    number_of_models[affClass] = 0

                else:
                    number_of_models[affClass] = r[afferent_populations[foot_region][affClass]].psth(1).shape[0]

                    data = r[afferent_populations[foot_region][affClass]].psth(1)

                    # initialise array to store spike times
                    spike_times = np.zeros((data.shape[0], data.shape[1]))

                    # get spike times
                    for i in range(data.shape[0]):

                        spike_times_index = np.array(np.where(data[i] != 0))

                        for j in range(len(spike_times_index[0])):

                            spike_times[i][j] = spike_times_index[0][j]

                    spikes = np.vstack((spike_times,spikes))


        spikes = np.delete(spikes, -1, 0)

        affClass_colours = [[0,0,0]]

        for affClass in number_of_models:

            class_colours = [fs.constants.affcol[affClass]] * number_of_models[affClass]

            affClass_colours = class_colours + affClass_colours

        affClass_colours = affClass_colours[:-1]

        if len(affClass_colours) == 0:

            continue

        else:

            plt.title("Afferent responses to ramp and hold stimuli", fontsize=20)
            plt.eventplot(spikes, linelengths=0.3, color=affClass_colours)
            plt.ylabel('Afferent', fontsize=15)
            plt.yticks([])
            plt.xticks([])
            plt.xlabel('Time (ms)', fontsize=15)

        q += 1

    #plt.plot(s.trace[0],color='black')
    #plt.xticks([])
    #plt.yticks([])
    #plt.title('Ramp stimulus',fontsize=10)
    #plt.ylabel('Indentation',fontsize=5)
    plt.savefig(output_path + 'ramp_rasters_rodrigo010422.png', dpi=300)

    return spikes, affClass_colours


def ramps_singlefigure(filepath, output_path, **args):

    """

    Args:
        filepath (str): path to location of microneurography_nocomments.xlsx file
        output_path (str) = path to location to store output files - ideally a path to a folder
        **args:
            amplitude (float): amplitude of indentation - how far into the skin is the ramp applied (mm)
            pin_radius (float): radius of the stimulus pin used (mm)
            ramp_length (float): length of time the ramp is applied to the foot (sec)
            foot_region_index (list): list containing indexes referring the regions of the foot. When set to 'all',
                    all regions will be investigated

    Returns:

    """

    amplitude = args.get('amplitude', 1.)  # amplitude of indentation (mm)
    pin_radius = args.get('pin_radius', 1.5)  # radius of stimulation pin (mm)
    ramp_length = args.get('ramp_length', 2.)  # length of time ramp is applied for (s)
    foot_region_index = args.get('foot_region_index',
                                 'all')  # list containing the indexes of the regions to be stimulated
    afferent_classes = ['FA1', 'FA2', 'SA1', 'SA2']  # list of afferent classes to investigate

    # generate afferent population

    afferent_populations = empirical_afferent_positioning(filepath=filepath)

    # generate full list of region indexes

    if foot_region_index == 'all':

        foot_region_index = list(range(13))

    # get location of the centre of each region on the foot surface

    centres = fs.foot_surface.centers

    # generate list of region names

    regions = list(afferent_populations.keys())

    number_of_models = {'FA1': 0, 'FA2': 0, 'SA1': 0, 'SA2': 0}

    q = 0

    fig = plt.figure(figsize=(14, 7), dpi=500)
    plt.suptitle('Responses to ramp and hold stimuli', fontsize=15)

    for index in foot_region_index:

        spikes = np.zeros((1, 2000))  # time window

        # get name of foot region
        foot_region = regions[index]

        # generate ramp to centre of foot region
        s = fs.generators.stim_ramp(amp=amplitude, loc=centres[index], ramp_type='lin', len=ramp_length,
                                    ramp_len=0.2,
                                    pin_radius=pin_radius)

        # check to see if there are any afferents in this region

        if len(afferent_populations[regions[index]].afferents) == 0:

            affClass_colours = []

            continue

        else:

            # generate response
            r = afferent_populations[foot_region].response(s)

            for affClass in afferent_classes:

                # check to see if there are models of the specific afferent class in this region

                # PSTH = Peri-stimulus time histogram #

                if r[afferent_populations[foot_region][affClass]].psth(1).T.shape[0] == 0:

                    number_of_models[affClass] = 0

                else:

                    number_of_models[affClass] = r[afferent_populations[foot_region][affClass]].psth(1).shape[0]

                    data = r[afferent_populations[foot_region][affClass]].psth(1)

                    # initialise array to store spike times

                    spike_times = np.zeros((data.shape[0], data.shape[1]))

                    # get spike times

                    for i in range(data.shape[0]):

                        spike_times_index = np.array(np.where(data[i] != 0))

                        for j in range(len(spike_times_index[0])):

                            spike_times[i][j] = spike_times_index[0][j]

                    spikes = np.vstack((spike_times, spikes))

        spikes = np.delete(spikes, -1, 0)

        affClass_colours = [[0, 0, 0]]

        for affClass in number_of_models:

            class_colours = [fs.constants.affcol[affClass]] * number_of_models[affClass]

            affClass_colours = class_colours + affClass_colours

        affClass_colours = affClass_colours[:-1]

        if len(affClass_colours) == 0:

            continue

        else:

            plt.title("All", fontsize=20)
            plt.eventplot(spikes, linelengths=0.3, color=affClass_colours)
            plt.ylabel('Afferent', fontsize=15)
            plt.yticks([])
            plt.xticks([])
            plt.xlabel('Time (ms)', fontsize=15)

        q += 1

    plt.savefig(output_path + str(q), dpi=300)

    return spikes, affClass_colours


def ramps_byclass(figpath, figname):

    fig = plt.figure(figsize=(7, 5.5))
    t = np.linspace(0,1,100)

    ramps = dict()
    for aix,aff in enumerate(['SA1','SA2','FA1','FA2']):
        ramps[aff] = np.zeros((fs.constants.affparams[aff].shape[0],100))
        for i in range(fs.constants.affparams[aff].shape[0]):
                        
            reg = fs.constants.affreg[aff][i]
            sur = fs.Surface(tags=[reg])

            # place Afferent
            a = fs.Afferent(affclass=aff,idx=i,surface=sur)

            # generate ramp to centre of foot region
            s = fs.stim_ramp(amp=2, ramp_type='sin', len=0.9, ramp_len=0.25, 
                             pad_len=0.05, pin_radius=3, surface=sur)

            r = a.response(s)
            ramps[aff][i] = r.psth()

        #resp = np.mean(ramps[aff],axis=0)
        resp = np.convolve(np.mean(ramps[aff],axis=0), [.25, .5, .25], mode='same')*100.

        ax = plt.subplot(2,2,aix+1)
        if aix==3:
            plt.ylim((0.,220))
            plt.plot(t,100.*scipy.signal.resample(s.trace[0],100),color='k')
        else:
            plt.ylim((0.,110))
            plt.plot(t,50.*scipy.signal.resample(s.trace[0],100),color='k')
        plt.plot(t,resp,color=fs.constants.affcol[aff], linewidth=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='svg')

    return ramps, s


def empirical_rfs(filepath):

    d = pd.read_excel(filepath)

    RF = dict()
    RF['SA1'] = dict()
    RF['SA2'] = dict()
    RF['FA1'] = dict()
    RF['FA2'] = dict()
    ids = d['Afferent ID'].unique()
    for id in ids:
        dat = d[d['Afferent ID'] == id].reset_index(drop=True)
        affclass = fs.constants.affclass_mapping[dat['class'][0]]
        try:
            #RF[affclass][id] = float(dat['RF Size '][0])
            area = float(dat['RF Size '][0])
            RF[affclass][id] = math.sqrt(area/math.pi)
        except:
            continue

    return RF


def single_rf(affclass, idx):

    """ Computes the receptive field area in mm^2 for a given afferent.

        Arguments:

            affclass(str): Afferent class of the population being tested.
            idx(int): Afferent model to be tested

        Returns:

            Receptive field area for the selected afferent

        """

    reg = fs.constants.affreg[affclass][idx]
    sur = fs.Surface(tags=[reg])

    ff = {'SA1':5, 'SA2':5, 'FA1':40, 'FA2':150}

    depths = np.linspace(0.005,0.5,45)
    min_loc = 0
    max_loc = 60
    thres_rate = 2

    # find threshold
    a = fs.Afferent(affclass=affclass,idx=idx,surface=sur)

    rr_thres = np.zeros(depths.shape)
    for i,d in enumerate(depths):
        s = fs.stim_sine(amp=d,freq=ff[affclass],pin_radius=1.,surface=sur)
        r = a.response(s)
        rr_thres[i] = r.rate()[0,0]

    try:
        thres_ind = np.argwhere(rr_thres>thres_rate)[0]
    except:
        return np.nan

    # find minimal responding distance
    s = fs.stim_sine(amp=depths[thres_ind]*3,freq=ff[affclass],pin_radius=1.,surface=sur)
    for i in range(10):
        a = fs.affpop_linear(min_dist=min_loc,max_dist=max_loc,num=3,affclass=affclass,idx=idx,surface=sur)
        r = a.response(s)
        rr = r.rate().flatten()

        try:
            ix = np.argwhere(rr<=thres_rate)[0]
        except:
            return np.nan

        if ix is None or ix==0:
            break
        max_loc = a.location[ix,0]
        min_loc = a.location[ix-1,0]

    print("Min: ", min_loc, ", Max: ", max_loc, ", Threshold: ", depths[thres_ind])

    #return ((max_loc+min_loc)/2.+1.)**2*math.pi
    return ((max_loc+min_loc)/2.+1.)


def model_rfs():
    RF_mod = dict()
    for a in fs.constants.affclasses:
        RF_mod[a] = dict()
        for i in range(fs.constants.affparams[a].shape[0]):
            RF_mod[a][fs.constants.affid[a][i]] = single_rf(a,i)

    return RF_mod


def comparative_rfs(figpath, figname, RF_emp, RF_mod):

    plotcount = 1

    classes = ['SA1', 'SA2', 'FA1', 'FA2']
    
    fig = plt.figure(figsize=(28, 22), dpi=200)

    for c in classes:

        ax = plt.subplot(6, 9, plotcount)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.set_aspect('equal')
        sns.set_style("ticks")


        aff = RF_mod[c].keys()
        rf_emp = np.zeros((len(aff)))
        rf_mod = np.zeros((len(aff)))
        for i,a in enumerate(aff):
            rf_mod[i] = RF_mod[c][a]
            rf_emp[i] = RF_emp[c][a]

        sns.scatterplot(x=rf_emp, y=rf_mod, color='black')

        plotcount = plotcount + 1

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='svg')

    return rf_emp, rf_mod


def comparative_rfs_swarm(figpath, figname, RF_emp, RF_mod):

    classes = ['SA1', 'SA2', 'FA1', 'FA2']

    rf_emp = pd.DataFrame(columns=['Class','RF size'])
    rf_mod = pd.DataFrame(columns=['Class','RF size'])

    counter = 0
    for c in classes:
        aff = RF_mod[c].keys()
        for a in aff:
            if np.isnan(RF_mod[c][a]):
                continue
            rf_mod.loc[counter] = [c, RF_mod[c][a]]
            counter += 1
    
    counter = 0
    for c in classes:
        aff = RF_emp[c].keys()
        for a in aff:
            rf_emp.loc[counter] = [c, RF_emp[c][a]]
            counter += 1

    fig = plt.figure(figsize=(7, 5.5), dpi=200)
    ax = plt.subplot(1, 2, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(ylim=(0, 31))
    sns.swarmplot(x='Class',y='RF size', data=rf_emp, hue='Class', palette=fs.constants.affcol)
    
    ax = plt.subplot(1, 2, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set(ylim=(0, 31))
    sns.swarmplot(x='Class',y='RF size', data=rf_mod, hue='Class', palette=fs.constants.affcol)
    ax.get_legend().remove()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figpath+figname, format='svg')

    return rf_emp, rf_mod
