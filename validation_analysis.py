from validation_toolbox import *

# Change the paths! #

filepath = 'C:\\Users\\pc1rss\\Dropbox\\Active Touch Laboratory\\Empirical data\\Foot_Guelph\\' \
           'microneurography_nocomments.xlsx'

figpath = "C:\\Users\\pc1rss\\Dropbox\\Active Touch Laboratory\\Software\\TouchSim - different versions\\footsim" \
          "\\footsim local\\python\\Model behaviour comparison\\Complete validation analysis\\43 models\\"

ImpPath = figpath + 'ImpCycle_43models_27_Nov20.xlsx'

amps = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2])

freqs = np.array([3, 5, 8, 10, 20, 30, 60, 100, 150, 250])

optfreqs = np.array([5, 30, 60])

valid_models = {'SA1': fs.constants.affidSA1,
                'SA2':  fs.constants.affidSA2,
                'FA1': fs.constants.affidFA1,
                'FA2': fs.constants.affidFA2}

valid_frequencies = {'FA1': 60, 'FA2': 30, 'SA1': 5, 'SA2': 5}

# You have to save the impcycles spreadsheet as an excel workbook

ImpCycle(figpath=figpath, filepath=filepath, valid_models=valid_models, amps=amps, freqs=freqs)


multiple_class_absolute_thresholds(filepath=filepath, figpath=figpath, valid_models=valid_models, amps=amps,
                                   freqs=freqs)

multiple_individual_absolute_thresholds(filepath=filepath, figpath=figpath, valid_models=valid_models, amps=amps, freqs=freqs)

ImpCycle_model_vs_empirical(figpath=figpath, filepath=filepath, valid_models=valid_models, amps=amps, freqs=freqs)

ImpCycle_visualisation(ImpPath=ImpPath, figpath=figpath)

FR_model_vs_empirical(figpath=figpath, filepath=filepath, valid_models=valid_models, amps=amps, freqs=freqs)

"""
RMSE_AFTindividualmodels(amps=amps, absolute=10, freqs=freqs, valid_models=valid_models, filepath=filepath,
                         figpath=figpath)
"""