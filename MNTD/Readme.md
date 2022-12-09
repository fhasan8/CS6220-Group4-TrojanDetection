# MNTD Code
This is the section for the MNTD model. 

## Source Code
We made use of the official MNTDbaseline implementation - https://github.com/mmazeika/tdc-starter-kit/blob/main/detection/example_submission.ipynb

MNTD python notebook contains the definition, training, evaluation and analysis code for the MNTD network.
It does expect dataset at a specific location "../../tdc_datasets/detection" for both trojan and clean folders of the dataset.
## Our Work

Our work on this model was adding Multiple MNTD networks which were created for experimentation. Some of them are saved in the notebook and can directly to be defined and used for training and evaluation.
The hyperparameters can be updated and the model can be trained in the notebook.

The new experimental networks along with the analysis code is separately added in the notebook.
The requriments.txt contains all the packages installed in the conda environment used for training MNTD.
