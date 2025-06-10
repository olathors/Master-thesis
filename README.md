# Master-thesis

Welcome to the repository for my master's thesis:

CNN Ensemble Multiclass Alzheimer’s Disease Stage Classification with Human Interpretable Results: A Single Modality Approach

### Abstract:

*In an age defined by the blistering pace of technological innovation,
Alzheimer’s Disease remains a bastion of unsolved medical mystery. With
cutting-edge artificial intelligence tools in hand, researchers worldwide
are making herculean efforts, chipping away at this most stubborn of
problems. Intending to contribute to these efforts, in this thesis, we
explore the utilization of an ensemble of multiple Convolutional Neural
Networks to classify the four disease stages: Cognitive Normal, Stable
Mild Cognitive Impairment, Progressive Mild Cognitive Impairment, and
Alzheimer’s Disease. By relying on only a single data modality, we maximize
the size of our usable dataset. In addition, by performing a feature
selection process tailored to each ensemble, we reduce the training data
volume while increasing the human-centered interpretability of our ensemble
model predictions. We propose and test three different EfficientNetV2-
based ensemble architectures, achieving a macro average Area Under the
Curves of 0.712, 0,724, and 0.735, respectively, on the four-class classification
problem. Our best performing ensemble, aptly named the 3&2-Class
ensemble, achieves a recall score of 0.419 on the Alzheimer’s Disease class
compared to the baseline of 0.239. Furthermore, using the state-of-the-
art post-hoc interpretability algorithm Shapley Additive Explanations, we
incorporate Shapley values into our proposed human-centered explainability
methods that allow for both local and global explanations. Together, these
form a robust and interpretable framework for early detection and disease
stage classification of Alzheimer’s disease.*

### Reproducing results:

1. To create the dataset, first use the notebook patient_selection.ipynb to create dictionaries containing the patients in each class. Then, use the notebook image_selection.ipynb to select relevant images for each patient in each class. Next, run the mri_dataset_general_reference.py script to create a reference csv for all images in the dataset, and finally run the mri_dataset_specific_reference.py scipt to create reference files for each class with train/test/val splits.

2. To preprocess the data, run the execute_preprocessing.py script. This executes the entire pre-processing pipeline.

3. To train the models, you could use your training script of choice with the ensemble models from combined_classifier.py with the corresponding custom dataset class. For the 4-Class ensemble, use the CombinedClassifierLogReg model with the custom dataset class in mri_dataset_combined.py. For the 3&2-Class ensemble, use the DoubleCombinedClassifierLogReg model with the custom dataset class in mri_dataset_combined_2.py. For the 1 vs Rest ensemble, use the FourCombinedClassifierLogReg model with the custom dataset class in mri_dataset_combined_3.py. Apologies for the naming conventions.

4. To create the explanations, run the test_shap_combined_fox.py script. This script was written for the fox cumpute cluster, and may require some changes to run on a different machine. This script requires a custom ensemble model and dataset class, and the ones found in the directory correspond to the 3&2-Class ensemble. It also requires the custom_image_shap.py script, which is a slightly altered script from the SHAP library.

### Python packages and versions:

Because of some compatability problems, the pre-processing pipeline was run in a dedicated virtual environment. The python packages and corresponding versions used for this envorionment are as follows:

Python version 3.11.10

Package                      Version
---------------------------- ---------
antspyx                      0.5.4
dicom2nifti                  2.5.0
itk                          5.4.0
itk-core                     5.4.0
itk-filtering                5.4.0
itk-io                       5.4.0
itk-numerics                 5.4.0
itk-registration             5.4.0
itk-segmentation             5.4.0
keras                        2.12.0
matplotlib                   3.9.2
matplotlib-inline            0.1.7
nibabel                      5.3.2
numpy                        2.1.3
pandas                       2.2.3
pillow                       11.0.0
pip                          24.3.1

All other scripts were run in another virtual environment. The python packages and corresponding versions used for this envorionment are as follows:

Python version 3.12.7

Package                           Version
--------------------------------- ------------------

keras                             3.7.0
matplotlib                        3.9.2
matplotlib-inline                 0.1.6
neptune                           1.13.0
nibabel                           5.3.2
numpy                             1.26.4
opencv-python                     4.11.0.86
pandas                            2.2.2
pillow                            10.4.0
pip                               24.2
scikit-image                      0.24.0
scikit-learn                      1.5.1
scipy                             1.13.1
seaborn                           0.13.2
shap                              0.46.0
tensorboard                       2.18.0
tensorflow                        2.18.0
torch                             2.6.0
torchvision                       0.21.0
tqdm                              4.66.5