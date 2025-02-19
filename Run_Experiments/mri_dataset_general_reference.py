import os
import pandas as pd
import numpy as np
import csv
import random


def main():
    cn_dict_with_images = np.load("/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/data/cn_dict_with_images.npy",allow_pickle='TRUE').item()
    ad_dict_with_images = np.load("/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/data/ad_dict_with_images.npy",allow_pickle='TRUE').item()
    smci_dict_with_images = np.load("/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/data/smci_dict_with_images.npy",allow_pickle='TRUE').item()
    pmci_dict_with_images = np.load("/Users/olath/Documents/GitHub/Master-thesis/Data_Experimentation/data/pmci_dict_with_images.npy",allow_pickle='TRUE').item()

    dicts = [cn_dict_with_images, ad_dict_with_images, smci_dict_with_images, pmci_dict_with_images]

    split = [0.6, 0.2, 0.2]

    bad_images = pd.read_csv("/Users/olath/Documents/GitHub/Master-thesis/Run_Experiments/bad_images_saggital.csv")

    bad_images = bad_images['IMAGE_ID'].tolist()

    #Non existant images
    bad_images.extend([396649, 397160])

    create_reference_file_patients(split, dicts, bad_images)
    
def create_reference_file_patients(split, dicts, bad_images):

    train_split, test_split, val_split = split

    with open('reference_all_classes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['IMAGE_ID','PATIENT_ID','CLASS','TYPE'])

        for i in range(0,4):

            patients = list(dicts[i].keys())
            random.shuffle(patients)

            train, test, val = np.split(patients, [int(len(patients)*train_split), int(len(patients)* (train_split + test_split))])

            for patient in train:
                for visit in dicts[i][patient]:
                    if len(visit) == 4 and len(visit[3]) != 0:
                        for image in visit[3]:
                            if image not in bad_images:
                                writer.writerow([str(image),str(patient), i, 'TRAIN'])

            for patient in test:
                for visit in dicts[i][patient]:
                    if len(visit) == 4 and len(visit[3]) != 0:
                        for image in visit[3]:
                            if image not in bad_images:
                                writer.writerow([str(image),str(patient), i, 'TEST'])

            for patient in val:

                for visit in dicts[i][patient]:
                    if len(visit) == 4 and len(visit[3]) != 0:
                        for image in visit[3]:
                            if image not in bad_images:
                                writer.writerow([str(image),str(patient), i, 'VAL'])


main()