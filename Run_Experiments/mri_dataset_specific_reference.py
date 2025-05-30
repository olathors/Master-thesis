import pandas as pd

def main(reference):



    generate_mri_dataset_reference(reference, 'Datasets/train-CN-allMCI-AD', [[0],[4],[1]], 'TRAIN', 'CORONAL', 43)
    generate_mri_dataset_reference(reference, 'Datasets/test-CN-allMCI-AD', [[0],[4],[1]], 'TEST', 'CORONAL', 43)
    generate_mri_dataset_reference(reference, 'Datasets/val-CN-allMCI-AD', [[0],[4],[1]], 'VAL', 'CORONAL', 43)

def generate_mri_dataset_reference(mri_reference_path,
                                output_path,
                                classes,
                                type,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_sampled_images = 5,
                                sampling_range = 3,
                                num_rotations = 3,
                                save_reference_file = True):
    
    df_mri_reference = pd.read_csv(mri_reference_path)

    #print(df_mri_reference)
    label_counter = 0
    temp_sets = [0] * len(classes)
    for label in classes:
        
        df_mri_dataset_temp = df_mri_reference.query("CLASS == @label[0] and TYPE == @type")
        df_mri_dataset_temp['CLASS'] = label_counter

        if (len(label) == 2):

            df_mri_dataset_temp2 = df_mri_reference.query("CLASS == @label[1] and TYPE == @type")
            df_mri_dataset_temp2['CLASS'] = label_counter
            df_mri_dataset_temp = pd.concat([df_mri_dataset_temp, df_mri_dataset_temp2])

        temp_sets[label_counter] = df_mri_dataset_temp
        label_counter += 1

    df_mri_dataset = pd.concat(temp_sets)
    #df_mri_dataset['ORIENTATION'] = orientation
    #df_mri_dataset['SLICE'] = orientation_slice
    df_mri_dataset.to_csv(output_path, index=False)

main('/Users/olath/Documents/GitHub/Master-thesis/Datasets/reference_all_classes_timewindow_singular.csv')
    
