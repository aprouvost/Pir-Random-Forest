# Importing the required packages
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.affines import apply_affine
import matplotlib.image as mpimg
import numpy.linalg as npl
import csv
import io


# dans volume --> les patients
# dans segmentation --> les cluster


# Function importing Dataset from patients
def importdata_patients():
    epi_img = nib.load('10000100_1_CTce_ThAb_0_0.nii.gz')
    print('DEBUG- Image loaded')
    epi_img_data = epi_img.get_fdata()  # turns img to Numpy numpy.ndarray
    print('DEBUG- Image turned to array')
    # Printing the dataswet shape
    print("Dataset Length: ", len(epi_img_data))
    # print("Dataset Shape: ", epi_img_data.shape)
    return epi_img_data


# Function importing infos on dataset's types
def importdata_types():
    epi_img_data = nib.load('10000100_1_CTce_ThAb.nii.gz')
    print('DEBUG- Data file loaded')
    epi_img_data_data = epi_img_data.get_fdata()  # turns img to Numpy numpy.ndarray
    print("Dataset Length info: ", len(epi_img_data_data))
    return epi_img_data


# Function importing CSV file. First row: label , Others: pixels
def import_CSV():
    pass


# Function to split the dataset
def splitdataset(balance_data):
    # Separating the target variable
    X = balance_data.values[:, 1:]  # the variable X contains the pixels
    Y = balance_data.values[:, 0]  # the variable Y contains the target label

    print(" X= ", X)
    print(" Y= ", Y)

    # Splitting the dataset into train and test

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test


# Function to return X, Y, Z coordinates for i, j, k
# def get_xyz_coordinates(i, j, k):
#    return M.dot([i, j, k]) + abc


# Function to map between voxels in the EPI image and voxels in the anatomical image (vox to anat. vox)
def get_image_voxels(image_affine, info_affine):
    new_vox = npl.inv(image_affine).dot(info_affine)
    return new_vox


# function to rescale the image using the affine matrix of this image
# NOT OPERATIONNAL YET
def rescale(scaling_coeff, affine):
    scaling_affine = np.array(
        [[scaling_coeff, 0, 0, 0], [scaling_coeff, 3, 0, 0], [scaling_coeff, 0, 3, 0], [0, 0, 0, 1]])
    one_vox_axis_0 = [1, 0, 0]
    apply_affine(scaling_affine, one_vox_axis_0)
    return affine.dot(scaling_affine)


# function to rotate the image using the affine matrix of this image
def rotate(rotation_coeff, affine):
    cos_gamma = np.cos(rotation_coeff)
    sin_gamma = np.sin(rotation_coeff)
    rotation_affine = np.array(
        [[1, 0, 0, 0], [0, cos_gamma, -sin_gamma, 0], [0, sin_gamma, cos_gamma, 0], [0, 0, 0, 1]])
    return rotation_affine.dot(affine)


# Function used to translate the image usinf the affine function of this image
def translate(x_translation, y_translation, z_translation, affine):
    translation_affine = np.array(
        [[1, 0, 0, x_translation], [0, 1, 0, y_translation], [0, 0, 1, z_translation], [0, 0, 0, 1]])
    return translation_affine.dot(affine)


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    # print("Predicted values:")
    # print(y_pred)
    return y_pred


if __name__ == "__main__":
    # epi_img = nib.load('10000100_1_CTce_ThAb_0_0.nii.gz')
    # epi_img_data = nib.load('10000100_1_CTce_ThAb.nii.gz')
    # epi_img_data = importdata_patients()
    # epi_img_data_types = importdata_types()
    # affine = epi_img_data_types.affine
    print('============================================================== \n')
    # print("---> Affine matrix of the patient : ")
    # print(affine)

    # split affine matrix into M and (a,b,c)
    # M = affine[:3, :3]
    # abc = affine[:3, 3]

    # print("\n DEBUG APPLYING TRANSLATE")
    # affine_transl =translate(1,1,1,affine)
    # print("---> new affine matrix")
    # print(affine_transl)
    # print('============================================================== \n')

    # 1080  collonnes
    # 23000 rows

    Organsdict = {
        0: 0,  # "background"
        1: 1,  # "body envelope",
        2: 2,  # "thorax-abdomen",
        86: 3,  # "spleen (rate)",
        58: 4,  # "liver (foie)",
        480: 5,  # "aorta (aorte)",
        7578: 6,  # "thyroid gland",
        1247: 7,  # "trachea",
        1302: 8,  # "right lung",
        1326: 9,  # "left lung",
        170: 10,  # "pancreas",
        187: 11,  # "gallbladder (vésicule biliaire)",
        237: 12,  # "urinary bladder (vessie)",
        2473: 13,  # "sternum",
        29193: 14,  # "first lumbar vertebra",
        29662: 15,  # "right kidney",
        29663: 16,  # "left kidney",
        30324: 17,  # "right adrenal gland",
        30325: 18,  # "left adrenal gland",
        32248: 19,  # "right psoas major",
        32249: 20,  # "left psoas major",
        40357: 21,  # "muscle body of right rectus abdominis",
        40358: 22,  # "muscle body of left rectus abdominis",
    }
    inverted_dict = dict(map(reversed, Organsdict.items()))
    OrgansNames = {
        0: "background",
        1: "body envelope",
        2: "thorax-abdomen",
        86: "spleen (rate)",
        58: "liver (foie)",
        480: "aorta (aorte)",
        7578: "thyroid gland",
        1247: "trachea",
        1302: "right lung",
        1326: "left lung",
        170: "pancreas",
        187: "gallbladder (vésicule biliaire)",
        237: "urinary bladder (vessie)",
        2473: "sternum",
        29193: "first lumbar vertebra",
        29662: "right kidney",
        29663: "left kidney",
        30324: "right adrenal gland",
        30325: "left adrenal gland",
        32248: "right psoas major",
        32249: "left psoas major",
        40357: "muscle body of right rectus abdominis",
        40358: "muscle body of left rectus abdominis",
    }

    path = os.path.realpath('')
    data_path = path + "/CTce_ThAb_b33x33_n1000_8bit"
    a = 0
    numMax = 9
    files_data = []
    data_loaded = []
    X = []
    Y = []

    for file in os.listdir(data_path):
        if file.endswith(".csv") and a < numMax:

            print(file + " détecté")
            files_data.append(data_path + "/" + file)
            a += 1

    for file in files_data:
        organs_blocks = pd.read_csv(file, nrows=23000)
        for row in organs_blocks.itertuples(index=False):
            data_loaded.append(
                [Organsdict[row[0]], list(row[1:1090])])
            Y.append(Organsdict[row[0]])
            X.append(list(row[1:1090]))
        print("Fichier | " + file + " | chargé")

    # csv_file = pd.read_csv("10000105_1_CTce_ThAb.csv", header=None)

    # X, Y, X_train, X_test, y_train, y_test = splitdataset(data_loaded)

    # print(data_loaded[:][0])
    # print(data_loaded[:][1:])

    # X_train, X_test, y_train, y_test = train_test_split(data_loaded[:][0], data_loaded[:][1:], test_size=0.9, random_state=100)
    # clf_gini = train_using_gini(X_train, X_test, y_train)

    # Prediction using gini
    # y_pred_gini = prediction(X_test, clf_gini)
    # cal_accuracy(y_test, y_pred_gini)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    sc = clf.score(X, Y)
    print(" Classifying score is " + str(sc))
    pred = clf.predict([X[0]])
    print(" Prediction is " + str(pred))
    print(" had to predict" + str(Y[0]))
