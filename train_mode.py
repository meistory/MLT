import input_data as Id
import modelClassification
import parser
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import pprint
from sklearn.metrics import accuracy_score
import census_income_demo as cid
import csv

def score(y_pred, y):
    # y_pred = self.prediction(X)
    return 1. - np.sqrt(mean_squared_error(y[:, 0], y_pred[:, 0])) / (np.max(y[:, 0]) - np.min(y[:, 0]))


def get_hid2():
    NewCheck = tf.train.NewCheckpointReader("./checkpoint_1/latent_model-54001")

    print('debug_string:\n')
    pprint.pprint(NewCheck.debug_string().decode('utf-8'))
    print('get_tensor:\n')
    pprint.pprint(NewCheck.get_tensor('hidden2/weights'))

    pprint.pprint(NewCheck.get_tensor('hidden2/Relu'))



def y_labels(y_label):

    list = []
    label = []

    for i in y_label:
        list.append(i)

    for h in range(len(y_label[i])):
        list_1 = []
        for j in list:
            list_1.append(y_label[j][h][0])

        label.append(list_1)

    return label


def train(train_data,y_label):
    FNN = modelClassification.modelclassification()
    FNN.build_model(train_data.shape[1])
    FNN.build_loss()
    FNN.build_training(0.0001)
    FNN.train_model(train_data, y_label)
    out_hidden = FNN.out_model()
    return  out_hidden

def infer(train_data):
    FNN = modelClassification.modelclassification()
    FNN.build_model(train_data.shape[1])
    FNN.build_loss()
    y_pred = FNN.infer(train_data)

    return y_pred

if __name__ == "__main__":

    train_data, train_label, validation_data, validation_label, test_data, test_label, \
    output_info,dict_train_labels,dict_test_labels,dict_validation_labels= cid.data_preparation()

    dict_label = {}
    for key in dict_test_labels:
        dict_label[key] = dict_test_labels[key]

    y_label = y_labels(dict_label)
    out_hidden = train(test_data,y_label)
    # print(out_hidden)
    # y_pred = infer(test_data)
    # print('y_pred')
    #
    # print(y_pred)
    #
    # print(np.shape(y_pred))
    #
    # with open('test_result.csv','w') as fp:
    #     csv_writer = csv.writer(fp)
    #     for i in range(np.shape(y_pred)[0]):
    #         csv_writer.writerow(y_pred[i])
    #
    #     print('write over')
    #
    # fp.close()
    #
    # get_hid2()

