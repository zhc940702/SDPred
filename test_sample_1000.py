import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from network import ConvNCF

from sklearn.metrics.pairwise import cosine_similarity

def read_raw_data(rawdata_dir):
    gii = open(rawdata_dir + '/' + 'Text_similarity_one.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()
    # drug_feature_one = Normalize(drug_feature_one)
    # drug_feature_one = preprocessing.scale(drug_feature_one)

    gii = open(rawdata_dir + '/' + 'Text_similarity_two.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()
    # drug_feature_two = Normalize(drug_feature_two)
    # drug_feature_two = preprocessing.scale(drug_feature_two)

    gii = open(rawdata_dir + '/' + 'Text_similarity_three.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()
    # drug_feature_three = Normalize(drug_feature_three)
    # drug_feature_three = preprocessing.scale(drug_feature_three)

    gii = open(rawdata_dir + '/' + 'Text_similarity_four.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()
    # drug_feature_four = Normalize(drug_feature_four)
    # drug_feature_four = preprocessing.scale(drug_feature_four)

    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()
    # drug_feature_five = Normalize(drug_feature_five)
    # drug_feature_five = preprocessing.scale(drug_feature_five)

    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()
    # drug_feature_six = Normalize(drug_feature_six)
    # drug_feature_six = preprocessing.scale(drug_feature_six)

    gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    Drug_word_sim = cosine_similarity(Drug_word2vec)

    # drug_feature_seven = Normalize(drug_feature_seven)
    # drug_feature_seven = preprocessing.scale(drug_feature_seven)

    # 药物的分类标签矩阵
    gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_sim = cosine_similarity(glove_word)

    gii = open(rawdata_dir + '/' + 'drug_target.pkl', 'rb')
    drug_target = pickle.load(gii)
    gii.close()
    drug_target_sim = cosine_similarity(drug_target)


    gii = open(rawdata_dir + '/' + 'fingerprint_similarity.pkl', 'rb')
    drug_f_sim = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()

    drug_side_sim = cosine_similarity(drug_side)



    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1
    drug_side_label_sim = cosine_similarity(drug_side_label)

    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(Drug_word_sim)
    # drug_features.append(drug_feature_fingerprint)
    drug_features.append(drug_target_sim)
    drug_features.append(drug_f_sim)
    drug_features.append(drug_side_sim)
    drug_features.append(drug_side_label_sim)

    side_drug_sim = cosine_similarity(drug_side.T)
    side_drug_label_sim = cosine_similarity(drug_side_label.T)


    side_features.append(effect_side_semantic)
    side_features.append(side_glove_sim)
    side_features.append(side_drug_sim)
    side_features.append(side_drug_label_sim)


    return drug_features, side_features

def fold_files(args):
    rawdata_dir = args.rawpath

    drug_features, side_features = read_raw_data(rawdata_dir)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    two_cell = []
    for i in range(drug_features_matrix.shape[0]):
        for j in range(side_features_matrix.shape[0]):
            two_cell.append([i, j])

    # two_cell = two_cell[0:1000]

    two_cell = np.array(two_cell)

    drug_test = drug_features_matrix[two_cell[:, 0]]
    side_test = side_features_matrix[two_cell[:, 1]]

    return drug_test, side_test, two_cell

def test_data(args):
    drug_test, side_test, two_cell = fold_files(args)
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ConvNCF(7570, 3976, args.embed_dim, args.batch_size).to(device)
    model_file = args.rawpath + '/' + 'my_model.dat'
    cheeckpoint = torch.load(model_file, map_location = device)
    model.load_state_dict(cheeckpoint['model'])
    model.eval()
    pred1 = []
    pred2 = []
    for test_drug, test_side in _test:
        scores_one, scores_two = model(test_drug, test_side, device)
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    print('Output_data')
    output = []
    output.append(['drug_id', 'side_effect_id', 'Sample_association_score', 'Sample_frequency_score'])
    for i in range(pred1.shape[0]):
        if pred1[i] < 0.5:
            pred2[i] = 0
        output.append([str(two_cell[i][0]), str(two_cell[i][1]), str(pred1[i]), str(pred2[i])])

    t = ''
    with open('Prediction.txt', 'w') as q:
        for i in output:
            for e in range(len(output[0])):
                t = t + str(i[e]) + ' '
            q.write(t.strip(' '))
            q.write('\n')
            t = ''


def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 220,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'learning rate')
    # 64是L0层映射出的维度数
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'yelp',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='\data',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('Dataset: ' + args.dataset)
    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    test_data(args)

if __name__ == "__main__":
    main()
