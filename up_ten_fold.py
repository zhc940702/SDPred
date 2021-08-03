#换损失函数，两个损失函数相同
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity



def read_raw_data(rawdata_dir, data_train, data_test):
    gii = open(rawdata_dir + '/' + 'Text_similarity_one.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_two.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_three.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_four.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()

    gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    Drug_word_sim = cosine_similarity(Drug_word2vec)

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

    for i in range(data_test.shape[0]):
        drug_side[data_test[i, 0], data_test[i, 1]] = 0

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


def fold_files(data_train, data_test, data_neg, args):
    rawdata_dir = args.rawpath
    data_train = np.array(data_train)
    data_test = np.array(data_test)

    drug_features, side_features = read_raw_data(rawdata_dir, data_train, data_test)

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))

    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))

    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]
    f_test = data_test[:, 2]

    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]
    f_train = data_train[:, 2]

    return drug_test, side_test, f_test, drug_train, side_train, f_train

def train_test(data_train, data_test, data_neg, fold, args):
    drug_test, side_test, f_test, drug_train, side_train, f_train = fold_files(data_train, data_test, data_neg, args)
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                              torch.FloatTensor(f_train))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                             torch.FloatTensor(f_test))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=16, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=16, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = ConvNCF(7570, 3976, args.embed_dim, args.batch_size).to(device)
    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    AUC_mn = 0
    AUPR_mn = 0

    rms_mn = 100000
    mae_mn = 100000
    endure_count = 0

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, Classification_criterion, Regression_criterion, device)
        # ====================     test       ====================

        t_i_auc, t_iPR_auc, t_rmse, t_mae, t_ground_i, t_ground_u, t_ground_truth, t_pred1, t_pred2 = test(model,
                                                                                                           _train,
                                                                                                           _train,
                                                                                                           device)
        if AUC_mn < t_i_auc and AUPR_mn < t_iPR_auc:
            AUC_mn = t_i_auc
            AUPR_mn = t_iPR_auc
            rms_mn = t_rmse
            mae_mn = t_mae
            endure_count = 0

        else:
            endure_count += 1

        print("Epoch: %d <Train> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f " % (
        epoch, t_rmse, t_mae, t_i_auc, t_iPR_auc))
        start = time.time()

        if endure_count > 10:
            break
    i_auc, iPR_auc, rmse, mae, ground_i, ground_u, ground_truth, pred1, pred2 = test(model, _test, _test, device)

    time_cost = time.time() - start
    print("Time: %.2f Epoch: %d <Test> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f " % (
        time_cost, epoch, rmse, mae, i_auc, iPR_auc))
    print('The best AUC/AUPR: %.5f / %.5f' % (i_auc, iPR_auc))
    print('The best RMSE/MAE: %.5f / %.5f' % (rmse, mae))

    return i_auc, iPR_auc, rmse, mae

def train(model, train_loader, optimizer, lossfunction1, lossfunction2, device):

    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_drug, batch_side, batch_ratings = data
        batch_labels = batch_ratings.clone().float()
        for k in range(batch_ratings.data.size()[0]):
            if batch_ratings.data[k] > 0:
                batch_labels.data[k] = 1
        optimizer.zero_grad()

        one_label_index = np.nonzero(batch_labels.data.numpy())
        logits, reconstruction = model(batch_drug, batch_side, device)
        loss1 = lossfunction1(logits, batch_labels.to(device))
        loss2 = lossfunction2(reconstruction[one_label_index], batch_ratings[one_label_index].to(device))
        total_loss = loss1 * loss2
        total_loss.backward(retain_graph = True)
        optimizer.step()
        avg_loss += total_loss.item()

    return 0

def test(model, test_loader, neg_loader, device):
    model.eval()
    pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    ground_u = []
    ground_i = []

    for test_drug, test_side, test_ratings in test_loader:

        test_labels = test_ratings.clone().long()
        for k in range(test_ratings.data.size()[0]):
            if test_ratings.data[k] > 0:
                test_labels.data[k] = 1
        ground_i.append(list(test_drug.data.cpu().numpy()))
        ground_u.append(list(test_side.data.cpu().numpy()))
        test_u, test_i, test_ratings = test_drug.to(device), test_side.to(device), test_ratings.to(device)
        scores_one, scores_two = model(test_drug, test_side, device)
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))
        label_truth.append(list(test_labels.data.cpu().numpy()))

    pred1 = np.array(sum(pred1, []), dtype = np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)



    iprecision, irecall, ithresholds = metrics.precision_recall_curve(label_truth,
                                                                      pred1,
                                                                      pos_label=1,
                                                                      sample_weight=None)
    iPR_auc = metrics.auc(irecall, iprecision)

    try:
        i_auc = metrics.roc_auc_score(label_truth, pred1)
    except ValueError:
        i_auc = 0

    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])


    return i_auc, iPR_auc, rmse, mae, ground_i, ground_u, ground_truth, pred1, pred2

def ten_fold(args):
    rawpath = args.rawpath
    gii = open(rawpath+'/drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)
    total_auc, total_pr_auc, total_rmse, total_mae = [], [], [], []
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        auc, PR_auc, rmse, mae = train_test(data[train].tolist(), data[test].tolist(), data_neg, fold, args)

        total_rmse.append(rmse)
        total_mae.append(mae)
        total_auc.append(auc)
        total_pr_auc.append(PR_auc)
        print("==================================fold {} end".format(fold))
        fold += 1
        print('Total_AUC:')
        print(np.mean(total_auc))
        print('Total_AUPR:')
        print(np.mean(total_pr_auc))
        print('Total_RMSE:')
        print(np.mean(total_rmse))
        print('Total_MAE:')
        print(np.mean(total_mae))
        sys.stdout.flush()

def denovo_test(args):
    rawpath = args.rawpath
    gii = open(rawpath+'/drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)
    total_auc, total_pr_auc, total_rmse, total_mae = [], [], [], []
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        auc, PR_auc, rmse, mae = train_test(data[train].tolist(), data[test].tolist(), data_neg, fold, args)

        total_rmse.append(rmse)
        total_mae.append(mae)
        total_auc.append(auc)
        total_pr_auc.append(PR_auc)
        print("==================================fold {} end".format(fold))
        fold += 1
        print('Total_AUC:')
        print(np.mean(total_auc))
        print('Total_AUPR:')
        print(np.mean(total_pr_auc))
        print('Total_RMSE:')
        print(np.mean(total_rmse))
        print('Total_MAE:')
        print(np.mean(total_mae))
        sys.stdout.flush()

def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    return addition_negative_sample, final_positive_sample, final_negtive_sample

def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 220,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0005,
                        metavar = 'FLOAT', help = 'learning rate')
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
    parser.add_argument('--dataset', type = str, default = 'hh',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='/home/zhaohc/four/data',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('N: ' + str(args.N))
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    ten_fold(args)

if __name__ == "__main__":
    main()
