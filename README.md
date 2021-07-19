# SDPred



# Requirements
* python == 3.6
* pytorch == 1.5
* Numpy == 1.16.2
* scikit-learn == 0.21.3


# Files:
1.data

This folder contains 11 input files needed by our model.

drug_mol.pkl: The word embedding matrix of drugs. We use Mol2vec model to learn the word embedding of drugs. Mol2vec can learn vector representations of molecular substructures pointing to similar directions of chemically related substructures. Each row of the matrix represents the word vector encoding of a drug.

glove_wordEmbedding.pkl: The word embedding matrix of side effects. We use the 300-dimensional Global Vectors (GloVe) trained on the Wikipedia dataset to represent the information of side effects. Each row of the matrix represents the word vector encoding of a side effect.

side_effect_semantic.pkl: The semantic similarity matrix of side effects. We download side effect descriptors from Adverse Drug Reaction Classification System (ADReCS, http://bioinf.xmu.edu.cn/ADReCS/index.jsp), and construct a novel model to calculate the semantic similarity of side effects. Each row of the matrix represents the similarity value between a side effect and all side effects in the benchmark dataset. The range of values is from 0 to 1.

Text_similarity_one.pkl, Text_similarity_two.pkl, Text_similarity_three.pkl, Text_similarity_four.pkl, Text_similarity_five.pkl: Five similarity matrices of drugs. These matrices are collected from the file "Chemical_chemical.links.detailed.v5.0.tsv.gz" in STITCH database (http://stitch.embl.de/). Each row of the matrices represents the similarity value between a drug and all drugs in the datasets respectively. The range of values is from 0 to 1.

drug_side.pkl: The matrix has 757 rows and 994 columns to store the known drug-side effect frequency pairs. The element at the corresponding position of the matrix is set to the frequency value, otherwise 0.
drug_target.pkl: The target protein information of the drugs is obtained from DrugBank database.

fingerprint_similarity.pkl: The structure similarity matrix of the drugs. We calculate the structure similarities between drugs according to the Jaccard scores.



If you want to view the value stored in the file, you can run the following command:

```bash
import pickle
import numpy as np
gii = open(‘data’ + '/' + ' drug_side.pkl ', 'rb')
drug_side_effect = pickle.load(gii)
```


2.Code
network.py: This function contains the network framework of our entire model.

up_ten_fold.py: This function can test the predictive performance of our model under ten-fold cross-validation.


# Train and test folds
python up_ten_fold.py --rawpath /Your path --epochs Your number --batch_size Your number

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

All files of Data and Code should be stored in the same folder to run the model.

Example:

```bash
python up_ten_fold.py --rawdata_dir /data --epochs 100 --batch_size 256
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn
