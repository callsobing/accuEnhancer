# accuEnhancer
Enhancers are one class of the regulatory elements that have been shown to act as key components to assist promoters in modulating the gene expression in living cells. accuEnhancer utilize DNase data to build a deep learning model for predicting the H3K27ac peaks as the active enhancers in selected cell types. Moreover, we also tested joint training for multiple cell types to boost the model performance. 

accuEnhancer shows the general feasibility to predict the within cell type enhancer activities, which the accuracy and F1 score can achieve 0.97 and 0.9, through the proposed deep learning model. But the results of the cross cell type prediction based on the single cell type model are only adequate, where the F1 score is only around 0.5. We then attempted to further improve the performance of cross cell type predictions by integrating the training data from different cell types. The F1 score on predicting the independent cell type increased from 0.48 to 0.81 as we combined more training data from different cell types. The results demonstrated that by incorporating more datasets across cell types, the complex regulatory patterns could be captured by the deep learning models and delivered better performances. 
Furthermore, we used the pre-trained filter weights from the DeepC, which could assist the training process and yields to a better result. Lastly, we tested the effectiveness of the model on predicting active enhancers in the VISTA database. The results indicates that accuEnhancer outperforms the previous works in predicting VISTA enhancers with the ability to make cross-cell type predictions.

accuEnhancer utilized the pre-trained weights from deepHaem (https://github.com/rschwess/deepHaem), which predicts chromatin features from DNA sequence, to assist the model training process. The pre-trained model data can be assesed from deephaem github repository and their webpage. (http://userweb.molbiol.ox.ac.uk/public/rschwess/deepHaem/model_deephaem_endpool_erythroid_model.tar.gz)

## Contents

* **accuEnhancer.py** : Main function of the accuEnhancer packages. 
                        The accuEnhancer package provides functions to train and to predict the enhancers from the given DNase files.

## Requirements
### Dependencies
* Python 3.5 +
* Keras
* Tensorflow (Tested on 1.15.0)
* h5py
* numpy
* argparse
* Bedtools (Tested on 2.28.0)

### Input data
* Pre-processed peak files
* DNase peaks in narrowpeak format.

## Usage
```bash
usage: accuEnhancers.py  [-h help] 
                  [--in_file IN_FILE] 
                  [--out_name OUT_NAME]
                  [--epochs EPOCHS]
                  [--trainable TRAINABLE]
                  [--test_file TEST_FILE]
                  [--deephaem_model_path DEEPHAEM_MODEL_PATH]
                  [--train_from_ckpt TRAIN_FROM_CKPT]
                                          
                  
Required arguments:
  --in_file IN_FILE     Six column file. [<chr> <start> <end> <comma separated IDs> <raw sequence> <epi mark>].
  --out_name OUT_NAME   Output file destination name, suffix
  --epochs EPOCHS       Number of epochs
    
Optional arguments:
  -h, --help            show this help message and exit
  --trainable TRAINABLE
                        pretrined weight retrainable ("T" or "F")
  --test_file TEST_FILE
                        using in-file testing or specified test file
  --deephaem_model_path DEEPHAEM_MODEL_PATH
                        using deephaem pre-trained model weights
  --train_from_ckpt TRAIN_FROM_CKPT
                        load from self-trained checkpoint, int number
```


## Installation
1) Download/Clone accuEnhancer
```bash
git clone https://github.com/callsobing/accuEnhancer.git

cd accuEnhancer
```

2) Install required packages
```bash
pip3 install torch
apt-get install bedtools
```

## Models
**./models** contains links to already trained models.

## Tutorial
A step-by-step tutorial from data preparation, preprocessing to model training and testing is demonstrated in `accuEnhancer_tutorial.ipynb`. 

## References
1) Schwessinger, R., Gosden, M., Downes, D., Brown, R., Telenius, J., Teh, Y. W., ... & Hughes, J. R. (2019). DeepC: Predicting chromatin interactions using megabase scaled deep neural networks and transfer learning. bioRxiv, 724005.
2) Kelley, D. R., Snoek, J., & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. Genome research, 26(7), 990-999.
