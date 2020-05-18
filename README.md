# accuEnhancer



### Contents

* **accuEnhancer.py** : Main function of the accuEnhancer packages. 
                        The accuEnhancer package provides functions to train and to predict the enhancers from the given DNase files.

### Requirements
#### Packages
* Python 3.5 +
* Keras +
* Bedtools (Tested on 2.28.0)
#### Input data
* Pre-processed peak files
* DNase peaks in narrowpeak format.

## Usage
```bash
usage: accuEnhancers.py  [-h help] 
                  [--positive_training_data POSITIVE_TR_DATA] 
                  [--negative_training_data NEGATIVE_TR_DATA]
                  [--out_path OUT_PATH]
                  
                  
Required arguments:
  --positive_training_data    
                        positive training data in fasta (*.fa, *.fasta) format. 
                        [Type: String]  
Optional arguments:
  -h, --help            
                        Show this help message and exit
  --negative_data_method NEGATIVE_DATA_METHOD  
                        If not given the negative training data, ezGeno will generate 
                        negative data based on the selected methods.
                        "random": random sampling from the human genome.
                        "dinucl": generate negative sequence based on same dinucleotide
                                  composition with the positive training data.
                        [Type: String, Default:"dinucl", options: "random, dinucl"]
  --out_path OUT_PATH   
                        The output directory for the trained model. 
                        [Type: String, default: "output_dir"]
```


## Installation
1) Download/Clone ezGeno
```bash
git clone https://github.com/ailabstw/accuEnhancer.git

cd accuEnhancer
```

2) Install required packages
```bash
pip3 install torch
apt-get install bedtools
```

### Dataset


### Models
**./models** contains links to already trained models.
