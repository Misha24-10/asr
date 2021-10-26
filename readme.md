# ASR (Automatic Speech Recognition)
In this project trained QuartzNet model in Rus speech

Paper: https://arxiv.org/abs/1910.10261

## Installation

1. Clone the repository
```
git clone https://github.com/Misha24-10/asr.git
cd asr
```
2. Create an new environment and install the dependencies
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
3. Insatll Dataset mozilla common voice RU and unzip it(https://commonvoice.mozilla.org/ru/datasets)
```
if you use colab use:
!gdown --id  1HzNFciQ62g64QMjlKlKeBaJ-jm3XmwZF
!tar -C "/content/" -xf  "/content/cv-corpus-7.0-2021-07-21-ru.tar.gz" 
```

4. Change paths to datset and model if it need in scr/configs/config.py
```
main_path_to_dataset_common_voice = '..'
path_to_csv_train_dataset = '..'
path_to_csv_valid_dataset = '..'
path_to_csv_test_dataset = '..'
model_weights = '..' # path to state_dict_model_commonvoice_part12__added_Aadam_lr_0_00005_del_to_every_2_iter.pt
```
5. run test.py or train.py