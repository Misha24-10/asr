numper_exmapes_to_print = 2
test_batch_size = 30
train_batch_size = 40
valid_batch_size = 40
num_iter_in_epoch_train = 500
num_iter_in_epoch_valid = 100
num_iter_in_epoch_test = 250


main_path_to_dataset_common_voice = "/content/cv-corpus-7.0-2021-07-21/ru/clips/"
path_to_csv_train_dataset = '/content/cv-corpus-7.0-2021-07-21/ru/train.tsv'
path_to_csv_valid_dataset = '/content/cv-corpus-7.0-2021-07-21/ru/validated.tsv'
path_to_csv_test_dataset = '/content/cv-corpus-7.0-2021-07-21/ru/test.tsv' # Just for get score
model_weights = '/content/asr/state_dict_model_commonvoice_part12__added_Aadam_lr_0_00005_del_to_every_2_iter.pt'

number_epoch = 100

opt_learning_rate = 0.05
opt_betas= (0.95, 0.5)
opt_weight_decay = 0.001

isAdam = True
adam_learning_rate = 0.5e-4
adam_betas = (0.93, 0.9995)
norm_clap = 5