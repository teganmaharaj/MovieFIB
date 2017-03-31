import socket
from jobman import DD

load_model_from = '/data/lisatmp4/yaoli/exps/video_qa/overfitting/y2t/gnet_f10/'
params_file = load_model_from + 'model_params_mb54000.pkl'
options_file = load_model_from + 'options.pkl'

hostname = socket.gethostname()
if 'gpu' in hostname or 'helios' in hostname:
    HOST = 'helios'
    save_dir = '/scratch/jvb-000-aa/yaoli001/exp/'
else:
    HOST = 'lisa'
    save_dir = '/data/lisatmp4/maharajt/CVPRmovieQA/'#'/Tmp/ballasn/LSMDC/model/baseline_text_h_320_wemb_data_10_nofine/'

options = DD({

         ### Loop
        'text_only': True,
        'do_eval': False,
        'save_model_dir': save_dir, #+'LSMDC/',
        'load_model_from':  None, #'/Tmp/ballasn/LSMDC/model/baseline_textonly_h_320_wemb_data_10/LSMDC/', #None, #'/Tmp/ballasn/LSMDC/model/baseline_textonyl_h_320/LSMDC/',
        'load_options_from': None,
        'erase': False,
        'max_epoch': 300,
        'dispFreq': 10,
        'estimate_population_statistics': False,
        'debug': True,

        ### Dataset
        'data_path' : "/Tmp/ballasn/LSMDC/LSMDC2016.pkl",
         ### Full None, 10%=>9511, 50%=>47559, 100%=>95118
        'max_train_example': 9511,
        'input_dim':  4096+1024, # 1024 gnet, 4096 C3D
        'features_type' : "Fuse", # 2D/3D/Fuse
        'features_path' : "/Tmp/ballasn/LSMDC/feat/", #"/data/lisatmp4/ballasn/datasets/LSMDC2016/LSMDC_googlenetfeatures.pkl",
        'n_subframes': 15,
        'batch_size': 24,
        'max_n_epoch': 1000,

        ### Vocabulary
        'train_emb': True, # Use only word present > 50 times in the training sets for the output vocabulary
        'use_out_vocab': True, # Use only word present > 50 times in the training sets for the output vocabulary
        'reduce_vocabulary': False, # Use only word present > 3 times in the training sets
        'n_words': 26818,
        'dim_word': 512,
        'hdims': 320,
        'use_dropout': True,
        'use_residual': False,
        'use_zoneout': True,
        'use_bn': True,
        'initial_gamma': 0.1,
        'initial_beta': 0.,
        'use_popstats': False, ### required to be false


        # Model: standard, momentum, adagrad, rmsprop
        'memory_update_rule': 'standard',
        'lstm_alpha': 0.95,

        ### Optimization
        'ita': 0.001,
        'optimizer': 'adam',
        'lr': 0.001,
        'clip_c': 10.,
        'patience': 5,
        'valid_freq': -1,
        })
