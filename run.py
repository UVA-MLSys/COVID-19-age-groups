import warnings
warnings.filterwarnings('ignore')

import argparse, torch, random
from exp.exp_forecasting import Exp_Forecast
from exp.config import DataConfig
import numpy as np
from datetime import datetime
        
def initial_setup(args):
    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # setup gpu devices
    use_gpu = True if torch.cuda.is_available() and not args.no_gpu else False

    if use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    # setup feature numbers
    args.n_features = len(set(DataConfig.static_reals+DataConfig.observed_reals+DataConfig.targets))
    args.enc_in = args.dec_in = args.c_out = args.n_features
    args.n_targets = len(DataConfig.targets)

def main(args):
    start = datetime.now()
    print(f'Experiment started at {start}')
    initial_setup(args)
    
    print('Args in experiment:')
    print(args)

    setting = stringify_setting(args)
    exp = Exp_Forecast(args, setting)

    if args.test:
        print(f'>>>>>>> testing : {setting} <<<<<<<<')
        exp.test(setting, load_model=True, flag='test')
        exp.test(setting, flag='val')
        exp.test(setting, flag='train')
    else:
        # setting record of experiments
        print(f'>>>>>>> training : {setting} >>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>> testing : {setting} <<<<<<<<')
        exp.test(setting, flag='test')
        exp.test(setting, flag='val')
        exp.test(setting, flag='train')
        
    torch.cuda.empty_cache()
    print(f'Experiment ended at {datetime.now()}, runtime {datetime.now() - start}')

def stringify_setting(args):
    setting = f"{args.model}_{args.data_path.split('.')[0]}"
    if args.des and args.des != '':
        setting += '_des_' + args.des
        
    return setting

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run Timeseries Models', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # basic config
    parser.add_argument('--test', action='store_true', help='test the checkpointed best model, train otherwise')
    parser.add_argument('--model', type=str, required=True, default='DLinear',
        choices=(Exp_Forecast.model_dict.keys()), help='model name')
    parser.add_argument('--seed', default=7, help='random seed')

    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/processed/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Top_20.csv', help='data file')
    parser.add_argument('--result_path', type=str, default='results', help='result folder')
    parser.add_argument('--freq', type=str, default='d', choices=['s', 't', 'h', 'd', 'b', 'w', 'm'],
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, \
            b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--no-scale', action='store_true', help='do not scale the dataset')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=14, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=7, help='start token length')
    parser.add_argument('--pred_len', type=int, default=14, help='prediction sequence length')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    # parser.add_argument('--enc_in', type=int, default=10, help='encoder input size, equal to number of past fetures.')
    # parser.add_argument('--dec_in', type=int, default=10, help='decoder input size, same as enc_in')
    # parser.add_argument('--c_out', type=int, default=10, help='output size, same as enc_in')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=7, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
        help='whether to use distilling in encoder, using this argument means not using distilling',
    )
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', choices=['timeF', 'fixed', 'learned'],
        help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', choices=['type1', 'type2'], default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--no_gpu', action='store_true', help='do not use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    parser.add_argument('--disable_progress', action='store_true', help='disable progress bar')
    
    return parser

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)