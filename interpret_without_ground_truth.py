from run import initial_setup, stringify_setting
from interpret_with_ground_truth import get_parser as get_main_parser
import os
from exp.exp_forecasting import *
from utils.interpreter import *
from exp.exp_interpret import Exp_Interpret, explainer_name_map

def main(args):
    initial_setup(args)

    # Disable cudnn if using cuda accelerator throws error.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # args.use_gpu = False
        
    settings = stringify_setting(args)
    exp = Exp_Forecast(args, settings)  # set experiments
    _, dataloader = exp.get_data(flag=args.flag, train=False)

    exp.load_model()

    # some models might not work with gradient based explainers
    interpreter = Exp_Interpret(exp, dataloader) 
    interpreter.interpret(dataloader)
    
def get_parser():
    parser = get_main_parser()
    parser.description = 'Interpret timeseries model'
    
    parser.add_argument('--metrics', nargs='*', type=str, default=['mae', 'mse'], 
        help='interpretation evaluation metrics')
    parser.add_argument('--areas', nargs='*', type=float, default=[0.05, 0.1],
        help='top k features to keep or mask during evaluation')
    
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)