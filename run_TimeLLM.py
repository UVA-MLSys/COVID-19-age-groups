from exp.exp_forecasting import *
import os, warnings
warnings.filterwarnings('ignore')

from run import main, get_parser as get_basic_parser

def load_content(args):
    df = pd.read_csv(os.path.join(args.root_path, 'prompt_bank.csv'))
    data_name = args.data_path.split('.')[0] 
    content = df[df['data']==data_name]['prompt'].values[0]
    return content
            
def get_parser():

    parser = get_basic_parser()
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=1, help='')
    
    parser.add_argument(
        '--llm_model', type=str, default='GPT2', help='LLM model',
        choices=['LLAMA', 'GPT2', 'BERT']) # 
    parser.add_argument('--llm_dim', type=int, default='768', 
        help='LLM model dimension. LLama7b:4096; GPT2-small:768; BERT-base:768')
    parser.add_argument('--llm_layers', type=int, default=6)
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    args.model = 'TimeLLM'
    args.content = 'Daily COVID-19 cases forecast at US county level for the next 14 days, based on previous 14 days'
    main(args)