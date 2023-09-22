# This file converts the cleaned raw dataset into a 
# single merged file that the model can work on.

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from exp.config import FeatureFiles, DataConfig
import os, pandas as pd
from data.datamerger import DataMerger

def main(args):
    args = parser.parse_args()

    # create output path if it doesn't exist
    if not os.path.exists(args.output):
        print(f'Creating output directory {args.output}')
        os.makedirs(args.output, exist_ok=True)

    # get merger class
    dataMerger = DataMerger(DataConfig, args.input)

    # %%
    # if you have already created the total df one, and now just want to 
    # reuse it to create different population cut
    output_path_total = os.path.join(args.output, 'Total.csv') 

    # whether to use the cached file
    if (not args.replace) and os.path.exists(output_path_total):
        total_df = pd.read_csv(output_path_total)
        print(f'Total.csv already exists in path {output_path_total}. Skipping...')
    else:
        total_df = dataMerger.get_all_features()
        print(f'Writing total data to {output_path_total}\n')
        
        # rounding up to reduce the file size
        total_df.round(4).to_csv(output_path_total, index=False)

    # you can define 'Population cut' in 'data'->'support'
    # this means how many of top counties you want to keep
    population_cuts = dataMerger.population_cut(total_df)
    for index, population_cut in enumerate(population_cuts):
        top_counties = FeatureFiles.population_cut[index]
        filename = f"Top_{top_counties}.csv"

        output_path_population_cut = os.path.join(args.output, filename)

        if (not args.replace) and os.path.exists(output_path_population_cut):
            print(f'{filename} already exists at {output_path_population_cut}. Skipped.')
            continue

        print(f'Writing top {top_counties} populated counties data to {output_path_population_cut}.')
        population_cut.round(4).to_csv(output_path_population_cut, index=False)  

def get_argparser():
    parser = ArgumentParser(
        description='Prepare Age Groups Dataset',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
    '--input', default=FeatureFiles.root_folder,
    help='input folder for the raw feature file'
    )

    parser.add_argument(
    '--output', default=DataConfig.root_folder,
    help='output folder for the merged feature file'
    )
    parser.add_argument(
    '--replace', help='whether to replace the existing features files',
    action='store_true'
    )
    
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)