import pandas as pd
from pandas import DataFrame
import os, math, sys

from exp.config import DataConfig, FeatureFiles
from utils.utils import *

class DataMerger:
    """
    Read input features and merge them into a single feature file.
    """

    def __init__(self, config:DataConfig, dataPath:str):
        self.config = config
        self.dataPath = dataPath

    def get_static_features(self) -> DataFrame:
        """Loads and merges the static features

        Returns:
            DataFrame: static features
        """
        static_df = self.get_population()[self.config.group_ids]

        """## Merge"""

        static_features_map = FeatureFiles.static_features
        for file_name in static_features_map.keys():
            feature_df = read_feature_file(self.dataPath, file_name)
            print(f'Merging feature {file_name} with length {feature_df.shape[0]}')

            has_date_columns = False
            for column in feature_df.columns:
                if valid_date(column):
                    has_date_columns = True
                    break

            # if static feature has date column, convert the first date column into feature of that name
            # this is for PVI data, and in that case static_features_map[file_name] is a single value
            if has_date_columns:
                feature_column = static_features_map[file_name]
                feature_df.rename({column: feature_column}, axis=1, inplace=True)
                feature_df = feature_df[['FIPS', feature_column]]
            else: 
                feature_columns = static_features_map[file_name]
                if type(feature_columns) == list:
                    feature_df = feature_df[['FIPS'] + feature_columns]
                else:
                    feature_df = feature_df[['FIPS', feature_columns]]

            static_df = static_df.merge(feature_df, how='inner', on='FIPS')

        print(f"\nMerged static features have {static_df['FIPS'].nunique()} counties")
        # print(static_df.head())
        
        return static_df

    def get_dynamic_features(self) -> DataFrame:
        """Loads and merges the dynamic features

        Returns:
            DataFrame: dynamic features
        """

        dynamic_features_map = FeatureFiles.dynamic_features

        dynamic_df = None
        merge_keys = ['FIPS', 'Date']
        remove_input_outliers = FeatureFiles.remove_input_outliers
        if remove_input_outliers:
            print('Removing outliers from dynamic inputs.')

        first_date = FeatureFiles.first_date
        last_date = FeatureFiles.last_date

        for file_name in dynamic_features_map.keys():
            print(f'Reading {file_name}')
            df = read_feature_file(self.dataPath, file_name)
            
            # check whether the Date column has been pivoted
            if 'Date' not in df.columns:
                if remove_input_outliers: 
                    df = remove_outliers(df)
                
                # technically this should be set of common columns
                id_vars = [col for col in df.columns if not valid_date(col)]
                df = df.melt(
                    id_vars= id_vars,
                    var_name='Date', value_name=dynamic_features_map[file_name]
                ).reset_index(drop=True)
            else:
                print('Warning ! Removing outliers is not still implemented for this case.')

            # can be needed as some feature files may have different date format
            df['Date'] = pd.to_datetime(df['Date'])

            print(f'Min date {df["Date"].min()}, max date {df["Date"].max()}')
            print(f'Filtering out dynamic features outside range {first_date} and {last_date}.')
            df = df[(first_date <= df['Date']) & (df['Date']<= last_date)]

            print(f'Length {df.shape[0]}.')

            if dynamic_df is None: dynamic_df = df
            else:
                # if a single file has multiple features
                if type(dynamic_features_map[file_name]) == list:
                    selected_columns = merge_keys + dynamic_features_map[file_name]
                else:
                    selected_columns = merge_keys + [dynamic_features_map[file_name]]

                dynamic_df = dynamic_df.merge(df[selected_columns], how='outer',on=merge_keys)
                # we don't need to keep mismatch of FIPS
                dynamic_df = dynamic_df[~dynamic_df['FIPS'].isna()]
            print()

        print(f'Total dynamic feature shape {dynamic_df.shape}')
        # print(dynamic_df.head())
        
        return dynamic_df

    def get_target_feature(self) -> DataFrame:
        """Loads and converts the target feature

        Returns:
            DataFrame: daily covid cases for each county
        """

        target_df = None
        merge_keys = ['FIPS', 'Date']
        remove_target_outliers = FeatureFiles.remove_target_outliers
        if remove_target_outliers:
            print('Removing outliers from target.')

        for file_name in FeatureFiles.targets.keys():
            print(f'Reading {file_name}')
            df = read_feature_file(self.dataPath, file_name)
            feature_name = FeatureFiles.targets[file_name]
            
            # check whether the Date column has been pivoted
            if 'Date' not in df.columns:
                df = convert_cumulative_to_daily(df)
                df.fillna(0, inplace=True)
                # technically this should be set of common columns
                id_vars = [col for col in df.columns if not valid_date(col)]

                if remove_target_outliers:
                    df = remove_outliers(df)

                df = df.melt(
                    id_vars= id_vars,
                    var_name='Date', value_name=feature_name
                ).reset_index(drop=True)
            else:
                print('Warning ! Removing outliers and moving average are not still implemented for this case.')
                df.fillna(0, inplace=True)
                # df = remove_outliers(df)

            # can be needed as some feature files may have different date format
            df['Date'] = pd.to_datetime(df['Date'])

            # some days had old covid cases fixed by adding neg values
            print(f'Setting negative daily {feature_name} counts to zero.')
            df.loc[df[feature_name]<0, feature_name] = 0
            
            print(f'Min date {df["Date"].min()}, max date {df["Date"].max()}')
            first_date = FeatureFiles.first_date
            last_date = FeatureFiles.last_date
            print(f'Filtering out target data outside range {first_date} and {last_date}.')
            df = df[(first_date <= df['Date']) & (df['Date']<= last_date)]

            print(f'Length {df.shape[0]}.')

            if target_df is None: target_df = df
            else:
                # if a single file has multiple features
                if type(feature_name) != list:
                    feature_name = [feature_name]
                selected_columns = merge_keys + feature_name

                # using outer to keep the union of dates
                target_df = target_df.merge(df[selected_columns], how='outer', on=merge_keys)

                # however, we don't need to keep mismatch of FIPS
                target_df = target_df[~target_df['FIPS'].isna()]
            print()

        print(f'Total target feature shape {target_df.shape}')
        # print(target_df.head())
        
        return target_df

    def get_all_features(self) -> DataFrame:
        """Loads and merges all features

        Returns:
            DataFrame: the merged file of all features 
        """

        static_df = self.get_static_features()
        dynamic_df = self.get_dynamic_features()
        target_df = self.get_target_feature()

        # the joint types should be inner for consistency
        print('Merging all features')

        total_df = dynamic_df.merge(target_df, how='outer', on=['FIPS', 'Date'])
        total_df = static_df.merge(total_df, how='inner', on='FIPS').reset_index(drop=True)

        print(f'Total merged data shape {total_df.shape}')
        print('Missing percentage in total data')
        print(missing_percentage(total_df))
        # print(total_df[total_df[self.data_config.sta].isnull().any(axis=1)])
        
        print('Filling null values with 0')
        total_df.fillna(0, inplace=True)

        # add future known features
        theta = total_df['Date'].dt.day_of_week * 2.0 * math.pi / 7.0
        if 'SinWeekly' in self.config.known_reals:
            total_df['SinWeekly'] = theta.apply(math.sin)
        if 'CosWeekly' in self.config.known_reals:
            total_df['CosWeekly'] = theta.apply(math.cos)
            
        return total_df

    def get_population(self) -> DataFrame:
        """Loads the population file

        Returns:
            DataFrame: population file
        """
        support_file = FeatureFiles.population
        population = pd.read_csv(
            os.path.join(FeatureFiles.root_folder, f'{support_file}')
        )
        
        return population

    def population_cut(self, total_df:DataFrame) -> List[DataFrame]:
        """Slices the total feature file based on number of top counties by population, 
        mentioned in `Population cut`

        Args:
            total_df: total feature file

        Returns:
            List[DataFrame]: list selected feature files
        """

        population_cuts = []
        # number of top counties (by population) to keep
        for top_counties in FeatureFiles.population_cut:
            print(f'Slicing based on top {top_counties} counties by population')
            
            population = self.get_population()
            sorted_fips = population.sort_values(by=['POPESTIMATE'], ascending=False)['FIPS'].values

            df = total_df[total_df['FIPS'].isin(sorted_fips[:top_counties])]
            population_cuts.append(df)

        return population_cuts