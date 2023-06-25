"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import logging
import logging.config



class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)
        # Get the logger specified in the file
        self.logger = logging.getLogger(__name__)

    def read_data(self) -> pd.DataFrame:
        """
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        data_train = pd.read_csv(self.input_path + 'Train_BigMart.csv')
        data_test = pd.read_csv(self.input_path + 'Test_BigMart.csv')
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'
        
        pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
        self.logger.debug("pandas_df DataFrame loaded, shape_ "+str(pandas_df.shape))
        
        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function deals with all the feature engineering process.
        :return pandas_df: a DataFrame ready for training process.
        :rtype: pd.DataFrame
        """
        
        # FEATURE ENGINEERING: para los años de establecimiento
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        # LIMPIEZA: unificando etiquetas para 'Item_Fat_Content'
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        
        # LIMPIEZA: faltantes en el peso de los productos
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # LIMPIEZA: faltantes en el tamaño de las tiendas
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        # FEATURE ENGINEERING: asignación de nueva categoría para 'Item_Fat_Content'

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        'Seafood': 'Meats', 'Meat': 'Meats',
        'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
        'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # FEATURE ENGINEERING: codificación los niveles de precios de los productos
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels = [1, 2, 3, 4])

        # FEATURE ENGINEERING: codificación de variables nominales
        
        dataframe = df.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
        dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) 
        dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])


        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        return dataset

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Writes the argument transformed_dataframe to the disk as a .csv file.
        It creates two files: train_final.csv and test_final.csv on the self.output_path
        
        """
        # Split the dataset as train and test sets
        df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']
        # SettingWithCopyWarning in Pandas
        pd.options.mode.chained_assignment = None

        # deleting columns without data
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

        df_train.drop(df_train.columns[[0]],axis=1,inplace=True)
        df_test.drop(df_test.columns[[0]],axis=1,inplace=True)
        
        # saving datasets
        df_train.to_csv(self.output_path + "train_final.csv")
        df_test.to_csv(self.output_path + "test_final.csv")
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = '../data/raw_dataset/',
                               output_path = '../data/train_dataset/').run()