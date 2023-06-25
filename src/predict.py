"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import pickle
import spark 
import json

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        Load test set dataset 
        """

        data = pd.read_csv(self.input_path + 'test_final.csv')        

        return data

    def load_model(self) -> None:
        """
        Load model from self.model_path
        """    
        # load classifier using pickle
        self.model = pickle.load(open(self.model_path + "model_clf_pickle", 'rb'))

        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
   
        new_data = self.model.predict(data)
        n = pd.DataFrame(new_data)

        return n


    def write_predictions(self, predicted_data: pd.DataFrame ,file_name="predicted_data.csv" ) -> None:
        """
        Saves the prediction as a .csv file
        """
        predicted_data.to_csv(self.output_path + file_name)

        return None
    
 #   def test_json(self,parsed_json):        
 #       #print(parsed_json)
 #       df = pd.json_normalize(parsed_json)
 # # FALTA LA INGENIERIA DE FEATURES 
 #       self.write_predictions(self.make_predictions(df),file_name="json_file_predictions.csv")

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)
    
#    def run_test_json(self):
#        with open('../Notebook/data_scientist/example.json') as user_file:
#            file_contents = user_file.read()
#        parsed_json = json.loads(file_contents)
#        self.test_json(parsed_json)
        


if __name__ == "__main__":

# uncomment when Spark is implemented
# spark = Spark()

    pipeline = MakePredictionPipeline(input_path = '../data/train_dataset/',
                                    output_path = '../data/train_dataset/',
                                    model_path = './')
    pipeline.run()  
    #pipeline.run_test_json()