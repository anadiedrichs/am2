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

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """

        return data

    def load_model(self) -> None:
        """
        COMPLETAR DOCSTRING
        """    
        # load classifier using pickle
        self.model = pickle.load(open(self.model_path + "model_clf_pickle", 'rb'))
        #result_score = my_model_clf.score(X_test,y_test)
        #print("Score: ",result_score)
        return None


    def make_predictions(self, data: DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
   
        new_data = self.model.predict(data)

        return new_data


    def write_predictions(self, predicted_data: DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        """

        return None


    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    spark = Spark()
    
    pipeline = MakePredictionPipeline(input_path = '../data/train_dataset/',
                                      output_path = './',
                                      model_path = './')
    pipeline.run()  