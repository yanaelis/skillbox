import pandas as pd
import dill
import os
from pydantic import BaseModel
from glob import glob

path = os.environ.get('PROJECT_PATH', '..')
def predict():
    class Form(BaseModel):
        description: str
        fuel: str
        id: int
        image_url: str
        lat: float
        long: float
        manufacturer: str
        model: str
        odometer: float
        posting_date: str
        price: float
        region: str
        region_url: str
        state: str
        title_status: str
        transmission: str
        url: str
        year: int

    class Prediction(BaseModel):
        id: int
        price: float
        predicted_category: str

    def open_model():
        #загружаем модель
        with open (f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
            model = dill.load(file)
        return model

    #предсказания для всех объектов папок data / test,
    def predict_form(form, model):
        df = pd.DataFrame([form.model_dump()])  # словарь из формы преобразуем в датафрейм
        y = model.predict(df)

        return {
            'id': form.id,
            'price': form.price,
            'predicted_category': y[0]
        }
    # объединяем предсказания в один DF
    # сохраняем их в csv - формате в папку data / predictions
    def join_predicts(model):
        data = []
        for datapath in glob(f'{path}/data/test/*.json'):
            json_data = open(datapath).read()
            form = Form.model_validate_json(json_data)
            data.append(predict_form(form, model))

        df=pd.DataFrame(data)
        df.to_csv(f'{path}/data/predictions/result.csv')


    model=open_model()
    join_predicts(model)

if __name__ == '__main__':
    predict()

