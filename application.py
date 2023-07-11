from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging


application = Flask(__name__)

app = application


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            Airline=request.form.get('Airline'),
            Source=request.form.get('Source'),
            Destination=request.form.get('Destination'),
            Total_Stops=request.form.get('Total_Stops'),
            Date_of_Journey=request.form.get('Date_of_Journey'),
            Arrival_Time=request.form.get('Arrival_Time'),
            Dep_Time=request.form.get('Departure_Time'),
            Additional_Info=request.form.get('Additional'),
        )
        final_new_data = data.get_data_as_dataframe()
        logging.info(final_new_data)
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('form.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
