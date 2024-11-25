from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('gender_classification.pkl','rb'))
list = ['Female','Male']

@app.route('/')
def hello_world():
    return render_template("gender.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    return render_template('gender.html',pred='{}'.format(list[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)