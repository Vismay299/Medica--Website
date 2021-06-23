from sqlite3.dbapi2 import OperationalError
from flask import Flask, app, render_template, request
import sqlite3
import pickle,joblib
import numpy as np

liver_perdict = joblib.load(r'/Users/vismayrathod/Desktop/medicalwebsite/med/liver_model.pkl')
heart_predict = joblib.load(r'/Users/vismayrathod/Desktop/medicalwebsite/med/heart_model.pkl')

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predict',methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = heart_predict.predict(final_features)
    output = prediction[0]
    if output==1:
        return render_template('heart.html',prediction_text='You Have Heart disease')
    elif output==0:
        return render_template('heart.html',prediction_text='You Do Not Have Heart disease')

@app.route('/predict-liv',methods=["POST"])
def predict_liv():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = liver_perdict.predict(final_features)
    output = prediction[0]
    if output==1:
        return render_template('liver.html',prediction_text='You Dont Have liver disease')
    elif output==2:
        return render_template('liver.html',prediction_text='You Have liver disease')


@app.route("/signup", methods=['GET',"POST"])
def signup():
    msg = None
    if (request.method=='POST'):
        if (request.form['username']!='' and request.form['email']!='' and request.form['password']!=''):
            username=request.form['username']
            email=request.form['email']
            password=request.form['password']
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO person VALUES('"+username+"','"+email+"','"+password+"')")
            msg = 'Your account has been created'
            conn.commit()
            conn.close()
        else:
            msg = 'Something went wrong'

    return render_template("signup.html",msg=msg)

if __name__ == "__main__":
    app.run(debug=True)

