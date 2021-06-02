from sqlite3.dbapi2 import OperationalError
from flask import Flask, app, render_template, request
import sqlite3




app = Flask(__name__)

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

