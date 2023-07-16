from flask import Flask , render_template 
from check_db import check_db
from submit_details import submit_details
from sign import sign
import psycopg2

app = Flask(__name__)
app.register_blueprint(check_db,url_prefix="/details")
app.register_blueprint(sign,url_prefix="/sign_details")
app.register_blueprint(submit_details,url_prefix="/upload")

@app.route("/",methods=['GET','POST'])
def index():
    return render_template("index.html")

@app.route("/signup")
def signup():
    return render_template("Pages/index-signup.html")

if __name__ == '__main__':
    app.run(debug=True)