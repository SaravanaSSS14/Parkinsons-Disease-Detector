from flask import Flask , render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/Pages")
def signup():
    return render_template("Pages/index-signup.html")
if __name__ == '__main__':
    app.run(debug=True)