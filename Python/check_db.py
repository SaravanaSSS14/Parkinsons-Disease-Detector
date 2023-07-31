from pickle import FALSE
from flask import Blueprint , render_template,request
import psycopg2
import pyautogui
check_db = Blueprint("check_db" , __name__ , static_folder="static" , template_folder="templates")

@check_db.route("/details" ,methods=['GET','POST'])
@check_db.route("/",methods=['GET','POST'])
def details():
    
    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    password = "410455"  # Replace with your actual password

    if request.method == 'POST':
        # Retrieve form data
        email = request.form['email']
        psword = request.form['password']

        # Connect to the PostgreSQL database
        connection = psycopg2.connect(host=host, database=database, user=user, password=password)
        cursor = connection.cursor()

        # Check presence of form data in table
        select_query = "SELECT email FROM users WHERE email=%s AND password=crypt(%s,password)"
        cursor.execute(select_query,(email,psword))

        selected_emails = cursor.fetchall()
    
        # Close the database connection
        cursor.close()
        connection.close()

    
        #Code for redirecting pages 
        if selected_emails :
            # Code for entering into details page
            return render_template("Pages/details.html") 
        else:
            # Code for indicating incorrect username or password
            pyautogui.alert("Invalid Username or Password")
            return render_template("index.html")

    