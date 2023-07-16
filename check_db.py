from flask import Blueprint , render_template,request
import psycopg2
check_db = Blueprint("check_db" , __name__ , static_folder="static" , template_folder="templates")

@check_db.route("/details" ,methods=['GET','POST'])
@check_db.route("/",methods=['GET','POST'])
def details():
    return render_template("Pages/details.html")

# Database connection configuration
'''host = "localhost"
database = "patient_details"
user = "postgres"
password = "your_password"  # Replace with your actual password

if request.method == 'POST':
    # Retrieve form data
    email = request.form['email']
    password = request.form['password']

    # Connect to the PostgreSQL database
    connection = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = connection.cursor()

    # Check presence of form data in table
    select_query = "SELECT email FROM patient_details WHERE email=%s AND password=crypt(%s,password)"
    cursor.execute(select_query,(email,password))

    selected_emails = cursor.fetchall()

    """
    Code for redirecting pages 
    if (!selected_emails){
        # Code for indicating incorrect email or password
    } else {
        # Code for entering into details page
    }
    """


    # Close the database connection
    cursor.close()
    connection.close()'''