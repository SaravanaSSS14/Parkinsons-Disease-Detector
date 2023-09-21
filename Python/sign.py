from flask import Blueprint , render_template,request
import psycopg2
sign = Blueprint("sign" , __name__ , static_folder="static" , template_folder="templates")

@sign.route("/sign_details" ,methods=['GET','POST'])
@sign.route("/",methods=['GET','POST'])
def signup():
    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    pssword = "410455"  # Replace with your actual password

    if request.method == 'POST':
        # Retrieve form data
        email = request.form['email']
        psword = request.form['password']

        # Connect to the PostgreSQL database
        connection = psycopg2.connect(host=host, database=database, user=user, password=pssword)
        cursor = connection.cursor()

        # Insert form data into the table
        insert_query = "INSERT INTO users (email, password) VALUES (%s,crypt(%s, gen_salt('bf')));"
        cursor.execute(insert_query,(email,psword))
        connection.commit()

        # Close the database connection
        cursor.close()
        connection.close()
    return render_template("Pages/details.html")




    