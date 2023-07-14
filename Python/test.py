import psycopg2
# Database connection configuration
host = "localhost"
database = "patient_details"
user = "postgres"
password = "410455"  # Replace with your actual password

# Connect to the PostgreSQL database
connection = psycopg2.connect(host=host, database=database, user=user, password=password)
cursor = connection.cursor()

    # Insert form data into the table
cursor.execute("INSERT INTO users (email, password) VALUES ('john1@gmail.com',crypt('johnpassword', gen_salt('bf')));")
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())
connection.commit()

    # Close the database connection
cursor.close()
connection.close()