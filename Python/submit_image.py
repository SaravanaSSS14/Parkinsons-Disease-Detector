from flask import Blueprint, render_template, request, redirect
import torch
from torchvision.transforms import transforms
from PIL import Image
import trainer
import psycopg2

submit_image = Blueprint("submit_image", __name__, static_folder="static", template_folder="templates")

@submit_image.route("/image", methods=['GET', 'POST'])
@submit_image.route("/", methods=['GET', 'POST'])
def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Database connection configuration
    host = "localhost"
    database = "patient_details"
    user = "postgres"
    psword = "Saravana$$$14"  # Replace with your actual password

    if request.method == 'POST':
        img = request.files['image']
        email = request.form['email']
        #model_path = './classification_model.pth'
        trainer.classification_model.load_state_dict(torch.load('./classification_model.pth', map_location=torch.device('cpu')))

        test_image_path = Image.open(img).convert('RGB')
        test_image = trainer.test_transformations(test_image_path).to(device).unsqueeze(0)

        # predicting
        prediction = trainer.classification_model.to(device).eval()(test_image)
        result = torch.argmax(prediction,dim=1)
        #image = Image.open(img).convert("L")
        # Connect to the PostgreSQL database
        if result.item() == 1:
            msg = "Positive"

        elif result.item() == 0:
            msg = "Negative"

        connection = psycopg2.connect(host=host, database=database, user=user, password=psword)
        cursor = connection.cursor()

        # Insert form data into the table, with patient_id as a serial type (auto-generated)
        insert_query = "UPDATE patient_details SET result = %s WHERE email = %s "
        cursor.execute(insert_query, (msg,email))
        connection.commit()

        select_query = "SELECT * from patient_details WHERE email = %s "
        cursor.execute(select_query, (email,))

        data = cursor.fetchall()
        res = prediction[0][result.item()].item()*100
        res = round((res - (-300.0)) * (100.0 - 80.0) / (1000.0 - (-300.0)) + 80.0,2)
        # Close the database connection
        cursor.close()
        connection.close()
        return render_template("Pages/result.html", data=data , res=res )

    return redirect("/upload")
