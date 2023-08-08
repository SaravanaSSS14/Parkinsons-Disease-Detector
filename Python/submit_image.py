from flask import Blueprint , render_template,request,redirect
import psycopg2
import pyautogui
import os
import torch
import trainer
from PIL import Image

submit_image = Blueprint("submit_image" , __name__ , static_folder="static" , template_folder="templates")

@submit_image.route("/image" ,methods=['GET','POST'])
@submit_image.route("/",methods=['GET','POST'])
def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request.method == 'POST':
        img = request.files['image']

        image = Image.open(img).convert("L")

        image_transformation = trainer.transforms.Compose([  
            trainer.transforms.Resize((256,256)),
            trainer.transforms.ToTensor(),
        ])

        img = image_transformation(image).unsqueeze(0).to(device)
        # passing to model
        prediction = torch.softmax(trainer.classification_model.eval().to(device)(img),dim=1)
        pyautogui.alert(f"prediction : {torch.argmax(prediction)}")
        # print(f"prediction : {torch.argmax(prediction)}")

    return redirect("/upload")

