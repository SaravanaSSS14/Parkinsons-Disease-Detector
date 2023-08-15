from flask import Blueprint , render_template,request,redirect
import torch
import trainer
import pickle_trainer
from torchvision.transforms import transforms
from PIL import Image

submit_image = Blueprint("submit_image" , __name__ , static_folder="static" , template_folder="templates")

trainer_instance = pickle_trainer.Trained.load('saved_trainer.pkl')

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
        prediction = trainer_instance.model.eval()(img)
        result = torch.argmax(prediction, dim=1).item()
        if result == 1 :
            message = " Parkinson's Positive "
            return render_template("Pages/upload.html" , message=message)
        elif result == 0 :
            message = " Parkinson's Negative"
            return render_template("Pages/upload.html" , message=message)
        
    return redirect("/upload")

