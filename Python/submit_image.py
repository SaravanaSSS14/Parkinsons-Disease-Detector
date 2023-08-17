from flask import Blueprint, render_template, request, redirect
import torch
from torchvision.transforms import transforms
from PIL import Image
import trainer

submit_image = Blueprint("submit_image", __name__, static_folder="static", template_folder="templates")

@submit_image.route("/image", methods=['GET', 'POST'])
@submit_image.route("/", methods=['GET', 'POST'])
def check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request.method == 'POST':
        img = request.files['image']
        #model_path = './classification_model.pth'
        trainer.classification_model.load_state_dict( torch.load('./classification_model.pth') )

        test_image_path = Image.open(img).convert('RGB')
        test_image = trainer.test_transformations(test_image_path).to(device).unsqueeze(0)

        # predicting
        prediction = trainer.classification_model.to(device).eval()(test_image)
        result = torch.argmax(prediction,dim=1)
        #image = Image.open(img).convert("L")
        if result.item() == 1:
            message = " Parkinson's Positive "
            return render_template("Pages/upload.html", message=message)
        elif result.item() == 0:
            message = " Parkinson's Negative"
            return render_template("Pages/upload.html", message=message)

    return redirect("/upload")
