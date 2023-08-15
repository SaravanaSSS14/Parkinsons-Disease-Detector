import pickle
import trainer 

trainer_instance = trainer.Trainer(model = trainer.classification_model, 
                   optimizer = trainer.optimizer, 
                   loss = trainer.loss, 
                   train_dataloader = trainer.spiral_train_dataloader, 
                   test_dataloader = trainer.spiral_test_dataloader, 
                   optimizer_params = trainer.optimizer_params, 
                   name = 'spiral')

# Save the entire module to a file
with open('saved_trainer.pkl', 'wb') as file:
    pickle.dump(trainer_instance, file)

# Load the saved module from the file
class Trained:
    # Your existing Trainer class code

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
            return loaded_instance
# import pickle
# try:
#     with open('saved_trainer.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     print("Model loaded successfully!")
#     # You can further inspect or use the loaded_model here
# except Exception as e:
#     print("Error:", e)
