# import pickle
# import trainer 

# trainer_instance = trainer.Trainer(model = trainer.classification_model, 
#                    optimizer = trainer.optimizer, 
#                    loss = trainer.loss, 
#                    train_dataloader = trainer.spiral_train_dataloader, 
#                    test_dataloader = trainer.spiral_test_dataloader, 
#                    optimizer_params = trainer.optimizer_params, 
#                    name = 'spiral')

# # Save the entire module to a file
# with open('saved_trainer.pkl', 'wb') as file:
#     pickle.dump(trainer_instance, file)

# # Load the saved module from the file
# class Trained:
#     # Your existing Trainer class code

#     @classmethod
#     def load(cls, filename):
#         with open(filename, 'rb') as file:
#             loaded_instance = pickle.load(file)
#             return loaded_instance
import pandas as pd

pickleFile = open("saved_trainer.pkl","rb")

obj = pd.read_pickle(pickleFile)
print (obj)

