
import pandas as pd

pickleFile = open("saved_trainer.pkl","rb")

obj = pd.read_pickle(pickleFile)
print (obj)

