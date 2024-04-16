# this data normalization is for the file that needed to be uploaded into MTALAB
import pandas as pd
import numpy as np

dst = '/content/drive/MyDrive/ce889_dataCollection.csv'
data = dst
data = pd.read_csv(dst)

def nrmlzd(dta):
  normalized = (dta - dta.min()) / (dta.max() - dta.min())
  return  normalized
newdta = nrmlzd(data)
# outputing the normalized and saving it into the disk
np.savetxt("Gnormalized.csv", newdta,delimiter =", ", fmt ='% s')