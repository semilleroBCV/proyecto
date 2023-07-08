import numpy as np
import pandas as pd

dic = {'MEL': 1, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}

data = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
data['final_label'] = data['MEL'].apply(lambda x: 1 if x == 1 else 0)

new_data = data[['image', 'final_label']]
new_data.to_csv('ISIC_2019_Final_GroundTruth_Binary.csv', index=False)
