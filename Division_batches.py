import numpy as np
import pandas as pd

new_data = pd.read_csv('ISIC_2019_Final_GroundTruth.csv')

numeros_unicos = new_data['final_label'].unique()
porcentaje_1 = 0.7
porcentaje_2 = 0.15

lotes = []

for numero in numeros_unicos:
    datos_filtrados = new_data[new_data['final_label'] == numero]
    total_filas = len(datos_filtrados)
    filas_lote_1 = int(total_filas * porcentaje_1)
    filas_lote_2 = int(total_filas * porcentaje_2)
    filas_lote_3 = total_filas - filas_lote_1 - filas_lote_2
    
    lote_1 = datos_filtrados[:filas_lote_1]
    lote_2 = datos_filtrados[filas_lote_1:filas_lote_1+filas_lote_2]
    lote_3 = datos_filtrados[filas_lote_1+filas_lote_2:]
    
    lotes.append((lote_1, lote_2, lote_3))

print(lotes)
