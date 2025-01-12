import pandas as pd


df = pd.read_csv('your relation between tissue and sample.csv')
#replace your tissue
muscle_samples = df[df['Tissue_class'] == 'Uterus']['Sample']

muscle_sampleslist=muscle_samples.tolist()

file_path2 = './your gene expression file'
data = pd.read_csv(file_path2, delimiter='\t')
matching_rows = data[data.index.isin(muscle_sampleslist)]
#replace your tissue
output_file = 'muscle_samples.txt'
matching_rows.to_csv(output_file, sep='\t')


