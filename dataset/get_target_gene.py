import pandas as pd
file_path = './train_gene.txt'
data= pd.read_csv(file_path, delimiter='\t', index_col=0)
all_genename = data.columns.tolist()
file_path2 = './gene_for_test.txt'
data2 = pd.read_csv(file_path2, delimiter='\t', index_col=0)
predict_gene = data2.columns.tolist()
fold = 0
for fold in range(5):
    fold += 1
    with open(f'./fold/top_genes{fold}.txt', 'r') as file:
        top_genes = [int(line.strip()) for line in file]
    columns_to_drop = data.columns[top_genes].tolist()
    # drop the columns that are chosen
    data3_all_target_gene = data.drop(columns_to_drop, axis=1)
    all_target_gene = data3_all_target_gene.columns.tolist()
    target_genes = [item for index, item in enumerate(all_genename) if index not in top_genes]
    target_genes = list(set(target_genes) & set(predict_gene))
    target_genes_indexes = [all_target_gene.index(gene) for gene in target_genes if gene in all_target_gene]
    with open(f'./fold/target_gene{fold}.txt', 'w') as file:
        for gene_index in target_genes_indexes:
            file.write(f"{gene_index}\n")


