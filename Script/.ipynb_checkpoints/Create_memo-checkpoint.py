import os
import csv

def create_memo(col_name,desc):
    
    file_path = './Data/Features/000_features_memo.csv'
    
    if not os.path.isfile(file_path):
        with open(file_path,'w'):pass
        
    with open(file_path,'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return
        
        writer = csv.writer(f)
        writer.writerow([col_name,desc])