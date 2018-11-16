import csv
with open('/Volumes/GoogleDrive/My Drive/1. UCSC/1. 2018 Fall Quarter/CS242 Machine Learning/HWs/hw4/diabetes.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)