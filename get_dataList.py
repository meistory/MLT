
import csv

def get_list(phars):
    result = []
    with open(phars,'r') as fp:

        reader = csv.reader(fp)
        for line in reader:
            list = []
            for i in line:
                list.append(float(i))
            result.append(list)
    return result
