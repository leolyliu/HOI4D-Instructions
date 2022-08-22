import csv

# The i-th number of each category represents the semantic label number corresponding to the i-th color
category2label_map = {
    "C1" : [24, 47],
    "C2" : [17, 47, 17],
    "C3" : [25, 47, 25],
    "C4" : [19, 47, 19, 19, 16], 
    "C5" : [16, 47, 17],
    "C6" : [27, 47, 27, 16],
    "C7" : [15, 47, 46],
    "C8" : [9, 47, 9, 9, 9, 47],
    "C9" : [48, 47, 48],
    "C10" : [22, 47],
    "C11": [23, 47, 23],
    "C12": [21, 47, 17],
    "C13": [31, 47, 46],
    "C14": [10, 47, 10, 16],
    "C15": [20, 47, 20, 20, 20, 20],
    "C16": [30, 47, 30, 30],
    "C17": [29, 47, 29, 29, 29, 47],
    "C18": [28, 47, 28, 28, 46],
    "C19": [26, 47, 26, 26, 26, 26],
    "C20": [2, 47, 47]
}

# The i-th number of each category represents the instance label number corresponding to the ith color (extended backward based on the original instance number)
category2label_map_instanceseg = {
    "C1" : [1, 2],
    "C2" : [1, 2, 3],
    "C3" : [1, 2, 1],
    "C4" : [1, 2, 1, 1, 3], 
    "C5" : [1, 2, 3],
    "C6" : [1, 2, 1, 3],
    "C7" : [1, 2, 3],
    "C8" : [1, 2, 1, 3, 3, 4],
    "C9" : [1, 2, 1],
    "C10" : [1, 2],
    "C11": [1, 2, 1],
    "C12": [1, 2, 3],
    "C13": [1, 2, 3],
    "C14": [1, 2, 1, 3],
    "C15": [1, 2, 1, 1, 1, 1],
    "C16": [1, 2, 1, 1],
    "C17": [1, 2, 1, 1, 1, 3],
    "C18": [1, 2, 1, 1, 3],
    "C19": [1, 2, 1, 1, 1, 1],
    "C20": [1, 2, 3]
}


def get_csv(path):
    with open(path) as f:
        f_csv = csv.reader(f)
    return f_csv
