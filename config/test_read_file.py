import ast

with open('./../FileListOfDictNameNumSentences2.txt', 'rb') as f:
    #list_files_x_names = ast.literal_eval(f.read())
    list_files_x_names = f.read()
    print(len(list_files_x_names))
    #list_files_x_names = ast.literal_eval(list_files_x_names)
    auxListString = list_files_x_names[0:len(list_files_x_names)-2]
    auxList = ast.literal_eval(auxListString)
    print(len(auxList))
    f.close()