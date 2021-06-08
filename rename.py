import os 

path = os.getcwd() + '/small_data/'
lst = os.listdir(path)


convlist = []
for img in lst:
    convert = ''
    for idx in range(len(img)):
        if img[idx] == 'A':
            convert += img[:idx] +'_'
            cut = idx
        if img[idx] == 'G':
            convert += img[cut:idx] + '_' + img[idx:]
    convlist.append(convert)

for idx, oldimg in enumerate(lst):
    os.rename(path + oldimg, convlist[idx])




#print(lst)
#print(check)