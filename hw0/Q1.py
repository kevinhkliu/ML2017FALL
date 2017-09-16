import sys
fRead = open(sys.argv[1],'r')
strRead = fRead.read()
strRead = strRead.rstrip()
strSplit = strRead.split()
dictCount={}
listIndex=[]
for words in strSplit:
    if words not in dictCount:
        dictCount[words] = 1
        listIndex.append( words )
    else:
        dictCount[words] = dictCount[words] + 1
    
fWrite = open('Q1.txt', 'w')
index = 0
for words in listIndex:    
    if index == 0:
        StringWrite = words + " " + str(index) + " " + str(dictCount[words])
        fWrite.write(StringWrite)
    else:
        StringWrite = '\n' + words + " " + str(index) + " " + str(dictCount[words])
        fWrite.write(StringWrite)
    index = index + 1
fRead.close()
fWrite.close()