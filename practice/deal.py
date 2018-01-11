import os
import re

# 去除空行
def delblankline(infile, outfile):
    """ Delete blanklines of infile """
    infp = open(infile+".txt", "r", encoding='utf-8')
    outfp = open(infile+outfile, "w", encoding='utf-8')
    lines = infp.readlines()
    for li in lines:
        if li.split():
            outfp.writelines(li)
    infp.close()
    outfp.close()

# 添加中文逗号
def addPunctuation(infile, outfile):
    infp = open(infile+".txt", "r", encoding='utf-8')
    outfp = open(infile+outfile, "w", encoding='utf-8')
    lines = infp.readlines()
    i = 0
    j = 0
    for li in lines:
        # if j == 4*i+2:
        if j == 2 * i:
            # +"，"
            outfp.writelines(li+"，")
            i = i + 1
        j = j+1
    infp.close()
    outfp.close()


# 添加英文文逗号
def addengPunctuation(infile, outfile):
    infp = open(infile+".txt", "r", encoding='utf-8')
    outfp = open(infile+outfile, "w", encoding='utf-8')
    lines = infp.readlines()
    i = 0
    j = 0
    for li in lines:
        # if j == 4*i+3:
        if j == 2 * i+1:
            # +","
            outfp.writelines(li)
            i = i + 1
        j = j+1
    infp.close()
    outfp.close()

# 去除换行
def deleteln(infile, outfile):
    """ Delete blanklines of infile """
    infp = open(infile+".txt", "r", encoding='utf-8')
    outfp = open(infile+outfile, "w", encoding='utf-8')
    lines = infp.readlines()
    for li in lines:
        li = li.strip('\n')
        outfp.writelines(li)
    infp.close()
    outfp.close()

# 调用示例
if __name__ == "__main__":
        delblankline("1_1", "no.txt")
        # addPunctuation("1_1no", "ch.txt")
        addengPunctuation("1_1no", "eng.txt")
        # deleteln("1_1noch","ch.txt")
        deleteln("1_1noeng","en.txt")
