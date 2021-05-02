# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import xml.etree.ElementTree as ET
import requests
import pandas as pd

def formatClass():
    # keep words of class <= 50 and of unknown class (-1)
    file_words = "./kotus-sanalista_v1/kotus-sanalista_v1.xml"
    root = ET.parse(file_words).getroot()
    file_out = open("./kotus-sanalista_v1/kotus-sanalista-class.txt", 'w')

    for w in root.findall('st'):
        s = w.find('s').text # word
        node_tn = w.find('t/tn') # class
        
        if node_tn is not None:
            if (int(node_tn.text) <= 50):
                file_out.write(s + "," + node_tn.text + '\n')
        else:
            file_out.write(s + ",-1\n")
    file_out.close()
    

def readWords(filepath = "./kotus-sanalista_v1/kotus-sanalista-class.txt"):
    df = pd.read_csv(filepath, header=None)
    return df

def getStats(df):
    return df[1].value_counts()

def scrapGenitiveFIWiktionary(word):
    r = requests.get("https://fi.wiktionary.org/wiki/" + word)
    if ("genetiivi" not in r.text.lower()):
        return ""
    
    root = ET.fromstring(r.text)
    _tr = root.findall(".//tr")
    
    for tr in _tr:
        _td = root.findall(".//td")

        for i in range(0, len(_td)):
            _txt = list(_td[i].itertext())
            txt = [x.lower().rstrip() for x in _txt]
            if 'genetiivi' in txt:
                if (len(_td) > i+1):
                    _gen = list(_td[i+1].itertext())
                    if (len(_gen)>0):
                        return _gen[0]
    return ""


def scrapCaseFIWiktionary(word, case):
    """
    case: ex: genetiivi
    """
    r = requests.get("https://fi.wiktionary.org/wiki/" + word)
    if (case not in r.text.lower()):
        return ""
    
    root = ET.fromstring(r.text)
    _tr = root.findall(".//tr")
    
    for tr in _tr:
        _td = root.findall(".//td")

        for i in range(0, len(_td)):
            _txt = list(_td[i].itertext())
            txt = [x.lower().rstrip() for x in _txt]
            if case in txt:
                if (len(_td) > i+1):
                    _gen = list(_td[i+1].itertext())
                    if (len(_gen)>0):
                        return _gen[0]
    return "" 

def scrapGenitiveWiktionary(word):
    r = requests.get("https://wiktionary.org/wiki/" + word)
    if ("inflection-table fi-decl vsSwitcher" not in r.text):
        return ""
    
    root = ET.fromstring(r.text)
    table = root.find(".//table[@class='inflection-table fi-decl vsSwitcher']")
    # table = _table[0]
    _tr = table.findall(".//tr")
    
    for i in range(0, len(_tr)):
        l = [x.lower().rstrip() for x in list(_tr[i].itertext())]
        if ('genitive' in l):
            return l[3]

    return ""

def scrapCaseWiktionary(word, case):
    """
    case: ex: genitive
    """
    r = requests.get("https://wiktionary.org/wiki/" + word)
    if ("inflection-table fi-decl vsSwitcher" not in r.text):
        return ""
    
    root = ET.fromstring(r.text)
    table = root.find(".//table[@class='inflection-table fi-decl vsSwitcher']")
    # table = _table[0]
    _tr = table.findall(".//tr")
    
    for i in range(0, len(_tr)):
        l = [x.lower().rstrip() for x in list(_tr[i].itertext())]
        if (case in l):
            return l[3]

    return ""

############################################################################


def scrap(name, short_name, case_fi, case_en, range_a=2000):
    """
    name: Genitive
    short_name: gen
    case_fi: genetiivi
    case_en: genitive
    """
    df = readWords()

    # Remove words of class -1
    df = df[df[1] != -1]
    df.reset_index(inplace=True, drop=True)
    
    
    # 2000
    range_b = min(range_a + 999, df.shape[0]-1)
    print(range_b)
    
    file_out = open("./" + name + "/" + short_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')
    file_err = open("./" + name + "/err_" + short_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')
    
    r = range(range_a, range_b + 1)
    for i in r:
        gen = scrapCaseFIWiktionary(df[0][i], case_fi).rstrip() # genitive
        if (gen != ""):
            file_out.write(df[0][i] + "," + gen + "," + str(df[1][i]) + "\n")
        else:
            gen2 = scrapCaseWiktionary(df[0][i], case_en).rstrip()
            if (gen2 != ""):
                file_out.write(df[0][i] + "," + gen2 + "," + str(df[1][i]) + "\n")
            else:
                file_err.write(df[0][i] + "," + str(df[1][i]) + "\n")
    
    file_out.close()
    file_err.close()




def scrap_genitive_full():
    # formatClass()
    
    
    df = readWords()
    
    # Remove words of class -1
    df = df[df[1] != -1]
    df.reset_index(inplace=True, drop=True)
    
    file_name = "gen"
    range_a = int(sys.argv[1])
    # 2000
    range_b = min(range_a + 999, df.shape[0]-1)
    print(range_b)
    
    file_out = open("./Genitive/" + file_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')
    file_err = open("./Genitive/err_" + file_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')
    
    r = range(range_a, range_b + 1)
    for i in r:
        gen = scrapGenitiveFIWiktionary(df[0][i]).rstrip() # genitive
        if (gen != ""):
            file_out.write(df[0][i] + "," + gen + "," + str(df[1][i]) + "\n")
        else:
            gen2 = scrapGenitiveWiktionary(df[0][i]).rstrip()
            if (gen2 != ""):
                file_out.write(df[0][i] + "," + gen2 + "," + str(df[1][i]) + "\n")
            else:
                file_err.write(df[0][i] + "," + str(df[1][i]) + "\n")
    
    file_out.close()
    file_err.close()
    
    
    
    
scrap("Inessive", "ine", "inessiivi", "inessive")