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

def scrapGenitive(word):
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

############################################################################

# formatClass()

df = readWords()

# Remove words of class -1
df = df[df[1] != -1]
df.reset_index(inplace=True, drop=True)

file_name = "gen"
range_a = 29000
range_b = min(range_a + 999, df.shape[0]-1)
print(range_b)

file_out = open("./Genitive/" + file_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')
file_err = open("./Genitive/err_" + file_name + "_" + str(range_a) + "_" + str(range_b) + ".txt", 'w')

r = range(range_a, range_b + 1)
for i in r:
    gen = scrapGenitive(df[0][i]).rstrip() # genitive
    if (gen == ""):
        file_err.write(df[0][i] + "," + str(df[1][i]) + "\n")
    else:
        file_out.write(df[0][i] + "," + gen + "," + str(df[1][i]) + "\n")

file_out.close()
file_err.close()