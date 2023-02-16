import time, cv2, json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def TIME(func: callable):
    '''Decorator for time record'''
    global timeres
    print("-"*3, "START", "-"*3)
    launch = time.time()
    func()
    finish = time.time()
    print("-"*3, "END", "-"*3)
    timeres = finish-launch
    print("Time taken:", timeres, "second(s).")


class IMAGE:
    def __init__(self, path: str) -> None:
        self._img = self.image(path)
        self.size = (self._img.shape[0]*self._img.shape[1], (self._img.shape))
        
    def image(self, path: str):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def PreProcess(self, method):
        if method == None:
            self.processed = self._preProcess(self._img)
        elif callable(method):
            self.processed = self._preProcess(method(self._img))
        else:
            raise Exception("""No method as {}. Avaliable methods are "nlm" and None.""".format(method))
    
    def _preProcess(self, object):
        img = {}
        for i in object:
            for o in i:
                o = tuple(o)
                if o in img:
                    img[o]+=1
                else:
                    img[o]=1
        return img

    def PROCESS(self, mode: str="knc", Percentage_output=False, color_set: str=None, pre_process=None):
        self.PreProcess(pre_process)
        cs = ColorSet(color_set)
        if mode == "log":
            func = models().LOG(cs.colorValue, cs.colorName, self.size[0])
        elif mode == "knc":
            func = models().KNC(cs.colorValue, cs.colorName)
        else:
            raise Exception("No such mode {}.".format(mode))
        colors = {}
        for i in self.processed:
            result = func.predict([[i[0], i[1], i[2]]])[0]
            if result in colors:
                colors[result]+=self.processed[i]
            else:
                colors[result]=self.processed[i]
        if Percentage_output:
            return {i:f"{colors[i]/self.size[0]*100}%" for i in colors}
        else:
            return colors



class ColorSet:
    def __init__(self, setName: str=None, csvPath="./color_names.csv", colorPath="./Colors.json") -> None:
        self.rawSet = pd.read_csv(csvPath)
        with open(colorPath) as file:
            self.rawColor = json.load(file)
        self.colset(setName)
    
    def colset(self, setName):
        if setName in self.rawColor:
            self.rawColorSet = pd.DataFrame([self.rawSet.loc[self.rawSet[(self.rawSet.Name == i)].index[0]] for i in self.rawColor[setName]])
        elif setName == None:
            self.rawColorSet = self.rawSet
        else:
            raise Exception("Given set name not found in json file")
        self.colorName = np.array(self.rawColorSet["Name"])
        self.colorValue = np.array(self.rawColorSet[["Red", "Green", "Blue"]])

class models:
    def KNC(self, X, Y):
        knc = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
        knc.fit(X, Y)
        return knc

    def LOG(self, X, Y, iteration):
        log = LogisticRegression(solver="liblinear", max_iter=iteration)
        log.fit(X, Y)
        return log