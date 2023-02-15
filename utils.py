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

    def PROCESS(self, mode: str="knc", color_set: str=None, pre_process=None):
        self.PreProcess(pre_process)
        color_set = ColorSet().colset(color_set)
        if mode == "log":
            func = models().LOG(color_set, len(color_set))
        elif mode == "knc":
            func = models().KNC(color_set)
        colors = {}
        for i in self.processed:
            result = func.predict([[i[0], i[1], i[2]]])[0]
            if result in colors:
                colors[result]+=self.processed[i]
            else:
                colors[result]=self.processed[i]
        return colors



class ColorSet:
    def __init__(self, csvPath="./color_names.csv", colorPath="./Colors.json") -> None:
        self.rawSet = pd.read_csv(csvPath)
        with open(colorPath) as file:
            self.rawColor = json.load(file)
    
    def colset(self, setName: str=None):
        if setName in self.rawColor:
            return pd.DataFrame([self.rawSet.loc[self.rawSet[(self.rawSet.Name == i)].index[0]] for i in self.rawColor[setName]])
        elif setName == None:
            return self.rawSet
        else:
            raise Exception("Given set name not found in json file")

class models:
    def KNC(self, data: pd.DataFrame):
        knc = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
        knc.fit(np.array(data[["Red", "Green", "Blue"]]), data["Name"])
        return knc

    def LOG(self, data: pd.DataFrame, iter):
        log = LogisticRegression(solver="liblinear", max_iter=iter)
        log.fit(np.array(data[["Red", "Green", "Blue"]]), data["Name"])
        return log