# Color analyzation tool

## Codes

### Utils.py

> ```python
> import time, cv2, json
> import numpy as np
> import pandas as pd
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.linear_model import LogisticRegression
> ```
>
> `time` for time record  
> `numpy` for data processing  
> `pandas` for csv file(data set) processing  
> `KNeighborsClassifier` and `LogisticRegression` for results

> ```python
> def TIME(func: callable):
>     '''Decorator for time record'''
>     print("-"*3, "START", "-"*3)
>     launch = time.time()
>     func()
>     finish = time.time()
>     print("-"*3, "END", "-"*3)
>     print("Time taken:", finish-launch, "second(s).")
> ```
> A simple time decorator for processing time record.

> ```python
> class IMAGE:
>     def __init__(self, path: str) -> None:
>         self._img = self.image(path)
>         self.size = (self._img.shape[0]*self._img.shape[1], (self._img.shape))
>   
>     def image(self, path: str):
>         return cv2.imread(path, cv2.IMREAD_COLOR)
>
>     def PreProcess(self, mode):
>         if mode == "nlm":
>             self._preProcess(cv2.fastNlMeansDenoisingColored(self._img))
>         elif mode == None:
>             self._preProcess(self._img)
>         else:
>             raise Exception("""No method as {}. Avaliable methods are "nlm" and None.""".format(mode))
>   
>     def _preProcess(self, object):
>         img = {}
>         for i in object:
>             for o in i:
>                 o = tuple(o)
>                 if o in img:
>                     img[o]+=1
>                 else:
>                     img[o]=1
>         self.processed = img
>
>     def PROCESS(self, mode: str="knc", color_set: str=None, pre_process: str=None):
>         self.PreProcess(pre_process)
>         if mode == "log":
>             func = algorithm().LOG(ColorSet(color_set).color_set, len(self.processed))
>         elif mode == "knc":
>             func = algorithm().KNC(ColorSet(color_set).color_set)
>         colors = {}
>         for i in self.processed:
>             result = func.predict([[i[0], i[1], i[2]]])[0]
>             if result in colors:
>                 colors[result]+=self.processed[i]
>             else:
>                 colors[result]=self.processed[i]
>         return colors
> ```
>
> `IMAGE` class, for image related operation.  
> `PIL` was used first, but it lacks built in denoising method so `cv2` was used insted.
>
> Class initialization takes the path to image file as input, which then calls `image` function to store image file as a variable.
>
> Two preprocessing function is used to convert image into python dictionary, with RGB as key, and counts of that RGB as value. `{(255, 12, 81): 7, (255, 255, 255): 241}`  
> If preprocessing mode was given, function will return the preprocessed image variable. Currently only NLMeans denoising method is implemented.  
> The seperation of preprocessing functions is just for simplification.  
> Also, this preprocessing setp greatly improves overall process time. Since only one computation is required for each unique color, bigger the image is, greater the time it saves.
>
> `PROCESS` function is used as final computation of image. It takes `mode`, `color_set`, `pre_process` as input. `mode` is used to determine which approach is using.  
> It first runns `PreProcess` function to simplify the computation, and initializes the model that is using.  
> A empty dict variable `color` is created for counting results, then iterating though the preprocessed image. If the result is in the `color`, add them up, and if otherwise, create a new key. After the iteration finishes, returns `color` variable.

> ```python
> class ColorSet:
>     def __init__(self, setName: str = None, csvPath="./color_names.csv", colorPath="./Colors.json") -> None:
>         self.rawSet = pd.read_csv(csvPath)
>         with open(colorPath) as file:
>             self.rawColor = json.load(file)
>
>     def colset(self, setName: str):
>         if setName != None:
>             return pd.DataFrame([self.rawSet.loc[self.rawSet[(self.rawSet.Name == i)].index[0]] for i in self.rawColor[setName]])
>         else:
>             return self.rawSet
> ```
>
> `ColorSet` class is used to resolve the raw csv table.
>
> It initializes with csv file as `rawSet`, and opens stored `Colors.json` file as `rawColor`.
>
> `colset` will return the set of colors. If the given name of color-set is not empty, it will return a new `DataFrame` object with the given colors in `Colors.json`. Else the `rawSet` will be returned.

> ```python
> class models:
>     def KNC(self, data: pd.DataFrame):
>         knc = KNeighborsClassifier(n_neighbors=1, algorithm="brute")
>         knc.fit(np.array(data[["Red", "Green", "Blue"]]), data["Name"])
>         return knc
>
>     def LOG(self, data: pd.DataFrame, iter):
>         log = LogisticRegression(solver="liblinear", max_iter=iter)
>         log.fit(np.array(data[["Red", "Green", "Blue"]]), data["Name"])
>         return log
> ```
>
> `models` class is only used for, models.
>
> It first initializes the model in `sklearn`, then fits the color-set with names, and other required data. Initialized object will then be returned.
>
> For `KNeighborsClassifier`, brute approach is used. BallTree and KDTree will insted take longer time to finish.  
> For `LogisticRegression`, liblinear is used since other approaches will cause issue.

### Main.py

> ```python
> from utils import *
>
> @TIME
> def test():
>     print(IMAGE("./Picture/800.png").PROCESS("knc", None, None))
> ```
>
> Imports all from `utils.py`
>
> Uses `TIME` to record processing time.  
> Since `PROCESS` returns the result, `print` is used to output the result.

## Usage

For image color description, helping artists or color-blind people.  
It can be used in stylized designing, pixel art creation, low-poly design, or just to see what colors are in this image.

## Conclutions?

This implementation is best used for given/limited output target.  
For example, narrowing down an image into few given colors. Or, converting an image into given colors.  
By editing `Names` or adding new role in the csv file, custom names as output is also possible.


To optimize codes, reorganizing functions and classes is important. Also, changing all image processing codes to support PIL or/and cv2 will make future changes easier.  
Reordering output into descending order and adding percentage as output result can be very helpful. 

The maximum iteration in final processing loop is `255*255*255=16581375`, which takes around 3600 seconds on a rather high-end device. The NLMeans denoising can reduce the total amount of unique color, thus reducing the iteration count.  
Still, preprocessing though the original image can't be more simplfied, so big image will still take a long time to process.

KNC method is way more accurate than LogisticRegression. For example, LogisticRegression will classify `(80, 80, 80)` into `Peach`, which has the RGB value of `(255, 229, 180)`, that is completely different.  
As for KNC, the result is `Dark liver`, with RGB value `(83, 75, 79)`. It is indeed the closest value in the csv file.  
Also, for some how KNC is about twice as fast as LogisticRegression, which is opposite from the last version(iterating though each pixel one).


Time it takes is directly proprotional to the colors inside an image. The example images are simple, with only few colors. As for an photos or other complex images, it will take way longer to process.  
Which means, for massive images, pixel count isn't as important as colors inside. A small image can take longer due to all it's unique colors, and a large image can take only seconds since it is monotone.

> Example image
>
> 200*200 : 0.17 Seconds  
> 400\*400 : 0.30 Seconds  
> 800\*800 : 0.65 Seconds

> Normal image
>
> 2480*3508 : 53.66 Seconds  
> 849\*1200 : 34.01 Seconds

The example image used is shown below, with 800*800 as it's width and height.  
![1676090944769](image/Report/1676090944769.png)

With mode set to `knc` and other two set to `None`, the result is as shown.
```
--- START ---
{'Dark liver': 200833, 'Wenge': 78, 'Beaver': 441, 'Pale taupe': 147, 'Tan': 189267, 'Shadow': 95, 'Light French beige': 215, 'Dark vanilla': 321, 'Desert sand': 138539, 'Light taupe': 135, "Davy's grey": 31, 'Khaki (HTML/CSS) (Khaki)': 26, 'Pastel brown': 303, 'Deep Taupe': 17, 'Granite Gray': 8, 'Café au lait': 8853, 'Chamoisee': 24923, 'Camel': 446, 'Deer': 5, 'Dirt': 16449, 'Quartz': 4836, 'Outer Space': 606, 'Raw umber': 32, 'Arsenic': 101, 'Black olive': 691, 'Umber': 23, 'Onyx': 114, 'Jet': 755, 'Coffee': 198, 'Charleston green': 17665, 'Raisin black': 33570, 'Dark lava': 22, 'French bistre': 32, 'Donkey brown': 10, 'Pale brown': 55, 'Bistre': 25, 'Coyote brown': 36, 'Old burgundy': 21, 'Dark liver (horses)': 18, 'Dark brown-tangelo': 4, 'Olive Drab #7': 6, 'Van Dyke Brown': 3, 'Café noir': 1, 'Rifle green': 42, 'Liver chestnut': 2}
--- END ---
Time taken: 0.6805756092071533 second(s).
```
