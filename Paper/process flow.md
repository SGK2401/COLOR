# Flow

Open image

Preprocess image (Denoising)
Reduce complexity (count unique pixel value)

Open color table and color json file
Return matched colors

Initialize models from sklearn (KNeighborsClassifier, LogisticRegressionClassifier)
Fit RGB value and Names
(KNC->"brute") (LOG->"liblinear", input length)

Interate through preprocessed image
Stacks up returned value of each pixel
