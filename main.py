from utils import *
from cv2 import fastNlMeansDenoisingColored

@TIME
def test():
    IMAGE("./Picture/200.png").PROCESS("knc", None, None)