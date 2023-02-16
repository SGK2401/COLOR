from utils import *
from cv2 import fastNlMeansDenoisingColored

@TIME
def test():
    print(IMAGE("./Picture/200.png").PROCESS("knc", True, None, None))