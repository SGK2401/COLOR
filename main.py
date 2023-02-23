from utils import *
from cv2 import fastNlMeansDenoisingColored

@TIME
def test():
    print(IMAGE("./Picture/RGB1.png").PROCESS("knc", True, None, None))
    print(IMAGE("./Picture/RGB2.png").PROCESS("knc", True, None, None))
    print(IMAGE("./Picture/RGB3.png").PROCESS("knc", True, None, None))
    print("================")
    print(IMAGE("./Picture/RGB1.png").PROCESS("log", True, None, None))
    print(IMAGE("./Picture/RGB2.png").PROCESS("log", True, None, None))
    print(IMAGE("./Picture/RGB3.png").PROCESS("log", True, None, None))
