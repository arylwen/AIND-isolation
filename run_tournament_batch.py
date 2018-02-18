import os
from subprocess import *

for i in range(10):
    p = Popen(['python','tournament.py'], stdout=PIPE)
    output = p.communicate()
    print (output[0])