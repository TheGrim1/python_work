from __future__ import print_function
from builtins import str
from builtins import input
from builtins import range
import sys

def input():
    arguments={}
    arguments['firstindex'] = input('first index of the array: ')
    arguments['lastindex'] = input('second index of the array: ')
    arguments['height'] = input('height of array: ')
    arguments['linelength'] = input('original line length: ')
    return arguments


def calc(arguments):
    print('array dimension = (' +arguments['height']+','+str(int(arguments['lastindex'])-int(arguments['firstindex'])+1)+')')
    roiarr=''
    
    for l in range(0,int(arguments['height'])):
        roiarr = roiarr + str(int(arguments['firstindex'])+l*int(arguments['linelength'])) +'-' + str(int(arguments['lastindex'])+l*int(arguments['linelength'])) +','
        
    print(roiarr)  
    
if __name__=="__main__":
    if len(sys.argv) != 5:
        calc(eval(input()))
    else:
        print('using  ' + sys.argv[1] + ' as first index, '+ sys.argv[2] + ' as last index , '+ sys.argv[3] +' as the heigt of the array. ' + sys.argv[4]+' is the length of each line in the original plot:')
        arguments={}
        arguments['firstindex'] = sys.argv[1]
        arguments['lastindex'] = sys.argv[2]
        arguments['height'] = sys.argv[3]
        arguments['linelength'] = sys.argv[4]
        calc(arguments)

  
