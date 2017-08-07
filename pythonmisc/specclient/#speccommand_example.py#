import sys
path_to_specclient = './specclient'
sys.path.append(path_to_specclient)

from SpecClient import SpecCommand

def init_mvr(host='lid13ctrleh3',
             session='eh3',
             timeout=6000):
    
    specVersion = host + ':' + session
    cmd = SpecCommand.SpecCommand('mvr', specVersion, timeout)
    return cmd

def my_mvr(motor, distance):
    cmd = init_mvr()
    return cmd(motor, distance)

    
    
    




    
