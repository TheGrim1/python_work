import sys

import SimpleXMLRPCServer
import xmlrpclib

class ServerFunctions:
    def __init__(self):
        pass
    
    def _privateFunction(self):
        # This function cannot be called through XML-RPC because it
        # starts with an '_'
        pass
    
    def server_print(self, msg):
        print (msg)
        return True
    
    def repeat(self, astr, times):
        return astr * times

def main(arg):
    if arg.upper() == 'SERVER':
        start_server()
    elif arg.upper() == 'CLIENT':
        start_client()

def start_server():
    lid13lab1_ip = '160.103.33.180'
    nanofocus_ip = '160.103.33.55'
    server = SimpleXMLRPCServer.SimpleXMLRPCServer((lid13lab1_ip, 8000))
    server.register_instance(ServerFunctions())
    print 'started XML server, waiting for instructions' 
    server.serve_forever()


# Client code

def start_client():
    server = xmlrpclib.Server('http://160.103.33.180:8000')
    msg = 'Hello world!'  
    print 'asking the server to print %s' %msg
    server.server_print(msg)
    
    
if __name__=='__main__':
    if len(sys.argv) >2:
        raise ValueError('to many args, expected server or client')
    elif len(sys.argv)==0:
        arg=input('server or client')
    arg = sys.argv[1]
    main(arg)
