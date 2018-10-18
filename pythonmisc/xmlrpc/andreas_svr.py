import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

import time

def andreas_action():
    print "call the thing ..."
    time.sleep(200.0)
    return "done."

#s = xmlrpclib.ServerProxy('http://localhost:8024')


def start_andreas_server():
    server = SimpleXMLRPCServer(("nanofocus", 8024))
    server.register_function(andreas_action)
    return server

if __name__ == '__main__':
    print "starting Andreas server ..."
    svr = start_andreas_server()
    svr.serve_forever()
