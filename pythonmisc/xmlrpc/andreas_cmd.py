import sys
from cmd import Cmd

import xmlrpclib


class AndreasCmd(Cmd):

    def __init__(self):
        Cmd.__init__(self)
        self.client = xmlrpclib.ServerProxy('http://nanofocus:8024')

    def emptyline(self):
        return

    def preloop(self):
        self.prompt = 'ANDREAS > '

    def do_andreas_action(self, *p, **kw):
        print self.client.andreas_action()

    def do_exit(self, *p, **kw):
        sys.exit(0)

if __name__ == '__main__':
    ac = AndreasCmd()
    ac.cmdloop()


