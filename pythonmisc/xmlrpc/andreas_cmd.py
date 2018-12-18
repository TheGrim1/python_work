import sys
from cmd import Cmd

import xmlrpclib


class AndreasCmd(Cmd):

    def __init__(self):
        Cmd.__init__(self)
        self.client = xmlrpclib.ServerProxy('http://lid13eh31:8020')

    def emptyline(self):
        return

    def preloop(self):
        self.prompt = 'ANDREAS > '

    def do_andreas_action(self, *p, **kw):
        try:
            msg = str(p)
        except TypeError:
            print('please type a valid str')
            
        print self.client.andreas_action()

    def do_exit(self, *p, **kw):
        sys.exit(0)

    def do_print(self, *p, **kw):
        for msg in p:
            self.client.server_print(str(msg))

    def do_load_lookuptable(self, *p, **kw):
        self.client.load_lookuptable(p[0])
        print('lookuptable loaded')

    def do_move_phi_with_lookup(self, *p, **kw):
        self.client.lut_mv_phi(float(p[0]))
        print('done')

    def do_where_is_fluoa(self, *p, **kw):
        fluoa_pos = self.client.where_is_fluoa()

        for (name, pos) in zip(['fluoax','fluoay','fluoaz'],fluoa_pos):
            print('{} is at {}'.format(name,pos))

            
if __name__== '__main__':
    ac = AndreasCmd()
    ac.cmdloop()


