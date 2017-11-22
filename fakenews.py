class Void(object): pass

class Unerhoert(Exception): pass

class Dummy(object):

    def __init__(self, *p, **kw):
        return

    def __setattr__(self, k, v):
	pass

    def __getattr__(self, k):
        if k == 'verboten':
		raise Unerhoert


	return Void

# if x is Void


def _test():
    try:
        import NotExist
    except ImportError:
        d = Dummy()

    a = d.wichtig
    b = d.unbenutzt
    print a, b
    print d.verboten

if __name__ == '__main__':
    _test()
