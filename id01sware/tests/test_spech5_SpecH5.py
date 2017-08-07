import silx.io.spech5

with open('/data/visitor/ma3331/id01/knno-47-008-SSO.spec') as fh:
    s5 = silx.io.spech5.SpecH5(fh)
    #print(s5.keys())