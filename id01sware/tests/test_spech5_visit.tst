import silx.io
print(silx.version)
import h5py
#from id01lib import id01h5



f = silx.io.open("/data/visitor/ma3331/id01/knno-47-008-SSO.spec")
#f = silx.io.open("/data/id01/inhouse/crichter/ih/hc3147/Gesl3.spec")


s1 = f["1.1"]["instrument"]


h5f = h5py.File("test.h5")
h5f.clear()
g1 = h5f.create_group("g1")
g11 = g1.create_group("g11")

g11.create_dataset("d111", dtype=float)
g11.create_dataset("d112", dtype=float)
g11.create_dataset("d113", dtype=float)

#g1.visititems(lambda k,v: print(k))
#--> g11
#    g11/d111
#    g11/d112
#    g11/d113
#    (relative paths)


#s1.visititems(lambda k,v: print(k))
#--> /1.1/instrument/specfile
#    /1.1/instrument/specfile/file_header
#    /1.1/instrument/specfile/scan_header
#    /1.1/instrument/positioners
#    ... (absolute paths)

g1.visit(lambda k: print(k))
#--> g11
#    g11/d111
#    g11/d112
#    g11/d113
#    (relative paths)


s1.visit(lambda k: print(k))
#--> /1.1/instrument/specfile
#    /1.1/instrument/specfile/file_header
#    /1.1/instrument/specfile/scan_header
#    /1.1/instrument/positioners
#    ... (absolute paths)
