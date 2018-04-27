from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()

import os
import sys
import subprocess;
import shlex
import time
import timeit
import datetime
import ast
import operator
import _thread
import threading

# home: /data/id13/inhouse2/AJ/skript/fileIO/sync/syncodisk.py

################################################### syncodisk.py
#   
#  Designed to help allocate a large amount of data to seperate disks (called parts in this program) in a sensibe way. 
#  The details of the folder alocations are saved to each part and in the source folder.
#
#  Usage:
#  
#  python syncodisk.py -<mode> <sourcepath> <path of part1> <optional size of part1> <path of part2> etc.'
#  All arguments are optional if syncodisk is run in a folder containing a valid config file, or such a path is specified as <sourcepath> i.e. after running syncodisk sucessfully once, you can rerun it in the same folder to update the disks.
#  Optional modes:
#  Default use: <mode> contains "d". 
#    -  An existing allocation will be updated, or a new allocation is created. A "df" in each <path of part> finds the available space.
#  If "t" in <mode> 
#    - test mode - rsync will not write anything
#  If "n" in <mode>
#    - delete previous config files. Note: no data files are deleted, previously transferred data may be duplicated to different parts.'
#  If "m" in <mode> 
#    - manually specify a space allocation (in Bytes) after each part
#  e.g. \n "python syncodisk.py -nst /data/id13/inhouse2/AJ/skript/fileIO/test /data/id13/inhouse2/AJ/skript/fileIO/disc0/ 200 /data/id13/inhouse2/AJ/skript/fileIO/disc1 200"'
#  Not finished:
#  If "s" in <mode>
#    - scan <path of parts> to find which folders are allready in these parts
#    - if some of the files have been written to a part without creating an appropriate config file this can be used.
#    - doesn't work beacuse of partial and keyword folders.
#
###################################################
        


class part(object):
    # Each filter instance is going to be written to one destination specified in .partpath
    # the individual paths that will be synced are saved in self.filterfname in .srcpath
    def __init__(self):
        self.partno          = 0
        self.partpath        = '/data/id13/inhouse2/AJ/skript/fileIO/disc'
        self.allocatedsize   = 100           #bytes 
        self.writesize       = 0             #bytes
        self.srcpath         = '/data/id13/inhouse2/AJ/skript/fileIO/test'
        self.filterfname     = 'part1.txt'
        self.keywords        = {"eg. path with reserved size":100}
        self.dirs            = [{"path":"","timestamp":0,"foldersize":0,"branchsize":0,"flag":0}]

    def setpartno(self,newpartno):
        self.partno=newpartno
        self.filterfname     = 'part%s.txt' % newpartno
#        print "setpartno:"
#        print (self.filterfname,self.partno)
    
    def getfolderlist(self,pathtree):
        folderlist           = []
        folderlist           = getfolderlist(pathtree,"partno",self.partno)       
        self.dirs            = folderlist

############################################################# Functions copied from the Interweb and other output functions :

def getinput(prompt,typ):
# Forces user to raw_input an "int" or "float" (specify in typ) after <prompt>
# works
    while True:
        try:
            totalpartno   =input(prompt)
            if typ=='float':
                userinput =float(userinput)
            elif typ =='int':
                userinput =int(userinput)
            else:
                raise AssertionError('invalid type specified %s') % typ
                
            break
        except ValueError:
            print("That is not a valid %s input, try again " %typ)
            
    return userinput

def raw_input_with_timeout(prompt, timeout=30.0):
    timer = threading.Timer(timeout, _thread.interrupt_main)
    astring = None
    try:
        timer.start()
        astring = input(prompt)
    except KeyboardInterrupt:
        pass
    timer.cancel()
    return astring

def dump(obj, nested_level=0, output=sys.stdout):
#### for debug visualization of the huge dicts

    spacing = '   '
    if type(obj) == dict:
        print('%s{' % ((nested_level) * spacing), file=output)
        for k, v in list(obj.items()):
            if hasattr(v, '__iter__'):
                print('%s%s:' % ((nested_level + 1) * spacing, k), file=output)
                dump(v, nested_level + 1, output)
            else:
                print('%s%s: %s' % ((nested_level + 1) * spacing, k, v), file=output)
        print('%s}' % (nested_level * spacing), file=output)
    elif type(obj) == list:
        print('%s[' % ((nested_level) * spacing), file=output)
        for v in obj:
            if hasattr(v, '__iter__'):
                dump(v, nested_level + 1, output)
            else:
                print('%s%s' % ((nested_level + 1) * spacing, v), file=output)
        print('%s]' % ((nested_level) * spacing), file=output)
    else:
        print('%s%s' % (nested_level * spacing, obj), file=output)


def printparts(parts):
## used for debugging and informing the user
    for i in range(len(parts)):
        part = parts[i]
        print("")
        print("Configuration of part %s saved in source-path: %s" % (part.partno,part.srcpath))
        print("File:                                          %s" % (part.filterfname))
        print("Destinaton path of part%s:                     %s" % (part.partno,part.partpath))
        print("Space on this part:                            %s" % (part.allocatedsize))
        print("Space allocated to folders:                    %s" % (part.writesize))
        print("Folders in this part: ")
        dump(part.dirs)



################################################################ Concerned with folder structure:
           
def recursivefoldersearch(pathtree, folderlist, spath):
#like recursice search but returns those paths (after dest/..) that appear in pathtree after (src/..)
    
    print("folderlist : ")
    dump(folderlist)
    for path in pathtree:
        if path !="info":
            for subpath in pathtree[path]:
                if subpath != "info":
                    folderlist        =recursivefoldersearch(pathtree[path], folderlist, spath)
    
                    print("looking for %s \n in %s" % (spath, subpath))
                    if subpath == spath:
                        print('found\n\n')
                        info=pathtree[path][subpath]["info"]
                        folderlist.append({"path":path,
                                           "flag":info["flag"],
                                           "branchsize":info["branchsize"],
                                           "foldersize":info["foldersize"],
                                           "timestamp":info["timestamp"]})
    return folderlist

                
def recursivesearch(pathtree,folderlist,skey,svalue="path"):
# recursive part of getfolderlist
    for path in pathtree:
        if path !="info":
            for subpath in pathtree[path]:
                
                if subpath != "info":
                    folderlist        =recursivesearch(pathtree[path],folderlist,skey,svalue)                                   

                elif pathtree[path]["info"][skey]==svalue:
                    info=pathtree[path]["info"]
                    folderlist.append({"path":path,
                                       "flag":info["flag"],
                                       "branchsize":info["branchsize"],
                                       "foldersize":info["foldersize"],
                                       "timestamp":info["timestamp"]})

    return folderlist

    

def getfolderlist(pathtree,skey,svalue):
# finds svalue for skey in pathtree and lists the corresponding path in folderlist
# e.g. used to get the contents of partx 
    folderlist               = []
    folderlist               = recursivesearch(pathtree,folderlist,skey,svalue)
    folderlist               = get_distinct(folderlist)
    return folderlist

def get_distinct(folderlist):
# removes duplicated from a list for getfolderlist
    distinct_list = []
    for each in folderlist:
        if each not in distinct_list:
            distinct_list.append(each)
    return distinct_list
  
        
def add_to_recursivedict(newpath,pathtree):
# turn a path tree into a recursive dictionary construct see scanscource
# does not overwrite existing dictitems (no update of "info")
# commented prints show function

    
#    print "recursing on newpath = %s " % newpath
#    print "with superpath = %s " % superpath

    if type(newpath) == str:
        norm_path     = os.path.normpath(newpath)
        path_list     = norm_path.split(os.sep)
        superpath     = os.sep.join(path_list[:-1])
        for expath in pathtree:
    #        print "comparing %s with %s in the pathtree" % (superpath,expath)
            if superpath == expath:
                if newpath not in pathtree[expath]:
                    pathtree[expath].update({newpath:{"info":{"timestamp":0,"foldersize":0,"branchsize":0,"partno":99,"flag":0}}})
    #            print "written: "
    #            print pathtree[expath]

            elif superpath.find(expath)!=-1:
                add_to_recursivedict(newpath,pathtree[expath])
    #            print "recursion: "
    #            print pathtree[expath]
                 
    elif type(newpath) == dict:
        norm_path     = os.path.normpath(newpath["path"])
        path_list     = norm_path.split(os.sep)
        superpath     = os.sep.join(path_list[:-1])
        for expath in pathtree:
    #        print "comparing %s with %s in the pathtree" % (superpath,expath)
            if superpath == expath:
                if newpath["path"] not in pathtree[expath]:
                    pathtree[expath].update({newpath["path"]:{"info":{"timestamp":newpath["timestamp"],"foldersize":newpath["foldersize"],"branchsize":newpath["branchsize"],"partno":newpath["partno"],"flag":newpath["flag"]}}})
    #            print "written: "
    #            print pathtree[expath]

            elif superpath.find(expath)!=-1:
                add_to_recursivedict(newpath,pathtree[expath])
    #            print "recursion: "
    #            print pathtree[expath]
                 
    return(pathtree)
        

################################################################# File/folder sizes
       
def check_access(path):

    touch   =["touch", os.path.sep.join([path,"test.file"])]
    remove  =["rm",os.path.sep.join([path,"test.file"])]

    try:
        subprocess.check_call(touch)
        subprocess.check_call(remove)
        flag = True        
    except subprocess.CalledProcessError:
        flag = False

    return flag

def df_in_path(path):
    args   =[]
    args.append("df")
    args.append(path)
    print("doing: %s" % (" ".join(args)))
    output = shlex.split(subprocess.check_output(args))
    space  = output[10]

    print("Found %s free space on %s "%(output[10],output[12]))

    return space
 
def fill_branchsize(pathtree):
# walk along the recursicely created dict and find all the individual folder sizes and sum them up into branchsize
# updates the timestamp
    for path in pathtree:
        branchsize  = 0
        flagsum     = 0

        if path != "info":
            for subpath in pathtree[path]:
                if subpath != "info":
                    if not pathtree[path][subpath]["info"]["flag"]:
                        fill_branchsize(pathtree[path])
                    branchsize          += pathtree[path][subpath]["info"]["branchsize"]
                    flagsum             += pathtree[path][subpath]["info"]["flag"]

            if flagsum == len(pathtree[path])-1:
                if not pathtree[path]["info"]["flag"]:
                    pathtree[path]["info"]["timestamp"]        = int(round(time.time()))
                    (size,timestamp)                           = foldersize(path)
                    pathtree[path]["info"]["foldersize"]       = size
    #                print "folder %s has size %s, modified at %s s" %(path,size,timestamp)
                    pathtree[path]["info"]["timestamp"]        = timestamp
                    pathtree[path]["info"]["branchsize"]       = size+branchsize
        #            print "the whole branch %s has size %s" %(path,branchsize+size)
    #                 flag
                    pathtree[path]["info"]["flag"]             = 1
    #                print "brach subpathpath %s flag set : " % path
    #                print pathtree
            
                    
    return pathtree
            
    



def modtime(path):
# not used because it is not faster than du which also gets the foldersize (see foldersize)
# get time a folder and its contents were modified
#  time files in test/  were modified
# find . -type d |xargs stat -c "%Y" 


    t           = 0
    args        = []
    args.append("find")
    args.append(path)
#    args.append("-type")
#    args.append("d")
    args.append("|")
    args.append("xargs")
    args.append("stat")
    args.append("-c")
    args.append("%Y")
#    print "doing %s" % " ".join(args)
#    print subprocess.check_output(["find",path,"-type","d"])
#    print subprocess.check_output(" ".join(args),shell="True")
    t = max(shlex.split(subprocess.check_output(" ".join(args),shell="True")))
#    print  type(t)
    return t


               
def foldersize(path):
# gets the size of the contents of <path> without following subfolders
# returns size in B and time in seconds since epoch (to compare with time())
    a            =[]
    args         =[]
    args.append("du")
    args.append("-sS")
    args.append("--time")
    args.append("--time-style=full-iso")
    args.append(path)
#    print "doing: %s" % " ".join(args[:])
    a            = shlex.split(subprocess.check_output(args))
#    print "got size %s" %a[0]
    size = a[0] 
   
# reformat the time from string to seconds since epoch:

    a1=a[1].split("-")
    Y    = int(a1[0])
    m    = int(a1[1])
    d    = int(a1[2])
    
#   print "%s-%s-%s" %(Y,m,d)

    a2=a[2].split(":") 
    h    = int(a2[0]) - int(int(a[3])/100)
    if h < 0:
        h += 24
        d -= 1
    if d  == 0:
        d = 28
        m -= 1
    mn    = int(a2[1])
    s    = int(round(float(a2[2])))
#    print "%s:%s:%s"%(h,m,s)
    

    timestamp = int((datetime.datetime(Y,m,d,h,mn) - datetime.datetime(1970,1,1)).total_seconds() + s)

    return int(size), timestamp



######################################################################### Stuff regarding rsync opperation

def largestpart(parts):
    freespace   = 0
    size        = 0
    reservation = 0
    largepart   = len(parts)-1

    for partno in range(len(parts)):
        for i in parts[partno]["keywords"]:
            reservation += parts[partno]["keywords"][i]
        freespace = (parts[partno]["size"] - parts[partno]["tobewritten"] - reservation)
#        print "part number %s has \n size %s \n tobewritten %s \n and reserved space %s" %(partno, parts[partno]["size"], parts[partno]["tobewritten"], reservation)
#        print (size,freespace)
        if size < freespace:
            size      = freespace
            largepart = partno
#            print "part %s has most free space" % partno
            

    return (largepart,parts[largepart])

def recursivesplit(sections, pathtree):
# flag = "full" - the whole of this folder fits on one part, it wont be split (for now)
 
##sorting the paths so that larges paths get allocated first
    sortedpaths = []
    for path in pathtree:
        if path!="info":
            sortedpaths.append([pathtree[path]["info"]["branchsize"],path])
    sortedpaths.sort(key=operator.itemgetter(0),reverse=True)
#    print sortedpaths
    
    for i in range(len(sortedpaths)):       
        path = sortedpaths[i][1]
#        print (i, path)
        if path!="info":
            (lsectionno,lsection)  =  largestpart(sections)
            for sectionno in range(len(sections)):             
                for keyword in sections[sectionno]["keywords"]:
                    keyword = keyword.rstrip(os.path.sep)
                    if keyword.find(path) != -1:                        # i.e. keyword is a subfolder of path 
                        pathtree[path]["info"]["flag"]         = "keypath"
#                        print "comparing \n%s with\n%s "  %(keyword,path)
                    if keyword == path:                                 # i.e. keyword is this path
#                        print "Special treatment of path\n %s \n = \n %s" % (keyword,path)
#                        print " sectionno = %s" %sectionno  #debug
#                        dump(pathtree[path])
                        if pathtree[path]["info"]["branchsize"] > sections[sectionno]["size"]-sections[sectionno]["tobewritten"]:
                            print("dumping path tree at exit:")
                            dump(pathtree)
                            print("Unable to allocate data to parts, the folder %s does not fit on part %s" %(path,sectionno))
                            print("You have some thinking to do, I quit.")
                            sys.exit(0)                    
                        pathtree[path]["info"]["partno"]     = sectionno
                        pathtree[path]["info"]["flag"]       = "keyfolder"
                        sections[sectionno]["tobewritten"]  += pathtree[path]["info"]["branchsize"]                
                        (lsectionno,lsection)                = largestpart(sections)
                            
            if pathtree[path]["info"]["partno"]== 99 and pathtree[path]["info"]["flag"] !="keypath":
                (lsectionno,lsection)                       =  largestpart(sections)
                if pathtree[path]["info"]["branchsize"] < (lsection["size"]-lsection["tobewritten"]):
                    pathtree[path]["info"]["flag"]          = "full"
                    lsection["tobewritten"]                += pathtree[path]["info"]["branchsize"]                
                    pathtree[path]["info"]["partno"]        = lsectionno                    
                    (lsectionno,lsection)                   = largestpart(sections)
                else:   
                    pathtree[path]["info"]["flag"]          = "partial"   
                     
            elif pathtree[path]["info"]["flag"] not in ["keyfolder","keypath"]:
                prevsection = sections[pathtree[path]["info"]["partno"]]
                if pathtree[path]["info"]["branchsize"] < prevsection["size"]-prevsection["tobewritten"]:
                    pathtree[path]["info"]["flag"]             = "full"              
                    prevsection["tobewritten"]                += pathtree[path]["info"]["branchsize"]                
                    sections[pathtree[path]["info"]["partno"]] = prevsection
                    (lsectionno,lsection)                      = largestpart(sections)
                else:
                    pathtree[path]["info"]["flag"]          = "partial"              
                    
            if pathtree[path]["info"]["flag"] in ["keypath","partial"]:
                try:
                    recursivesplit(sections,pathtree[path])
                except KeyError:
                    print("dumping path tree at exit:")
                    dump(pathtree)
                    print("It was not possible to allocate all the folders to your parts. Possibe cause: there is a single folder larger than the remaining space on any part")
                    print("folder:")
                    print(path)
                    print("size : %s, largest partspace found on part %s: %s" %(pathtree[path]["info"]["branchsize"], lsectionno, (lsection["size"]-lsection["tobewritten"])))
                    print("possible solution: move files into subfolders and/or delete files on part and refine the file allocation.")
                    print("You have some thinking to do, I quit.")
                    sys.exit(0)
                (lsectionno,lsection)                   =  largestpart(sections)
                pathtree[path]["info"]["partno"]        = lsectionno                    
                lsection["tobewritten"]                += pathtree[path]["info"]["foldersize"]                
                (lsectionno,lsection)                   =  largestpart(sections)
 
  
                              
    return (sections,pathtree)        
                    
    

def findsplit(parts, pathtree):
# calculate the split of the folders


# synces parts and pathtree:    
    for i in range(len(parts)):
        for pathdict in parts[i].dirs:
            print("adding %s" % pathdict["path"])
            add_to_recursivedict(pathdict,pathtree)

    superpath = parts[0].srcpath
    try:
        totalsize = pathtree[superpath]["info"]["branchsize"]
    except KeyError:
        print("dumping path tree at exit:")
        dump(pathtree)
#        printparts(parts)
        print("Could not find a valid path, possible errors: \n not running this skript in the root path of the folder allocation \n or \n corrupted config files")
        sys.exit(0)
             
    availablespace     = 0
    for i in range(len(parts)):
        availablespace+=parts[i].allocatedsize
    if availablespace<totalsize:
        print("dumping path tree at exit:")
        dump(pathtree)
        printparts(parts)
        print("Sorry, I could not find a solution to the allocation problem or there is simply not enough part space to save all files. You have some thinking to do. I quit.")   
        sys.exit(0)
    
# list of keywords (foldernames) to be put on a certain diskno
# The space of 1000000000B will be held free on part 0 for "<wholepath>" if you do:
#  part.keywords={"<wholepath>":1000000000}
# prime eg. the path to the eiger1 directory or the whole "DATA"
    sections={} # This is stupid and could have been done consistently using the part class, sorry   
    for i in range(len(parts)):
        sections.update({parts[i].partno:{"partno":parts[i].partno,"size":parts[i].allocatedsize,"tobewritten":0,"keywords":parts[i].keywords}})
    
#    dump(parts)
# allocate subbranches of the pathree to different sections
    (sections,pathtree) = recursivesplit(sections, pathtree)

# update the parts with the new folder allococations (sections)
    for i in range(len(parts)):
        part=parts[i]
        part.writesize = sections[i]["tobewritten"]
        part.getfolderlist(pathtree)

    return (parts, pathtree)

def get_folders(src):
# sorted list of folders in src
    arg       = []
    arg.append("find")
    arg.append(src)
    arg.append("-type")
    arg.append("d")

    allfolders=shlex.split(subprocess.check_output(arg))
    allfolders.sort()

    return allfolders

def scanpart(part, pathtree):
# scan the part.partpath and add all folders that also appear in pathtree to part.dirs
# not integrated into the standard workflow. Part.dirs is overwritten in the scanning process. To force folders into a certain path you will need to use keywords. This will be difficult for a large number of folders.
#I recommend starting over with the allocation and writing.
    oldfolderlist        = get_folders(part.partpath)
    for i in part.dirs:
        oldfolderlist.append(part.dirs["path"])
    oldfolderlist        = get_distinct(oldfolderlist)
    newfolderlist = []

#    print "oldfolderlist:"
#    print oldfolderlist
    nspath          = ""
    dest            = part.partpath
    src             = part.srcpath
    dest_list       = []
    src_list        = []
    new_spath_list  = []

# folderlist paths have to start like src:
    for spath in oldfolderlist:
        dest_list          = dest.split(os.sep)
        spath_list         = spath.split(os.sep) 
        new_spath_list     = [itm for itm in spath_list if itm not in dest_list]
        nspath             = os.path.sep.join(new_spath_list)
        nspath             = src + os.path.sep + nspath
        newfolderlist.append(nspath)
#    print newfolderlist
    oldfolderlist          = newfolderlist
    newfolderlist          = []
    
    for skey in oldfolderlist:
        for path in pathtree:
            if path !="info":
                for subpath in pathtree[path]:
#                    print "looking for %s in %s" % (skey,subpath)
                    if subpath == skey:
#                        dump(pathtree[path][subpath])
                        info=pathtree[path][subpath]["info"]
                        newfolderlist.append({"path":path,
                                              "flag":info["flag"],
                                              "branchsize":info["branchsize"],
                                              "foldersize":info["foldersize"],
                                              "timestamp":info["timestamp"]})
                    if subpath != "info" and skey.find(subpath)!=-1:
                        newfolderlist        =recursivefoldersearch(pathtree[path], newfolderlist, skey)
                        

#    print "newfolderlist:"
#    print newfolderlist
    part.dirs = newfolderlist    
    return part


def scansource(src):
# scan the folder src to get its folder structure and size
# return a nested dict representing the pathtree
# {'dir/': {'dir1/dir2/':{<etc.>}},{"info":{<timestamp>,<foldersize>,<totalbranchsize>,<partno>,<flag>]}}}
# <partno>            = 99      - unallocated
# <foldersize>        = X       - if <syncedbool> then on part, else at source
# <branchsize>        = X       - foldersize + all sub (and subsub foldersizes)
# <flag>              = X       - use to control recursive loops
#                     
    norm_path         = os.path.normpath(src)
    path_list         = src.split(os.sep)
    superpath         = os.sep.join(path_list[:])
    startingpathtree  = {superpath:{"info":{"timestamp":0,"foldersize":0,"branchsize":0,"partno":99,"flag":0}}}

    allfolders = get_folders(src)

    pathtree     = startingpathtree
    for path in allfolders:
        print("looking at %s" %path)
        pathtree.update(add_to_recursivedict(path,pathtree))  
#    dump(pathtree)
    pathtree = fill_branchsize(pathtree)

    return pathtree
#    print "final result:"
#    print pathtree


def syncit(part,mode):
# construct and execute the rsync commands
#    for i in range(len(parts)):
#    print mode

    problem = False
    args            = []
    args.append("rsync")
    args.append("-arvz")
    args.append("--progress")
#    args.append("--delete-before")
    
    if mode == "test":
        args.append("--dry-run")
    

    sourcelist  = part.srcpath.split(os.path.sep)
    
    for item in part.dirs:
        if item["flag"] not in ["keyfolder","full","keypath","partial","syncofix"]:
            continue
        subargs = args[0:len(args)]
        if item["flag"] not in ["full","keyfolder"]:
            subargs.append("--exclude=\"*" + os.path.sep + "\"")
    
        path           = item["path"]+os.path.sep
        subargs.append(path)

        pathlist       = path.split(os.path.sep)

        relpathlist    = []
        
        for each in pathlist:
            if each not in sourcelist:
                relpathlist.append(each)
        
        relpath = ""
        for each in relpathlist:
            relpath = os.path.sep.join([relpath,each])
            try:
                subprocess.check_call(["mkdir",os.path.sep.join([part.partpath,relpath])])
#                print "doing: %s" % cmd.join(["mkdir",os.path.sep.join([part.partpath,relpath])])
            except:
#                print "Folder allready there, continuing"
                pass

#        print os.path.join(part.partpath,relpath)
        subargs.append(os.path.sep.join([part.partpath,relpath]))
        cmd=" "
        print("doing: %s" % cmd.join(subargs))
        try:
            subprocess.check_call([cmd.join(subargs)], shell=True)
        except:
            problem = True
 #       
        
    if problem:
        print("------------------------------")
        print("\n rsync had a problem here \n")
        print("------------------------------")
        return False
    else:
        return True


############################################################ FilterfileIO

def deletefilter(src):
    flag       = 0
#    print src
    dircontent = os.listdir(src)
#    print dircontent
    for fname in os.listdir(src):
        if fname.endswith(".txt") and fname.find("part")!=-1:
            flag = 1
            try:
                print("deleting %s " % os.path.join(src,fname))
                cmd   = "rm %s"% os.path.join(src,fname)
                os.system(cmd) 
            except:
                print("error deleting %s" % os.path.join(src,fname))
                print("please remove manually")
                sys.exit(0)
    if flag !=1:
        print("No config files found in %s, continuing" %src)


def find_dest_from_filter(src):
    flag       = 0
    destpaths  = []
    dircontent = os.listdir(src).sort()
#    print dircontent
    for fname in dircontent:
        if fname.endswith(".txt") and fname.find("part")!=-1:
            flag = 1
            try:
                f = open(os.path.join(src,fname),"r")
                cfg=f.readlines()
                for i in range(13):
                    if cfg[i].find("part_path")     !=-1:
                        path  = cfg[i+1].rstrip()
                    if cfg[i].find("allocated_size")!=-1:
                        space = int(cfg[i+1].rstrip())
                destpaths.append([path,space])
                f.close()
            except IndexError: 
                print("Error reading diskXX.txt files in the specified path. Please check the format of or remove files:")
                print("path specified : %s" %src)
                print("filename tried:   %s" %fname)
                sys.exit(0)
    if flag != 1:
        print("No configfiles found in %s " % src)
        raise IndexError

    return destpaths

def findfilter(src):
    parts      = []
    flag       = 0
#    print src
    dircontent = os.listdir(src)
#    print dircontent
    for fname in os.listdir(src):
        if fname.endswith(".txt") and fname.find("part")!=-1:
            flag = 1
            try:
#                print("reading %s " % os.path.join(src,fname))
                f = open(os.path.join(src,fname),"r")
                cfg=f.readlines()

                new_filter          = part()
                for i in range(13):
                    if cfg[i].find("part_path")     !=-1:
                        new_filter.partpath         = cfg[i+1].rstrip()
                    if cfg[i].find("part_no")       !=-1:
                        new_filter.partno           = int(cfg[i+1].rstrip())
                        new_filter.setpartno(new_filter.partno)
                    if cfg[i].find("allocated_size")!=-1:
                        new_filter.allocatedsize    = int(cfg[i+1].rstrip())
                    if cfg[i].find("write_size")    !=-1:
                        new_filter.writesize        = int(cfg[i+1].rstrip())
                    if cfg[i].find("source_path")   !=-1:
                        new_filter.srcpath          = cfg[i+1].rstrip()     
                    if cfg[i].find("keywords")      !=-1:
                        new_filter.keywords         = ast.literal_eval(cfg[i+1].rstrip())
                    if cfg[i].find("folders")       !=-1:
                        try:
                            for n in range(i+1,len(cfg)):
                                new_filter.dirs.append(ast.literal_eval(cfg[n].rstrip()))
                        except:
                            new_filter.dirs={}
                parts.append(new_filter)
                f.close()
            except IndexError: 
                print("Error reading diskXX.txt files in the specified path. Please check the format of or remove files:")
                print("path specified : %s" %src)
                print("filename tried:   %s" %fname)
                sys.exit(0)
    if flag != 1:
        print("No configfiles found in %s " % src)
        
    return parts

def save(part):
# save folder list
    try:
        f = open(os.path.join(part.srcpath,part.filterfname), "w")
        try:
            tbw=[]
            tbw.append("part_path: \n%s"       %  part.partpath)
            tbw.append("part_no: \n%s"         %  part.partno)
            tbw.append("source_path: \n%s"     %  part.srcpath )
            tbw.append("allocated_size: \n%s"  %  part.allocatedsize )
            tbw.append("write_size: \n%s"      %  part.writesize )
            tbw.append("keywords: \n%s"        %  part.keywords )            
            tbw.append("folders:")
            for i in range(len(part.dirs)):
                tbw.append(part.dirs[i])
            f.writelines("%s \n" % l for l in tbw) # Write a sequence of strings to a file
        finally:
            f.close()
    except IOError:
        print("could not write file %s , quitting" %  os.path.join(part.srcpath,part.filterfname))
        sys.exit(0)




def doall(parts,
             mode="test"#or not containing test to actually write with rsync
             ):

    done = []
    for i in range(len(parts)):
        print("Trying to sync part %s" %i)
        if check_access(parts[i].partpath):
            if mode.find("test")!=-1:
                worked = syncit(part = parts[i], mode = "test")
            else:
                worked = syncit(part = parts[i], mode = "")
                if worked:
                    done.append(i)
        else:
            print("Could not access destination path of part%s at %s" %(i,parts[i].partpath))
   #    except:
   #        choice    = raw_input_with_timeout("Something happened! To quit press q, will continue in 30s ")
   #        if choice == "q":
   #            print "bye"
   #            sys.exit(0)
    
    doneconfig = []
    if len(done) ==  len(parts):
        print("SUCCESS")
        for i in range(len(parts)):
            doneconfig.append("/".join([parts[i].srcpath,parts[i].filterfname]))
        if mode.find("test")!=-1:
            print("The files listed in these configfiles can be synced: \n%s" % "\n".join(doneconfig))
        else:
            print("The files listed in these configfiles were synced: \n%s" % "\n".join(doneconfig))
    else:
        for i in range(len(parts)):
            if i not in done:
                doneconfig.append("/".join([parts[i].srcpath,parts[i].filterfname]))                
        print("The files listed in these configfiles were NOT synced: \n%s" % "\n".join(doneconfig))
                
            
           
################################################################ User IO type stuff

   
def printusage():
    print('Usage: \npython syncodisk.py -<mode> <sourcepath> <path of part1> <optional size of part1> <path of part2> etc.')
    print('All arguments are optional if syncodisk is run again in a folder containing a valid config file, or such a path is specified as <sourcepath>')
    print('\nDefault use: <mode> contains "d" \n  - An existing allocation will be updated, or a new allocation is created. A "df" in each part finds the available space.')
    print('\nIf "t" in <mode> - test mode\n  - rsync will not write anything')
    print('\nIf "n" in <mode>\n  - delete previous allocations. Note: no data files are deleted, previously transferred data my be duplicated to different parts.')
    print('\nIf "m" in <mode>\n  - manually specify a space allocation (in Bytes) after each part')
    print('e.g. \n "python syncodisk.py -nst /data/id13/inhouse2/AJ/skript/fileIO/test /data/id13/inhouse2/AJ/skript/fileIO/disc0/ 200 /data/id13/inhouse2/AJ/skript/fileIO/disc1 200"')
    sys.exit(0)
        

def initiate_parts(src, mode, destpaths):
    
    parts=findfilter(src)

    partpaths = []

           
    if mode.find("m")!=-1:
        parts = []
        try:
            for i in range(0,len(destpaths)):
                partpaths.append({"path":destpaths[i][0],"size":int(destpaths[i][1]),"no":int(i),"mode":"m"})
#                print i
#                print partpaths[int(i/2)]
        except IndexError:
            print("Invalid number of arguments, please enter a size in B after every path")
        except ValueError:
            print('Not a number, please enter a size in B (eg. "1000") after every path')
            printusage()

    else:
        try:
            for i in range(len(destpaths)):                
                partpaths.append({"path":destpaths[i][0],"size":destpaths[i][1],"no":int(i),"mode":"d"}) 
        except IndexError:
            printusage()
  ## the following part is buggy: redoing a sync run does not seem to work well
    
    newparts  = []
    try:
        for path in partpaths:  
            new_part              = part()
            new_part.setpartno(path["no"])
            new_part.srcpath      = src
            new_part.partpath     = path["path"]
            new_part.allocatedsize= int(path["size"])
            newparts.append(new_part)
    except IndexError:
        newparts=parts
    
    if len(parts)==0 and len(newparts)==0:
        print("No destination paths specified and no config files found! \n \n")
        printusage()

    if len(parts)!=len(newparts):
        print("Creating %s new, default configfiles" % len(newparts))
        for i in range(len(newparts)):
            save(newparts[i])
        parts=findfilter(src)
    else:
        print("Using %s configfiles found in: %s " % (len(parts),src))
        if mode.find("m")!=-1:
            print("Allocating manually specified space")
            for i in range(len(partpaths)):
                parts[i].allocatedsize = newparts[i].allocatedsize
        elif mode.find("d")!=-1:
            print("Saving found space")
            for i in range(len(partpaths)):              
                parts[i].allocatedsize = max(parts[i].allocatedsize, newparts[i].allocatedsize)
                    
    i = 0
    while True:
        try:
            dump(parts[i].dirs)
            i+=1
        except IndexError:
            break


    pathtree          ={}

    print("Scanning source folder %s\n ... (this can take a while) " % src)

    pathtree          = scansource(src)
#    dump(pathtree)
    
   
        
# Not finished:
#    if mode.find("s")!=-1:
#        for i in range(len(parts)):
#            print "Scanning part%s in %s for folders allready copied ..." % (i, parts[i].partpath)
#            parts[i] = scanpart(parts[i],pathtree)
#            dump(parts[i].dirs)

    print("Allocating folders from source ... ")
    (parts, pathtree) = findsplit(parts,pathtree)
 
    tbwsize = 0
    for i in range(len(parts)):
        tbwsize+=parts[i].writesize
        save(parts[i])

    printparts(parts)

    print("total number of bytes found:              %s" % pathtree[src]["info"]["branchsize"])
    print("total number of bytes allocated to parts: %s" % tbwsize)
    print("-------------------------------------------------")

    return parts


def define_destpaths(src, mode, argv):
# sort out how the user gave this info and df if neccessary
    
    destpaths = []
    if mode.find("d")!=-1 and mode.find("m")==-1:
     
        try:
            if len(argv) == 1:
                destpaths = find_dest_from_filter(src)
            else:
                for i in range(1,len(argv)):
                    try:
                        destpaths.append([argv[i],df_in_path(argv[i])])
                    except subprocess.CalledProcessError:
                        print("The size of this path could not be assessed: \n %s" % argv[i])
                        printusage()
                        sys.exit(0)
        except IndexError:
            printusage()
            sys.exit(0)
            

    if mode.find("m")!=-1 and mode.find("d")==-1:
        try:
            for i in range(1,len(argv),2):
                try:
                    destpaths.append([argv[i],argv[i+1]])
                except IndexError:
                    print("I did not understand path %s /n with size %s" % (argv[i],argv[i+1]))
                    printusage()
        except IndexError:
            printusage()

    return destpaths

    
def main(userargv):

    print("")
    print("")

    argv=[]
    userpaths = [] 
    mode = " "
    for arg in userargv:
        if arg.find(os.path.sep)!=-1 or arg.isdigit():
            argv.append(arg)
        if arg.find("-")!= -1 and arg.find("/")==-1 :
            mode = arg
            
    if len(argv)>0:
        src = argv[0]
    else:
        src = os.getcwd()

    if mode.find("n")!=-1:
        deletefilter(src)
        if mode.find("m")==-1:
            mode = mode + "d"
            
    if mode == "-t":
        mode = "-dt"
    if len(mode) == 1:
        mode = "-d"

    try:
        src  = argv[0]
    except IndexError: 
        src  = os.getcwd()
    
    destpaths = define_destpaths(src, mode, argv)


    parts = initiate_parts(src, mode, destpaths)
 
    if mode.find("t")!=-1:
        rsyncmode="test"
    else:
        rsyncmode="write"

    doall(parts,rsyncmode)
    
    

if __name__ == '__main__':

    main(sys.argv[1:])
    sys.exit(0)
