from __future__ import print_function
import os
import sys
import ast

import syncodisk as synco


def read_part(partfname):

    new_filter          = synco.part()
    
    if partfname.endswith(".txt") and partfname.find("part")!=-1:
        flag = 1
        try:
            #                print("reading %s " % os.path.join(src,fname))
            f = open(partfname,"r")
            cfg=f.readlines()

            
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
                if cfg[i].find("folders:")       !=-1:
                    try:
#                        print "trying to read folders from line %s" %i
                        for n in range(i+1,len(cfg)):
                            new_filter.dirs.append(ast.literal_eval(cfg[n].rstrip()))
#                            print (ast.literal_eval(cfg[n].rstrip()))
                    except IndexError:
                        print("index error on reading folders listed in file %s "  %partfname)
                        new_filter.dirs={}
            f.close()
        except IndexError: 
            print("Error reading diskXX.txt files in the specified path. Please check the format of or remove files:")
            print("path specified : %s" %  partfname)
            sys.exit(0)
    
    return new_filter

           
def is_subdir(path1,path2):
    path1    = os.path.realpath(path1)
    path1    = os.path.realpath(path2)
    relative = os.path.relpath(path1,path2)
    return not relative.startswith(os.pardir + os.sep)

def compare_sourcefolder_to_folderlist(src, folderlist):

    allfolders = synco.get_folders(src)
    print("found %s folders " %len(allfolders))

 
#    print "folderlist found in %s " %partfname
#    synco.dump(folderlist)
    
#    print type(folderlist)
    notinthispart = []
    fullpaths     = [] 

    for path in allfolders:
        found = False
        
        # check whether path is a subpath of a full path
        for fullpath in fullpaths:
            if is_subdir(path,fullpath):
                found = True
                print("not copying subpath %s of a full path %s" %(path,fullpath))
                break

#        print "found source path :\n%s" %path
        for savedpath in folderlist:

#            print "comparing %s \nwith %s"%(os.path.normpath(path),os.path.normpath(savedpath['path']))
            if os.path.normpath(path) == os.path.normpath(savedpath['path']):
                if savedpath["flag"]=="full":
                    print("not copying full path %s" %path)
                    found = True
                    fullpaths.append(savedpath['path'])
                    break
                else:
                    print("copying %s path %s" %(path,savedpath["path"]))
                    notinthispart.append({'path':path,'flag':savedpath["flag"]})
                    found = True
                    break
                

        if found == False:
            print("copying %s path %s" %("new" ,path))
            notinthispart.append({'path':path,'flag':'syncofix'})
            


    return notinthispart


def update_part(referencepart,correctedpart):

    src          = os.path.dirname(referencepart)
    oldpart      = read_part(correctedpart)
    savedpart    = read_part(referencepart)
    oldpart.dirs = compare_sourcefolder_to_folderlist(src, savedpart.dirs)

    synco.dump(oldpart.dirs)
    synco.save(oldpart)
    syncpart     = oldpart
    return syncpart


def main(args):

    referencepart  = args[0]
    correctedpart  = args[1]
#    syncpart       = update_part(referencepart,correctedpart)
    syncpart       = read_part(correctedpart)
    synco.dump(syncpart.dirs)
    synco.syncit(syncpart,"doit")



if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc. 
\n2) python <thisfile.py> -f <file containing args as lines> 
\n3) find <*yoursearch* -> arg1 etc.> | python <thisfile.py> 
"""

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)
