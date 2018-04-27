import configparser
import os
from scipy.spatial.distance import pdist, squareform
import numpy as np
import transforms3d as tfs
import id01lib as id01

# TODO skip deleted groups in spec grabber


# parse the config file to something python can parse in a uniform way
def parseTXT2config(file_name):  # use scale =1000 for microns

    infile=open(file_name,'r')
    lines=infile.readlines()
    file_name_cfg=file_name.split('.')[0]+'.cfg'
    outfile=open(file_name_cfg,'w')
    
    # split the StagMapPositions into identifiable sections
    ID=0
    for line in lines:
        if line.find('StageMapPosition')==1: 
            outfile.write('[Position_%i]\n'%ID)
            ID+=1
        else:
            outfile.write(line)

    infile.close()
    outfile.close()

    # parse with configparser and organise correctly
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    _config.remove_section('StagePositionMapFile')

    for section in _config.sections():
        section_newname = _config.get(section,'positionname')
        _config.add_section(section_newname)
        for option,value in _config.items(section):
            if option !='positionname':
                _config.set(section_newname, option, value)
        _config.remove_section(section)

    # write the config parser file
    outfile=open(file_name_cfg,'w')
    _config.write(outfile)
    outfile.close()
'm'
def get_config(file_name_cfg):
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)
    return _config	

def get_pos_config(file_name_cfg,pos_name):
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    A=np.array([0,0])

    for section in _config.sections():
        print(section)
        if section == pos_name:
            A=np.vstack((A,np.array([np.float(_config.get(section,'xcoordinate')),np.float(_config.get(section,'ycoordinate'))])))

    return pos_name, A[1:]

def get_pars_pos_config(file_name_cfg,pos_name,scale=1.0): # scale=1.0 assumes metres
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    par=dict(_config.items(pos_name))
    names=[]
    formats=[]
    values=[]
    for entry in par:
        names.append(entry)
        formats.append('f8')
        values.append(np.float(par[entry])*scale)

    dtype = dict(names = names, formats=formats)
    array = np.array(tuple(values), dtype=dtype)

    return array

def gen_distmatrix4mpoints(file_name_cfg):
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    A=np.array([0,0])

    for section in _config.sections():
        A=np.vstack((A,np.array([np.float(_config.get(section,'xcoordinate')),np.float(_config.get(section,'ycoordinate'))])))
    dist_mat = squareform(pdist(A[1:], metric="euclidean"))
    return dist_mat

def gen_matrix4mpoints(file_name_cfg):
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    A=np.array([0,0])

    for section in _config.sections():
        A=np.vstack((A,np.array([np.float(_config.get(section,'xcoordinate')),np.float(_config.get(section,'ycoordinate'))])))

    Xarr=A[1:].copy()
    Xarr[:,1]=0
    Yarr=A[1:].copy()
    Yarr[:,0]=0
    # workaround set alternate axis= 0 to get individual components
    Xdist_mat = squareform(pdist(Xarr, metric="euclidean"))
    Ydist_mat = squareform(pdist(Yarr, metric="euclidean"))

    outarr = np.zeros((Xdist_mat.shape[0],Xdist_mat.shape[1],2)) 
    outarr[:,:,0]=Xdist_mat
    outarr[:,:,1]=Ydist_mat
    #X=squareform(pdist(A[0,1:], metric="euclidean"))
    #Y=squareform(pdist(A[1,1:], metric="euclidean"))

    #return X,Y,dist_mat
    return outarr

def apply_corr_factors(file_name_cfg,angle,x_factor,y_factor, outfile_append='_scaled.cfg'):
    #apply directly to the config file - convert to matrix and offset * multiply
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    for section in _config.sections():
        # read config values
        X,Y = np.float(_config.get(section,'xcoordinate')),np.float(_config.get(section,'ycoordinate'))
        # apply rotation and mutiplaction factor in matrix form so can take any type of correction in future
        #newmat = (Rz(angle) @ np.array([X,Y,0])) @ np.array([[x_factor,0,0],[0,y_factor,0],[0,0,0]])
        newmat = np.dot(np.dot(Rz(angle),np.array([X,Y,0])),np.array([[x_factor,0,0],[0,y_factor,0],[0,0,0]]))
        # overwrite existing section in config
        _config.set(section,'xcoordinate',str(newmat[0]))
        _config.set(section,'ycoordinate',str(newmat[1]))

    # write the new config parser file with '_scaled.cfg' appended to the name
    out_fn = file_name_cfg.split('.')[0]+outfile_append
    outfile=open(out_fn,'w')
    _config.write(outfile)
    outfile.close()
    return gen_matrix4mpoints(out_fn)

def Rz(angle):
    return tfs.euler.euler2mat(0,0,np.deg2rad(angle))

def Ry(angle):
    return tfs.euler.euler2mat(0,np.deg2rad(angle),0)

def Rx(angle):
    return tfs.euler.euler2mat(np.deg2rad(angle),0,0)

def SIFT_find_correction_matrix():
    pass

def gen_output_file(file_name_cfg,out_fn='result.txt',array_id='test'):
    _config = configparser.ConfigParser()
    _config.read(file_name_cfg)

    array = gen_matrix4mpoints(file_name_cfg)
    outfile=open(out_fn,'a+')
    outfile.write(array_id+'\t')
    [outfile.write(section+'\t') for section in _config.sections()]
    outfile.write('\n')
    for i,section in enumerate(_config.sections()):
        outfile.write(section+'\t')
        for j in np.arange(array.shape[0]):
            outfile.write('(%.6f,%.6f)\t'%(array[i,j,0],array[i,j,1]))
        outfile.write('\n')
    outfile.close()

def get_group_from_spec(groupname,group_dict):
	mot_names=[]
	outdict={}
	for j in range(1,int(group_dict[groupname]['number'])+1):
		mot_names.append(group_dict[groupname]['motor_%i'%j])
	for i in range(1,int(group_dict[groupname]['positions'])+1):
		pos=[float(x) for x in group_dict[groupname]['position_%i'%i].split()]
		formats=['f8']*len(pos)
		dtype = dict(names = mot_names, formats=formats)
		outdict[group_dict[groupname][str(i)]]=np.array(tuple(pos), dtype=dtype)
	return outdict
	


if __name__ == '__main__':
    input_file_name = "SampleonSTMholder.txt"
    # parse the txt file in a sensible way
    parseTXT2config(input_file_name)
    file_name_cfg = "SampleonSTMholder.cfg"
    #X,Y,dist_mat=gen_matrix4mpoints(file_name_cfg)
    mat=gen_matrix4mpoints(file_name_cfg)
    #mat_scaled=apply_corr_factors(file_name_cfg,90,1,1,outfile_append='_scaled.cfg')

    output_file_name = "result_file.txt"
    gen_output_file(file_name_cfg,out_fn=output_file_name,array_id='MGERMANY')
    #file_name_cfg = "SampleonSTMholder_scaled.cfg"
    #gen_output_file(file_name_cfg,out_fn=output_file_name,array_id='MFRANCE')



