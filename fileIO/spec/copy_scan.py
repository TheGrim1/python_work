import subprocess
import os

def readlines_scan(source_fname, scanno):
    '''
    returns a list of strings corresponding to the entire scan in a specfile
    '''
    
    scan_str_list = []
    scantpl = '#S {}'
    spec_f = open(source_fname)
    all_lines = spec_f.readlines()    
    end = False
    in_scan = False
    i = 0 
    while not end and i< len(all_lines):
        current_line = all_lines[i]
        if current_line.startswith(scantpl.format(scanno)):
            in_scan = True
        if current_line.startswith(scantpl.format(scanno+1)):
            end = True
        
        if in_scan and not end:
            scan_str_list.append(current_line)

        i+=1
    
    spec_f.close()        
    return scan_str_list


def readlines_fileheader(source_fname):
    header_str_list = []
    header_start = '#F '
    scan_start = '#S '
    spec_f = open(source_fname)
    all_lines = spec_f.readlines()    
    end = False
    in_scan = False
    i = 0 
    while not end and i< len(all_lines):
        current_line = all_lines[i]
        if current_line.startswith(header_start):
            in_header = True
        if current_line.startswith(scan_start):
            end = True
        
        if in_header and not end:
            header_str_list.append(current_line)

        i+=1
    spec_f.close()
    
    return header_str_list


def append_scans_for_xsocs(source_fname, source_scanno_list, source_edf_path, dest_fname, dest_scanno_list, dest_edf_path, edf_prefix='vo2', edf_idxFmt='%05d', edf_suffix='.edf.gz', copy_edf=True):
    source_scanno_list.sort()
    dest_f = open(dest_fname,'a')
    imagefile_tpl = '#C imageFile dir[{}] prefix[{}] idxFmt[{}] nextNr[{}] suffix[{}]\n'
    edf_fname_tpl = edf_prefix+edf_idxFmt+edf_suffix
    
    for source_index, scanno in enumerate(source_scanno_list):
        dest_scanno = dest_scanno_list[source_index]
        current_scan_list = readlines_scan(source_fname, scanno)
        no_lines = len(current_scan_list)
        things_done = 0
        for i in range(no_lines):
            curr_line = current_scan_list[i]
            
            if curr_line.startswith('#S '):
                replace_line_list = curr_line.split(' ')
                replace_line_list[1] = str(dest_scanno_list[source_index])
                replace_line = ' '.join(replace_line_list)
                current_scan_list[i] = replace_line
                things_done+=1
            if curr_line.startswith('#C imageFile'):
                current_scan_list[i] = imagefile_tpl.format(
                    dest_edf_path,edf_prefix,edf_idxFmt, dest_scanno, edf_suffix)
                things_done+=1
            if things_done==2:
                break
                    
            

        print('\nappending scan no {} in {}\nas scan {} to file {}'.format(scanno, source_fname, dest_scanno, dest_fname))
        dest_f.writelines(current_scan_list)

        if copy_edf:
            for fname in os.listdir(source_edf_path):
                if fname==edf_fname_tpl % source_index:
                    copy_source = source_edf_path + os.path.sep + fname
                    copy_dest = dest_edf_path + os.path.sep + edf_fname_tpl % dest_scanno
                    call_list = ['cp', copy_source, copy_dest]

                    print('calling: {}'.format(' '.join(call_list)))
                    call_return = subprocess.call(call_list)
                    if call_return == 0:
                        things_done+=1

        else:
            things_done+=1
            
        
        if things_done!=3:
            print('something missing here')
        else:
            print('SUCCESS') 
    dest_f.close()
        
    
def make_header_file(source_fname, dest_fname):
    
    dest_f = open(dest_fname,'w')
    print('writing header of file {} to file {}'.format(source_fname, dest_fname))

    header_list = readlines_fileheader(source_fname)
    dest_f.writelines(header_list)
    dest_f.close()


def shift_and_copy(source_fname, source_scanno_list, dest_fname, dest_scanno_list, copy_file_header = True, data_col_to_shift = 4, shift_magnitude = 2.2, verbose=False):
    '''
    appends scans to dest_fname if copy_file_header = False, otherwise overwrites!
    '''

 

    if copy_file_header:
        dest_f = open(dest_fname,'w')
        file_header = readlines_fileheader(source_fname)
        dest_f.writelines(file_header)
    else:
        dest_f = open(dest_fname,'a')

    for source_index, scanno in enumerate(source_scanno_list):
        dest_scanno = dest_scanno_list[source_index]
        
        if verbose:
            print('scan {} in file {}'.format(scanno,source_fname))
            print('shifting col {} by {}'.format(data_col_to_shift,shift_magnitude))
            print('saving to scan {} in {}'.format(dest_scanno,dest_fname))
        

        current_scan_list = readlines_scan(source_fname, scanno)
        no_lines = len(current_scan_list)

        for i in range(no_lines):
            curr_line = current_scan_list[i]

            if curr_line.startswith('#') or curr_line.startswith('\n'):
                pass
            else:
                replace_line_list = curr_line.split(' ')
                oldval = float(replace_line_list[data_col_to_shift])
                oldval += shift_magnitude
                replace_line_list[data_col_to_shift] = str(oldval)
                replace_line = ' '.join(replace_line_list)
                current_scan_list[i] = replace_line

        dest_f.writelines(current_scan_list)

    dest_f.close()

        
