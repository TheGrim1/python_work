import subprocess


def main():
    for i, eig_no in enumerate(range(41,159,2)):
        last_r3_run = 7
        
        cmd1 = 'x {} mxs'.format(eig_no)
        cmd2 = 'f {}'.format(1 + i + last_r3_run)
        print('**'*10+ '\n'*3+' calling \n' + cmd1 +'\n'+ cmd2 +'\n'*3+'**'*10)
        subprocess.call(cmd1,shell=True)
        subprocess.call(cmd2,shell=True)

if __name__=='__main__':
    main()
