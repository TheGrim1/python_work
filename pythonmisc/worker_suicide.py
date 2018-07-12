import os
import psutil
import signal

# parent_id = os.getpid()
def worker_init(parent_id):
    '''
    facilitates ctrl - c for multiprocess.Pool processes
    usage:
    from pythonmisc.worker_suicide import worker_init
    pool= Pool(12, worker_init)
    '''
    def sig_int(signal_num, frame):
        print('signal: %s' % signal_num)
        parent = psutil.Process(parent_id)
        for child in parent.children():
            if child.pid != os.getpid():
                print("killing child: %s" % child.pid)
                child.kill()
        print("killing parent: %s" % parent_id)
        parent.kill()
        print("suicide: %s" % os.getpid())
        psutil.Process(os.getpid()).kill()
    signal.signal(signal.SIGINT, sig_int)
