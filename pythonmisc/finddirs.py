
def listdirs(folder):
    return [
        d for d in [os.path.join(folder, d1) for d1 in os.listdir(folder)] 
        if os.path.isdir(d)
    ]
