import fabio
import sys
import matplotlib.pyplot as plt
import scipy.misc


def main(filename):
# plot the mask
    mask1 = fabio.open(filename).data[range(40)]
    fig1 = plt.figure(figsize=(7,7), dpi=80)
    plt.imshow(mask1, cmap='jet')
    #plt.axis([0, 517, 0, 517])  
    plt.colorbar()
    plt.title("first image")
    ax = plt.gca()
    ax.set_aspect('equal') 
    plt.show()  

    for i in range(80):
        zrange= range(i*51+10, (i+1)*51)
        mask1 = fabio.open(filename).data[zrange]
        scipy.misc.imsave(''.join([filename,str(i),'.jpg']),mask1)



def plot(data, vmin=1, vmax=100):
    fig1 = plt.figure(figsize=(10,10), dpi=80)
    plt.imshow(data, vmin, vmax, cmap='gray')
    #plt.axis([0, 517, 0, 517])  
    plt.colorbar()
    plt.title("data")
    ax = plt.gca()
    ax.set_aspect('equal') 
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) !=2 :
        print 'usage: python plotedf.py <your .edf>'
        sys.exit(0)
    else:
        main(sys.argv[1])
