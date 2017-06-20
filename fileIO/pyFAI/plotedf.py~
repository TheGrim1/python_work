import fabio
import sys
import matplotlib.pyplot as plt

def main(filename):
# plot the mask
    mask1 = fabio.open(filename).data
    fig1 = plt.figure(figsize=(7,7), dpi=80)
    plt.imshow(mask1, vmin=0, vmax=1, cmap='gray')
    #plt.axis([0, 517, 0, 517])  
    plt.colorbar()
    plt.title("module mask")
    ax = plt.gca()
    ax.set_aspect('equal') 
    plt.show()  

def plot(data, vmin=0, vmax=10)
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
