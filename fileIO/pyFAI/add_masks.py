import fabio
import sys
import matplotlib.pyplot as plt

   

def plotmask(mask1):
# plot the mask
    # mask1 = fabio.open(filelist[0]).data
    fig1 = plt.figure(figsize=(7,7), dpi=80)
    plt.imshow(mask1, vmin=0, vmax=1, cmap='gray')
    #plt.axis([0, 517, 0, 517])  
    plt.colorbar()
    plt.title("module mask")
    ax = plt.gca()
    ax.set_aspect('equal') 
    plt.show()  


def main(filelist):
     mask1 = fabio.open(filelist[1])
     for f in filelist[2:]:
         mask2 = fabio.open(f).data
         mask1.data += mask2
         mask1.data[mask1.data!=0]=1         
     mask1.write(filelist[1])
     plotmask(mask1.data)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'usage: python integrate.py <your new maskname.edf> <at least 2 mask files .edf>'
        sys.exit(0)
    else:
        main(sys.argv[1:])

    
