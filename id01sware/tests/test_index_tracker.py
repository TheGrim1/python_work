from id01lib.PScanTracker import *
if __name__=="__main__":
    # just a test
    data = np.arange(400).reshape(5,10,8)**2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("test1")
    ax.set_ylabel("test2")
    
    
    tracker = GenericIndexTracker(ax, data, norm="log")
    tracker.set_extent(0,5,-2,2)
    tracker.set_axes_properties(title=map(str, range(5)))
    
    try:
        plt.show()
    except AttributeError: # plot closed
        pass
    
    print(tracker.POI)
