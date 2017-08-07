import matplotlib.pyplot as plt

def plot_result(x,y,ordering,iteration_history, energy_history,kT_history):
    """
        Plots the results of simulated annealing from metropolis_function
    """
    plt.subplot(1, 2, 1)
    plt.plot(x[ordering], y[ordering], 'k')
    #plot([x[ordering[-1]], x[ordering[0]]], [y[ordering[-1]], y[ordering[0]]], 'k') # close the path
    plt.plot(x, y, 'bo') # put a blue dot at each city

    # put a text label next to each of the cities
    HOME_CITY = 0
    for city in ordering:
        if city == HOME_CITY:
            label = '  Home'
        else:
            label = '  %s' % city
        plt.text(x[city], y[city], label)

    plt.axis('image') # scale the axes equally
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('final path after simulated annealing')

    # Plot the energy and temperature history in the right panel
    plt.subplot(1, 2, 2)
    plt.semilogy(iteration_history, energy_history, 'k,', label='system energy', markersize=0.001, alpha = 0.5)
    plt.semilogy(iteration_history, kT_history, 'b', label='thermal energy kT', linewidth=0.7, alpha = 0.5)

    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.title('energy history')
    plt.legend(loc='lower center')

    plt.show()


def plot_mesh(meshx,meshy,savefile = ''):
    plt.figure(1)
    plt.scatter(meshx,meshy)
    if savefile:
        plt.savefig(savefile)
    else:
        plt.show()
    plt.clf()
    plt.close(1)
