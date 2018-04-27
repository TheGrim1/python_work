from silx import sx



def plot_and_wait_for_coords(frame):
    '''
    plots frame using sx and waits for input of to numbers
    '''
    sx.imshow(frame)
    while not success:
        try:
            x = float(input('please enter X-coordinate'))
            success = True
        except ValueError as msg:
            print(msg)
    success = False
    while not success:
        try:
            x = float(input('please enter X-coordinate'))
            success = True
        except ValueError as msg:
            print(msg)

    return (x,y)
