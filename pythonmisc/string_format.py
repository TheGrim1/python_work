
# Create a function to easily repeat on many lists:
def ListToFormattedString(alist, space):
    ## from https://stackoverflow.com/questions/7568627/using-python-string-formatting-with-lists

    # Each item is right-adjusted, width=space
    format_str = '{:>'+str(space)+'}'
    # print(format_str)
    formatted_list = [format_str for item in alist] 
    s = ','.join(formatted_list)
    # print s
    return s.format(*alist)
