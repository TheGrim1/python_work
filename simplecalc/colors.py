def random_color(bla)
'''
TODO and stolen from 
http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
'''
# use golden ratio
golden_ratio_conjugate = 0.618033988749895
h = rand # use random start value
gen_html {
  h += golden_ratio_conjugate
  h %= 1
  hsv_to_rgb(h, 0.5, 0.95)
}


def color_list(number):
    colors = ['r','b','g','black','yellow','magenta','orange','purple','darkblue','darkred','darkyellow','darkgreen']
    return colors[number % len(colors)]
