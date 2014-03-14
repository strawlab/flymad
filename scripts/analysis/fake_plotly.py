import matplotlib.pyplot as plt

def plot( graph_data, layout=None ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data_row in graph_data:
        ax.plot( data_row['x'],
                 data_row['y'],
                 label=data_row['name'],
                 )
    if 'xaxis' in layout and 'title' in layout['xaxis']:
        ax.set_xlabel( layout['xaxis']['title'] )
    if 'yaxis' in layout and 'title' in layout['yaxis']:
        ax.set_ylabel( layout['yaxis']['title'] )
    if 'title' in layout:
        ax.set_title(layout['title'])
    ax.legend()

    results = {'fig':fig,
               }
    return results
