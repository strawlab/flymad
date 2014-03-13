import matplotlib.pyplot as plt

def plot( graph_data, layout=None ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data_row in graph_data:
        ax.plot( data_row['x'],
                 data_row['y'],
                 label=data_row['name'],
                 )
    ax.set_xlabel( layout['xaxis']['title'] )
    ax.set_ylabel( layout['yaxis']['title'] )
    ax.legend()

    results = {'fig':fig,
               }
    return results
