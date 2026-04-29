'''copied from https://github.com/limix/limix-plot/blob/master/limix_plot/_show.py'''

def show():
    import matplotlib as mpl

    if mpl.get_backend().lower() == "agg":
        return None

    from ._plt import get_pyplot

    return get_pyplot().show()
