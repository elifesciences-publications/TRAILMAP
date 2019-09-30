


def add_edge_labels(vol):

    for i in range(vol.shape[0]):
        slice = vol[i]

        if 2 in slice:
            for x in range(1, slice.shape[0] - 1):
                for y in range(1, slice.shape[1] - 1):
                    if slice[x][y] == 1 and is_axon_close(slice, x, y):
                        slice[x][y] = 4
    return vol


def is_axon_close(slice, x, y):
    return slice[x][y+1] == 2 or slice[x+1][y+1] == 2 or slice[x+1][y] == 2 or slice[x+1][y-1] == 2 or slice[x][y-1] == 2 or slice[x-1][y-1] == 2 or slice[x-1][y] == 2 or slice[x-1][y+1] == 2
