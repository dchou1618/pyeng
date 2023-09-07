import numpy as np


primary_palette = {'cephalopod ink': (35/255, 40/255, 60/255),
                   'moonstone': (95/255, 135/255, 160/255),
                   'moonstone_40': (159/255, 183/255, 198/255),
                   'moonstone_60': (191/255, 207/255, 217/255),
                   'gold': (207/255, 160/255, 82/255),
                   'gold_40': (226/255, 198/255, 151/255),
                   'gold_60': (236/255, 217/255, 186/255),
                   'shadow': (190/255, 190/255, 190/255),
                   'white': (255/255, 255/255, 255/255),
                   'black': (0, 0, 0)
                   }

secondary_palette = {'monterey': (3/255, 97/255, 142/255),
                     'shadow': (190/255, 190/255, 190/255),
                     'mist': (205/255, 220/255, 230/255),
                     'moonstone': (95/255, 135/255, 160/255),
                     'gold': (207/255, 160/255, 82/255),
                     'moss': (122/255, 167/255, 120/255),
                     'moonstone_40': (159/255, 183/255, 198/255),
                     'ice': (215/255, 220/255, 220/255),
                     'moonstone_60': (223/255, 231/255, 236/255)
                     }

extended_palette = {'monterey': (3/255, 97/255, 142/255),
                    'moonstone': (95/255, 135/255, 160/255),
                    'pacific': (95/255, 135/255, 160/255),
                    'mist': (205/255, 220/255, 230/255),
                    'turquoise': (0, 180/255, 175/255),
                    'hunter': (0, 100/255, 35/255),
                    'grass': (90/255, 140/255, 35/255),
                    'moss': (122/255, 167/255, 120/255),
                    'sprout': (125/255, 189/255, 66/255),
                    'olive': (186/255, 201/255, 131/255),
                    'ocean': (0, 55/255, 100/255),
                    'chocolate': (76/255, 0, 0),
                    'bark': (115/255, 100/255, 0),
                    'chestnut': (120/255, 65/255, 0),
                    'dark red': (192/255, 0, 0),
                    'scarlet': (255/255, 0, 0),
                    'salmon': (235/255, 150/255, 131/255),
                    'light red': (230/255, 205/255, 225/255),
                    'burnt orange': (211/255, 76/255, 8/255),
                    'tangerine': (250/255, 165/255, 50/255),
                    'gold': (207/255, 160/255, 82/255),
                    'butternuet': (230/255, 180/255, 60/255),
                    'daffodil': (255/255, 200/255, 115/255),
                    'lavendar': (192/255, 150/255, 187/255),
                    'lilac': (133/255, 80/255, 127/255),
                    'midnight': (11/255, 10/255, 11/255),
                    'gray': (130/255, 130/255, 130/255),
                    'shadow': (190/255, 190/255, 190/255),
                    'ice': (215/255, 220/255, 220/255),
                    'warmth': (205/255, 195/255, 180/255)
                   }

primary = [
             primary_palette['gold'],
             primary_palette['moonstone'],
             primary_palette['moonstone_40'],
             primary_palette['gold_40'],
             primary_palette['moonstone_60'],
             primary_palette['gold_60'],
             primary_palette['cephalopod ink'], 
		     primary_palette['shadow']]

secondary = [
			 secondary_palette['monterey'],
             secondary_palette['shadow'],
             secondary_palette['mist'],
             secondary_palette['moonstone'],
             secondary_palette['gold'],
             secondary_palette['moss'],
             secondary_palette['moonstone_40'],
             secondary_palette['ice'],
             secondary_palette['moonstone_60']
             ]

extended = [extended_palette['monterey'],
                    extended_palette['moonstone'],
                    extended_palette['pacific'],
                    extended_palette['mist'],
                    extended_palette['turquoise'],
                    extended_palette['hunter'],
                    extended_palette['grass'],
                    extended_palette['moss'],
                    extended_palette['sprout'],
                    extended_palette['olive'],
                    extended_palette['ocean'],
                    extended_palette['chocolate'],
                    extended_palette['bark'],
                    extended_palette['chestnut'],
                    extended_palette['dark red'],
                    extended_palette['scarlet'],
                    extended_palette['salmon'],
                    extended_palette['light red'],
                    extended_palette['burnt orange'],
                    extended_palette['tangerine'],
                    extended_palette['gold'],
                    extended_palette['butternuet'],
                    extended_palette['daffodil'],
                    extended_palette['lavendar'],
                    extended_palette['lilac'],
                    extended_palette['midnight'],
                    extended_palette['gray'],
                    extended_palette['shadow'],
                    extended_palette['ice'],
                    extended_palette['warmth']
                   ]



def create_colormap(colors, position=None, bit=False, reverse=False, name='custom_colormap'):
    """
    returns a linear custom colormap
    Parameters
    ----------
    colors : array-like
        contain RGB values. The RGB values may either be in 8-bit [0 to 255]
        or arithmetic [0 to 1] (default).
        Arrange your tuples so that the first color is the lowest value for the
        colorbar and the last is the highest.
    position : array like
        contains values from 0 to 1 to dictate the location of each color.
    bit : Boolean
        8-bit [0 to 255] (in which bit must be set to
        True when called) or arithmetic [0 to 1] (default)
    reverse : Boolean
        If you want to flip the scheme
    name : string
        name of the scheme if you plan to save it
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        cmap with equally spaced colors
    """
    from matplotlib.colors import LinearSegmentedColormap
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f')
    if reverse:
        colors = colors[::-1]
    if position is not None and not isinstance(position, np.ndarray):
        position = np.array(position)
    elif position is None:
        position = np.linspace(0, 1, colors.shape[0])
    else:
        if position.size != colors.shape[0]:
            raise ValueError("position length must be the same as colors")
        elif not np.isclose(position[0], 0) and not np.isclose(position[-1], 1):
            raise ValueError("position must start with 0 and end with 1")
    if bit:
        colors[:] = [tuple(map(lambda x: x / 255., color)) for color in colors]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return LinearSegmentedColormap(name, cdict, 256)
