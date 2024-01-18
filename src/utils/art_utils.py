
def rgba(c: str):
    from matplotlib import colors as mcolors
    return mcolors.to_rgba(c)

def rgb(c: str):
    from matplotlib import colors as mcolors
    return mcolors.to_rgb(c)

color_map = {
             'source_motion': rgba('darkred'),
             'source': rgba('darkred'),
             'target_motion': rgba('olivedrab'),
             'input': rgba('olivedrab'),
             'target': rgba('olivedrab'),
             'generation': rgba('purple'),
             'generated': rgba('steelblue'),
             'denoised': rgba('purple'),
             'noised': rgba('darkgrey'),
             }