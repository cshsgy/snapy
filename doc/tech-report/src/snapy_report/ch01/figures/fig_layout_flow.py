"""Storage cartoon at snapy@73c45a7c8c991209924e1137992352c335575062; no data."""
from snapy_report import figstyle as fs


def make_fig():
    fig, ax = fs.new_figure(fs.DOUBLE, 2.2)
    for x, label in enumerate([r'$i-1$', r'$i$', r'$i+1$']):
        fs.cell_box(ax, x, 0, ghost=(x == 0))
        fs.cell_value(ax, x+0.5, 0.5)
        ax.text(x+0.5, 0.76, label, ha='center')
    fs.wall(ax, 1, 0, 1)
    for x, label in [(1,r'$i-\frac{1}{2}$'),(2,r'$i+\frac{1}{2}$')]:
        fs.face_value(ax, x, 0.25)
        ax.text(x, -0.17, label, ha='center')
    ax.text(0.5, 1.12, 'ghost', ha='center')
    ax.text(2, 1.12, 'interior cell averages', ha='center')
    ax.set(xlim=(-0.1,3.1), ylim=(-0.45,1.4))
    ax.axis('off')
    return fig
