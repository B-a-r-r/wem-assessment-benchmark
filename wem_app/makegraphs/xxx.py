import matplotlib.pyplot as plt
import numpy as np
from WemVisualsBoarding import WemVisualsBoarding
from matplotlib.animation import FuncAnimation

class xxx(WemVisualsBoarding):
    def __init__(self):
        super().__init__(self)
        
    def create_figure(self):
        # Créer une figure simple avec un seul sous-graphe
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], label='Ligne droite')
        ax.set_title('Figure simple')
        ax.legend()
        return fig

    def create_axes(self):
        # Créer un Axes simple
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 4], 'ro-', label='Quadratic')
        ax.set_title('Axes simple')
        ax.legend()
        return ax

    def create_funcanimation(self):
        # Créer une animation simple
        fig, ax = plt.subplots()
        x = np.linspace(0, 2 * np.pi, 100)
        line, = ax.plot(x, np.sin(x))

        def update(frame):
            line.set_ydata(np.sin(x + frame / 10.0))  # Mettre à jour les données y
            return line,

        # Créer l'animation
        ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=50)
        return ani
    

if __name__ == "__main__":
    # Exemple d'utilisation
    visual = xxx()
    fig = visual.create_figure()
    fig.get_axes()[0].set_label("Axe1 Simple")
    ax = visual.create_axes()
    ax.set_label("Axes2 Simple")
    #ani = visual.create_funcanimation()
    
    visual.make_board(board_size=(16, 8), visuals=[fig.get_axes()[0], ax])
    f = visual.return_visuals_board(show=True)
    print(f.get_axes())
    