import numpy as np
import pygame
from game import SnakeGameAI

# Utility to get the current game screen as a numpy array (RGB)
def get_screen_image(game: SnakeGameAI) -> np.ndarray:
    """
    Returns the current game screen as a numpy array (H, W, 3) in RGB format.
    """
    # Ensure the display is updated
    pygame.display.flip()
    # Get the screen as a numpy array (W, H, 3)
    screen = pygame.surfarray.array3d(game.display)
    # Transpose to (H, W, 3)
    screen = np.transpose(screen, (1, 0, 2))
    return screen 