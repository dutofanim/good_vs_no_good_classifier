# utils/__init__.py

# Import visualization utilities
from .visualization import MLVisualization

# Define any utility-specific constants
DEFAULT_PLOT_STYLE = 'default'
DEFAULT_FIGURE_SIZE = (12, 8)

# Configure default plotting settings
import matplotlib.pyplot as plt
plt.style.use(DEFAULT_PLOT_STYLE)
