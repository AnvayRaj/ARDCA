import matplotlib.pyplot as plt

# Using built-in colormap
cmap = plt.cm.viridis  # Choose any colormap you like (e.g., inferno, plasma)

# Create some data (replace this with your actual data)
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 2, 1]

# Create a scatter plot with the chosen colormap
plt.scatter(x, y, c=y, cmap=cmap)  # Use data values for color intensity

# Add a colorbar to show the colormap legend
plt.colorbar(label='Color Intensity')

# Label axis and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example using built-in colormap (viridis)')
plt.show()


# Creating a custom colormap (red to blue)
colors = [(1, 0, 0), (0.5, 0.5, 0.5), (0, 0, 1)]  # RGB color values
custom_cmap = plt.cm.LinearSegmentedColormap.from_list("", colors)

# Create another scatter plot with the custom colormap
plt.scatter(x, y, c=y, cmap=custom_cmap)
plt.colorbar(label='Color Intensity')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Example using custom colormap (red to blue)')
plt.show()
