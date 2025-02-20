import numpy as np
import matplotlib.pyplot as plt

# Simulated positions (posX, posY) over time
positions = [
    (10, 10),
    (11, 10),
    (11, 11),
    (12, 11),
    (12, 12),
    (13, 12),
    (13, 13),
    (14, 13),
    (14, 14),
    (15, 14),
    (15, 15),
    (16, 15),
    (16, 16)
]

# Extract posX and posY
posX = [pos[0] for pos in positions]
posY = [pos[1] for pos in positions]
timesteps = list(range(len(positions)))

# Plotting the positions over time
plt.figure(figsize=(10, 4))
plt.plot(timesteps, posX, marker='o', label='posX')
plt.plot(timesteps, posY, marker='s', label='posY')
plt.title('Agent Position Over Time')
plt.xlabel('Timestep')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()