import matplotlib.pyplot as plt

# # Data points
# epsilon_values = [0.00001, 0.1, 0.15, 0.2, 0.3, 0.5, 1]
# speed_values = [0.66, 0.67, 0.71, 0.74, 0.75, 0.93, 10.46]
# # Increase text size for the plots
# plt.rcParams.update({'font.size': 18})
# # Plot the line graph
# plt.figure(figsize=(8, 6))
# plt.plot(epsilon_values, speed_values, marker='o', linestyle='-', color='b', label="Epsilon")

# # Adding labels and title
# plt.xlabel("Epsilon", fontsize=18)
# plt.ylabel("Convergence time (s)", fontsize=18)
# # plt.title("Epsilon vs Speed", fontsize=14)
# plt.grid(True)
# plt.legend()

# # Show the graph
# plt.xscale('log')  # Use a logarithmic scale for epsilon values
# plt.show()

# Data points
epsilon_values = [0.1, 0.15, 0.2, 0.3, 0.5, 1]
speed_values = [0.67, 0.62, 0.57, 0.56, 0.51, 0.46]
# Increase text size for the plots
plt.rcParams.update({'font.size': 18})
# Plot the line graph
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, speed_values, marker='o', linestyle='-', color='b', label="Learning rate")

# Adding labels and title
plt.xlabel("Learning rate", fontsize=18)
plt.ylabel("Convergence time (s)", fontsize=18)
# plt.title("Learning rate vs Convergence time (s)", fontsize=14)
plt.grid(True)
plt.legend()

# Show the graph
plt.xscale('log')  # Use a logarithmic scale for epsilon values
plt.show()