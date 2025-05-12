import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Increase text size for the plots
plt.rcParams.update({'font.size': 18})
def plot_box(csv_file):
    """
    Reads a CSV file and plots box plots for different methods grouped by Seed.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Seed'].astype(str), y=df['Result'], hue=df['Method'])
    
    # plt.title('Box Plot of Results by Seed and Method')
    plt.xlabel('Map Seed')
    plt.ylabel('Time (s)')
    plt.legend(title='Method')
    # plt.xticks(rotation=45)
    plt.show()

def plot_box_2(csv_file):
    """
    Reads a CSV file and plots box plots for different methods grouped by Seed with dual y-axes.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Separate data for better visualization
    method1 = df[df['Method'] == df['Method'].unique()[0]]
    method2 = df[df['Method'] == df['Method'].unique()[1]]
    
    # Plot box plots
    sns.boxplot(x=method1['Seed'].astype(str), y=method1['Result'], ax=ax1, color='blue', width=0.4, position=1)
    sns.boxplot(x=method2['Seed'].astype(str), y=method2['Result'], ax=ax2, color='red', width=0.4, position=0)
    
    # Labels and title
    ax1.set_xlabel('Seed')
    ax1.set_ylabel(f'Result ({df["Method"].unique()[0]})', color='blue')
    ax2.set_ylabel(f'Result ({df["Method"].unique()[1]})', color='red')
    ax1.set_title('Box Plot of Results by Seed and Method with Dual Y-Axes')
    
    plt.xticks(rotation=45)
    plt.show()

def plot_box_3(csv_file):
    """
    Reads a CSV file and plots box plots for different methods grouped by Seed.

    Parameters:
    csv_file (str): Path to the CSV file.
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Seed'].astype(str), y=df['Result'], hue=df['Method'])
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    plt.xlabel('Map Seed')
    plt.ylabel('Time (s)')
    plt.legend(title='Method')
    plt.show()

def plot_box_with_parallel_axes(csv_file):
    """
    Reads a CSV file and plots box plots with parallel y-axes for different methods on the same figure.

    Parameters:
    csv_file (str): Path to the CSV file.
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter data for each method
    methods = df['Method'].unique()
    colors = ['blue', 'orange']  # Adjust the color palette if more methods are added
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the first method
    method_1_data = df[df['Method'] == methods[0]]
    sns.boxplot(ax=ax1, x=method_1_data['Seed'].astype(str), y=method_1_data['Result'], color=colors[0])
    ax1.set_ylabel(f'{methods[0]} - Time (s)', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    
    # Create a secondary y-axis for the second method
    ax2 = ax1.twinx()
    method_2_data = df[df['Method'] == methods[1]]
    sns.boxplot(ax=ax2, x=method_2_data['Seed'].astype(str), y=method_2_data['Result'], color=colors[1])
    ax2.set_ylabel(f'{methods[1]} - Time (s)', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    # Shared labels and legend
    ax1.set_xlabel('Map Seed')
    ax1.set_title('Box Plots with Parallel Y-Axes for Each Method')
    
    plt.tight_layout()
    plt.show()

def plot_parallel_axes(csv_file):
    """
    Reads a CSV file and creates parallel plots for different methods, each with its own y-axis.

    Parameters:
    csv_file (str): Path to the CSV file.
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Split the data by Method
    methods = df['Method'].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 6), sharex=True)
    
    # Create a box plot for each method with its own y-axis
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        sns.boxplot(ax=axes[i], x=method_data['Seed'].astype(str), y=method_data['Result'])
        axes[i].set_title(method)
        axes[i].set_xlabel('Map Seed')
        axes[i].set_ylabel('Covergence speed (s)')
        axes[i].set_yscale('log')  # Optional if you want logarithmic scaling
        
    plt.tight_layout()
    plt.show()

# Example usage
plot_parallel_axes('coursework_B_model_based.csv')
# plot_box_with_parallel_axes('coursework_B_model_based.csv')
