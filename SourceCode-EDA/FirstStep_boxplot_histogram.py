import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load the preprocessed Neflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")

#store three key variables for analysis
selected_vars = ["duration_scaled", "rating_encoded", "cast_count"]

#Generate descriptive staticstics summary table 
#.describe() method calculate count, mean, std, min, 25%, 50%, 75%, max 
#.T transposes the table to show variables as row
summary_stats = df[selected_vars].describe().T 


# create dual visualization (boxplot+ histogram)
# provides distribution shape and outlier
for var in selected_vars:
    plt.figure(figsize=(12, 4))


    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[var])
    plt.title(f'Boxplot of {var}')


    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')

    #automaticlly adjust subplot
    plt.tight_layout()
    plt.show()



