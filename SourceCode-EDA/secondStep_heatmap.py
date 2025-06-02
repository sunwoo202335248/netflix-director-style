import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


#Load the preprocessed Neflix dataset
df = pd.read_csv("netflix_preprocessed_final.csv")


#analyze the correlation between the key variables
corr_matrix = df[["duration_scaled", "rating_encoded", "cast_count"]].corr()
a

# control the overall size of the plot 
plt.figure(figsize=(6, 4))
# visualize heatmap, display the number in the sell, and use blue to red color spectrum, display to second decimal place
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Key Variables")
plt.tight_layout()
plt.show()



