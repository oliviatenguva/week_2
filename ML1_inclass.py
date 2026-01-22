"""
ML1 In-Class
.py file
"""
# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3021/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.py#scrollTo=9723a7ee)

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3001/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.ipynb#scrollTo=9723a7ee)
# %%

# %%
# import packages
from turtle import color
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# set the dimension on the images
import plotly.io as pio
pio.templates.default = "plotly_dark" # set dark theme

# %%
iris = data('iris')
iris.head()

# %%
# What mental models can we see from these data sets? 
# Iris species can be differentiated by their sepal and petal lengths and widths. 
# What data science questions can we ask? 
# Can we predict the species of an iris flower based on its sepal and petal dimensions?

# %%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this,
# we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent
# dataframes.

X = iris.drop(['Species'], axis=1).values  # features
y = iris['Species'].values  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for now

# %%
# Now, we use the scikitlearn k-NN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

# %%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

# %%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a
# classification model point these out?
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

# %%
patterns['Sepal.Width'].describe()

# %%
patterns['Petal.Length'].describe()

# %%
patterns['Petal.Width'].describe()

# %%
# scatter plot using plotly
fig = px.scatter_3d(iris, x='Sepal.Length', y='Sepal.Width', z='Petal.Length',
                 color='Species', title='Iris Sepal Dimensions')
fig.show()
# %%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example!
# We go over kNN models at length later in the course; you do not need to
# understand exactly what the model is doing quite yet.
# For now, ask yourself:

# 1. What is the purpose of data splitting? The purpose of data splitting is to evaluate the performance of a machine learning model on unseen data. By dividing the dataset into training and testing sets, we can train the model on one portion of the data and then test its accuracy and generalization on another portion that it has not seen before. This helps to prevent overfitting and provides a more realistic assessment of how the model will perform in real-world scenarios.
# 2. What can we learn from data testing/validation? We can learn several important things from data testing and validation:
#    - Model Performance: Testing allows us to evaluate how well our model performs on unseen data, providing insights into its accuracy, precision, recall, and other relevant metrics.
#    - Generalization: Validation helps us understand how well our model generalizes to new, unseen data, which is crucial for real-world applications.
#    - Hyperparameter Tuning: Validation sets can be used to fine-tune model hyperparameters, helping to optimize performance without overfitting.
#    - Error Analysis: By analyzing the errors made during testing, we can identify patterns and areas where the model may need improvement.
# 3. How do we know if a model is working? We know the model is working by evaluating its performance on a separate test dataset that was not used during training. Key indicators of a working model include:
#    - High accuracy or other relevant performance metrics (e.g., precision, recall, F1-score) on the test data.
#    - Consistent performance across different subsets of the data.
#    - The model's ability to generalize well to new, unseen data without overf
# 4. How could we find the model error? We can find the model error by comparing the predicted outputs of the model to the actual outputs in the test dataset.

# If you want, try changing the size of the test data
# or the number of n_neighbors and see what changes!
