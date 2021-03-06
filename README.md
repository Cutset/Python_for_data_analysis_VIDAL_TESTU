# Python_for_data_analysis_VIDAL_TESTU

WARNING 1: the notebook cannot be displayed, use this link to visualize it properly : 
https://nbviewer.org/github/Cutset/Python_for_data_analysis_VIDAL_TESTU/blob/main/PythonForDataScience_project_VIDAL_Sara_TESTU_Constantin.ipynb
(Note that the plotly express graphs cannot be displayed)

WARNING 2: The API file was too heavy for GitHub, so the files are dispatched in the repository.

Authors : VIDAL Sara & TESTU Constantin (A4 DIA7)

Group project at the end of the semester for the following course : Python for data analysis.

We picked the News Popularity Dataset. The goal was to pedict the popularity (measured by its number of shares) of articles from a numeric media called Mashable. 

We first looked at the dataset. The data was already clean : there were no missing values, and all categorical features were encoded (e.g. The week days).
The next steps were :
- Data Visualization
- Splitting the data
- Scaling the data (Robust scaler)
- Training the regression models 
- Optimizing the hyperparameters (GridSearch)

After getting terrible scores with the regressions (Root Squared Mean Error = 8000 while the range of most of the articles was between 500 and 15000 shares), by reading the scientific paper of those who studied the dataset, we realized that they were actually doing a classification : each article was either popular or not popular based on a threshold number of shares. 

We thus went through the same steps but with classification models. As the target feature had about 80% '0' and 20% '0', using the accuracy as a score would have been pointless, so we used ROC, and AUC to evaluate our models. 

Adaboost ended up with the best score with a bit more than 70%.
