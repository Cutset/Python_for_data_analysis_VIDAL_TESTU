# Python_for_data_analysis_VIDAL_TESTU
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
