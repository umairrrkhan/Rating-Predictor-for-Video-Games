# Rating Predictor for Video Games

This project aims to predict the Entertainment Software Rating Board (ESRB) ratings for video games based on various features. The project utilizes machine learning algorithms and a combination of libraries to achieve this prediction task.

## Libraries Used

The following libraries have been used in this project:

- `numpy`: A library for numerical computations in Python.
- `pandas`: A library for data manipulation and analysis.
- `sklearn`: A comprehensive library for machine learning in Python.
- - `train_test_split`: Used to split the dataset into training and testing sets.
  - `StandardScaler`: Used for feature scaling.
  - `train_test_split`: Used to split the dataset into training and testing sets.
  - `StandardScaler`: Used for feature scaling.
  - `LogisticRegression`: A linear classification algorithm.
  - `KNeighborsClassifier`: A k-nearest neighbors classification algorithm.
  - `DecisionTreeClassifier`: A decision tree classification algorithm.
  - `LinearSVC, SVC`: Linear and non-linear support vector classifiers.
  - `MLPClassifier`: A multi-layer perceptron neural network classifier.
  - `RandomForestClassifier`: A random forest classification algorithm.
  - `GradientBoostingClassifier`: A gradient boosting classification algorithm.
  - `XGBClassifier`: XGBoost classifier.
  - `LGBMClassifier`: LightGBM classifier.
  - `CatBoostClassifier`: CatBoost classifier.

## Databases Used

The project utilizes the following databases:

1. `Video_games_esrb_rating.csv`: This dataset contains information about video games, including various features and the corresponding ESRB ratings.
2. `test_esrb.csv`: A separate test dataset for evaluating the model's performance.

## Project Workflow

The project involves the following steps:

1. Data Preprocessing: The raw data is loaded and cleaned using pandas.
2. Feature Selection: Relevant features are chosen to build the model.
3. Data Splitting: The dataset is split into training and testing sets using `train_test_split`.
4. Feature Scaling: StandardScaler is used to scale the features.
5. Model Selection: A variety of classifiers are trained and tested on the training dataset.
6. Model Evaluation: The model's performance is evaluated on the testing dataset using appropriate metrics.

## Conclusion

This project demonstrates the process of predicting ESRB ratings for video games using machine learning techniques. The models are evaluated based on their accuracy on the testing dataset, and the best-performing model can be selected for further deployment or fine-tuning.



