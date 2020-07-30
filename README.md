### Gasdrift_Data_Classification_with_openML
Reproducibility of ML models have become a very major concerns in the ML in production systems. This is a very simple implementation of reproducible ML model using Open ML. The model outcomes are returned as a JSON which could be used later to produce similar result.

#### An implementation of classification using the data and task information from openML

1. Implemented a multiclass classification using gas-drift dataset.
2. The dataset has been extracted from openML using the data ID.
3. The task has been also extracted from openML using task ID.
4. Classification has been performed using SVM with polynomial degree 7.
5. Finally a JSON file has been created with all the results and implementation details for the purpose of total regeneration of the model.

##### FINAL JSON OUTPUT (use [JSON LINT](https://jsonlint.com/) for better visualization of the following JSON result)
{"svm-openml": {"dataset": "gas-drift", "settings:": {"parameters": {"number_folds": "10", "percentage": "", "number_repeats": "1", "stratified_sampling": "true"}, "data_splits_url": "https://www.openml.org/api_splits/get/9986/Task_9986_splits.arff", "type": "crossvalidation"}, "classifier:": "sklearn.svm.SVC(gamma=scale, kernel=poly, degree=7)", "result": {"precision": 0.9930329104361378, "recall": 0.9929906542056075, "time": "1.4088271", "confusion_matrix": [[1011, 8, 0, 3, 4, 0], [1, 1168, 0, 0, 2, 0], [0, 2, 647, 0, 7, 0], [0, 0, 0, 773, 0, 1], [0, 2, 2, 2, 1198, 0], [0, 0, 0, 4, 1, 728]]}}}

