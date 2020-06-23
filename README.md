### Gasdrift_Data_Classification_with_openML

#### An implementation of classification using the data and task information from openML

1. Implemented a multiclass classification using gas-drift dataset.
2. The dataset has been extracted from openML using the data ID.
3. The task has been also extracted from openML using task ID.
4. Classification has been performed using SVM with polynomial degree 7.
5. Finally a JSON file has been created with all the results and implementation details for the purpose of total regeneration of the model.

##### FINAL JSON OUTPUT (use [JSON LINT](https://jsonlint.com/) for better visualization of the following JSON result)
{
	"svm-openml": {
		"result": {
			"confusion_matrix": [
				[1011, 8, 0, 3, 4, 0],
				[1, 1168, 0, 0, 2, 0],
				[0, 2, 647, 0, 7, 0],
				[0, 0, 0, 773, 0, 1],
				[0, 2, 2, 2, 1198, 0],
				[0, 0, 0, 4, 1, 728]
			],
			"recall": [0.9853801169590644, 0.9974380871050385, 0.9862804878048781, 0.9987080103359173, 0.9950166112956811, 0.9931787175989086],
			"precision": [0.9990118577075099, 0.9898305084745763, 0.9969183359013868, 0.9884910485933504, 0.9884488448844885, 0.9986282578875172],
			"time": "1.4878681"
		},
		"dataset": "gas-drift",
		"classifier:": "sklearn.svm.SVC(gamma=scale, kernel=poly, degree=7)",
		"settings:": {
			"type": "crossvalidation",
			"parameters": {
				"percentage": "",
				"number_folds": "10",
				"stratified_sampling": "true",
				"number_repeats": "1"
			},
			"data_splits_url": "https://www.openml.org/api_splits/get/9986/Task_9986_splits.arff"
		}
	}
}


