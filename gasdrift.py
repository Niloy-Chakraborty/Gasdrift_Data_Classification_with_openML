# import deendencies
import openml
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn import metrics

'''
Function to extract dataset and task from openML
'''


def extract_data_openml(data_id,task_id):
    data = openml.datasets.get_dataset(data_id)
    print(data.features)
    task = openml.tasks.get_task(task_id)
    print ("task details: ", task)
    print("estimation Procedure: ", task.estimation_procedure)
    print('evaluation_measure:', task.estimation_procedure)
    X, y, categorical_indicator, attribute_names = data.get_data(dataset_format='array',target=data.default_target_attribute)
    print(pd.DataFrame(X).head())
    print(pd.DataFrame(y).head())

    return  task.estimation_procedure,X,y


'''
Function to preprocess the data
'''


def data_preprocess(X,y):

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    print(pd.DataFrame(X).head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_test, y_train, y_test


'''
Function to Model Creation
'''


def model(X_train,y_train):
    # model = GaussianNB()
    model = svm.SVC(gamma="scale", kernel="poly", degree=7)


    '''
    The following commented lines could b used if we need to run out model on the task of openML.
    '''
    # r, f = openml.runs.run_model_on_task(model=gnb, task=task, avoid_duplicate_runs=False, upload_flow=False,
    # return_flow=True)
    # score = []
    # evaluations = r.fold_evaluations['predictive_accuracy'][0]
    # print(evaluations)
    # for key in evaluations:
    #     print(key)
    #     print(evaluations[key])
    #     score.append(evaluations[key])
    # print("Mean Accuracy:",np.mean(score))

    model.fit(X_train,y_train)

    # save the model to disk as Pickle File
    Pkl_Filename = "Model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return Pkl_Filename


'''
Function to test model performance
'''


def test(X_test ,y_test,Pkl_Filename):

    with open(Pkl_Filename, 'rb') as file:
        model1 = pickle.load(file)

    y_pred= model1.predict(X_test)
    result = model1.score(X_test, y_test)
    print("Result on Test Data:",result)

    # Matrices
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    precision= metrics.precision_score(y_test, y_pred,average='weighted')
    recall = metrics.recall_score(y_test, y_pred,average='weighted')
    print(precision, recall)
    print(confusion_matrix)
    confusion_matrix = confusion_matrix.tolist()
    report = {}
    report["precision"]= precision
    report["recall"]= recall
    report["confusion_matrix"] = confusion_matrix
    return report


'''
Function to create JSON
'''


def create_json(report, time,task):
    dict1={}
    dict= {}
    report["time"]= str(time)
    dict["dataset"]='gas-drift'
    dict["classifier:"]="sklearn.svm.SVC(gamma="'scale'", kernel="'poly'", degree=7)"
    dict["settings:"]= task
    dict["result"]= report

    dict1["svm-openml"]= dict
    dict1 = json.dumps(dict1)

    return dict1


'''
Function to call all other methods
'''


def main():
    task, X, y = extract_data_openml(1476,9986)
    start_time = time.clock()
    X_train, X_test, y_train, y_test = data_preprocess(X,y)
    Pkl_Filename = model(X_train,y_train)
    tim = (time.clock() - start_time)
    print("Training Time:",tim ,"seconds")
    report = test(X_test,y_test, Pkl_Filename)
    jsn = create_json(report, tim,task )
    print(jsn)


if __name__ == "__main__":
    main()
