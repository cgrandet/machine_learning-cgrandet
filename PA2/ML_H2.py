import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as pl
import itertools 
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import accuracy_score
from sklearn import linear_model, decomposition, datasets 
 


CRITERIA = {"DebtRatio":[("less than 50%",0,.5),("less than 100%",5,1),("less than 5 times",1,5),("less than 10 times",5,10),("more than 10 times",10,float("inf"))]}
Y_VARIABLE = 'SeriousDlqin2yrs'

def read_csv(filename):
	'''
	Read a csv and return a datafile  
	'''
	data_file = pd.read_csv(filename)

	return data_file


def obtain_basic_statistics(data_file):
	'''
	Provide descriptive statistics and save them in
	a csv
	'''
	stats = data_file.describe().round(2)
	missing_data = data_file.iloc[0,:].apply(lambda x : len(data_file) - x)
	missing_data.name = "missing values"
	stats = stats.append(missing_data)
	stats.to_csv("descriptive_statistics.csv")


def generate_histogram(data_file):
	'''
	Generate histogram graphs and save them in png files
	'''
	for variable in data_file.describe().keys():
		histograms = data_file[variable].hist(figsize=(15, 10))
		plt.savefig(variable+"_histogram.png", bbox_inches = "tight")
		plt.close()

def generate_correlations(data_file):
	correlations = data_file.corr()
	correlations.to_csv("correlations.csv")

	for x in data_file.describe().keys():
		for y in [i for i in data_file.describe().keys() if i != x]:
			plt.scatter(data_file[x],data_file[y])
			plt.xlabel(x)
			plt.ylabel(y)
			plt.savefig(x+"-"+y+"_histogram.png", bbox_inches = "tight")
			plt.close()


def preprocess_data(data_file, method, list_grouping = []):
	'''
	Fill in missing values 
	'''
	if method == "A":
		data_file = data_file.fillna(data_file.mean())

	if method == "B":
		l2 = itertools.combinations(list_grouping,2)
		list_grouping += l2 
		for variable in list_grouping:
			data_file.fillna(datafile.groupby(variable).transform("mean"), inplace = True)

	data_file.to_csv("clean_database.csv")
	return data_file


def generate_discrete_variable(data, criteria_dict):
	'''
	Write a sample function that can discretize a continuous variable 
	'''
	#Generate categorical values from continous variable 
	for column, criteria in criteria_dict.items():
		#The parameter list contains the labels
		parameter_list = []
		#The range set contains the values for the different parameters
		range_set = set()
		for parameter in range(len(criteria)):
			parameter_list.append(criteria[parameter][0])
			range_set.add(criteria[parameter][1])
			range_set.add(criteria[parameter][2])

		range_list = list(range_set)
		range_list.sort()

		#Generate categorical variables, the "right" option
		#creates set [a,b) to satisfy greater or equal restriction
		#for lower limit. 
		data[column] = pd.cut(data[column],range_list,
						right = False, labels = parameter_list)
		
		#Drop rows that did not have a categorical match
		data = data[~data[column].isnull()]

	return data 

def generate_continous_variable(data, variable_list):
	'''
	function that can take a categorical variable and create 
	binary variables from it
	'''
	for variable in variable_list:
		list_values = list(data.groupby(variable).groups.keys())
		for i,value in enumerate(list_values):
			data[variable].replace(value,i)

	return data 


def build_logistic_classifier(data, y_variable, x_variables):
	'''
	Write a sample function that can discretize a continuous variable 
	and one function that can take a categorical variable and create 
	binary variables from it.
	'''
	model_list = []
	for i  in range(len(x_variables)):
		if i < len(x_variables) - 1:
			run_list = list(itertools.combinations(x_variables,i+2))
			model_list += run_list
	
	general_results = np.array([y_variable])
	general_results = np.append(general_results,data[y_variable])
	general_results = general_results.reshape((-1,1))
	for i, model_variable in enumerate(model_list):
		logit = sm.Logit(data[y_variable], model_variable)
		result = logit.fit()
		result.save(i+"_results.pickle")
		header = np.array([model_variable])
		result_table = np.append(header, result.predict())
		model_result = np.append(header,result_table)
		model_result = model_result.reshape((-1,1))
		general_results = np.append(general_results,model_result, axis = 1)


	np.savetxt("predictions.csv", general_results, delimiter=",")


def evaluate_classifier(predictions_file):
	predictions = read_csv(predictions_file)
	accuracy_dict = {}
	for variable in predictions.columns:
		accuracy_score = accuracy_score(predictions.iloc[:,0],predictions[variable])
	accuracy_dict[variable] = accuracy_score

	return accuracy_dict



