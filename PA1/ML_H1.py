import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import requests
import json

FILENAME = "mock_student_data.csv"
COLUMNS = 1
ROW = 0

data_file = pd.read_csv(FILENAME)

criteria = {"DebtRatio":[("less than 50%",0,.5),("less than 100%",5,1),("less than 5 times",1,5),("less than 10 times",5,10),("more than 10 times",10,float("inf"))]}

def obtain_statistics(data_file):
	'''
	Obtain basic statistics for data 
	'''
	median = data_file.median(ROW)
	mean = data_file.mean(ROW)
	mode = data_file.mode(ROW)
	sd = data_file.std(ROW)
	missing_values_list = []
	for variable in data_file.columns:
		missing_values = len(data_file[variable][data_file[variable].isnull()])
		missing_values_list.append((variable,missing_values))

	final = pd.concat([median,mean,sd],axis=1)

	return final, mode, missing_values_list


def plot_histograms(data_file):
	'''
	Obtain plots  
	'''
	histograms= data_file.hist(figsize=(15, 10))
	for i,graph in enumerate(histograms):
		plt.savefig("graph"+str(i)+".png", bbox_inches = "tight")
		plt.close()

	data_file.groupby("State")["ID"].count().sort_values().plot(kind = "bar")
	plt.savefig("state_histogram.png", bbox_inches = "tight")
	plt.close()

	data_file.groupby("Gender")["ID"].count().sort_values().plot(kind = "bar")
	plt.savefig("gender_histogram.png", bbox_inches = "tight")
	plt.close()

	data_file.groupby("Graduated")["ID"].count().sort_values().plot(kind = "bar")
	plt.savefig("graduated_histogram.png", bbox_inches = "tight")
	plt.close()
	

def predict_gender(data_file):
	'''
	Predict gender based on genderize API and replace data file
	'''
	missing_names = data_file[data_file["Gender"].isnull()]
	gender_list = []
	for row in missing_names.iterrows():
		name = row[1]["First_name"]
		req = requests.get("http://api.genderize.io?name=" + name)
		result = json.loads(req.text)["gender"]
		gender_list.append(result)
	
	missing_names["Gender"] = gender_list
	final_data = pd.concat([missing_names,data_file[~data_file["Gender"].isnull()]],axis=0)
	final_data.to_csv("names_predictions.csv",header = True)


def fill_missing_values(data_file, method):
	'''
	Replace missing values with different models 
	'''
	if method == "A":
		data_file = data_file.fillna(data_file.mean())

	if method == "B":
		group_variables = ["Graduated"]
		grouped = data_file.groupby(group_variables)
		f = lambda x: x.fillna(x.mean())
		transform = grouped.transform(f)
		variables = [0,1,2,3,4,8]
		short_data = data_file.iloc[:,variables]
		data_file = pd.concat([short_data, transform], axis=1)

	if method == "C":
		group_variables = ["Gender","Graduated"]
		grouped = data_file.groupby(group_variables)
		f = lambda x: x.fillna(x.mean())
		transform = grouped.transform(f)
		variables = [0,1,2,3,4,8]
		short_data = data_file.iloc[:,variables]
		data_file = pd.concat([short_data, transform], axis=1)

	return data_file


new_data_file = pd.read_csv("names_predictions.csv")

def generate_files(new_data_file):
	treated_data = fill_missing_values(new_data_file,"A")
	treated_data.to_csv("Model_A.csv",header = True)
	treated_data = fill_missing_values(new_data_file,"B")
	treated_data.to_csv("Model_B.csv",header = True)
	treated_data = fill_missing_values(new_data_file,"C")
	treated_data.to_csv("Model_C.csv",header = True)





