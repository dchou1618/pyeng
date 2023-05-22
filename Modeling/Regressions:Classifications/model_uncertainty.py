# Data Wrangling and Manipulation
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import datasets
import statsmodels.api as sm
# For basic linear modelling 
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

# Reducing the estimation of Pproj from P0
# as a convex optimization problem.
import cvxpy as cp
from scipy.special import entr
import os

from scipy.stats import binom
from scipy.stats import norm
from functools import reduce

class UncertaintyEstimator:
	def __init__(self, model, train_data, test_data):
		self.model = model
		self.train_data = train_data
		self.test_data = test_data
	def est_distr_divergence(self, p1: np.array, p2: np.array):
		hist1, bin_edges1 = np.histogram(p1, bins='auto', density=True)
		pmf1 = hist1 / np.sum(hist1)

		hist2, bin_edges2 = np.histogram(p2, bins='auto', density=True)
		pmf2 = hist2 / np.sum(hist2)
		kl_div = np.sum(entr(pmf1) - np.dot(pmf2, np.log(pmf1)))
		return kl_div


class DistributionInstability(UncertaintyEstimator):
	"""
	At its core, an unstable parameter wrt shifts in marginal distributions of
	certain covariates requires knowledge about covariate parameter estimates
	in the out-of-sample data. How can we estimate the performance of a 
	mdoel in the out-of-sample data given limited information? What if we only
	know the average age in the out-of-sample data? We can run over infinitely
	many reweighting methods to achieve the same average age. 

	Procedure comes from page 14 in 
	https://arxiv.org/pdf/2105.03067.pdf#page=13&zoom=100,113,765.
	
	(1) Identify the subset of features that contribute most to the parameter
	of interest (or average mean square error) - feature importance
	(2) Estimate the projected probability distribution (reweigh) the 
	observations Z_i.
	(3) Estimate beta(Pproj) - sample from Pproj for some "x" number of 
	repetitions to get "x" beta estimates and sample from Pproj for the test
	dataset to get "x" corresponding beta test estimates, then take the absolute
	differences of the ones in the projected dataset and those of 
	the working dataset.
	
	Note that Z is a random variable sampled from the distribution of interest
	(could be the variable associated with the parameter we're estimating 
	the instability of).
	(1/n)*sum_{i=1}^{n} \delta_{i}, where delta_i is the dirac measure on Z_i
	, denotes that each observation for variable Z has an equal likelihood of 
	appearing.
	"""
	def __init__(self, model, train_data, test_data):
		super().__init__(model, train_data, test_data)
		self.X = np.array(self.train_data[self.train_data.columns[:-1]])
	def _compute_lambdas(self, gamma_vector: np.array):
		n = len(self.train_data.columns)-1
		lambda_vals = cp.Variable(n)
		
		num_obs = len(self.train_data)
		obj = cp.Minimize((1/num_obs)*cp.sum([cp.exp(lambda_vals.T @ (self.X[i, :]-gamma_vector)) for i in range(num_obs)] ) )
		constraints = []
		prob = cp.Problem(obj, constraints)
		res = prob.solve()	
		return lambda_vals.value
	def _resample_from_projected(self, proj_weights):
		"""
		_resample_from_project
		:param proj_weights List[float]: the probability per observation from a shifted distribution.
		:return:
		"""
		return np.random.choice(list(range(len(proj_weights))), len(proj_weights), proj_weights, replace=True)
	def plot_projected_uncertainty(self, first_moments):
		"""
		plot_projected_uncertainty - this is to allow for inference on unseen populations given that 
		we have found that parameters are unstable to distribution shifts. 
		We observe difference in model parameters from one probability measure (P0) to another
		(Pproj).
		:param first_moments List[float]:
		:return:
		"""
		lambda_vals = self._compute_lambdas(first_moments)
		num_obs = len(self.train_data)

		obs_weights = [np.exp(lambda_vals.T @ (self.X[i, :]-first_moments)) for i in range(num_obs)]
		projected_distribution_weights = obs_weights/np.sum(obs_weights)
					
def summary_table_to_df(table):
	df = pd.DataFrame(table)
	df.columns = [str(elem) for elem in df.iloc[0]]
	df = df.iloc[1:]

	df.set_index("",inplace=True)
	return df

class MCMC(UncertaintyEstimator):
	"""
	Markov Chain Monte Carlo (MCMC) can be used to sample from a target posterior
	distribution that is difficult to sample directly.
	"""
	def __init__(self, model, train_data, test_data, target_var):
		super().__init__(model, train_data, test_data)
		logit_df = summary_table_to_df(self.model.summary().tables[-1])

		input_df = self.train_data[self.train_data.columns]
		input_df.insert(loc=0, column='const', value=1)
		self.input_df = input_df
		
		test_df = self.test_data[self.test_data.columns]
		test_df.insert(loc=0, column="const", value=1)
		self.test_df = test_df

		self.coefs = [float(str(elem)) for elem in logit_df.loc[:,"coef"].tolist()]
		self.vars = [float(str(elem)) for elem in logit_df.loc[:, "std err"].tolist() ]
		self.b = np.exp(self.coefs[0]+np.euler_gamma)
		self.target_var = target_var

	def run_mh_algorithm(self, n: int, b: int):
		"""
		run_mh_algorithm runs metropolis hastings after 
		specifying a prior distribution of the parameters and 
		runs monte carlo simulation to sample data points from the 
		posterior.

		For the multivariate case, we will 
		have the proposal distribution be 
		g(alpha, beta_1,...,beta_n) =
			pi(alpha,b_hat=e^{alpha_hat+euler's constant})phi(beta_1)...phi(beta_n)
		Each of the phi distributions would be gaussian distributions centered at
		the estimated coefficient beta_i and standard error std_i.
		
		Initial x0 are the coefficients in self.coefs.
		"""
		
		x = np.array([[0 for col_idx in range(len(self.coefs))] for row_idx in range(n+b)],
				     dtype=float)
		x = np.reshape(x, (n+b, len(self.coefs)))
		posterior_x = [0 for i in range(n+b)]

		x[0] = self.coefs
		posterior_x[0] = self._get_posterior(self.coefs)
		accepted_count = 0
		i = 1
		while i < n+b:
			print("Value of i: ", i)
			x_prime = self._get_random_proposal()
			
			posterior_prime = self._get_posterior(x_prime)
			
			if posterior_prime == 0:
				continue
			if self._get_proposal(x[(i-1)]) == 0:
				continue
			if posterior_x[i-1] == 0:
				continue
			if self._get_proposal(x_prime) == 0:
				continue
			r = np.log(posterior_prime)+np.log(self._get_proposal(x[(i-1)]))-\
				np.log(posterior_x[i-1])-np.log(self._get_proposal(x_prime))
			acceptance_prob = min(np.exp(r),1)
			# print(r, acceptance_prob)
			if np.random.uniform(0, 1) <= acceptance_prob:
				accepted_count += 1
				x[i,:] = x_prime
				posterior_x[i] = posterior_prime
			else:
				x[i,:] = x[(i-1),:]
				posterior_x[i] = posterior_x[i-1]
			i += 1

		return x, posterior_x, accepted_count/(n+b)
	def _return_pi_alpha(self, theta):
		return reduce(lambda a,b:a*b, [(1/self.b)*np.exp(theta[0])]+[np.exp(-1*np.exp(theta[0])/self.b)])
	def _get_posterior(self, theta):
		"""
		_get_posterior - obtain the product of the likelihood
		and prior to get posterior p(theta|x)p(theta).
		p(theta|x) = p(x|theta)p(theta)/p(x)
		"""
		
		p_lst = self.input_df.loc[:, ~self.input_df.columns.isin({self.target_var})]\
				.apply(lambda row: np.exp(np.dot(theta, row.tolist()))/(1+np.exp(np.dot(theta, row.tolist() ))),
							    axis=1).tolist()

		y = self.train_data[self.target_var].tolist()

		# e^(sum of log probabilities)
		likelihood = np.exp(sum([np.log(binom.pmf(y[i],n=1,p=prop)) for i, prop in enumerate(p_lst)]))

		dprior = self._return_pi_alpha(theta)

		return likelihood*dprior
	def _get_proposal(self, theta):
		proposal_distribution = reduce(lambda a,b: a*b, [self._return_pi_alpha(theta)]+\
								[norm.pdf(theta[i], self.coefs[i], self.vars[i]) for i in range(1,len(theta))])
		
		return proposal_distribution
	def _get_random_proposal(self):
		intercept = [np.log(np.random.exponential(scale=self.b))]
		coefs = [np.random.normal(c,v) for c, v in zip(self.coefs[1:], self.vars[1:])]
		#print(intercept, coefs, self.coefs)
		return intercept+coefs

class ConformalPrediction(UncertaintyEstimator):
	"""
	Conformal Prediction is a way of estimating the uncertainty in model
	predictions that is distribution-free and has guarantees on coverage.

	Implementation based on http://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf
	
	In the case of predicting diabetes scores, we will be using conformal quantile 
	regression, followed by inflation/deflation of the bounds depending on relative locations of predicted scores in 
	the calibration dataset.
	"""
	def __init__(self, model, train_data, test_data, lower_model, upper_model):
		super().__init__(model, train_data, test_data)
		self.lower_model = lower_model
		self.upper_model = upper_model
	def estimate_bounds(self):
		pass


def load_sample_data():
	"""
	load_sample_data - loads in a dataset from scikit-learn's
	datasets - in this case, the diabetes dataset.

	Based on the correlation matrix, we have evidence for computing the s-values of the 
	parameters wrt these corresponding features (correlation above 0.3):
	age: bp, s6
	sex: s3, s4
	bmi: bp, s3, s4, s5	
	bp: age, bmi, s5, s6 
	
	s1: s2, s4, s5, s6
	s2: s1, s4, s5 
	s3: sex, bmi, s4, s5
	s4: sex, bmi, s1, s2, s3, s5, s6
	s5: bmi, bp, s1, s2, s3, s4, s6
	s6: age, bmi, bp, s1, s4, s5 
	:return: pd.DataFrame
	"""
	diabetes_data = datasets.load_diabetes()
	df = pd.DataFrame(diabetes_data["data"], columns=diabetes_data["feature_names"])

	#print(f"Correlation Matrix of sample data:\n{df.corr()}")
	df["target"] = diabetes_data["target"]
	return df


def randomize_train_test(data, prop_train=0.8):
	data_size = len(data)
	train_size = int(data_size*prop_train)
	selected_train_indices = np.random.choice(range(data_size), train_size, replace=False)
	train_data = data.iloc[selected_train_indices]
	test_data = data.loc[~data.index.isin(set(selected_train_indices))]
	return train_data, test_data

def train_lm(input_matrix, output_matrix, loss_obj):
	"""
	train_lm - takes in an input matrix, normalizes it, and 
	passes through a sequential tf model (regression).
	"""
	normalizer = layers.Normalization(input_shape=[None,len(input_matrix[0])], axis=None)
	normalizer.adapt(input_matrix)
	model = tf.keras.Sequential([
    		normalizer,
    		layers.Dense(units=1)
		])
	model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
				  loss=loss_obj)
	model.fit(input_matrix, output_matrix)

	return model

def train_logit_model(input_matrix, output_matrix):
	normalizer = layers.Normalization(input_shape=[None,len(input_matrix[0])], axis=None)
	normalizer.adapt(input_matrix)
	model = tf.keras.Sequential([normalizer,
							    layers.Dense(units=1, activation=activations.sigmoid)])
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
				  loss="binary_crossentropy")
	model.fit(input_matrix, output_matrix)
	return model

def fit_stats_logistic_model(input_matrix, output_matrix):
	logit_model = sm.GLM(output_matrix, sm.add_constant(input_matrix), family=sm.families.Binomial())
	logit_model = logit_model.fit(attach_wls=True, atol=1e-10)
	new_cols = input_matrix.columns.tolist()

	df = summary_table_to_df(logit_model.summary().tables[-1] )
	while len([x for x in df[df["P>|z|"].apply(lambda x:\
			   float(str(x)) >= 0.10)].index.tolist() if str(x) != "const" ]) > 0:
		sig_table = df[df["P>|z|"].apply(lambda x: float(str(x)) < 0.10)]
		
		new_cols = [str(x) for x in sig_table.index.tolist() if str(x) != "const"]
		logit_model = sm.GLM(output_matrix, sm.add_constant(input_matrix[new_cols]), family=sm.families.Binomial())
		logit_model = logit_model.fit(attach_wls=True, atol=1e-10)
		df = summary_table_to_df(logit_model.summary().tables[-1] )

	return new_cols, logit_model
	

def get_root_path(root_name):
    dir_lst = os.getcwd().split("/")
    idx = dir_lst.index(root_name)
    return "/".join(dir_lst[:(idx+1)])



def extract_pseudo_r2(table):
	stats_df = pd.DataFrame(table)
	stats_df[2] = stats_df[2].apply(str)
	pseudo_r2 = stats_df.loc[stats_df[2]=="  Pseudo R-squ. (CS):", 3].iloc[0]
	return float(str(pseudo_r2))

def load_data_file(train_fpaths, test_fpaths, variable_names, target_var,
				   threshold_na, mappings):
	"""
	load_data_file expects files in the .data format
	that most often come from the UCI data repository.
	"""
	train_dfs, test_dfs = [], []
	for fpath in train_fpaths:
		with open(fpath) as f:
			rows = []
			for line in f.readlines():
				rows.append([np.nan if elem == "?" else float(elem) for elem in line[:-1].split(",")])
			
			df = pd.DataFrame(data=rows,columns=variable_names)
			df[target_var] = df[target_var].map({1:1,2:1,3:1,4:1,0:0})
		train_dfs.append(df)
	for fpath in test_fpaths:
		with open(fpath) as f:
			rows = []
			for line in f.readlines():
				rows.append([np.nan if elem == "?" else float(elem) for elem in line[:-1].split(",")])
			df = pd.DataFrame(data=rows,columns=variable_names)
			df[target_var] = df[target_var].map({1:1,2:1,3:1,4:1,0:0})
		test_dfs.append(df)
	train_df = pd.concat(train_dfs)
	test_df = pd.concat(test_dfs)
	# Complete case removal	
	columns_to_drop = set()
	for df in [train_df, test_df]:
		na_df = df.isna().sum(axis=0).reset_index(name="NA count")

		na_df = na_df[na_df["NA count"] >= threshold_na*len(df)]
		columns_to_drop.update(na_df["index"].tolist())
	columns_to_drop = list(columns_to_drop)
	train_df = train_df.drop(columns_to_drop, axis=1)
	test_df = test_df.drop(columns_to_drop, axis=1)
	# Selecting the best feature set
	curr_feature_set = []
	
	best_r2 = 0
	for col in mappings:
		mapping = mappings[col]
		
		if col in train_df.columns and col in test_df.columns:
			train_df[col] = train_df[col].map(mapping)
			test_df[col] = test_df[col].map(mapping)
	train_df = pd.get_dummies(train_df, columns=list(mappings.keys()))
	test_df = pd.get_dummies(test_df, columns=list(mappings.keys()))

	features_to_consider = {col for col in train_df.columns if col != target_var}
	all_features = list(features_to_consider)
	while len(curr_feature_set) < len(train_df.columns)-1:
		best_feature = None
		for feature in features_to_consider:
			curr_input = train_df[curr_feature_set+[feature]]
			curr_output = train_df[target_var]
			res = sm.GLM(curr_output, sm.add_constant(curr_input), family=sm.families.Binomial())
			res = res.fit(attach_wls=True, atol=1e-10)
				
			pseudo_r2 = extract_pseudo_r2(res.summary().tables[0])
			if len(curr_feature_set) == 0:
				if pseudo_r2 > best_r2:
					best_feature = feature
					best_r2 = pseudo_r2
			else:
				multicollinear = False
				for i in range(len(curr_input.columns)):
					rhs = curr_input[list(curr_input.columns[:i])+list(curr_input.columns[(i+1):])]
					lhs = curr_input[curr_input.columns[i]]
					if type(lhs.dropna().iloc[0]) == str:
						res = sm.MNLogit(lhs, sm.add_constant(rhs)).fit()
						
					else:
						res = sm.GLM(lhs, sm.add_constant(rhs)).fit()
					
					pseudo_r2_i = extract_pseudo_r2(res.summary().tables[0])
					if pseudo_r2_i == 1:
						multicollinear = True
						continue
					vif = 1/(1-pseudo_r2_i)

					if vif > 2:
						multicollinear = True
				
				if not multicollinear:
					if pseudo_r2 > best_r2:
						best_feature = feature
						best_r2 = pseudo_r2
		if best_feature is None:
			break
		features_to_consider.remove(best_feature)
		curr_feature_set.append(best_feature)
	train_df = train_df[curr_feature_set+[target_var]]
	test_df = test_df[curr_feature_set+[target_var]]

	return train_df, test_df


def obtain_probability_dist(row, x):
	return 1-(1/(1+np.exp(list(reduce(lambda x,y: np.add(x,y),
								([row[i]*x[:, i] for i in range(len(row))]) ) ))))


def export_uncertainty(row, x, coefs):
	row_num = row.name
	diagnosis = "Has heart disease" if row["diagnosis"] == 1 else "Does not have heart disease"

	row = row.tolist()[:-1]
	prob_dist = obtain_probability_dist(row, x)

	curr_data = pd.DataFrame({"x": prob_dist})
	curr_data = curr_data.drop_duplicates().reset_index(drop=True)
	fig = px.histogram(curr_data, x="x", nbins=20,
                       color_discrete_sequence=['firebrick'])
	fig.update_traces(marker_line_width=2,marker_line_color="black")
	predicted_prob = np.exp(np.dot(coefs, row))/(1+np.exp(np.dot(coefs, row) ) )

	fig.add_vline(x=predicted_prob, line_width=3, line_dash="dash", line_color="green",
				  annotation_text=f"Predicted Probability: {predicted_prob}")


	fig.update_layout(title=f"Predicted Probability Distribution of Heart Disease for Subject {row_num}: {diagnosis}")
	fig.write_html(f"./data/probability_distribution_{row_num}.html")

if __name__ == "__main__":
	############################
	# Uncertainty for regression
	############################
	'''
	data = load_sample_data()
	train_data, test_data = randomize_train_test(data, prop_train=0.8)
	#print(f"Features: {', '.join(train_data.columns[:-1])}, Target: {train_data.columns[-1]}")
	X_train, y_train = np.array(train_data[train_data.columns[:-1]]), np.array(train_data["target"])
	X_test, y_test = np.array(train_data[test_data.columns[:-1]]), np.array(test_data["target"])
	model = train_lm(X_train, y_train, tf.keras.losses.MeanSquaredError())
	lower_model = train_lm(X_train, y_train, tfa.losses.PinballLoss(tau=0.05))
	upper_model = train_lm(X_train, y_train, tfa.losses.PinballLoss(tau=0.95))
	#print("Initializing distribution uncertainty object...")
	di = DistributionInstability(model, train_data, test_data)
	first_moments = [0.03, 0.05, 0.03, 0.03,0,0,0,0,0,0]
	#print(train_data.describe())
	print("Estimating shift in parameters over a shifted distribution "+\
		  "with older individuals with a higher bmi and bp than average"+" with only known first moments...")
	di.plot_projected_uncertainty(first_moments)
	'''
	################################
	# Uncertainty for classification
	################################
	rpath = get_root_path("pyeng")
	mappings = {'sex':{1:"male",0:"female"},
    'chest_pain_type':{1:"typical angina",2:"atypical angina",
                       3:"non-anginal pain", 4: "asymptomatic"},
                       'fasting_blood_sugar': {1: "above 120 mg/dl",
                       0:"below or equal to 120 mg/dl"},
                       'resting_ecg':{0:"normal",1:"S-T wave abnormality",
                       2:"left ventricular hypertrophy"},
       'exercise_induced_angina': {1:"exercise induced angina",
        0: "exercise did not induce angina"}}

	hd_train_data, hd_test_data = load_data_file([f"{rpath}/data/processed.cleveland.data"],
							 [f"{rpath}/{path}" for path in ["data/processed.hungarian.data",
							 "data/processed.switzerland.data",
							 "data/processed.va.data"]],
							 variable_names=["age","sex","chest_pain_type",
							 "resting_blood_pressure","cholesterol",
							 "fasting_blood_sugar","resting_ecg",
							 "max_heart_rate","exercise_induced_angina",
							 "depression_induced_by_exercise",
							 "slope_of_peak_exercise",
							 "number_of_colored_vessels",
							 "defects","diagnosis"], target_var="diagnosis",
							 threshold_na=0.50, mappings=mappings)
	##
	# Fitting statsmodel logistic regression for statistics 
	# of the model parameters, including its variance.
	##
	new_cols, hd_model = fit_stats_logistic_model(hd_train_data[hd_train_data.columns[:-1]], hd_train_data["diagnosis"])
	
	mcmc = MCMC(hd_model, hd_train_data[new_cols+["diagnosis"]], hd_test_data[new_cols+["diagnosis"]], target_var="diagnosis")	
	x, posterior_x, accepted_proportion = mcmc.run_mh_algorithm(n=10000, b=10000)
	

	mcmc.test_df.apply(lambda row: export_uncertainty(row, x, mcmc.coefs), axis=1)
