# Data Wrangling and Manipulation
import numpy as np
import pandas as pd
from sklearn import datasets

# For basic linear modelling 
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers

# Reducing the estimation of Pproj from P0
# as a convex optimization problem.
import cvxpy as cp
from scipy.special import entr


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

	print(f"Correlation Matrix of sample data:\n{df.corr()}")
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

if __name__ == "__main__":
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

