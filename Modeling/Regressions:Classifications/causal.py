import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
import statsmodels.api as sm

import os

class CausalInference:
	_effect_types = {"risk ratio", "risk difference"}
	def __init__(self, fname, modify_f, skiprows):
		self.data = pd.read_csv(fname, skiprows=skiprows)
		self.data = modify_f(self.data, "Sex", "Pclass")
	def ipw(self, strata_col, treatment_col, treatment_val, outcome_col, outcome_class):
		"""
		inverse probability weighting - we assume categorical strata and 
		categorical treatment.
		"""
		a1_per_strata = self.data.groupby([strata_col])\
			       .apply(lambda x: len(x[x[treatment_col]==treatment_val])/len(x) )
		a1_per_strata = a1_per_strata.reset_index(name="prop")
		strata_prop_dict = dict(zip(a1_per_strata[strata_col], a1_per_strata["prop"]))
		n = len(self.data)
			
		E_a1 = np.sum(self.data.apply(lambda row: apply_ipw_weight(row,
																   strata_prop_dict,
																   strata_col,
																   outcome_col,
																   outcome_class,
															       treatment_col,
																   treatment_val, 
																   treatment_exists=True), axis=1))/n
		E_a0 = np.sum(self.data.apply(lambda row: apply_ipw_weight(row,
																   strata_prop_dict,
																   strata_col,
																   outcome_col,
																   outcome_class,
																   treatment_col,
															       treatment_val, 
																   treatment_exists=False), axis=1))/n
		return E_a1-E_a0
	def standardization(self, strata_col, treatment_col, treatment_val, outcome_col, outcome_class, causal_effect_type):
		strata = set(self.data[strata_col].unique())
		treatment_cumulative = 0
		no_treatment_cumulative = 0
		n = len(self.data)
		for stratum in strata:
			stratum_data = self.data[self.data[strata_col]==stratum]
			prob_l = len(stratum_data)/n
			a1_data = stratum_data[stratum_data[treatment_col]==treatment_val]
			a0_data = stratum_data[stratum_data[treatment_col]!=treatment_val]
			y1_a1_data = a1_data[a1_data[outcome_col]==outcome_class]
			y1_a0_data = a0_data[a0_data[outcome_col]==outcome_class]
			prob_y1_a1 = len(y1_a1_data)/len(a1_data)
			prob_y1_a0 = len(y1_a0_data)/len(a0_data)
			treatment_cumulative += prob_y1_a1*prob_l
			no_treatment_cumulative += prob_y1_a0*prob_l
		if causal_effect_type not in CausalInference._effect_types:
			raise Exception("Causal effect type must be in valid effect types.")
		if causal_effect_type == "risk ratio":
			return treatment_cumulative/no_treatment_cumulative
		else:
			return treatment_cumulative-no_treatment_cumulative
	def nonparam_dr(self, treatment_col, outcome_col, covariate_cols):
		"""
		nonparam_dr is the nonparametric estimation of 
		treatment effect using doubly robust estimation.
		
		We first define the propensity score (PS) model, then
		the outcome model. The outcome model inherits the 
		propensity scores from the PS model to weigh the 
		outcome regression.

		:return:
		"""

		# (1) First split the data
		train_data = self.data.sample(frac=0.5, random_state=42)
		test_data = self.data.loc[~self.data.index.isin(set(train_data.index)), :]
		train_data = train_data[~pd.isna(train_data[treatment_col])]
		test_data = test_data[~pd.isna(test_data[treatment_col])]
		for col in covariate_cols:
			if type(train_data[col].dropna().iloc[0]) == str:
				train_data[col] = pd.factorize(train_data[col])[0]
				test_data[col] = pd.factorize(test_data[col])[0]
		
		# (2) Estimate the nuisance functions pi and mu to obtain predicted values.
		# pi gives the propensity score of treatment (probability) and mu returns
		# the mean outcome.
		pi_hat, mu_hat = estimate_nuisance_functions(self.data, 
													 covariate_cols,
													 treatment_col,
												     outcome_col)
		
		# (3) Construct pseudo-outcome and regress on the treatment variable A
		# The pseudo outcome is computed from (Y-mu)/pi*(mean(pi))+mean(mu)
		# mu would be the predicted outcome for the sample and pi is the predicted
		# propensity of treatment.
		
		# (4) The pseudo-outcome regressed on the treatment A can be 
		# estimated using a loess or kernel estimator, like in the paper.
		# We will do it with local linear regression.

def estimate_nuisance_functions(data, covariates, treatment, outcome):
	"""
	estimate_nuisance_functions - takes in X covariates,
	treatment A, and outcome Y. In the case of the titanic
	dataset, we have a binary outcome.
	:param covariates List[str]:
	:param treatment str:
	:param outcome str:
	:return:
	"""
	# propensity score.
	propensity_scores = kernel_propensity_scores(data, treatment, covariates,
											    kernel=kernel_f,
												bandwidth="silverman")
	
	predicted_outcomes = logistic_predicted_outcome(data, 
													covariates,
													treatment,
													outcome)

	return propensity_scores, predicted_outcomes


def logistic_predicted_outcome(data, covariates, treatment, outcome):
	logreg = _


def kernel_propensity_scores(data, treatment, covariates, kernel, bandwidth):
	X = data.loc[:, covariates].values
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
	log_density = kde.score_samples(data[covariates])
	propensity_scores = np.exp(log_density)
	return propensity_scores



def get_root_path(root_name):
    dir_lst = os.getcwd().split("/")
    idx = dir_lst.index(root_name)
    return "/".join(dir_lst[:(idx+1)])


def modify_titanic_data(data, col1, col2):
    data[col1+"-"+col2] = data.apply(lambda row: f"{row[col1]} {row[col2]}", axis=1)
    return data


def apply_ipw_weight(row, strata_prop_dict, strata_col, outcome_col, outcome_class,
				     treatment_col, treatment_val, treatment_exists):
	"""
	:param row:
	:param treatment_val str:
	:return:
	"""

	if treatment_exists:
		A_a_identity = int(row[treatment_col]==treatment_val)
	else:
		A_a_identity = int(row[treatment_col]!=treatment_val)
	return A_a_identity*int(row[outcome_col]==outcome_class)/\
              (strata_prop_dict[row[strata_col]]\
              if row[treatment_col]==treatment_val\
              else 1-strata_prop_dict[row[strata_col]])


if __name__ == "__main__":
	rpath = get_root_path("pyeng")
	inference1 = CausalInference(fname=f"{rpath}/data/titanic.csv", modify_f=modify_titanic_data, skiprows=0)
	print(inference1.standardization(strata_col="Sex-Pclass", treatment_col="Embarked", treatment_val="Q",
						       outcome_col="Survived", outcome_class=0, causal_effect_type="risk ratio"))

	print(inference1.ipw(strata_col="Sex-Pclass", treatment_col="Embarked", treatment_val="Q",
						 outcome_col="Survived", outcome_class=0))

	print(inference1.nonparam_dr(treatment_col="Age", outcome_col="Survived", 
								 covariate_cols=['Pclass',
												 'Sex',
												 'SibSp',
												 'Parch', 
												 'Fare', 
												 'Embarked']))
	
	









