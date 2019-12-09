parser = argparse.ArgumentParser(description='Arguments for training a discrete model')    
parser.add_argument('--prefix', type=str, default='GenoML_data', help='Prefix for your training data build.')
parser.add_argument('--max-tune', type=int, default=50, help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
parser.add_argument('--n-cv', type=int, default=5, help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')

args = parser.parse_args()

print("Here is some basic info on the command you are about to run.")
print("Python version info...")
print(sys.version)
print("CLI argument info...")
print("Working with the dataset and best model corresponding to prefix", args.prefix, "the timestamp from the merge is the prefix in most cases.")
print("Your maximum number of tuning iterations is", args.max_tune, "and if you are concerned about runtime, make this number smaller.")
print("You are running", args.n_cv, "rounds of cross-valdiation, and again... if you are concerned about runtime, make this number smaller.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 representing a positive case.")

print("")

run_prefix = args.prefix
infile_h5 = run_prefix + ".dataForML.h5"
df = pd.read_hdf(infile_h5, key = "dataForML")

print()
print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
print("#"*30)
print(df.describe())
print("#"*30)
print()

y_tune = df.PHENO
X_tune = df.drop(columns=['PHENO'])
IDs_tune = X_tune.ID
X_tune = X_tune.drop(columns=['ID'])

best_algo_name_in = run_prefix + '.best_algorithm.txt'
best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
best_algo = str(best_algo_df.iloc[0,0])

print("From previous analyses in the training phase, we've determined that the best algorithm for this application is", best_algo, " ... so lets tune it up and see what gains we can make!")

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix, roc_curve, auc, make_scorer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

algorithms = [
	LogisticRegression(),
	RandomForestClassifier(),
	AdaBoostClassifier(),
	GradientBoostingClassifier(),
	SGDClassifier(loss='modified_huber'),
	SVC(probability=True),
	ComplementNB(),
	MLPClassifier(),
	KNeighborsClassifier(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis(),
	BaggingClassifier(),
	XGBClassifier()
	]

if best_algo == 'LogisticRegression':
	algo = getattr(sklearn.linear_model, best_algo)()

if  best_algo == 'SGDClassifier':
	algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

if (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
	algo = getattr(sklearn.ensemble, best_algo)()

if best_algo == 'SVC':
	algo = getattr(sklearn.svm, best_algo)(probability=True, gamma='auto')

if best_algo == 'ComplementNB':
	algo = getattr(sklearn.naive_bayes, best_algo)()

if best_algo == 'MLPClassifier':
	algo = getattr(sklearn.neural_network, best_algo)()

if best_algo == 'XGBClassifier':
	algo = getattr(xgboost, best_algo)()

if best_algo == 'KNeighborsClassifier':
	algo = getattr(sklearn.neighbors, best_algo)()

if (best_algo == 'LinearDiscriminantAnalysis') or (best_algo == 'QuadraticDiscriminantAnalysis'):
	algo = getattr(sklearn.discriminant_analysis, best_algo)()

if best_algo == 'LogisticRegression':
	hyperparameters = {"penalty": ["l1", "l2"], "C": sp_randint(1, 10)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if  best_algo == 'SGDClassifier':
	hyperparameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "learning_rate": ["constant", "optimal", "invscaling", "adaptive"]}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
	hyperparameters = {"n_estimators": sp_randint(1, 1000)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if best_algo == 'SVC':
	hyperparameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": sp_randint(1, 10)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)
	
if best_algo == 'ComplementNB':
	hyperparameters = {"alpha": sp_randfloat(0,1)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if best_algo == 'MLPClassifier':
	hyperparameters = {"alpha": sp_randfloat(0,1), "learning_rate": ['constant', 'invscaling', 'adaptive']}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if best_algo == 'XGBClassifier':
	hyperparameters = {"max_depth": sp_randint(1, 100), "learning_rate": sp_randfloat(0,1), "n_estimators": sp_randint(1, 100), "gamma": sp_randfloat(0,1)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if best_algo == 'KNeighborsClassifier':
	hyperparameters = {"leaf_size": sp_randint(1, 100), "n_neighbors": sp_randint(1, 10)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

if (best_algo == 'LinearDiscriminantAnalysis') or (best_algo == 'QuadraticDiscriminantAnalysis'):
	hyperparameters = {"tol": sp_randfloat(0,1)}
	scoring_metric = make_scorer(roc_auc_score, needs_proba=True)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

max_iter = args.max_tune
cv_count = args.n_cv

print("Here is a summary of the top 10 iterations of the hyperparameter tune...")

rand_search = RandomizedSearchCV(estimator=algo, param_distributions=hyperparameters, scoring=scoring_metric, n_iter=max_iter, cv=cv_count, n_jobs=-1, random_state=153, verbose=0)
start = time()
rand_search.fit(X_tune, y_tune)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
	" parameter iterations." % ((time() - start), max_iter))

def report(results, n_top=10):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
			results['mean_test_score'][candidate],
			results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

report(rand_search.cv_results_)
rand_search.best_estimator_

print("Here is the cross-validation summary of your best tuned model hyperparameters...")
cv_tuned = cross_val_score(estimator=rand_search.best_estimator_, X=X_tune, y=y_tune, scoring=scoring_metric, cv=cv_count, n_jobs=-1, verbose=0)
print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
print(cv_tuned)
print("Mean cross-validation score:")
print(cv_tuned.mean())
print("Standard deviation of the cross-validation score:")
print(cv_tuned.std())

print()

print("Here is the cross-validation summary of your baseline/default hyperparamters for the same algorithm on the same data...")
cv_baseline = cross_val_score(estimator=algo, X=X_tune, y=y_tune, scoring=scoring_metric, cv=cv_count, n_jobs=-1, verbose=0)
print("Scores per cross-validation of the metric to be maximized, this scoring metric is AUC for discrete phenotypes and explained variance for continuous phenotypes:")
print(cv_baseline)
print("Mean cross-validation score:")
print(cv_baseline.mean())
print("Stnadard deviation of the cross-validation score:")
print(cv_baseline.std())

print()
print("Just a note, if you have a relatively small variance among the cross-validation iterations, there is a higher chance of your model being more generalizable to similar datasets.")

print()
if cv_baseline.mean() > cv_tuned.mean():
	print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
	print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
	print()
	print("Lets shut everythong down, thanks for trying to tune your model with GenoML.")

if cv_baseline.mean() < cv_tuned.mean():
	print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
	print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
	print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")
	
	algo_tuned = rand_search.best_estimator_
	
	### Save it using joblib
	from joblib import dump, load
	algo_tuned_out = run_prefix + '.tunedModel.joblib'
	dump(algo_tuned, algo_tuned_out)
	
	### Export the ROC curve
	import matplotlib.pyplot as plt

	plot_out = run_prefix + '.tunedModel_allSample_ROC.png'

	test_predictions = algo_tuned.predict_proba(X_tune)
	test_predictions = test_predictions[:, 1]

	fpr, tpr, thresholds = roc_curve(y_tune, test_predictions)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % roc_auc + '\nMean cross-validation ROC curve (area = %0.2f)' % cv_tuned.mean())
	plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('Receiver operating characteristic (ROC) - ' + best_algo + '- tuned' )
	plt.legend(loc="lower right")
	plt.savefig(plot_out, dpi = 600)

	print()
	print("We are also exporting a ROC curve for you here", plot_out, "this is a graphical representation of AUC in all samples for the best performing algorithm.")
	
	### Exporting tuned data, which is by nature overfit.
		
	tune_predicteds_probs = algo_tuned.predict_proba(X_tune)
	tune_case_probs = tune_predicteds_probs[:, 1]
	tune_predicted_cases = algo_tuned.predict(X_tune)

	tune_case_probs_df = pd.DataFrame(tune_case_probs)
	tune_predicted_cases_df = pd.DataFrame(tune_predicted_cases)
	y_tune_df = pd.DataFrame(y_tune)
	IDs_tune_df = pd.DataFrame(IDs_tune)

	tune_out = pd.concat([IDs_tune_df.reset_index(), y_tune_df.reset_index(drop=True), tune_case_probs_df.reset_index(drop=True), tune_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
	tune_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
	tune_out = tune_out.drop(columns=['INDEX'])

	tune_outfile = run_prefix + '.tunedModel_allSample_Predictions.csv'
	tune_out.to_csv(tune_outfile, index=False)

	print("")
	print("Preview of the exported predictions for the tuning samples which is naturally overfit and exported as", tune_outfile, "in the similar format as in the initial training phase of GenoML.")
	print("#"*30)
	print(tune_out.head())
	print("#"*30)
	
	## Export historgrams of probabilities.

	import seaborn as sns

	genoML_colors = ["cyan","purple"]

	g = sns.FacetGrid(tune_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
	g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=False, rug=True))
	g.add_legend()

	plot_out = run_prefix + '.tunedModel_allSample_probabilities.png'
	g.savefig(plot_out, dpi=600)

	print("")
	print("We are also exporting probability density plots to the file", plot_out, "this is a plot of the probability distributions of being a case, stratified by case and control status for all samples.")

print()
print("Lets shut everything down, thanks for trying to tune your model with GenoML.")
print()