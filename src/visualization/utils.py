

pre_pic = """

#### Regression

In this exercise we are tasked with predicting an Initial Production (IP) of an unconventional wells drilled in Western Canadian
Sedimentary Basin (WCSB) over several decades. Since an unconventional well rarely produces a single phase liquid, 
we are provided with IP values for Oil, Gas and Water. As a result the problem becomes that of multi-output regression:
that is, given a set of columns (features), predict **not one, but several targets (Oil, Gas and Water IPs)**.
"""

post_pic = """
The modelling approach can be conceptualized as the following iterative process:

1. Analyze the input features, and determine variables that should provide higher/lower predictive power
2. Analyze the target values, and determine whether any transform is necessary (e.g. check the scoring metric, underlying target distribution) 
3. Use creativity and domain knowledge to come up with new features - a non-linear function of those already in a dataset
4. Break down dataset into training and validation, which will be used for model training and model selection/benchmarking, respectively.
5. Create a Pipeline of feature generation with model training to make sure feature generation process uses training data ONLY
6. Train a model, analyze residuals, feature importance, compare with your expectations in 1, and rinse-repeat the process using new knowledge. 

Progress tracking and code version control was done via GitHub - that allows for reproducibility via marking commits with good score and reverting when/if necessary.
"""

modelling_approach_reg =[pre_pic,post_pic]


modelling_approach_class = """
#### Classification 

Most of the approach above applies, since the datasets provided were pretty much identical, except for the size/covered time periods. Classification dataset is a lot bigger, but
the techniques above are still valid, except that our target is now a label (Abandoned|Active|Suspended). Our model result analysis will also change from the residuals to confusion matrices.

"""

feature_engineering = """
Feature engineering step was mainly driven by the exploratory data analysis performed earlier:

 - All the categorical columns were encoded with Frequency encoder which assigns a number equal to the frequency of the category value in that column.
This helps a model in case of categorical feature with a high proportion of unique values. 
- Completions and geology both play an important role, and both were missing in the underlying dataset. Therefore we had to come up with proxies.
- Latitude and longitude were used to calculate a HZ length of a well, as well as its azimuth - the longer the well, the higher its production generally.
- Well length was also calculated using drilling days and metres drilled per day information
- Proppant intensities historically have generally been increasing - therefore we kept `SpudDate` as a proxy for proppant and other technologies that vary with time
- Some of the continuous columns (`DrillingDays`,`GroundElevation`) had large outlier values - a potential error while data gathering, which were clipped to a chosen quantile (5%-95%) range.
- For classification & regression, pair-wise date differences were calculated. A difference between a `StatusDate` and `SurfAbandonDate` helps pinpoint old wells, that
are more likely to be labeled `Abandoned`, as well as those wells that recently went into production, likely to be flagged as `Active`  
"""

modelconf_blurb = """
#### Regression & Classification
As a go-to model we picked an implementation of Gradient Boosted Trees Machine - a LightGBM, which proved itself successfull in multiple competitions. The algorithm benefits from a highly -efficient
C++ implementation, and is well parallelized, which makes it a great candidate for quick prototyping and experimentation.

As per our cross-validation discussion above, we decided to sort our data in time, and train in KFold, non-shuffled manner.
That way, we make sure that our train and test do not intersect in time, and hopefully the CV results are indicative of 
future performance (or at least closer to it, as opposed to random subsetting)

![gbm](https://littleml.files.wordpress.com/2017/03/boosted-trees-process.png?w=497 "Trees iteratively try to correct errors in prediction from a previous tree in the sequence")

*Trees iteratively try to correct errors in prediction from a previous tree in the sequence*

"""

intro_blurb = """
The purpose of this little webapp is threefold:

1. Introduce a reader to the modelling approach used in the Regression & Classification part of the Untapped challenge
2. Discuss in brief feature engineering, a class of models, and a cross-validation strategy employed at modelling stage
3. Analyze model performance (residuals, complexity of the inference process), and illustrate individual impact of 
selected features on a model prediction.  
"""

target_transform_blurb = """
#### Regression

It was demonstrated that in oil and gas industry, the distribution of well production values has long tails. 
This phenomenon is not unique for well production - a distribution of rainfall and earthquake magnitudes have a similar property.
If we train a model using raw target values trying to minimize RMSE, a large relative error on a smaller number will be given the same weight as small relative error on a large number.
E.g. for a well with true IP of *1 bbld* a prediction of 10 bbld provides same error as for a well with true IP 1000 bbld and predicted IP 1010 bbld. Thus, our model may not be
good at predicting low values where a bulk of the dataset sits at, and attempt at minimizing error at the right tail with high production values, that are rarely observed, and a hard to predict.



Presence of such outliers can *explode* the error term (as we do not want to penalize large errors when both ground truth and a prediction are large numbers), 
thus we need to alleviate this problem and direct out model towards minimizing a relative error. Therefore, we can switch to `log(y)`,
and directly minimize RMSLE. One of the downsides of RMSLE metric is that **it penalizes underestimation more than
 overestimation** - that's not something one would like to do when predicting well performance !


In Python, the easiest way to switch to a different target is to create a new LGBM-like model class (via inheritance) and redefine `fit` and `predict` methods.
That allows one to keep all the convenient functionality of underlying class, while reaping benefits of a custom target scaling (`log` in our case).

```
class LogLGBM(LGBMRegressor):
    def __init__(self, target=None, **kwargs):
        super().__init__(**kwargs)
        self.target_scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

    def fit(self, X, Y, **kwargs):
        self.target_scaler.fit(Y.values.reshape(-1, 1) + 1)
        super(LogLGBM, self).fit(X, y_train, **kwargs)
        return self

    def predict(self, X):
        preds = super(LogLGBM, self).predict(X).reshape(-1, 1)
        preds = self.target_scaler.inverse_transform(preds) - 1
        return preds
```

#### Classification

In classification the target is provided as a class ID, and the distribution of classes is rather balanced. Therefore no transform / resampling was applied.
However, a problem with the classification is that the balance of classes is time-dependent: obviously when a field is in early development, none of the wells are abandoned,
and vice versa - when a field is really old with no activity, then most of the wells will be abandoned. Therefore the model trained on a random sample over time will have
very questionable forecasting abilities.



"""

model_blurb = """ We can demonstrate model results using residual plots and individual model prediction break-downs, and feature importances.

Using SHAP method, we attempt at explaining at what feature values caused the model to make certain decisions - that is explanations in **local sense** (per data point).

If we aggregate SHAP explanations over the dataset, we can get an idea of average feature importance - that is explanations in **global sense**.
"""

feat_imp_blurb = """
#### Regression

Feature importance plot shows features in the order of decreasing importance. As mentioned earlier, 
`Max BOE` is a top feature due to the fact that it really *leaks* the target.
That is, it is strongly correlated to the target values, however may not be available in the reality (we simply do not
 know `Max BOE` prior to drilling, same applies to IP). 
Among other strong features are obviously the location of the well (`lat`, `long` **maps a rock quality, geological 
features**), its **vintage** (`SpudDate`) and horizontal **length** (`haversine_Length`).


Although for most of the features the trend is ambiguous and not interpretable, we can see a meaningful relationship 
for both `Max BOE` and `haversine_Length` - we do expect IP 
to **positively correlate with maximum of production**, as well as **increase as wells become longer**, which is
 captured in the figure. 
"""


classification_fi_blurb ="""
#### Classification


Similar to regression, features are shown here in the order of decreasing importance (i.e. their impact on a particular class probability, shown by a colored bar).
A brief examination of the feature importance plot and probability dependency plot for each class (e.g. how probability of 'Abandoned' class changes w.r.t. Max BOE value)
 may bring you to the following, quite reasonable conclusions:
 
- Non-NA `Max BOE` suggests that a well is still active (most likely because the number is available for later spud date wells


- The closer time difference between a `SurfAbandonDate` and a `StatusDate` is to 0, the more likely that a well is abandoned (there's no more status updates after a well is Abandoned)


- For some operators (= certain values in `CurrentOperatorParent`) the game of drilling in the Viking Basin is over, 
    as you might see by elevated chances of wells being abandoned if they belong to those operators.
        
"""

follow_up_blurb = """

#### Regression


Since the **high-score models** currently heavily **rely on `Max BOE`**, we are afraid there is **not much practical use** to the models created.
The problem is that `Max BOE` **is not available before the well is drilled**, as that is the time
when a prediction of IP is desired. Thus using this feature in modelling is a technically a `target leak` and should be avoided if 
future predictive power is sought after. 

However, if `Max BOE` was to be eliminated, and some completions / geology data was added, then a similar modelling procedure 
could be used in evaluating reserves and impact on production caused by various completions practices. From a risk point of view,
one could apply this model in a step-off (=clustered) manner and identify geographic areas / zones where predictions are less reliable (example is given below). 
Less reliable areas could be loosely defined as those characterized with high error in predictions. Those areas can then be marked as
risky, and could be given a higher discount when economical analysis is performed.

We could also quantitatively analyze effectiveness of completions practices and figure out which one leads to better IPs / EURs,
and help completions engineers run the jobs in a more sustainable manner (i.e. use less proppant if the excessive 
intensity is no longer justified by a forecasted increase in production)

#### Classification

Similar to the aforementioned issues, the models that score high in this competition are quite useless, since they directly use leaking proxies for the target, such as `SurfAbandonDate`.
We find it difficult to find a reasonable practical application for the models that score high on the leaderboard.
If that feature were to be eliminated, then the model ends up relying on an operator and their historical activity in the region. We still are facing a problem of a model
having no **predict-in-the-future** capabilities, since the dataset in its current current form is rather static (i.e. none of the columns are time-dependent).

However, if the dataset had time dependent features (days since a closure event, or days on production in the last 365
 days, etc.) , it would be interesting to develop a model that predicts days till abandonment. We could get the 
 probabilities of a well to be abandoned in the next N months. The main value here is to have this probability change
  over time, thus allowing the user to quickly identify areas where a large population is likely to be abandoned.
"""


cv_blurb = """

An important note - for the model to be useful in practice, it should be trained in a reality-reflecting scenario. That is,
if the data has a temporary character (i.e. Spud Date of the real test data is only increasing, or any other trends in features related to time),
the cross-validation technique employed should reflect that. Depending on the application of this model, the random train-test split used by organizers
might be an erroneous approach to estimate a model's predictive power, as the new data will not be coming randomly.

Indeed, wells are not drilled in a random pattern occuring randomly in time and space - E&P companies tend to over-develop good reservoir, and under-develop 
areas of poor production. Moreover, the underlying features might also change over time (think about HZ Length, Proppant used, various technologies of completion),
and models may not necessarily be good at extrapolating when trained on one vintage of wells and applied on the other. It is this performance that we are often interested in:
how good/bad is a model at predicting performance in the future, and how to make it depend on the features that contribute most towards metric score in that scenario.


Notice an evidently randomized split between the train/test/validation datasets and a clearly
non-stationary HZ length distribution in the figure below. Try selecting different windows along the time 
axis to see the distribution of selected wells around the basin (*very non-random*)
"""