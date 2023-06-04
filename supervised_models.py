import numpy as np

# This class is basically a wrapper on a supervised ML library.  It does the following in addition to the usual stuff:
# - Uses dummy 1-hot encoding to handle categorical data
# - lightly cleans the trainign and tet set
# - Keeps and returns some analytics (feature importances, performance stats/ROC curves)

# Use it like so:
# scorer = RevenueForecastModel(<Training_dataset>, <target_column_name>)
# ret = scorer.train_scorer(addRules = None,keepCols=<feature_column_list>,minsamples = <minsamples>,id_column=<UID_column_name>)

# Then Set the calibration:
# scores = scorer.score_records(<Training_dataset>, addRules = addRules, keepCols = <feature_column_list>,id_column=<UID_column_name>)

# df_raw_training_cut[target_col + '_score'] = scores
# calib_cofactor = scorer.calculate_calib_cofactor(df_raw_training_cut,target_col,target_col + '_score',calib_variable)
# #from now on, this calibration will tune the scores that come from this model
# scorer.set_calibration_vector(calib_cofactor)
# model_stack.append(scorer)

# #Score data later on:
# scores = scorer.score_records(<set_to_score>, addRules = addRules, keepCols = <feature_column_list>,id_column=<UID_column_name>)

class RevenueForecastModel:
   
    # init method or constructor 
    def __init__(self, df, targetColumn, allColsLabeled = True, model = None, scorePercEdges = [5,10,15,20,30,40,50,60,70,80]):
        self.load_data_from_dataframe(df)
        self.targetColumn = targetColumn
        self.model = model

        #Variables to underlie the transform->scaler->model pipeline structure
        self.col_trans = None
        #self.scaler = None
        self.pipeline = None
        #model performance and calibration details
        self.performance_results = None
        self.calibration_cofactor = 1.0
       
        self.set_model_params(scorePercEdges = scorePercEdges) # put the defaults in at first
        if self.targetColumn is not None:
            self.set_target_column(targetColumn)
        # If all the incoming columns are correctly labeled, we just hold the data and train the model with it.
        # But if not, then we expect the positive (1/True) labels to be very rare and not representative of the whole set
        # In this case, we want to first "enrich" the dataset by performing a dimensional reduction, projection and re-labeleing according
        # to some assumptions:
        self.allColsLabeled = allColsLabeled
        if self.allColsLabeled is False:
            self.enrich_and_relabel_dataset()
        # either way, here we complete the construction with a dataset we should be ready to train on - either explicitly
        # labeled, or synthetically labeled ("enriched") by extrapolating from the few provided labels.
    
    def get_data(self):
        return self.df

    def set_model_params(self,scorePercEdges):
        # The percentile-edges for the nonzero scores to be constructed.
        # e.g [30,50,70,90] will produce the following "score bins":
        # 0th percentile-30th percentile: Receive a score of 0
        # 30th-50th percentile: Receive a score of 1
        # ...
        # 90th+ percentile: Receive a score of 4.
        self.scorePercEdges = scorePercEdges

    def get_model_performance(self):
        return self.performance_results,

    def set_model_performance(self,mp):
        self.performance_results = mp

    def get_calibration_vector(self):
        return self.calibration_cofactor

    def set_calibration_vector(self, cc):
        self.calibration_cofactor = cc

    def load_data_from_dataframe(self,df):
        # set the dataset.
        # note: this is intended to be a pandas dataframe, with several feature columns
        # and a target column. The target should be a binary (False/True or 0/1), where
        # True/1 means "the thing has happened," where "the thing" is usually either
        # account expansion, churn or continued activity.  See "targetGoodness" below for more about the difference.
        self.df = df

    def load_data_from_file(self,filePath):
        #load the data as a csv
        self.df = pd.read_csv(filePath)

    def save_data_to_file(self,filePath):
        #save the data to a csv
        if self.df is not None:
            self.df.to_csv(filePath,index=False)
        
    def save_model_to_file(self,modelFilePath):
        #store the model as a file to be loaded 
        joblib.dump(self.pipeline, modelFilePath) 

    def load_model_from_file(self,modelFilePath):
        #load the model from a file location
        self.pipeline = joblib.load(modelFilePath)

    def set_target_column(self,targetColumn,targetGoodness=1):
        # set up the target we want to use (the rest of the
        # input columns will be used to train the model)
        # targetGoodness should be 0 or 1, where 0 means "the target being positive is bad"
        # (e.g. the target represents churn) and 1 means good (e.g. the terget represents expansion.)
        # this will affect which "direction" the score goes (high score should always be "good")
        self.targetColumn = targetColumn
        self.targetGoodness = targetGoodness

    def get_target_column(self):
        return self.targetColumn
    
    def centroid_np(self,arr):
        #get the centroid from a bunch of 2D vectors
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length
    
    def add_addrules(self, df_data, addRules):
        # Add columns to this dataset, using the set of rules and names from the input.
        df_ret = df_data.copy()

        #add these features to both sets here:
        for addRule in addRules:
            df_ret[addRule[1]] = df_ret.eval(addRule[0])
            # a little caution here - this use of "eval" allows basically any column to be made
            # so we'd like to clean up infinities, NaNs and any other odd thing here:
            df_ret[addRule[1]] = df_ret[addRule[1]].replace([np.inf, -np.inf], np.nan)
            df_ret[addRule[1]] = df_ret[addRule[1]].fillna(0) #safety and human-relevant answers
        return df_ret
    
    def test_model_perf(self,testData,testTarget,trainData,trainTarget,useRegressor = False):
        # check the model performance on some test data
        # Note: Do these tests with the zero-activity stuff
        # removed from your dataset for a "cleaner" performance comparison.
        # In actual use there are many zero-activity accounts, which skews
        # accuracy/ROC AUC up signifcantly.
        y_pred=self.pipeline.predict(testData)
        y_pred_train=self.pipeline.predict(trainData)
        featnames = self.pipeline['categoricals'].get_feature_names()
        if useRegressor:
            #return 0,0,self.pipeline['classifier'].feature_importances_,featnames,[[0],[0],[0]],0,0
            #return testData,testTarget,y_pred,self.pipeline['classifier'].feature_importances_,featnames,[[0],[0],[0]],0,0
            return testData,testTarget,y_pred,[],featnames,[[0],[0],[0]],0,0
        else:
            try:
                acc = metrics.accuracy_score(testTarget, y_pred)
                test_probs = self.pipeline.predict_proba(testData)[:, 1]
                auc = roc_auc_score(testTarget, test_probs)

                acc_train = metrics.accuracy_score(trainTarget, y_pred_train)
                train_probs = self.pipeline.predict_proba(trainData)[:, 1]
                auc_train = roc_auc_score(trainTarget, train_probs)
            except:
                return 0.0,0.0,self.pipeline['classifier'].feature_importances_,featnames,0,0,0,testData,testTarget,y_pred
            try:
                return auc,acc,self.pipeline['classifier'].feature_importances_,featnames,roc_curve(testTarget, test_probs),auc_train,acc_train,testData,testTarget,y_pred
            except: 
                return auc,acc,None,featnames,roc_curve(testTarget, test_probs),auc_train,acc_train,testData,testTarget,y_pred

        
    def train_scorer(self,addRules = None, keepCols = None,useRegressor = False,minsamples = 200,id_column = "opportunity_id"):
        # use the dataframe to train an RF method
        if self.df is None:
            print('BROKEN - NO INTERNAL DATAFRAME!!')
            return None
        else:

            #prep and segment the data for training
            df_data = self.df.loc[:, self.df.columns != self.targetColumn]
            if addRules is not None:
                # add the columns corresponding to the additional Rules - we'll make these for 
                # training, scoring and explanation
                df_data = self.add_addrules(df_data, addRules)
            if keepCols is not None:
                #df_data = df_data.drop(dropCols, axis=1, inplace=False)
                df_data = df_data[keepCols]

            features_to_encode = df_data.columns[df_data.dtypes==object].tolist()
            features_to_encode.remove(id_column)
            df_data[features_to_encode] = df_data[features_to_encode].fillna('None')
            df_data = df_data.fillna(0)
            
            gs = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=0) # ,random_state=0
            train_ix, test_ix = next(gs.split(df_data, self.df[self.targetColumn], groups=df_data[id_column]))
            X_train = df_data.iloc[train_ix]
            y_train = self.df[self.targetColumn].iloc[train_ix]
            X_test = df_data.iloc[test_ix]
            y_test = self.df[self.targetColumn].iloc[test_ix]
            X_train = X_train.drop(id_column,1)
            X_test = X_test.drop(id_column,1)

            # non-numeric columns can be encoded with a one-hot encoder.  If these data aren't actually useful,
            # then the RF method ought to find that and leave them out of most of the trees.

            self.col_trans = make_column_transformer(
                        (OneHotEncoder(handle_unknown='ignore'),features_to_encode),
                        remainder = "passthrough"
                        )
            #Hyperparameters - could be further tweaked/optimized
            if useRegressor:
                gbr_params = {'n_estimators': 1000,
                                'max_depth': 5, #3,
                                'min_samples_split': 7,
                                'learning_rate': 0.03,
                                #'loss': 'absolute_error'}
                                'loss': 'ls'}
                self.model = GradientBoostingRegressor(**gbr_params)
            else:
                # NOTE:  Can use balanced or unweighted class weights, our calibrator should handle it
                self.model=RandomForestClassifier(n_estimators=250,class_weight='balanced',min_samples_split=minsamples,random_state = 0)
            #make a pipeline out of the transformation and model, and train the model
            self.pipeline = Pipeline(steps= [('categoricals', self.col_trans),('classifier', self.model)])
            print('THE COLUMNS WE"RE TRAINING WITH ARE:')
            print(X_train.columns)

            self.pipeline.fit(X_train, y_train)

            #return some stats on model performance
            ret = self.test_model_perf(X_test,y_test,X_train,y_train,useRegressor)
            self.set_model_performance(ret)
            return ret,
        
    def scale_score(self,score_raw):
        return score_raw * self.calibration_cofactor

    def score_records(self,dfToScore,addRules = None, keepCols = None,useRegressor = False,useProbs=False,id_column = "opportunity_id"):
        if self.pipeline is None:
            print('BROKEN!! NO PIPELINE!')
            return 0 # should lead to "all scores 0 down the line"
        else:
            # use everything but the target column - these should be the same as the model was trained on
            score_data = dfToScore.loc[:, dfToScore.columns != self.targetColumn]#.fillna(0)
            if addRules is not None:
                # add the columns corresponding to the additional Rules - we'll make these for 
                # training, scoring and explanation
                score_data = self.add_addrules(score_data, addRules)
            if keepCols is not None:
                #score_data = score_data.drop(dropCols, axis=1, inplace=False, errors='ignore')
                score_data = score_data[keepCols]
            features_to_encode = score_data.columns[score_data.dtypes==object].tolist()
            # JUST ADDED
            features_to_encode.remove(id_column)
            score_data[features_to_encode] = score_data[features_to_encode].fillna('None')

            score_data = score_data.drop(id_column,1)
            score_data = score_data.fillna(0)

            print('THE COLUMNS WE\'RE SCORING WITH ARE:')
            print(score_data.columns)

            print('THE FEATURES WE\'RE ENCODING FIRST ARE:')
            print(features_to_encode)
            
            # note that we're not using model.predict here (which has a static threshold).
            # we're using predict_proba, which should give a range from 0-1 inclusive (essentially a probability)
            if useRegressor:
                scores_raw = self.pipeline.predict(score_data)
            elif useProbs:
                return self.pipeline.predict_proba(score_data)
            else:
                scores_raw = [1.0 - i for i in self.pipeline.predict_proba(score_data)[:, 0]]

            return [self.scale_score(i) for i in scores_raw]

    def explain_records(self,dfToScore,addRules = None, keepCols = None,useRegressor = False,id_column = "opportunity_id"):
        # Explanation technique #1: the SHAP parameters.
        # here we return the relative-importance values/directions for each data feature,
        # for each individual model inference.  
        if self.pipeline is None:
            return None
        else:
            # use everything but the target column - these should be the same as the model was trained on
            score_data = dfToScore.loc[:, dfToScore.columns != self.targetColumn]#.fillna(0)
            if addRules is not None:
                # add the columns corresponding to the additional Rules - we'll make these for 
                # training, scoring and explanation
                score_data = self.add_addrules(score_data, addRules)
            if keepCols is not None:
                #score_data = score_data.drop(dropCols, axis=1, inplace=False, errors='ignore')
                score_data = score_data[keepCols]
            features_to_encode = score_data.columns[score_data.dtypes==object].tolist()
            # JUST ADDED
            features_to_encode.remove(id_column)
            score_data[features_to_encode] = score_data[features_to_encode].fillna('None')

            score_data = score_data.drop(id_column,1)
            score_data = score_data.fillna(0)

            #set the tree explainer as the model of the pipeline
            explainer = shap.TreeExplainer(self.pipeline['classifier'])

            #apply the preprocessing to x_test
            transf_score_data = self.pipeline['categoricals'].transform(score_data)

            print(transf_score_data.dtype)

            #get Shap values from preprocessed data
            shap_values = explainer.shap_values(transf_score_data)
            #plot the feature importance (if desired)
            #shap.summary_plot(shap_values[0], score_data, plot_type="bar")
            return shap_values, self.pipeline['categoricals'].get_feature_names()


    def record_explanations(self,explanations,dfToScore,addRules = None, keepCols = None,useRegressor = False,id_column = "opportunity_id"):
        if self.pipeline is None:
            return None
        else:
            # use everything but the target column - these should be the same as the model was trained on
            score_data = dfToScore.loc[:, dfToScore.columns != self.targetColumn]#.fillna(0)
            if addRules is not None:
                # add the columns corresponding to the additional Rules - we'll make these for 
                # training, scoring and explanation
                score_data = self.add_addrules(score_data, addRules)
            if keepCols is not None:
                #score_data = score_data.drop(dropCols, axis=1, inplace=False, errors='ignore')
                score_data = score_data[keepCols]
            features_to_encode = score_data.columns[score_data.dtypes==object].tolist()
            # JUST ADDED
            features_to_encode.remove(id_column)
            score_data[features_to_encode] = score_data[features_to_encode].fillna('None')

            score_data = score_data.drop(id_column,1)
            score_data = score_data.fillna(0)

            top_inds = [sorted(range(len(slist)), key = lambda sub: slist[sub])[-3:] for slist in explanations[0][0]]
            bot_inds = [sorted(range(len(slist)), key = lambda sub: slist[sub])[-3:] for slist in explanations[0][1]]
            top_feats = [[explanations[1][t] for t in q] for q in top_inds]
            bot_feats = [[explanations[1][t] for t in q] for q in bot_inds]

            #now make the distribution for each non-one_hot_encoded feature, for this feat_most_recent_stage_prob,
            #    then go and get the percentile for each important feature within records with the same feat_most_recent_stage_prob.
            #    Finally, deliver this all as a structured list of lists(?) from a function.  And have a look/spot check.
            # Apologies to the reader:  This is a bit complex here, with all the list comprehensions, but it's fairly efficient.
            eligible_top_feats = [[ top_feats[i][j] for j in range(len(top_feats[i])) if (('feat_most_recent_stage_prob' not in top_feats[i][j]) and  ('onehotencoder' not in top_feats[i][j]) )] for i in range(len(top_feats)) ]
            eligible_top_feat_informal_names = [[ feature_informal_names[feature_column_names.index(eligible_top_feats[i][j])]for j in range(len(eligible_top_feats[i]))] for i in range(len(eligible_top_feats)) ]
            dwtd_top_medians = [[ np.median(score_data.loc[(score_data['feat_most_recent_stage_prob']==score_data.loc[score_data.index[i],'feat_most_recent_stage_prob']),eligible_top_feats[i][j]]) for j in range(len(eligible_top_feats[i]))] for i in range(len(eligible_top_feats)) ]
            dwtd_top_stds = [[ np.std(score_data.loc[(score_data['feat_most_recent_stage_prob']==score_data.loc[score_data.index[i],'feat_most_recent_stage_prob']),eligible_top_feats[i][j]]) for j in range(len(eligible_top_feats[i]))] for i in range(len(eligible_top_feats)) ]
            dwtd_top_levels = [[ score_data.loc[score_data.index[i],eligible_top_feats[i][j]] for j in range(len(eligible_top_feats[i]))] for i in range(len(eligible_top_feats)) ]
            dwtd_top_abovemedian = [[ (dwtd_top_levels[i][j] - dwtd_top_medians[i][j])/dwtd_top_stds[i][j] for j in range(len(eligible_top_feats[i]))] for i in range(len(eligible_top_feats)) ]
            dwtd_top_explan_strings = [''.join([ ', ' + eligible_top_feat_informal_names [i][j] + scorer.descriptor_str(dwtd_top_abovemedian[i][j]) + str(dwtd_top_medians[i][j]) for j in range(len(eligible_top_feats[i]))]).lstrip(string.punctuation) for i in range(len(eligible_top_feats)) ]

            eligible_bot_feats = [[ bot_feats[i][j] for j in range(len(bot_feats[i])) if (('feat_most_recent_stage_prob' not in bot_feats[i][j]) and  ('onehotencoder' not in bot_feats[i][j]) )] for i in range(len(bot_feats)) ]
            eligible_bot_feat_informal_names = [[ feature_informal_names[feature_column_names.index(eligible_bot_feats[i][j])]for j in range(len(eligible_bot_feats[i]))] for i in range(len(eligible_bot_feats)) ]
            dwtd_bot_medians = [[ np.median(score_data.loc[(score_data['feat_most_recent_stage_prob']==score_data.loc[score_data.index[i],'feat_most_recent_stage_prob']),eligible_bot_feats[i][j]]) for j in range(len(eligible_bot_feats[i]))] for i in range(len(eligible_bot_feats)) ]
            dwtd_bot_stds = [[ np.std(score_data.loc[(score_data['feat_most_recent_stage_prob']==score_data.loc[score_data.index[i],'feat_most_recent_stage_prob']),eligible_bot_feats[i][j]]) for j in range(len(eligible_bot_feats[i]))] for i in range(len(eligible_bot_feats)) ]
            dwtd_bot_levels = [[ score_data.loc[score_data.index[i],eligible_bot_feats[i][j]] for j in range(len(eligible_bot_feats[i]))] for i in range(len(eligible_bot_feats)) ]
            dwtd_bot_abovemedian = [[ (dwtd_bot_levels[i][j] - dwtd_bot_medians[i][j])/dwtd_bot_stds[i][j] for j in range(len(eligible_bot_feats[i]))] for i in range(len(eligible_bot_feats)) ]
            dwtd_bot_explan_strings = [''.join([ ', ' + eligible_bot_feat_informal_names [i][j] + scorer.descriptor_str(dwtd_bot_abovemedian[i][j]) + str(dwtd_bot_medians[i][j]) for j in range(len(eligible_bot_feats[i]))]).lstrip(string.punctuation) for i in range(len(eligible_bot_feats)) ]

            #Use this model to score our data (probably just the 1-year-back stuff here:)
            # dataset_withscores_training_df[target_col + '_score'] = scores
            return [dwtd_top_explan_strings[i] + dwtd_bot_explan_strings[i] for i in range(len(dwtd_top_explan_strings))]

    def descriptor_str(self,sigmas_above_median):
        if sigmas_above_median > 1.5:
            return ' is significantly above the median level of '
        elif sigmas_above_median > 0.5:
            return ' is above the median level of '
        elif sigmas_above_median > 0.0:
            return ' is somewhat above the median level of '
        if sigmas_above_median > -0.5:
            return ' is somewhat below the median level of '
        if sigmas_above_median > -1.5:
            return ' is below the median level of '
        else:
            return ' is significantly below the median level of '

    #XYZ_health_scores: These functions "bin up" scores by their opp-stage (prob) and sometiems whether the
    #  expected close date is before or after the next quarter.  Then they yield the percentile level
    #  each score is at within those resulting populations.
    # clean_health just puts those scores into a useful 1-5 star integer range, Then we'll use these as
    #  a "health-score-scaler" when we do the final scoring.
    #return the scaled percentile levels of each score, among the records with their same stated probability
    @classmethod
    def generate_health_scores(cls,opp_stage_probs,raw_scores):
        opp_stage_probs = np.array(list(opp_stage_probs))
        raw_scores = np.array(list(raw_scores))
        percs = [ percentileofscore(raw_scores[np.where(opp_stage_probs == opp_stage_probs[i])],raw_scores[i] )   for i in range(len(opp_stage_probs))]
        return [(i - 50) * (2.75/50.0) for i in percs]

    # clean up the health scores
    # currently binning based on the score into 1-5 
    @classmethod
    def clean_health(cls,val):
        if val >= 2: return 5
        elif val >= 1: return 4
        elif val >= -1: return 3
        elif val >= -2: return 2
        elif val < -2: return 1
    
    # TODO: Consider re-doing this in a more complex way (allow quatratic calibration curve, or use standard deviations at each opp-stage.)
    #calculate the calibration factor from actual and calculated score sets:
    #From now on this calibration will automatically tune the scores.
    def calculate_calib_cofactor(self,data_valid,target_col,uncalib_score_col,calib_col):
        if data_valid is None:
            return 1.0
        else:
            plt.scatter(data_valid[target_col],data_valid[uncalib_score_col],alpha=0.02)
            plt.ylabel('Uncalibrated_score')
            plt.xlabel('Real Value')
            plt.show()
            #return np.sum(data_valid[target_col] * data_valid[calib_col] )/np.sum(data_valid[uncalib_score_col] * data_valid[calib_col])
            return np.sum(data_valid[target_col] / data_valid[calib_col] )/np.sum(data_valid[uncalib_score_col] / data_valid[calib_col])
