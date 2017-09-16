---
layout: post
title:  "Wi-Fi Fingerprint Indoor Localization (Part III): Classification and Cascade Prediction"
date:   2017-04-28 12:00:00 -0600
comments: true
---

In the [previous notebook](#UJIIndoorLoc-preprocess.ipynb), we performed various transformations on the independent variables of the raw UJIIndoorLoc dataset to prepare it for the machine learning.

In this notebook, first, I focus on our response variables including building ID, floor ID, latitiude and longitude. Understanding the class imbalance in classification responses buildingID and floorID is important for training our machine learning models. Similarly, I analyze the distributions of our regression response variables latitude and longitude and their relationship with the building ID and floor ID. Second, I formulate the localization problem for the machine learning. Finally, I begin constructing machine learning framework first by focusing on regression without Floor and Building information. In future notebooks, I will model and evaluate cascade machine learning frameworks that perform building and floor classification before applying building and floor-optimized regression models.


## 5. Building and Floor Classification

In this sub-section, I quickly evaluate the performance of various models for building classification, floor classification and per-building floor classification. My project partner Kumail found Random Forests with 100 trees to provide the best performance and I will primarily focus on it's performance.

Before I begin the evaluation, let's write a few functions that will come handy.


```python
def multiclass_roc(estimator, X_train,y_train,X_test,y_test,
                   classes_list,
                   decision = 'predict_proba',
                   title = None):

    # Binarize the output
    y_train = label_binarize(y_train, classes=classes_list)
    n_classes = y_train.shape[1]

    y_test = label_binarize(y_test, classes=classes_list)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(X_train, y_train)

    if decision == 'decision_function':
        y_score = classifier.decision_function(X_test)
    elif decision == 'predict_proba':
        y_score = classifier.predict_proba(X_test)   

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10,8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
```


```python
y_building_train = y_crossval['BUILDINGID']
y_building_test = y_holdout['BUILDINGID']

y_building_train.shape,y_building_test.shape
```




    ((17874,), (1987,))




```python
RANDOM_STATE = np.random.RandomState(0)

sm = SMOTE(random_state=RANDOM_STATE)
nm = NearMiss(version=2, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
pipe_rf = make_pipeline(nm,rf)

plot_learning_curves(X_train, y_building_train,
                     X_test, y_building_test,
                     pipe_rf, scoring = 'accuracy',
                     print_model = False)
plt.show()
```


![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_87_0.png)



```python
multiclass_roc(pipe_rf, X_train,y_building_train,X_test,y_building_test,
                   [0,1,2],
                   decision = 'predict_proba',
                   title = 'Random Forests per-building ROC curve')
```


![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_88_0.png)


## 6. Floor Classification


```python
y_floor_train = y_crossval['FLOOR']
y_floor_test = y_holdout['FLOOR']

RANDOM_STATE = np.random.RandomState(0)

sm = SMOTE(random_state=RANDOM_STATE)
nm = NearMiss(version=2, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
pipe_rf_floor = make_pipeline(nm,rf)
```


```python
plot_learning_curves(X_train, y_floor_train,
                     X_test, y_floor_test,
                     pipe_rf, scoring = 'accuracy',
                     print_model = False)
```




    ([0.58869613878007832,
      0.62171236709569111,
      0.63465124953375607,
      0.63379493635473494,
      0.64674946850173431,
      0.64350988437150314,
      0.64215490368475736,
      0.65039513252675007,
      0.6692776327241079,
      0.65262392301667227],
     [0.5787619526925013,
      0.60895822848515346,
      0.61852038248616004,
      0.62254655259184699,
      0.6406643180674384,
      0.65525918470055355,
      0.63714141922496226,
      0.64569703069954709,
      0.65827881227981877,
      0.64418721690991443])




![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_91_1.png)



```python
multiclass_roc(pipe_rf, X_train,y_floor_train,X_test,y_floor_test,
                   [0,1,2,3,4],
                   decision = 'predict_proba',
                   title = 'Random Forests per-Floor ROC curve')
```


![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_92_0.png)


### 6.1 Per-Building Floor Classification


```python
# b indicates building
X_crossval_b = {}
y_crossval_b = {}
X_holdout_b = {}
y_holdout_b = {}

for building in buildings:
    # Finding index of samples with the building and floor

    index_crossval_b = y_crossval[y_crossval.BUILDINGID == building].index
    index_holdout_b  = y_holdout[y_holdout.BUILDINGID == building].index

    if len(index_crossval_b) == 0:
        continue

    key = (building)

    X_crossval_b[key] = X_pca_crossval.loc[index_crossval_b]
    y_crossval_b[key] = y_crossval.loc[index_crossval_b,'FLOOR']

    X_holdout_b[key] = X_pca_holdout.loc[index_holdout_b]
    y_holdout_b[key] = y_holdout.loc[index_holdout_b,'FLOOR']

    print("Building = {}".format(building))
    print("Crossval shape", len(index_crossval_b))
    print("Holdout shape", len(index_holdout_b))

#X_crossval_b.keys(), X_holdout_b.keys(),
```

    Building = 2
    Crossval shape 8511
    Holdout shape 943
    Building = 0
    Crossval shape 4704
    Holdout shape 544
    Building = 1
    Crossval shape 4659
    Holdout shape 500



```python
clf_knn = Pipeline([('scl', StandardScaler()),
            ('reg', KNeighborsClassifier(n_neighbors=3,
                                        metric='manhattan',
                                        weights = 'distance'))])

# Random Forests
clf_rf = RandomForestClassifier(random_state=1, n_estimators=50)

# Extra Trees
clf_et = ExtraTreesClassifier(random_state=1,n_estimators=50)

clfs = [clf_knn,clf_rf,clf_et]
clf_labels = ['KNN','Random Forests','Extra Trees']

clf_dict = dict(zip(clf_labels, clfs))

best_local_crossval_b = {}
best_local_train_b = {}
best_local_holdout_b = {}
best_local_model = {}

for key in X_crossval_b:
    print("Building = {}".format(key))

    X_train_b = X_crossval_b[key]
    y_train_b = y_crossval_b[key]

    X_test_b = X_holdout_b[key]
    y_test_b = y_holdout_b[key]

    max_accu= 0

    for clf_label in clf_dict:
        model = clf_dict[clf_label]

        crossval_score = np.sqrt(np.abs((cross_val_score(estimator=model,
                                            X=X_train_b,
                                            y=y_train_b,
                                            cv=10,
                                            n_jobs=1,
                                            scoring = 'accuracy'))))

        if np.mean(crossval_score) > max_accu:
            best_local_model[key] = clf_label
            best_local_crossval_b[key] = crossval_score

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(best_local_crossval_b[key]),
                                              np.std(best_local_crossval_b[key])))

    # Best RMSE
    best_model = clf_dict[best_local_model[key]]
    best_model.fit(X_train_b,y_train_b)

    y_predict_train = best_model.predict(X_train_b)
    best_local_train_b[key] = accuracy_score(y_train_b, y_predict_train)
    print('Local Training Accuracy: %.3f' % (best_local_train_b[key]))

    y_predict_holdout = best_model.predict(X_test_b)
    best_local_holdout_b[key] = accuracy_score(y_test_b, y_predict_holdout)
    print('Local Holdout RMSE: %.3f' % (best_local_holdout_b[key]))
```

    Building = 0
    CV accuracy: 0.978 +/- 0.002
    Local Training Accuracy: 1.000
    Local Holdout RMSE: 0.943
    Building = 1
    CV accuracy: 0.986 +/- 0.004
    Local Training Accuracy: 1.000
    Local Holdout RMSE: 0.956
    Building = 2
    CV accuracy: 0.989 +/- 0.002
    Local Training Accuracy: 1.000
    Local Holdout RMSE: 0.975



```python
best_local_crossval_b = pd.Series(best_local_crossval_b)
best_local_floor_clf = (pd.concat([pd.Series(best_local_model),
                                   best_local_crossval_b.apply(np.mean),
                                   best_local_crossval_b.apply(np.std),
                                   pd.Series(best_local_train_b),
                                   pd.Series(best_local_holdout_b)],
                                  axis=1))

best_local_floor_clf.columns = ['BEST_LOCAL_MODEL','BEST_LOCAL_CROSSVAL_MEAN','BEST_LOCAL_CROSSVAL_STD',
                         'BEST_LOCAL_TRAINING_ACCURACY','BEST_LOCAL_HOLDOUT_ACCURACY']
best_local_floor_clf = best_local_floor_clf.rename_axis(['BUILDING'])

best_local_floor_clf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BEST_LOCAL_MODEL</th>
      <th>BEST_LOCAL_CROSSVAL_MEAN</th>
      <th>BEST_LOCAL_CROSSVAL_STD</th>
      <th>BEST_LOCAL_TRAINING_ACCURACY</th>
      <th>BEST_LOCAL_HOLDOUT_ACCURACY</th>
    </tr>
    <tr>
      <th>BUILDING</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>0.978295</td>
      <td>0.002010</td>
      <td>1.0</td>
      <td>0.943015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>0.986483</td>
      <td>0.004177</td>
      <td>1.0</td>
      <td>0.956000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.989367</td>
      <td>0.002324</td>
      <td>1.0</td>
      <td>0.974549</td>
    </tr>
  </tbody>
</table>
</div>



We observe the per-building Floor Classification greatly improves the accuracy!


```python
# First, let's save our data into a file
f = open("best_local_floor_clf.pckl", "wb")
pickle.dump(best_local_floor_clf,f)
```

## 7. The Final Problem

In this final sub-section, I perform the complete cascade modeling and evaluation of our indoor positioning problem. For that purpose, first our models need to be trained combining the training and holdout set.

1. Building Classification

2. Per-Building Floor Classification

3. Per-Building and Per-Floor Regression


```python
X_train = pd.concat([X_pca_crossval,X_pca_holdout],axis=0)
y_train = pd.concat([y_crossval,y_holdout],axis=0)

## Final test data
X_test = pd.read_csv("data/X_pca_test.csv",index_col=0)
y_test = pd.read_csv("data/y_test.csv",index_col=0)

record_train = pd.concat([X_train,y_train],axis=1)
record_test = pd.concat([X_test,y_test],axis=1)
```


```python
X_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>140</th>
      <th>141</th>
      <th>142</th>
      <th>143</th>
      <th>144</th>
      <th>145</th>
      <th>146</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.163364</td>
      <td>-1.309429</td>
      <td>0.618297</td>
      <td>0.220110</td>
      <td>0.224450</td>
      <td>0.865842</td>
      <td>-0.308312</td>
      <td>-0.467345</td>
      <td>-0.688095</td>
      <td>-0.525474</td>
      <td>...</td>
      <td>-0.283669</td>
      <td>0.013979</td>
      <td>0.177229</td>
      <td>-0.234990</td>
      <td>0.016964</td>
      <td>-0.455687</td>
      <td>-0.156325</td>
      <td>-0.165567</td>
      <td>0.007310</td>
      <td>0.014424</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.129236</td>
      <td>1.391735</td>
      <td>0.140099</td>
      <td>0.214359</td>
      <td>0.029103</td>
      <td>1.705419</td>
      <td>-0.687835</td>
      <td>3.242107</td>
      <td>0.674838</td>
      <td>1.155327</td>
      <td>...</td>
      <td>-0.008547</td>
      <td>-0.115510</td>
      <td>-0.153102</td>
      <td>-0.017840</td>
      <td>0.173631</td>
      <td>-0.107054</td>
      <td>0.228907</td>
      <td>0.087466</td>
      <td>-0.128199</td>
      <td>0.319020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.043640</td>
      <td>0.573283</td>
      <td>0.095193</td>
      <td>0.246763</td>
      <td>-0.189658</td>
      <td>1.579516</td>
      <td>-0.002478</td>
      <td>2.107065</td>
      <td>-1.490188</td>
      <td>2.065422</td>
      <td>...</td>
      <td>-0.003664</td>
      <td>-0.183146</td>
      <td>0.053817</td>
      <td>0.059181</td>
      <td>-0.105436</td>
      <td>-0.413949</td>
      <td>0.484042</td>
      <td>-0.934152</td>
      <td>-0.003869</td>
      <td>0.126402</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.235647</td>
      <td>0.929559</td>
      <td>0.138743</td>
      <td>0.276892</td>
      <td>-0.160512</td>
      <td>1.115693</td>
      <td>-0.491587</td>
      <td>3.207946</td>
      <td>-2.449852</td>
      <td>1.735254</td>
      <td>...</td>
      <td>0.112100</td>
      <td>0.080984</td>
      <td>0.133420</td>
      <td>-0.056582</td>
      <td>0.319027</td>
      <td>-0.167541</td>
      <td>-0.250058</td>
      <td>0.789169</td>
      <td>-0.140382</td>
      <td>-0.017589</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.776334</td>
      <td>-5.780305</td>
      <td>-4.452061</td>
      <td>-6.585396</td>
      <td>-1.060150</td>
      <td>-0.262021</td>
      <td>-0.322183</td>
      <td>0.071958</td>
      <td>0.092279</td>
      <td>-0.441068</td>
      <td>...</td>
      <td>1.703087</td>
      <td>-0.215988</td>
      <td>0.380231</td>
      <td>0.655841</td>
      <td>0.322163</td>
      <td>0.175419</td>
      <td>0.038403</td>
      <td>0.020690</td>
      <td>0.066572</td>
      <td>0.078180</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 150 columns</p>
</div>



### 7.1 Model Fitting


```python
# Building Fitting

RANDOM_STATE = np.random.RandomState(0)

sm = SMOTE(random_state=RANDOM_STATE)
nm = NearMiss(version=2, random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

# Building
building_clf = make_pipeline(nm,rf)
building_clf.fit(X_train, y_train['BUILDINGID'])
```




    Pipeline(steps=[('nearmiss', NearMiss(n_jobs=1, n_neighbors=3, n_neighbors_ver3=3,
         random_state=<mtrand.RandomState object at 0x10ff9fee8>, ratio='auto',
         return_indices=False, size_ngh=None, ver3_samp_ngh=None, version=2)), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, class_wei... random_state=<mtrand.RandomState object at 0x10ff9fee8>,
                verbose=0, warm_start=False))])




```python
# Per-Building Floor Fitting

floor_clf_models = {}

for building in buildings:
    #model = clf_dict[model]
    model = clf_dict[best_local_floor_clf.loc[building]['BEST_LOCAL_MODEL']]
    print(model)

    X_train_b = pd.concat([X_crossval_b[building],X_holdout_b[building]],axis=0)
    y_train_b = pd.concat([y_crossval_b[building],y_holdout_b[building]],axis=0)

    model.fit(X_train_b,y_train_b)
    floor_clf_models[building] = model
```

    Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reg', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='distance'))])
    Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reg', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='distance'))])
    Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reg', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='distance'))])



```python
# Per-Building Per-Floor Regression Fitting

regression_models = {}

for key in X_crossval_bf:
    model = reg_dict[best_local_metric_df.loc[key[0],key[1]]['BEST_LOCAL_MODEL']]

    X_train_bf = pd.concat([X_crossval_bf[key],X_holdout_bf[key]],axis=0)
    y_train_bf = pd.concat([y_crossval_bf[key],y_holdout_bf[key]],axis=0)

    model.fit(X_train_bf,y_train_bf)
    regression_models[key] = model
```

### 7.2 Localization Prediction

For each sample in the training and test data, we compute the error metric given by

$positioning\_error(actual,predicted)= euclidean\_distance(actual,predicted) + penalty_{floor}*fail_{floor} + penalty_{building}*fail_{building}$


```python
gs_knn_best.fit(X_train,y_train[['LATITUDE','LONGITUDE']])
```




    Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reg', KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='manhattan',
              metric_params=None, n_jobs=-1, n_neighbors=3, p=2,
              weights='distance'))])




```python
def compute_final_error(record):    
    penalty_building = 50
    penalty_floor = 4

    X = record.iloc[:150]
    y = record.iloc[150:]

    # Building classification
    building_pred = building_clf.predict(X)[0]
    #print(building_pred)

    # Floor Classification
    floor_pred = floor_clf_models[building].predict(X)[0]
    #print(floor_pred)

    # Latitude and Longitude Prediction
    #lat_long_pred = regression_models[(building_pred,floor_pred)].predict(X)
    lat_long_pred = gs_knn_best.predict(X)
    lat_pred = lat_long_pred[0][0]
    long_pred = lat_long_pred[0][1]

    penalty = 0

    # Building Penalty
    if building_pred != y['BUILDINGID']:
        penalty += penalty_building

    #print("After building: ", penalty)

    # Floor Penalty
    if floor_pred != y['FLOOR']:
        penalty += penalty_floor

    #print("After floor: ", penalty)

    # Latitude-Longitude Penalty
    lat_actual, long_actual = y['LATITUDE'], y['LONGITUDE']
    #penalty += np.sqrt(((lat_long_pred - y[['LATITUDE','LONGITUDE']])**2).sum(axis=1))
    penalty += np.sqrt((long_pred - long_actual) ** 2 + (lat_pred - lat_actual) ** 2)

    return penalty
```


```python
record_train['ERROR'] = record_train.apply(compute_final_error,axis = 1)
record_test['ERROR'] = record_test.apply(compute_final_error,axis = 1)
```


```python
record_test.head(50)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>147</th>
      <th>148</th>
      <th>149</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>FLOOR</th>
      <th>BUILDINGID</th>
      <th>SPACEID</th>
      <th>RELATIVEPOSITION</th>
      <th>ERROR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.163364</td>
      <td>-1.309429</td>
      <td>0.618297</td>
      <td>0.220110</td>
      <td>0.224450</td>
      <td>0.865842</td>
      <td>-0.308312</td>
      <td>-0.467345</td>
      <td>-0.688095</td>
      <td>-0.525474</td>
      <td>...</td>
      <td>-0.165567</td>
      <td>0.007310</td>
      <td>0.014424</td>
      <td>-7515.916799</td>
      <td>4.864890e+06</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>16.920579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.129236</td>
      <td>1.391735</td>
      <td>0.140099</td>
      <td>0.214359</td>
      <td>0.029103</td>
      <td>1.705419</td>
      <td>-0.687835</td>
      <td>3.242107</td>
      <td>0.674838</td>
      <td>1.155327</td>
      <td>...</td>
      <td>0.087466</td>
      <td>-0.128199</td>
      <td>0.319020</td>
      <td>-7383.867221</td>
      <td>4.864840e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>20.125179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.043640</td>
      <td>0.573283</td>
      <td>0.095193</td>
      <td>0.246763</td>
      <td>-0.189658</td>
      <td>1.579516</td>
      <td>-0.002478</td>
      <td>2.107065</td>
      <td>-1.490188</td>
      <td>2.065422</td>
      <td>...</td>
      <td>-0.934152</td>
      <td>-0.003869</td>
      <td>0.126402</td>
      <td>-7374.302080</td>
      <td>4.864847e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.461367</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.235647</td>
      <td>0.929559</td>
      <td>0.138743</td>
      <td>0.276892</td>
      <td>-0.160512</td>
      <td>1.115693</td>
      <td>-0.491587</td>
      <td>3.207946</td>
      <td>-2.449852</td>
      <td>1.735254</td>
      <td>...</td>
      <td>0.789169</td>
      <td>-0.140382</td>
      <td>-0.017589</td>
      <td>-7365.824883</td>
      <td>4.864843e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>15.805552</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.776334</td>
      <td>-5.780305</td>
      <td>-4.452061</td>
      <td>-6.585396</td>
      <td>-1.060150</td>
      <td>-0.262021</td>
      <td>-0.322183</td>
      <td>0.071958</td>
      <td>0.092279</td>
      <td>-0.441068</td>
      <td>...</td>
      <td>0.020690</td>
      <td>0.066572</td>
      <td>0.078180</td>
      <td>-7641.499303</td>
      <td>4.864922e+06</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.350470</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.504230</td>
      <td>4.042574</td>
      <td>-0.102819</td>
      <td>-0.144219</td>
      <td>-0.106406</td>
      <td>-0.398325</td>
      <td>0.070869</td>
      <td>0.161131</td>
      <td>-3.288956</td>
      <td>1.101974</td>
      <td>...</td>
      <td>-0.711891</td>
      <td>0.114770</td>
      <td>0.002351</td>
      <td>-7338.807210</td>
      <td>4.864825e+06</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>14.860020</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-2.483697</td>
      <td>4.544346</td>
      <td>-0.012430</td>
      <td>0.030243</td>
      <td>-0.225547</td>
      <td>2.839085</td>
      <td>0.091003</td>
      <td>2.958616</td>
      <td>-0.234515</td>
      <td>5.498340</td>
      <td>...</td>
      <td>-1.077523</td>
      <td>1.141564</td>
      <td>-0.758575</td>
      <td>-7379.351683</td>
      <td>4.864849e+06</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6.478737</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.173294</td>
      <td>3.997709</td>
      <td>-0.208364</td>
      <td>-0.447606</td>
      <td>0.471752</td>
      <td>-1.262361</td>
      <td>-1.153710</td>
      <td>3.809793</td>
      <td>-1.239465</td>
      <td>-3.290923</td>
      <td>...</td>
      <td>-0.632331</td>
      <td>0.037082</td>
      <td>0.262750</td>
      <td>-7340.558777</td>
      <td>4.864759e+06</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>22.704712</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-2.929266</td>
      <td>6.099108</td>
      <td>-0.339470</td>
      <td>-0.733101</td>
      <td>0.656938</td>
      <td>-1.585132</td>
      <td>-1.182537</td>
      <td>3.369852</td>
      <td>1.197174</td>
      <td>-0.623949</td>
      <td>...</td>
      <td>-0.322556</td>
      <td>-0.108400</td>
      <td>0.268637</td>
      <td>-7357.531253</td>
      <td>4.864766e+06</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>42.695660</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.578400</td>
      <td>4.387787</td>
      <td>-0.306922</td>
      <td>-0.392237</td>
      <td>-0.132843</td>
      <td>-1.588002</td>
      <td>1.398522</td>
      <td>-6.137630</td>
      <td>1.532778</td>
      <td>-0.997848</td>
      <td>...</td>
      <td>-0.652459</td>
      <td>0.670900</td>
      <td>-0.515416</td>
      <td>-7345.085170</td>
      <td>4.864831e+06</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>17.891245</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-2.655942</td>
      <td>5.279138</td>
      <td>-0.296905</td>
      <td>-0.614430</td>
      <td>0.523824</td>
      <td>-1.498157</td>
      <td>-0.946050</td>
      <td>2.682172</td>
      <td>0.526420</td>
      <td>-1.136411</td>
      <td>...</td>
      <td>0.125958</td>
      <td>0.601778</td>
      <td>-0.353713</td>
      <td>-7344.182657</td>
      <td>4.864754e+06</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>22.229916</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-2.761414</td>
      <td>5.202649</td>
      <td>-0.371969</td>
      <td>-0.341600</td>
      <td>-0.372103</td>
      <td>-0.366171</td>
      <td>1.883151</td>
      <td>-6.017568</td>
      <td>2.323409</td>
      <td>0.920669</td>
      <td>...</td>
      <td>0.914770</td>
      <td>0.496106</td>
      <td>-0.899793</td>
      <td>-7372.664041</td>
      <td>4.864844e+06</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.209025</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-2.327280</td>
      <td>3.982863</td>
      <td>-0.279312</td>
      <td>-0.155028</td>
      <td>-0.487564</td>
      <td>0.127554</td>
      <td>1.965946</td>
      <td>-5.950479</td>
      <td>1.975299</td>
      <td>1.072088</td>
      <td>...</td>
      <td>0.431744</td>
      <td>-0.034116</td>
      <td>-0.497946</td>
      <td>-7377.067905</td>
      <td>4.864849e+06</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4.457323</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.963770</td>
      <td>3.271962</td>
      <td>-0.165783</td>
      <td>-0.278748</td>
      <td>0.164413</td>
      <td>-0.598598</td>
      <td>-0.148547</td>
      <td>0.307444</td>
      <td>1.194059</td>
      <td>-0.394381</td>
      <td>...</td>
      <td>0.138220</td>
      <td>-0.087859</td>
      <td>-0.231822</td>
      <td>-7331.100100</td>
      <td>4.864767e+06</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>51.888749</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-2.213421</td>
      <td>3.763820</td>
      <td>-0.222161</td>
      <td>-0.178405</td>
      <td>-0.320767</td>
      <td>0.150164</td>
      <td>1.641907</td>
      <td>-5.524932</td>
      <td>2.559093</td>
      <td>-0.074373</td>
      <td>...</td>
      <td>0.857767</td>
      <td>1.018078</td>
      <td>-1.238699</td>
      <td>-7385.871536</td>
      <td>4.864840e+06</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>7.474282</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-1.957033</td>
      <td>3.017905</td>
      <td>-0.146938</td>
      <td>-0.200150</td>
      <td>-0.006737</td>
      <td>-0.546971</td>
      <td>0.563973</td>
      <td>-2.481107</td>
      <td>1.133229</td>
      <td>1.109803</td>
      <td>...</td>
      <td>0.183686</td>
      <td>-0.186727</td>
      <td>0.136293</td>
      <td>-7393.435150</td>
      <td>4.864837e+06</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>57.166238</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.024677</td>
      <td>-5.215182</td>
      <td>10.645851</td>
      <td>-1.153351</td>
      <td>-5.733261</td>
      <td>-3.471800</td>
      <td>2.728561</td>
      <td>1.692353</td>
      <td>1.132387</td>
      <td>-0.230960</td>
      <td>...</td>
      <td>-0.275632</td>
      <td>-0.622804</td>
      <td>-0.541807</td>
      <td>-7559.678074</td>
      <td>4.864887e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.052196</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.360134</td>
      <td>2.464626</td>
      <td>0.226828</td>
      <td>-0.517850</td>
      <td>2.012452</td>
      <td>0.435149</td>
      <td>-3.355717</td>
      <td>5.852363</td>
      <td>6.898578</td>
      <td>-2.680229</td>
      <td>...</td>
      <td>0.953456</td>
      <td>-0.293292</td>
      <td>-0.045549</td>
      <td>-7414.450283</td>
      <td>4.864788e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.523325</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-2.454417</td>
      <td>4.661757</td>
      <td>-0.336508</td>
      <td>-0.495909</td>
      <td>0.116747</td>
      <td>-1.695365</td>
      <td>0.599785</td>
      <td>-3.484059</td>
      <td>2.588495</td>
      <td>-1.668927</td>
      <td>...</td>
      <td>0.731987</td>
      <td>0.777425</td>
      <td>-0.641325</td>
      <td>-7402.404284</td>
      <td>4.864806e+06</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>56.222774</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-2.207846</td>
      <td>3.683321</td>
      <td>-0.191143</td>
      <td>-0.271092</td>
      <td>0.005528</td>
      <td>-0.398883</td>
      <td>0.574555</td>
      <td>-2.089641</td>
      <td>1.576963</td>
      <td>2.239112</td>
      <td>...</td>
      <td>-0.554741</td>
      <td>-1.186625</td>
      <td>1.329196</td>
      <td>-7358.379579</td>
      <td>4.864837e+06</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>12.389318</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.659957</td>
      <td>-5.649527</td>
      <td>12.683451</td>
      <td>-3.339278</td>
      <td>-1.709504</td>
      <td>-4.349243</td>
      <td>8.929624</td>
      <td>3.099215</td>
      <td>1.660353</td>
      <td>-0.359571</td>
      <td>...</td>
      <td>0.027025</td>
      <td>-0.899899</td>
      <td>-0.578598</td>
      <td>-7558.859014</td>
      <td>4.864871e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9.467753</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.379455</td>
      <td>-1.616752</td>
      <td>-0.076774</td>
      <td>1.616193</td>
      <td>0.336750</td>
      <td>0.377830</td>
      <td>0.287480</td>
      <td>-0.077532</td>
      <td>-0.197827</td>
      <td>-0.913429</td>
      <td>...</td>
      <td>0.068135</td>
      <td>-0.109322</td>
      <td>-0.093770</td>
      <td>-7586.944816</td>
      <td>4.864986e+06</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>59.604483</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.393003</td>
      <td>3.057116</td>
      <td>0.252421</td>
      <td>-0.607130</td>
      <td>1.912155</td>
      <td>0.058216</td>
      <td>-3.025175</td>
      <td>5.130354</td>
      <td>6.527319</td>
      <td>-3.067726</td>
      <td>...</td>
      <td>0.041624</td>
      <td>-0.024235</td>
      <td>-0.547041</td>
      <td>-7410.584706</td>
      <td>4.864793e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>16.149533</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.690813</td>
      <td>2.682713</td>
      <td>0.214080</td>
      <td>-0.449441</td>
      <td>1.525425</td>
      <td>0.394383</td>
      <td>-2.743992</td>
      <td>6.123664</td>
      <td>5.206984</td>
      <td>-1.272807</td>
      <td>...</td>
      <td>-0.207608</td>
      <td>-0.342996</td>
      <td>0.339735</td>
      <td>-7394.226370</td>
      <td>4.864821e+06</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>41.564300</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.011255</td>
      <td>-6.217726</td>
      <td>13.903561</td>
      <td>-5.137462</td>
      <td>0.630790</td>
      <td>-4.990981</td>
      <td>12.225601</td>
      <td>3.645552</td>
      <td>1.356606</td>
      <td>-0.041414</td>
      <td>...</td>
      <td>0.590590</td>
      <td>1.167636</td>
      <td>1.160573</td>
      <td>-7568.985890</td>
      <td>4.864876e+06</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17.632690</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.193696</td>
      <td>-5.502712</td>
      <td>12.258207</td>
      <td>-5.201089</td>
      <td>2.574098</td>
      <td>-4.309809</td>
      <td>11.919660</td>
      <td>3.326382</td>
      <td>1.128274</td>
      <td>0.171792</td>
      <td>...</td>
      <td>-0.025117</td>
      <td>0.182480</td>
      <td>0.284191</td>
      <td>-7562.717100</td>
      <td>4.864866e+06</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4.971026</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-2.727191</td>
      <td>5.116988</td>
      <td>-0.248507</td>
      <td>-0.306127</td>
      <td>-0.176202</td>
      <td>0.542563</td>
      <td>1.330249</td>
      <td>-4.022283</td>
      <td>2.643586</td>
      <td>2.341796</td>
      <td>...</td>
      <td>0.801814</td>
      <td>0.076026</td>
      <td>-0.600433</td>
      <td>-7385.478494</td>
      <td>4.864838e+06</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>22.804966</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.279353</td>
      <td>3.668950</td>
      <td>-1.218911</td>
      <td>0.423161</td>
      <td>-1.955741</td>
      <td>0.596882</td>
      <td>2.195919</td>
      <td>-2.236994</td>
      <td>1.894694</td>
      <td>2.104408</td>
      <td>...</td>
      <td>0.801478</td>
      <td>0.756471</td>
      <td>0.397603</td>
      <td>-7397.033907</td>
      <td>4.864829e+06</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>28.726957</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.876248</td>
      <td>-6.382803</td>
      <td>-5.565912</td>
      <td>-5.705165</td>
      <td>-0.435607</td>
      <td>-0.441643</td>
      <td>-0.850032</td>
      <td>-0.094175</td>
      <td>0.038734</td>
      <td>0.206034</td>
      <td>...</td>
      <td>-0.021584</td>
      <td>0.194159</td>
      <td>0.239444</td>
      <td>-7674.785283</td>
      <td>4.864934e+06</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8.776038</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.857991</td>
      <td>-6.439435</td>
      <td>-5.484258</td>
      <td>-7.255466</td>
      <td>-0.954089</td>
      <td>-0.353486</td>
      <td>-0.667904</td>
      <td>-0.006557</td>
      <td>0.093347</td>
      <td>0.050152</td>
      <td>...</td>
      <td>0.126001</td>
      <td>0.286110</td>
      <td>0.487450</td>
      <td>-7656.475561</td>
      <td>4.864938e+06</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13.164138</td>
    </tr>
    <tr>
      <th>30</th>
      <td>-0.611572</td>
      <td>-3.535902</td>
      <td>-1.443688</td>
      <td>5.536384</td>
      <td>2.299946</td>
      <td>-0.944985</td>
      <td>0.688623</td>
      <td>0.120723</td>
      <td>0.064531</td>
      <td>-0.539900</td>
      <td>...</td>
      <td>0.046276</td>
      <td>-0.045512</td>
      <td>0.004632</td>
      <td>-7642.764986</td>
      <td>4.865005e+06</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.066918</td>
    </tr>
    <tr>
      <th>31</th>
      <td>-0.352496</td>
      <td>-4.125927</td>
      <td>7.021198</td>
      <td>0.751206</td>
      <td>-7.271909</td>
      <td>-1.952998</td>
      <td>-2.752468</td>
      <td>0.025893</td>
      <td>0.715191</td>
      <td>-0.290487</td>
      <td>...</td>
      <td>0.560198</td>
      <td>0.950462</td>
      <td>0.747049</td>
      <td>-7539.960817</td>
      <td>4.864920e+06</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.553796</td>
    </tr>
    <tr>
      <th>32</th>
      <td>-0.436716</td>
      <td>-3.603922</td>
      <td>5.655212</td>
      <td>1.658612</td>
      <td>-7.852439</td>
      <td>-1.188408</td>
      <td>-5.589602</td>
      <td>-0.860828</td>
      <td>0.287203</td>
      <td>-0.233510</td>
      <td>...</td>
      <td>0.327938</td>
      <td>0.511981</td>
      <td>0.515935</td>
      <td>-7525.314098</td>
      <td>4.864933e+06</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.366263</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-0.470452</td>
      <td>-4.003593</td>
      <td>5.951629</td>
      <td>2.389388</td>
      <td>-8.149318</td>
      <td>-1.391480</td>
      <td>-6.033577</td>
      <td>-0.939974</td>
      <td>0.283857</td>
      <td>-0.479516</td>
      <td>...</td>
      <td>-0.251361</td>
      <td>-0.307613</td>
      <td>-0.168068</td>
      <td>-7533.086327</td>
      <td>4.864938e+06</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.664836</td>
    </tr>
    <tr>
      <th>34</th>
      <td>-0.406119</td>
      <td>-3.804396</td>
      <td>6.279975</td>
      <td>1.407328</td>
      <td>-7.754700</td>
      <td>-1.394509</td>
      <td>-4.669328</td>
      <td>-0.578015</td>
      <td>0.384089</td>
      <td>-0.380336</td>
      <td>...</td>
      <td>0.025659</td>
      <td>0.040016</td>
      <td>0.182426</td>
      <td>-7521.608767</td>
      <td>4.864951e+06</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>16.771997</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2.451471</td>
      <td>-0.562586</td>
      <td>0.228143</td>
      <td>1.397301</td>
      <td>-1.657560</td>
      <td>11.119765</td>
      <td>2.322657</td>
      <td>0.673441</td>
      <td>1.462958</td>
      <td>-0.413364</td>
      <td>...</td>
      <td>-0.263059</td>
      <td>0.345298</td>
      <td>-0.204539</td>
      <td>-7424.812344</td>
      <td>4.864863e+06</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>24.026381</td>
    </tr>
    <tr>
      <th>36</th>
      <td>5.575980</td>
      <td>0.078463</td>
      <td>1.735410</td>
      <td>-0.650695</td>
      <td>4.022007</td>
      <td>4.920610</td>
      <td>-3.731345</td>
      <td>2.279152</td>
      <td>7.842826</td>
      <td>-2.872385</td>
      <td>...</td>
      <td>0.450539</td>
      <td>0.539838</td>
      <td>0.576401</td>
      <td>-7457.360650</td>
      <td>4.864831e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.144679</td>
    </tr>
    <tr>
      <th>37</th>
      <td>5.172227</td>
      <td>0.822983</td>
      <td>1.445478</td>
      <td>-1.114108</td>
      <td>4.840692</td>
      <td>1.033110</td>
      <td>-5.312631</td>
      <td>3.018514</td>
      <td>8.297729</td>
      <td>-2.836130</td>
      <td>...</td>
      <td>-0.431727</td>
      <td>0.616720</td>
      <td>1.513380</td>
      <td>-7460.452553</td>
      <td>4.864816e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.259578</td>
    </tr>
    <tr>
      <th>38</th>
      <td>4.085010</td>
      <td>0.703727</td>
      <td>1.660802</td>
      <td>-1.159088</td>
      <td>4.959997</td>
      <td>1.264625</td>
      <td>-5.144358</td>
      <td>2.552580</td>
      <td>7.396631</td>
      <td>-2.717711</td>
      <td>...</td>
      <td>-0.027003</td>
      <td>0.005075</td>
      <td>-0.285417</td>
      <td>-7460.037216</td>
      <td>4.864814e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8.135276</td>
    </tr>
    <tr>
      <th>39</th>
      <td>8.158658</td>
      <td>0.347535</td>
      <td>2.489705</td>
      <td>-1.643593</td>
      <td>6.159756</td>
      <td>0.772554</td>
      <td>-5.274828</td>
      <td>1.114975</td>
      <td>5.866219</td>
      <td>-1.515413</td>
      <td>...</td>
      <td>-0.229600</td>
      <td>0.782967</td>
      <td>-0.170419</td>
      <td>-7468.448112</td>
      <td>4.864818e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>13.489538</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.169497</td>
      <td>-1.023079</td>
      <td>1.280157</td>
      <td>1.183242</td>
      <td>-1.089330</td>
      <td>9.961731</td>
      <td>0.657944</td>
      <td>0.761521</td>
      <td>0.845453</td>
      <td>0.341669</td>
      <td>...</td>
      <td>0.024017</td>
      <td>-0.002822</td>
      <td>0.029337</td>
      <td>-7423.096844</td>
      <td>4.864878e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17.782138</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.171965</td>
      <td>-1.041229</td>
      <td>0.663425</td>
      <td>1.255742</td>
      <td>-1.143131</td>
      <td>9.801861</td>
      <td>1.588196</td>
      <td>0.686471</td>
      <td>1.249440</td>
      <td>-0.191640</td>
      <td>...</td>
      <td>0.125665</td>
      <td>-0.034248</td>
      <td>0.394542</td>
      <td>-7423.176200</td>
      <td>4.864896e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6.300249</td>
    </tr>
    <tr>
      <th>42</th>
      <td>-0.453924</td>
      <td>-4.206511</td>
      <td>7.372738</td>
      <td>1.048531</td>
      <td>-7.990413</td>
      <td>-1.862868</td>
      <td>-3.506348</td>
      <td>-0.041634</td>
      <td>0.505341</td>
      <td>-0.328532</td>
      <td>...</td>
      <td>0.131880</td>
      <td>-0.052967</td>
      <td>-0.029714</td>
      <td>-7538.841262</td>
      <td>4.864920e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.773671</td>
    </tr>
    <tr>
      <th>43</th>
      <td>-0.172258</td>
      <td>-4.318246</td>
      <td>7.366410</td>
      <td>-0.070108</td>
      <td>-5.766632</td>
      <td>-2.074357</td>
      <td>-0.834247</td>
      <td>0.410382</td>
      <td>0.663192</td>
      <td>-0.529542</td>
      <td>...</td>
      <td>0.265341</td>
      <td>0.645282</td>
      <td>0.216422</td>
      <td>-7538.918079</td>
      <td>4.864920e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.621833</td>
    </tr>
    <tr>
      <th>44</th>
      <td>-0.397273</td>
      <td>-3.989762</td>
      <td>7.021180</td>
      <td>0.553884</td>
      <td>-6.482952</td>
      <td>-1.787542</td>
      <td>-1.492962</td>
      <td>0.388273</td>
      <td>0.532106</td>
      <td>-0.514848</td>
      <td>...</td>
      <td>0.234182</td>
      <td>0.252943</td>
      <td>0.244591</td>
      <td>-7530.756500</td>
      <td>4.864922e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3.411329</td>
    </tr>
    <tr>
      <th>45</th>
      <td>-0.324643</td>
      <td>-3.850870</td>
      <td>6.103062</td>
      <td>0.168159</td>
      <td>-5.385025</td>
      <td>-1.575978</td>
      <td>-0.677392</td>
      <td>0.401478</td>
      <td>0.584686</td>
      <td>-0.792887</td>
      <td>...</td>
      <td>-0.051296</td>
      <td>0.130112</td>
      <td>0.065532</td>
      <td>-7521.829906</td>
      <td>4.864951e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>33.708080</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.325796</td>
      <td>-4.352615</td>
      <td>8.822327</td>
      <td>-2.382303</td>
      <td>-1.184922</td>
      <td>-2.914901</td>
      <td>6.736432</td>
      <td>2.273275</td>
      <td>1.160312</td>
      <td>-0.476186</td>
      <td>...</td>
      <td>-0.424977</td>
      <td>-1.047950</td>
      <td>-0.864736</td>
      <td>-7555.422550</td>
      <td>4.864886e+06</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.876299</td>
    </tr>
    <tr>
      <th>47</th>
      <td>-1.740780</td>
      <td>2.204319</td>
      <td>0.008299</td>
      <td>0.081431</td>
      <td>-0.154699</td>
      <td>1.561404</td>
      <td>-0.046771</td>
      <td>3.047253</td>
      <td>-2.965622</td>
      <td>4.011670</td>
      <td>...</td>
      <td>-1.319449</td>
      <td>-0.459669</td>
      <td>0.766223</td>
      <td>-7366.085279</td>
      <td>4.864843e+06</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6.995463</td>
    </tr>
    <tr>
      <th>48</th>
      <td>-1.953307</td>
      <td>3.099470</td>
      <td>-0.026155</td>
      <td>-0.100644</td>
      <td>0.148668</td>
      <td>-0.220857</td>
      <td>-0.830458</td>
      <td>3.141564</td>
      <td>-2.683536</td>
      <td>0.596444</td>
      <td>...</td>
      <td>0.854718</td>
      <td>0.293087</td>
      <td>-0.490817</td>
      <td>-7336.648692</td>
      <td>4.864827e+06</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5.049464</td>
    </tr>
    <tr>
      <th>49</th>
      <td>-1.332452</td>
      <td>1.015545</td>
      <td>0.113763</td>
      <td>0.232714</td>
      <td>-0.163153</td>
      <td>0.325559</td>
      <td>-0.518840</td>
      <td>3.010195</td>
      <td>-5.417671</td>
      <td>0.236935</td>
      <td>...</td>
      <td>-0.153477</td>
      <td>-0.455434</td>
      <td>0.532333</td>
      <td>-7323.657903</td>
      <td>4.864819e+06</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2.669695</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 157 columns</p>
</div>




```python
eval_train = record_train.iloc[:,152:]
eval_test = record_test.iloc[:,152:]

eval_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FLOOR</th>
      <th>BUILDINGID</th>
      <th>SPACEID</th>
      <th>RELATIVEPOSITION</th>
      <th>ERROR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14651</th>
      <td>1</td>
      <td>2</td>
      <td>117</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>16771</th>
      <td>3</td>
      <td>0</td>
      <td>108</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17601</th>
      <td>3</td>
      <td>0</td>
      <td>107</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6651</th>
      <td>0</td>
      <td>2</td>
      <td>103</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>3</td>
      <td>2</td>
      <td>222</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
(sns.factorplot(x="FLOOR",y="ERROR",hue="BUILDINGID",
               kind="violin",data=eval_train,
              size = 10,
              aspect = 1.0,
              legend_out = False))
plt.ylim([-3,15])
plt.xlim([-1,6])
plt.legend(loc='lower right', fontsize=20, title='BUILDING ID')
```




    <matplotlib.legend.Legend at 0x1351cc5f8>




![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_114_1.png)



```python
(sns.factorplot(x="FLOOR",y="ERROR",hue="BUILDINGID",
               kind="box",data=eval_test,
              size = 10,
              aspect = 1.0,
              legend_out = False))
plt.ylim([-5,40])
plt.xlim([-1,6])
plt.legend(loc='lower right', fontsize=20, title='BUILDING ID')
```




    <matplotlib.legend.Legend at 0x136f87320>




![png](UJIIndoorLoc-machine-learning_files/UJIIndoorLoc-machine-learning_115_1.png)


[wlan-loc-1]: https://sharan-naribole.github.io/2017/03/29/ujiindoorloc-part-I.html
[wlan-loc-2]: https://sharan-naribole.github.io/2017/04/21/ujiindoorloc-part-II.html
[wlan-loc-3]: https://sharan-naribole.github.io/2017/04/28/ujiindoorloc-part-III.html
