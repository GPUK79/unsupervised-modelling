#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:13:01 2023

@author: giampaololevorato
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing



# =============================================================================
# READ IN DATASET
# =============================================================================
if False:
    loc = ("./Data/")
    df = pd.read_csv("./Data/datasetForPCAandUnsupervised1.csv") 
# =============================================================================
#     df = df1.sample(frac=0.025, replace=True, random_state=1)
# =============================================================================
    
# =============================================================================
#     df.to_csv(loc+"datasetForPCAandUnsupervised.csv", index=False)
# =============================================================================
if False:    

    loc = ("./Data/")
    df = pd.read_csv(loc+"datasetForPCAandUnsupervised.csv")
    print("")
    print("")
    print(df.shape)
    print("")
    print("")
    
    
    # =============================================================================
    # IDENTIFY TARGET
    # =============================================================================
    
    # =============================================================================
    # print(df['loan_status'].value_counts(dropna=True))
    # =============================================================================
    
    
    def Target(row):
        if((row['loan_status'] == "Fully Paid") |
           (row['loan_status'] == "Current") |   
           (row['loan_status'] == "Late (16-30 days)") |  
           (row['loan_status'] == "Does not meet the credit policy. Status:Fully Paid")):
            return 1
        else:
            return 0
    
    df['Target'] = df.apply(Target, axis=1)
    
    print("Target distribution")
    print(df['Target'].value_counts(dropna=False))
    print("")
    
    # =============================================================================
    # DROP COLUMNS WITH DIRT IN IT 
    # =============================================================================
    
    
    # =============================================================================
    # Drop columns containing only missing values
    # =============================================================================
    col_list = df.columns
    col_filter = []
    
    for i in range(len(col_list)): 
        if(df[col_list[i]].isna().sum() == len(df)):
            col_filter.append(col_list[i])
    
    print("Feaures containing only missing values")
    print(col_filter)
    print("")
    df = df.drop(columns=col_filter, axis=1)
    
    # =============================================================================
    # Drop columns containing constant values
    # =============================================================================
    col_list = df.columns
    constant_columns = []
    for i in range(len(col_list)):
        if(df[col_list[i]].nunique(dropna=True) == 1):
            constant_columns.append(col_list[i])
    df = df.drop(columns=constant_columns)
    print("Feaures containing constant values")
    print(constant_columns)
    print("")
                
    # =============================================================================
    # Drop field used for Target definition
    # =============================================================================
    df = df.drop(columns=['loan_status'], axis=1)            
    
    
    # =============================================================================
    # Drop fields containing more than 50% of missing records
    # =============================================================================
    
    col_list = df.columns
    threshold = 0.50
    too_many_missing_columns =[]
    
    for i in range(len(col_list)):
        if(df[col_list[i]].isna().sum()/len(df) > threshold):
            too_many_missing_columns.append(col_list[i])
    
    print("Drop features containing % missing values > 50%")
    print(too_many_missing_columns)
    print("")
    df = df.drop(columns=too_many_missing_columns)
                
    print("")            
    # =============================================================================
    # Check shape of dataset
    # =============================================================================
    print("Dataframe shape")
    print(df.shape)
    print("")
    
    df.to_csv(loc+"dataSampleForUnsupervisedExercise.csv", index=False)
    
    
    # =============================================================================
    # Label encode the string features
    # =============================================================================
    
    features = df.columns
    
    stringFeatures = []
    for i in range(len(features)):
        if(df[features[i]].dtype == object):
            stringFeatures.append(features[i])
    
    print("Categorical features to label encode")
    print(stringFeatures)
    print("")
    
    label_encoder = preprocessing.LabelEncoder()
    for i in range(len(stringFeatures)):
        df[stringFeatures[i]] = df[stringFeatures[i]].astype(str)
        df[stringFeatures[i]] = label_encoder.fit_transform(df[stringFeatures[i]])
    
    
    
    
    # =============================================================================
    # Replace missing values with -99999
    # =============================================================================
    from sklearn.impute import SimpleImputer
    
    col_list = df.columns
    features_containing_NAN = []
    
    for i in range(len(features_containing_NAN)):
        if(df[col_list[i]].isna().sum() > 0):
            features_containing_NAN.append(col_list[i])
    
    print("")
    print("Features containing missing values")
    print(features_containing_NAN)
    print("")
    
    df = df.fillna(-99999)

if False:
    ###################################################################
    #
    #            UNIVARIATE ANALYSIS
    #
    ###################################################################
    
    print("")
    print("Univariate Analysis")
    print("")
    chars = df.columns
    
    d = pd.DataFrame(index = [0],columns=['Feature','Feature Type','# Records',"# Zero's","% Zero's","# Missing","% Missing","Minimum", "Mean", "Median", "Maximum", "Skewness","Kurtosis"])
    d.to_csv(loc+'univariateAnalysis.csv')        
    
    
    notApplicable = "N/A"
    
    for z in range(len(chars)):
        a = pd.read_csv(loc+'univariateAnalysis.csv',index_col=0)  
        if(df[chars[z]].dtype !='object'):
            dd = pd.DataFrame({'Feature':[chars[z]],
                               'Feature Type':df[chars[z]].dtype,
                               "# Records":df.size,
                               "# Zero's": (df[chars[z]] == 0).sum(),
                               "% Zero's" : round((df[chars[z]] == 0).sum()/df.size*100,2),
                               "# Missing" : (df[chars[z]].isna()).sum(),
                               "% Missing" : round((df[chars[z]].isna()).sum()/df.size*100,2),
                               "Minimum" : round(df[chars[z]].min(),2),
                               "Mean" : round(df[chars[z]].mean(),2),
                               "Median" : round(df[chars[z]].median(),2),
                               "Maximum" : round(df[chars[z]].max(),2),
                               "Skewness" : round(df[chars[z]].skew(axis = 0, skipna = True),2), #Positive Skewness -> LEFT SKEW
                               "Kurtosis" : round(df[chars[z]].kurtosis(axis=None, skipna=True, level=None, numeric_only=None),1)} #Negative Kurtosis -> FAT TAILS
                               )
            s = pd.concat([a, dd])
            s = s.dropna()
            s.to_csv(loc+'univariateAnalysis.csv')
    
        else:
            dd = pd.DataFrame({'Feature':[chars[z]],
                       'Feature Type':df[chars[z]].dtype,
                       "# Records": len(df),
                       "# Zero's": (df[chars[z]] == 0).sum(),
                       "% Zero's" : round((df[chars[z]] == 0).sum()/df.size*100,1),
                       "# Missing" : (df[chars[z]].isna()).sum(),
                       "% Missing" : round((df[chars[z]].isna()).sum()/df.size*100,1),
                       "Minimum" : [notApplicable],
                       "Mean" : [notApplicable],
                       "Median" : [notApplicable],
                       "Maximum" : [notApplicable],
                       "Skewness" : [notApplicable],
                       "Kurtosis" : [notApplicable]}
                       )
    
            s = pd.concat([a, dd])
            s = s.dropna()
            s.to_csv(loc+'univariateAnalysis.csv')
    
    def render_mpl_table(data, col_width=9.0, row_height=0.625, font_size=10.5,
                         header_color='#33B3FF', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                         ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
    
        for k, cell in mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        return ax.get_figure(), ax
    s = pd.read_csv(loc+'univariateAnalysis.csv', index_col=0)
    fig,ax = render_mpl_table(s, header_columns=0, col_width=2)
    fig.savefig(loc+"Univariate Analysis.png")#,auto_open=True
    
    
    from PIL import Image
    
    PNG_FILE = loc+"Univariate Analysis.png"
    PDF_FILE = loc+"Univariate Analysis.pdf"
    
    rgba = Image.open(PNG_FILE)
    rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
    rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
    rgb.save(PDF_FILE, 'PDF', resoultion=100.0)


if False:

    # =============================================================================
    # Select predictive features
    # =============================================================================
    
    
    col_list = list(df.columns)
    
    col_list.remove("Target")
    col_list.remove("id")
    col_list.remove("url")
    col_list.remove("zip_code")
    
    
    # =============================================================================
    # Correlation Matrix
    # =============================================================================
    
    def correlation_heatmap(dataframe,l,w):
        #correlations = dataframe.corr()
        correlation = dataframe.corr()
        plt.figure(figsize=(l,w))
        sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
        plt.title('Correlation between different fearures')
        plt.show();
        
    # Let's Drop Class column and see the correlation Matrix & Pairplot Before using this dataframe for PCA as PCA should only be perfromed on independent attribute
    
    dev = df[col_list]
    #print("After Dropping: ", cleandf)
    correlation_heatmap(dev, 30,15)
    
    # =============================================================================
    # # =============================================================================
    # #
    # #
    # #
    # #
    # # DATASET IS READY FOR MODELLING
    # #
    # #
    # #
    # #
    # # =============================================================================
    # =============================================================================
    
    
    
    
    # =============================================================================
    # Identify Train vs Test Sample
    # =============================================================================
    
    df['TrainFlag'] = 0
    df.iloc[pd.Series(df.index).sample(frac=0.5, random_state=1234), df.columns.get_loc('TrainFlag')] = 1
    print("Train vs Test sample")
    print(df['TrainFlag'].value_counts())
    train = df.loc[(df['TrainFlag'] == 0)]
    test = df.loc[(df['TrainFlag'] == 1)]
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    
    print("")
    
    # =============================================================================
    # Standardize train and test sample which means that we get a z-score
    # =============================================================================
    
    print("Standardize Features")
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().set_output(transform="pandas")
    
    # Fit only on Train sample
    train_transformed = sc.fit_transform(train[col_list])
    
    # Transform Test sample
    test_transformed = sc.transform(test[col_list])
    train_transformed.to_csv(loc+"trainTransformed.csv",index=False)   
    print("")
    # ===============================================================================
    # # =============================================================================
    # # Principal Component Analysis
    # # =============================================================================
    # # =============================================================================
    
    print("Explained variance ratio")
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(train_transformed)
    
    print(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10,8))
    plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker = "o", linestyle = "--")
    plt.title("Explained variance by components")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.show()
    
    # =============================================================================
    # A rule of thumb is to preserve around 80 % of the variance.
    # =============================================================================
    
    pca = PCA(n_components=15)
    print("Perform PCA with n components")
    print(pca.fit(train_transformed))
    print("Generate scores")
    scores_pca = pca.transform(train_transformed)
    print(scores_pca)
    Scores = pd.DataFrame(scores_pca)
    print(Scores)
    Scores.to_csv(loc+"ScoresTrainSample.csv", index=False)
    print("Export pickle file")
    import pickle
    filename = 'PCA.sav'
    pickle.dump(pca, open(filename, 'wb'))
    
    # =============================================================================
    # Incorporate the newly obtained PCA scores in the K-means algorithm
    # =============================================================================
    
    # =============================================================================
    # Select optimal number of clusters    
    # =============================================================================
    
    print("")    
    print("Select optimal number of clusters")
    from sklearn.cluster import KMeans
    wcss = []
    for k in range(1,20):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(scores_pca)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(12,6))    
    plt.grid()
    plt.plot(range(1,20),wcss, linewidth=2, color="red", marker ="8")
    plt.xlabel("Number of clusters")
    plt.title("K-means with PCA clustering")
    plt.xticks(np.arange(1,20,1))
    plt.ylabel("WCSS")
    plt.show()
    print("")
    
    # =============================================================================
    # Train the model
    # =============================================================================
         
    kmeans_pca = KMeans(n_clusters = 4, init="k-means++", random_state=42)
    kmeans_pca.fit(scores_pca)
    
    filename = 'kmeans_pca.sav'
    pickle.dump(kmeans_pca, open(filename, 'wb'))
            
    df_train = pd.concat([train.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_train.columns.values[-15:] = [
        
        "Component1",
        "Component2",
        "Component3",
        "Component4",
        "Component5",
        "Component6",
        "Component7",
        "Component8",
        "Component9",
        "Component10",
        "Component11",
        "Component12",
        "Component13",
        "Component14",
        "Component15"
        ]        
    
    print(df_train.info())     
    df_train['Segment K_means PCA'] = kmeans_pca.labels_
    df_train.to_csv(loc+"df_train.csv", index=False)
    
    # =============================================================================
    # Visualize clsuters for top 3 components - Train Sample
    # =============================================================================
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits import mplot3d
    
    fig = plt.figure(figsize=(15,15))
    
    ax = plt.axes(projection='3d')
    
    df_train = df_train.sort_values(by=['Segment K_means PCA'])
    
    for s in df_train['Segment K_means PCA'].unique():
        ax.scatter(df_train.Component1[df_train['Segment K_means PCA']==s],df_train.Component2[df_train['Segment K_means PCA']==s],df_train.Component3[df_train['Segment K_means PCA']==s],label=s, alpha=0.5)
    sns.set_style("darkgrid")
    sns.set_context("talk")   
    plt.xlabel("Component1")
    plt.ylabel("Component2")
    plt.title("K-means with PCA clustering - Top 3 Components - Train Sample")
    ax.set_zlabel("Component3")
    ax.legend()
    plt.show()
    
    
    # =============================================================================
    # Visualize clsuters for top 2 components - Train Sample
    # =============================================================================
    
    x_axis = df_train['Component2']
    y_axis = df_train['Component1']
    plt.figure(figsize=(30,30))
    
    sns.scatterplot(x=x_axis, y=y_axis, hue=df_train['Segment K_means PCA'], palette = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"])
    plt.title("Clusters by PCA Components - Train Sample")
    plt.show()
    
    
    
    # =============================================================================
    # Apply PCA to the Test Sample 
    # =============================================================================
    
    # =============================================================================
    # Load the pickle file
    # =============================================================================
    filename = 'PCA.sav'
    
    pickle.dump(pca, open(filename, 'wb'))
    pca = pickle.load(open(filename, 'rb'))
    
    scores_pca = pca.transform(test_transformed)
    Scores = pd.DataFrame(scores_pca)
    print(Scores)
    Scores.to_csv(loc+"ScoresTestSample.csv", index=False)
    
    # =============================================================================
    # Generate K-Means clusters on Test sample
    # =============================================================================
         
    filename = 'kmeans_pca.sav'
    KMeans_pca = pickle.load(open(filename, 'rb'))
    KMeans_pca.fit_predict(scores_pca)
            
    df_test = pd.concat([test.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
    df_test.columns.values[-15:] = [
        
        "Component1",
        "Component2",
        "Component3",
        "Component4",
        "Component5",
        "Component6",
        "Component7",
        "Component8",
        "Component9",
        "Component10",
        "Component11",
        "Component12",
        "Component13",
        "Component14",
        "Component15"
        ]        
    
    print(df_test.info())     
    df_test['Segment K_means PCA'] = KMeans_pca.labels_
    df_test.to_csv(loc+"df_test.csv", index=False)
    
    
    
    
    # =============================================================================
    # Visualize clusters for top 3 components - Test Sample
    # =============================================================================
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits import mplot3d
    
    fig = plt.figure(figsize=(15,15))
    
    ax = plt.axes(projection='3d')
    
    df_test = df_test.sort_values(by=['Segment K_means PCA'])
    
    for s in df_test['Segment K_means PCA'].unique():
        ax.scatter(df_test.Component1[df_test['Segment K_means PCA']==s],df_test.Component2[df_test['Segment K_means PCA']==s],df_test.Component3[df_test['Segment K_means PCA']==s],label=s, alpha=0.5)
    sns.set_style("darkgrid")
    sns.set_context("talk")   
    plt.xlabel("Component1")
    plt.ylabel("Component2")
    plt.title("K-means with PCA clustering - Top 3 Components - Test Sample")
    ax.set_zlabel("Component3")
    ax.legend()
    plt.show()
    
    
    # =============================================================================
    # Visualize clusters for top 2 components - Test Sample
    # =============================================================================
    
    x_axis = df_test['Component2']
    y_axis = df_test['Component1']
    plt.figure(figsize=(30,30))
    
    sns.scatterplot(x=x_axis, y=y_axis, hue=df_test['Segment K_means PCA'], palette = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"])
    plt.title("Clusters by PCA Components - Test Sample")
    plt.show()



if True:
    loc = "./Data/"
    df_test = pd.read_csv(loc+"df_test.csv")
    d = df_test[["num_tl_120dpd_2m",	"num_tl_30dpd",	"num_tl_90g_dpd_24m","tot_cur_bal","last_fico_range_high",'annual_inc','dti','inq_last_6mths','revol_util','total_acc','Segment K_means PCA']]
    e = d.groupby('Segment K_means PCA').median()
    e.to_csv(loc+"SegmentAnalysis.csv")
    print(e)


# =============================================================================
# 
# if False:
# 
#     
#     # =============================================================================
#     # pca.fit(train_transformed)
#     # cumsum = np.cumsum(pca.explained_variance_ratio_)
#     # d = np.argmax(cumsum >= 0.95)+1
#     # print(d)
#     # =============================================================================
#     
#     
#     #Calculating the covariance matrix
#     cov_matrix = np.cov(train.T)
#     print("")
#     print("Covariance Matrix")
#     print("cov_matrix shape:",cov_matrix.shape)
#     print("Covariance_matrix",cov_matrix)
#     print("")
#     
#     # =============================================================================
#     # check if any of the cov_matrix values is nan or inf
#     # =============================================================================
#     print(np.isnan(cov_matrix).any())
#     print(np.isinf(cov_matrix).any())
#     
#     # =============================================================================
#     # replace nan with 0
#     # =============================================================================
#     cov_matrix = np.nan_to_num(cov_matrix)
#     
#     
#     # =============================================================================
#     # Calculating Eigen Vectors & Eigen Values
#     # =============================================================================
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#     print("Eigen vectors and Eigen Values")
#     print('Eigen Vectors \n%s', eigenvectors)
#     print('\n Eigen Values \n%s', eigenvalues)
#     print("")
#     
#     # =============================================================================
#     # Sort eigenvalues in descending order
#     # =============================================================================
#     
#     # Make a set of (eigenvalue, eigenvector) pairs:
#     eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
#     # Sort the (eigenvalue, eigenvector) pairs from highest to lowest with respect to eigenvalue
#     print("Sort pairs")
#     eig_pairs.sort()
#     eig_pairs.reverse()
#     print("Sorted eig_pairs")
#     print(eig_pairs)
#     print("")
#     
#     # =============================================================================
#     #  Extract the descending ordered eigenvalues and eigenvectors
#     # =============================================================================
#     eigvalues_sorted = [eig_pairs[index][0] for index in range(len(eigenvalues))]
#     eigvectors_sorted = [eig_pairs[index][1] for index in range(len(eigenvalues))]
#     # Let's confirm our sorting worked, print out eigenvalues
#     print('Eigenvalues in descending order: \n%s' %eigvalues_sorted)
#     print("")
#     
#     # =============================================================================
#     # Calculate variance explained in percentage
#     # =============================================================================
#     tot = sum(eigenvalues)
#     var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]  # an array of variance explained by each  eigen vector... 
#     cum_var_exp = np.cumsum(var_explained)  # an array of cumulative variance. 
#     print(cum_var_exp)
#     
#     # =============================================================================
#     # Plot The Explained Variance and Principal Components
#     # =============================================================================
#     print("Number of eigenvectprs")
#     print(len(eigenvectors))
#     # =============================================================================
#     # plt.bar(range(1,103), var_explained, alpha=0.5, align='center', label='individual explained variance')
#     # =============================================================================
#     plt.step(range(1,103),cum_var_exp, where= 'mid', label='cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.legend(loc = 'best')
#     plt.show()
#     
#     
#     from sklearn.decomposition import PCA
#     
#     pca = PCA(n_components=4)
#     X_reduced = pca.fit_transform(train_transformed)
#     print(X_reduced)
#     # =============================================================================
#     # X = train.values
#     # 
#     # # P_reduce represents reduced mathematical space....
#     # P_reduce = np.array(eigvectors_sorted[0:4])   # Reducing from 102 to 4 dimension space
#     # X_std_8D = np.dot(X,P_reduce.T)   # projecting original data into principal component dimensions
#     # reduced_pca = pd.DataFrame(X_std_8D)  # converting array to dataframe for pairplot
#     # print(reduced_pca)
#     # 
#     # sns.pairplot(reduced_pca, diag_kind='kde') 
#     # 
#     # =============================================================================
#     
#     
#     
#     if False:
#         # =============================================================================
#         # Transform string features into numeric
#         # =============================================================================
#         
#         features = [
#             'Age',
#             'Income',
#             'OwnershipType',
#             'MonthsAtCurrentJob',
#             'LoanScope',
#             'InternalGrade',
#             'LoanAmount',
#             'InterestRate',
#             'DebtToIncome',
#             'TimeOnBooks'  
#         ]
#         
#         
#         
#     
#         dev.to_csv("dev.csv", index=False)
#         print(dev)
#                    
#         
#         
#         
#         # =============================================================================
#         # Select predictive features    
#         # =============================================================================
#                
#         col_list = [
#             'Age',
#         # =============================================================================
#         #     'InternalGrade',
#         # =============================================================================
#             'DebtToIncome'
#         ]
#         
#         dev = dev[col_list]
#         
#         # =============================================================================
#         # Standardize predictive fields    
#         # =============================================================================
#            
#         
#         from sklearn.preprocessing import StandardScaler
#         sc = StandardScaler().set_output(transform="pandas")
#         
#         train_transformed = sc.fit_transform(dev)
#         df = sc.transform(dev)
#         
#         print(df)
#             
#         
#         
# 
#         
#         
#         # =============================================================================
#         # Train K-Means
#         # =============================================================================
#             
#         km = KMeans(n_clusters=3)
#         #predict the labels of clusters.
#         clusters = km.fit_predict(df.iloc[:,1:])
#         df["label"] = clusters
#         
#         d = df.groupby('label').mean()
#         print(d)
#         print(clusters)
#         
#         
#         # =============================================================================
#         # Plot clusters
#         # =============================================================================
#         
#         import matplotlib.pyplot as plt
#          
#         #filter rows of original data
#         
#         
#         df = df.to_numpy()
#         filtered_label0 = df[clusters == 0]
#         
#         
#         
#         #plotting the results
#         # =============================================================================
#         # plt.scatter(filtered_label0[:,0] , filtered_label0[:,1])
#         # plt.show()
#         # 
#         # =============================================================================
#         #filter rows of original data
#         filtered_label1 = df[clusters == 1]
#         filtered_label2 = df[clusters == 2]
#          
#         #Plotting the results
#         plt.scatter(filtered_label0[:,0] , filtered_label0[:,1], color = 'yellow')
#         plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'green')
#         plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
#         plt.show()
#         
#         
#         
#         
#         if False:    
#             from mpl_toolkits.mplot3d import Axes3D
#             import matplotlib.pyplot as plt
#             import numpy as np
#             import pandas as pd
#              
#             fig = plt.figure(figsize=(20,10))
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(df.Age[df.label == 0], df["Income"][df.label == 0][df.label == 0], c='blue', s=60)
#             ax.scatter(df.Age[df.label == 1], df["Income"][df.label == 1][df.label == 1], c='red', s=60)
#             ax.scatter(df.Age[df.label == 2], df["Income"][df.label == 2][df.label == 2], c='green', s=60)
#             ax.scatter(df.Age[df.label == 3], df["Income"][df.label == 3][df.label == 3], c='orange', s=60)
#             # =============================================================================
#             # ax.scatter(df.Age[df.label == 4], df["Income"][df.label == 4][df.label == 4], c='purple', s=60)
#             # ax.scatter(df.Age[df.label == 0], df["Income"][df.label == 0][df.label == 0], c='blue', s=60)
#             # ax.scatter(df.Age[df.label == 1], df["Income"][df.label == 1][df.label == 1], c='red', s=60)
#             # ax.scatter(df.Age[df.label == 2], df["Income"][df.label == 2][df.label == 2], c='green', s=60)
#             # ax.scatter(df.Age[df.label == 3], df["Income"][df.label == 3][df.label == 3], c='orange', s=60)
#             # =============================================================================
#             ax.view_init(30, 185)
#             plt.xlabel("Age")
#             plt.ylabel("Income")
#             # =============================================================================
#             # ax.set_zlabel('MonthsAtCurrentJob')
#             # =============================================================================
#             plt.show()
#             
#             print(df)
#             
#             
#             
#     
#         
# 
# =============================================================================
