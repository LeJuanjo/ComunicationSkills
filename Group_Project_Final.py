# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD 2 - v1
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import numpy as np
import sweetviz as sv
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import math
import statsmodels.api as sm
from time import time
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from pandas_datareader import data as wb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import shap
import streamlit.components.v1 as components
shap.initjs()

#===================================================
# GLOBAL VARIABLES
#===================================================

   # Load Data Churn_Train

Churn_Train = pd.read_csv("./Data/churn_train.csv")
 

    # Load Data Churn_Test
Churn_Test = pd.read_csv("./Data/churn_test.csv")
  
# Label Encoder Churn
le_churn = LabelEncoder()
Churn_Train["churn"] = le_churn.fit_transform(Churn_Train["churn"])

# Label Encoder international_plan
le_international_plan = LabelEncoder()
Churn_Train["international_plan"] = le_international_plan.fit_transform(Churn_Train["international_plan"])

# Label Encoder voice_mail_plan
le_voice_mail_plan = LabelEncoder()
Churn_Train["voice_mail_plan"] = le_voice_mail_plan.fit_transform(Churn_Train["voice_mail_plan"])

# Label Encoder area_code
le_area_code = LabelEncoder()
Churn_Train["area_code"] = le_area_code.fit_transform(Churn_Train["area_code"])

# Label Encoder area_code
le_state = LabelEncoder()
Churn_Train["state"] = le_state.fit_transform(Churn_Train["state"])
    
# Label Encoder international_plan
le_international_plan = LabelEncoder()
Churn_Test["international_plan"] = le_international_plan.fit_transform(Churn_Test["international_plan"])

# Label Encoder voice_mail_plan
le_voice_mail_plan = LabelEncoder()
Churn_Test["voice_mail_plan"] = le_voice_mail_plan.fit_transform(Churn_Test["voice_mail_plan"])

# Label Encoder area_code
le_area_code = LabelEncoder()
Churn_Test["area_code"] = le_area_code.fit_transform(Churn_Test["area_code"])

# Label Encoder area_code
le_state = LabelEncoder()
Churn_Test["state"] = le_state.fit_transform(Churn_Test["state"])
    
# Feature Variables Creation
features = ['state', 
                'account_length', 
                'area_code', 
                'international_plan', 
                'voice_mail_plan', 
                'number_vmail_messages', 
                'total_day_minutes', 
                'total_day_calls', 
                'total_day_charge', 
                'total_eve_minutes', 
                'total_eve_calls', 
                'total_eve_charge', 
                'total_night_minutes', 
                'total_night_calls', 
                'total_night_charge', 
                'total_intl_minutes', 
                'total_intl_calls', 
                'total_intl_charge', 
                'number_customer_service_calls']

features_Sel = ['international_plan', 
            'number_vmail_messages', 
            'total_day_minutes', 
            'total_day_charge', 
            'total_eve_minutes', 
            'number_customer_service_calls']

global X_train, X_test, y_train, y_test
X, y = Churn_Train[features], Churn_Train["churn"]
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=42)
#==============================================================================
# Predictive Models
#==============================================================================

logreg = sm.Logit(y_train,X_train).fit()

    # Predict
pred_trainLG = logreg.predict(X_train)
pred_testLG = logreg.predict(X_test)

# Evaluate predictions
acc_trainlg = accuracy_score(y_train, np.round(pred_trainLG))
acc_testlg = accuracy_score(y_test, np.round(pred_testLG))

# split data in train and test (stratify y)
X, y = Churn_Train[features], Churn_Train["churn"]
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, random_state=42)
    
# define model
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train,y_train) 
    
# predict probabilities
pred_train = rf.predict(X_train)
pred_test = rf.predict_proba(X_test)

# evaluate predictions
acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(y_test, np.argmax(pred_test, axis=1))

# normalize data
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)
mlp = mlp.fit(X_train, y_train)
    
# predict probabilities
pred_trainNR = mlp.predict_proba(X_train)
pred_testNR = mlp.predict_proba(X_test)

# evaluate predictions
acc_train = accuracy_score(y_train, np.argmax(pred_trainNR, axis=1))
acc_test = accuracy_score(y_test, np.argmax(pred_testNR, axis=1))






Churn_Test = Churn_Test[features]
Churn_Test = sm.add_constant(Churn_Test)
pred_testT = mlp.predict(Churn_Test)
Churn_TestPred = pd.concat([Churn_Test, pd.DataFrame(pred_testT)], axis=1)
Churn_TestPred.columns = [*Churn_TestPred.columns[:-1], 'Churn']

sample_CSC= Churn_TestPred[Churn_TestPred["number_customer_service_calls"]>4]
sample_CSCX= Churn_TestPred[Churn_TestPred["number_customer_service_calls"]>4]
sample_CSC= sample_CSC.loc[:,:"number_customer_service_calls"]


#==============================================================================
# Tab 1
#==============================================================================

def tab1():
    
# Add dashboard title and description
    st.title("Customer Churn Behaviour")
    st.header('Abstract')
    
    st.markdown("""  Nowadays, companies are focused in maintain and improve the their profits. One of the biggest challenges for the retaintion  departments is to identy churners and prepare retaintion packages. Our target with this report is to present insights based on the Logistict Regresion, Random Forest and Neural Networks models to improve the Linear Regresion Model that is being used for the company. This way the company can offer a better retantion programs for customers that are churn prospects and keep their profits and improve the services that might be triggering the users to churn""")
    
    
  
    

#==============================================================================
# Tab 2
#==============================================================================

def tab2():
    
# Add dashboard title and description
    st.title("Current Insights:eyeglasses:")
    #Print interactive dashboard using Sweetviz
    #my_report = sv.analyze(Churn_Train)
    #my_report.show_html(filepath='C:/Users/jmedinamartinez/OneDrive - IESEG/Documents/Comunication Skills Project/Data/EDA.html', open_browser=False, layout='vertical', scale=1.0)
    HtmlFile = open("./Data/EDA.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    #components.iframe(src='C:/Users/jmedinamartinez/OneDrive - IESEG/Documents/Comunication Skills Project/Data/EDA.html', width=1100, height=1200, scrolling=True)
    components.html(source_code, width=1100, height=1200, scrolling=True)

    
# Add a Dashboard description
    st.markdown("""To date, the telecommunications company has 4250 customers. Of these customers 598 have unsubscribed from the service.  The churn rate is a very important measure because it is much more profitable to retain existing customers than to acquire new ones. This is because it saves marketing costs and sales costs. You will get a return on retention because you will gain the trust and loyalty of the customer. We will identify through different models why these customers churn and quantify the features that influence churn.  """)
    

    
    
    
#==============================================================================
# Tab 3
#==============================================================================

def tab3():
    


    
# Add dashboard title and description
    st.title("Logistic Regression")
    



    
# Logistic Regression

    
    st.write(logreg.summary())



# Print Accuracy
    st.markdown("""
    ### Logistic Regresessio Accuracy""")
    st.write(f"Train:\tACC={acc_trainlg:.4f}")
    st.write(f"Test:\tACC={acc_testlg:.4f}")
    
#==============================================================================
# Tab 4
#==============================================================================

def tab4():
    
    # Add dashboard title and description
    st.title("Neural Network")
    
    st.markdown("""
    ### Neural Network Accuracy""")
    st.write(f"Train:\tACC={acc_train:.4f}")
    st.write(f"Test:\tACC={acc_test:.4f}")
    st.balloons()
    
   
#==============================================================================
# Tab 5
#==============================================================================

def tab5():
    
    # Add dashboard title and description
    st.title("Random Forest")

# Important features well formatted
    rfdf = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Feature importance"]).sort_values(by= "Feature importance", ascending = False)
    
    st.write(rfdf)
    
    st.markdown("""
    ### Random Forest Accuracy""")
        
    st.write(f"Train:\tACC={acc_train:.4f}")
    st.write(f"Test:\tACC={acc_test:.4f}")
    
    
    st.markdown("""
    ### Feature Selection""")
    
# Feature Selection
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

# Concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

# Naming the dataframe columns
    featureScores.columns = ['Features','Score']  

# This prints the 10 best features
    st.write(featureScores.nlargest(10,'Score'))  
    
    # Gets the analysis for each ticker
#     analyst_info = si.get_analysts_info(ticker)
    
#     # Gets the keys from dictionary 'analyst_info'
#     keys = analyst_info.keys()
    
    
#     # This for loop gets the keys already extracted from keys = analyst_info.keys() and asks for each key value to build the table
#     for key in keys:
#         st.write(key)
#         st.table(analyst_info[key])
    
#==============================================================================
# Tab 6
#==============================================================================

def tab6():

# Add dashboard title and description
    st.title("Interpretability Techniques: Partial Dependence")
    #Churn_TestX = Churn_Test.iloc[:,1:]
   
    
  
    
  

    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(estimator=mlp, X=X_train, features=[9, 7, (9, 7),4, 19, (4,19), 6, 10, (6,10)], ax=ax)
    fig.tight_layout(pad=2.0)
    
    st.pyplot(fig)
    
    st.markdown("""These graphs make it possible to study how the variable weighs on the model's prediction. This graph measures the intensity of the overall impact of the variable on churn. This model allows you to use 2 features maximum.""")
 
#==============================================================================
# Tab 7
#==============================================================================

def tab7():
    
    # Add dashboard title and description
    st.title("Interpretability Techniques: ICE")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.tight_layout(pad=2.0)
    ice=PartialDependenceDisplay.from_estimator(estimator=mlp,
                                                    X=Churn_Test,
                                                    features=[7,6,9,10,4,19],
                                                    target=1,
                                                    kind="both",
                                                    ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                                                    pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},
                        # centered=True, # will be added in the future
                                                    ax=ax)
    st.pyplot(fig) 

#==============================================================================
# Tab 8
#==============================================================================

def tab8():
    
    # Add dashboard title and description
    st.title("Interpretability Techniques: Shapley values")
    
    # set up explainer for ".predict" method
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)

    mlp = mlp.fit(X_train[features_Sel], y_train)
# set up explainer for ".predict" method
    explainer = shap.Explainer(mlp.predict,X_test[features_Sel] )

# compute shap values
    shap_values = explainer(Churn_Test[features_Sel])
    
# feature importance (global)

# bar chart
    st.write("Feature Importance global")
    fig, ax =plt.subplots()
    shap.plots.bar(shap_values)
    st.pyplot(fig)
# individual dots for each instance
    fig, ax =plt.subplots()
    shap.plots.beeswarm(shap_values)
    st.pyplot(fig)

# split population in distinct groups (uses sklearn DecisionTree)
#shap.plots.bar(shap_values.cohorts(2).abs.mean(0))
    
    # initialize explainer
    
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)

    mlp = mlp.fit(X_train[features_Sel], y_train)
    explainer = shap.KernelExplainer(mlp.predict_proba,X_test[features_Sel])
    shap_values = explainer.shap_values(Churn_Test[features_Sel])
    
    # feature importance (global)

# average shapley values
    fig, ax =plt.subplots() 
    shap.summary_plot(shap_values, Churn_Test[features_Sel], class_names=["No Churn", "Churn"])
    st.pyplot(fig)
    st.write("feature importance for a specific target Churn")
    # feature importance for a specific target class
    fig, ax =plt.subplots() 
    target =1  # (Churn)

# distribution of shapley values for target
    shap.summary_plot(shap_values[target], Churn_Test[features_Sel], plot_type="bar")
    st.pyplot(fig)
    fig, ax =plt.subplots()
    shap.summary_plot(shap_values[target], Churn_Test[features_Sel], plot_type="dot")
    st.pyplot(fig)
# decision plot for target
    fig, ax =plt.subplots()
    shap.decision_plot(explainer.expected_value[target], shap_values[target],Churn_Test[features_Sel])
    st.pyplot(fig)

#==============================================================================
# Tab 7
#==============================================================================

def tab9():
    
    # Add dashboard title and description
    st.title('SubSet Interpretation')
    
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)

    mlp = mlp.fit(X_train, y_train)
    st.write('Subset PDP')
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(estimator=mlp, X=sample_CSC, features=[9, 7, (9, 7),4, 19, (4,19), 6, 10, (6,10)], ax=ax)
    fig.tight_layout(pad=2.0)
    st.pyplot(fig)
    
    st.write('Subset ICE')
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.tight_layout(pad=0.1)
    ice = PartialDependenceDisplay.from_estimator(estimator=mlp,
                        X=sample_CSC,
                        features=[7,6,9,10,4,19],
                        target=0,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},
                        # centered=True, # will be added in the future
                        ax=ax)
    st.pyplot(fig)
    
    st.write('Subset SHAP')
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), batch_size=32, early_stopping=False, random_state=42)

    mlp = mlp.fit(X_train[features_Sel], y_train)
    explainer = explainer = shap.Explainer(mlp.predict, sample_CSC[features_Sel])
    shap_values = explainer(sample_CSC[features_Sel])
    
    fig, ax =plt.subplots() 
    shap.plots.bar(shap_values)
    st.pyplot(fig)
        
    fig, ax =plt.subplots() 
    shap.plots.beeswarm(shap_values)
    st.pyplot(fig)
#==============================================================================
# Main body
#==============================================================================


def run():
     
 
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Abstract', 'Insight', 'Logistic Regression', 'Neural Network', 'Random Forest', 'Interpretability Techniques: PDP', 'Interpretability Techniques ICE', 'Interpretability Techniques: Shapley values','SubSet Interpretation'])
    
    # Show the selected tab
    if select_tab == 'Abstract':
        # Run tab 1
        tab1()
    elif select_tab == 'Insight':
        # Run tab 2
        tab2()
    elif select_tab == 'Logistic Regression':
        # Run tab 3
        tab3()
    elif select_tab == 'Neural Network':
        # Run tab 4
        tab4()
    elif select_tab == 'Random Forest':
        # Run tab 5
        tab5()
    elif select_tab == 'Interpretability Techniques: PDP':
        # Run tab 6
        tab6()
    elif select_tab == 'Interpretability Techniques ICE':
        # Run tab 7
        tab7()
    elif select_tab == 'Interpretability Techniques: Shapley values':
        # Run tab 8
        tab8()  
    elif select_tab == 'SubSet Interpretation':
        # Run tab 8
        tab9()          
    
        
        
if __name__ == "__main__":
    run()
    
###############################################################################
# END
###############################################################################