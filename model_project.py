import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
df=pd.read_excel(r"C:\Users\shrig\Documents\Documents\python\Telco-Customer-Churn-dataset.xlsx")
while True:
    print("\n")
    print("1.DATASET DETAILS")
    print("2.GRAPH OF CLASS IMBALANCE")
    print("3.MODEL TRAINING")
    print("4.MODEL EVALUTAION WITH HEATMAP")
    print("5.EXIT")
    n=int(input("Enter Ur Choice : "))
    if n==1:
        print("\n")
        print("--DATATYPE OF ALL COLUMNS--")
        print(df.dtypes)
        print("\n")
        print("---COLUMN NAMES---")
        print(df.columns)
        print("\n")
        print("----Total number of distinct things in each columns----")
        for col in df.columns:
            print(df[col].value_counts())
            print("\n")
    elif n==2:
        if "Churn" in df.columns:
            plt.figure(figsize=(10,20))
            sns.countplot(data=df,x="Churn",color="red")
            plt.xlabel("Churn")
            plt.ylabel("frequency")
            plt.show()
        else:
            print("Churn column not found")
    elif n==3:
        le=LabelEncoder()
        df_copy=df.copy()
        df_copy["TotalCharges"]=pd.to_numeric(df_copy["TotalCharges"],errors="coerce")#to convert from object to numeric
        df_copy["TotalCharges"]=df_copy["TotalCharges"].fillna(df_copy["TotalCharges"].mean())
        df_encode=df_copy.copy()
        for col in df.columns:
            if col=="customerID":
                continue
            elif df_copy[col].dtype=="object" and df_copy[col].nunique()==2:
                df_encode[col]=le.fit_transform(df_copy[col])
            elif df_copy[col].dtype=="object" and df_copy[col].nunique()>2:
                df_encode=pd.get_dummies(df_encode,columns=[col])
                
        X=df_encode.drop(columns=["customerID","Churn"])
        y=df_encode["Churn"]
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        numeric=["tenure", "MonthlyCharges", "TotalCharges"]
        scale=MinMaxScaler()
        x_scaled=scale.fit_transform(x_train[numeric])
        x_tested=scale.transform(x_test[numeric])
        x_train[numeric]=x_scaled
        x_test[numeric]=x_tested
        model=LogisticRegression(class_weight="balanced")
        model.fit(x_train,y_train)
        cutsomer_Id=input("cutomerId:")
        gender=input("enter gender --- options(male/female) : ").strip().lower() #strip() removes staring and ending spaces and lower makes whole string in lowercase
        citizen=int(input("enter ur citizenship ---options(1/0) : "))
        partner=input("enter ur partner status  --- options(Yes/no) : ").strip().lower()
        dependent=input("enter dependents --- options(Yes/no) : ").strip().lower()
        tenue=int(input("enter how many months of togetherness with compnay : "))
        phone=input("enter phoneservice--- options(Yes/no) : ").strip().lower()
        multiple=input("enter multipleservice--- options(Yes/no/NO phone service) : ").strip().lower()
        internetservice=input("enter internetservice--- options(fibre optic/dsl/no) : ").strip().lower()
        onlinesecurity=input("enter onlineservice--- options(Yes/no/NO internet service) : ").strip().lower()
        onlinebackup=input("enter onlinebackup--- options(Yes/no/NO internet service) : ").strip().lower()
        deviceprotection=input("enter deviceprotection--- options(Yes/no/NO internet service) : ").strip().lower()
        techsupport=input("enter techsupport--- options(Yes/no/NO internet service) : ").strip().lower()
        streamTV=input("enter streamingTV--- options(Yes/no/NO internet service) : ").strip().lower()
        streamMovies=input("enter streamingMovies--- options(Yes/no/NO internet service) : ").strip().lower()
        contract=input("enter contract with company--- options(Month-to-month/TWO year/one year) : ").strip().lower()
        paperbill=input("enter paperbilling--- options(Yes/no) : ").strip().lower()
        PaymentMethod=input("enter paymentmethod--- options(Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic) ) : ").strip().lower()
        monthcharge=int(input("enter monthlycharges : "))
        totalcharge=int(input("enter totalcharges : "))
# 1. Collect user input into a dictionary
        new_customer = {
            "gender": gender,
            "SeniorCitizen": citizen,
            "Partner": partner,
            "Dependents": dependent,
            "tenure": tenue,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internetservice,
            "OnlineSecurity": onlinesecurity,
            "OnlineBackup": onlinebackup,
            "DeviceProtection": deviceprotection,
            "TechSupport": techsupport,
            "StreamingTV": streamTV,
            "StreamingMovies": streamMovies,
            "Contract": contract,
            "PaperlessBilling": paperbill,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": monthcharge,
            "TotalCharges": totalcharge
        }
        new_df=pd.DataFrame([new_customer])
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            new_df[col]=le.fit_transform(new_df[col])
        multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
              'Contract','PaymentMethod']
        new_df=pd.get_dummies(new_df,columns=multi_cols).astype(int)
        new_df=new_df.reindex(columns=X.columns,fill_value=0)#required to match training data
        new_df[numeric] = scale.transform(new_df[numeric])
        prediction = model.predict(new_df)
        print("Customer_Id:",cutsomer_Id)
        print("predicted churn is 0(still active) and 1(not active):",prediction[0])
    elif n==4:
        le=LabelEncoder()
        df_copy=df.copy()
        df_copy["TotalCharges"]=pd.to_numeric(df_copy["TotalCharges"],errors="coerce")#to convert from object to numeric
        df_copy["TotalCharges"]=df_copy["TotalCharges"].fillna(df_copy["TotalCharges"].mean())
        df_encode=df_copy.copy()
        for col in df.columns:
            if col=="customerID":
                continue
            elif df_copy[col].dtype=="object" and df_copy[col].nunique()==2:
                df_encode[col]=le.fit_transform(df_copy[col])
            elif df_copy[col].dtype=="object" and df_copy[col].nunique()>2:
                df_encode=pd.get_dummies(df_encode,columns=[col])
                
        X=df_encode.drop(columns=["customerID","Churn"])
        y=df_encode["Churn"]
        x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        numeric=["tenure", "MonthlyCharges", "TotalCharges"]
        scale=MinMaxScaler()
        x_scaled=scale.fit_transform(x_train[numeric])
        x_tested=scale.transform(x_test[numeric])
        x_train[numeric]=x_scaled
        x_test[numeric]=x_tested
        model=LogisticRegression(max_iter=150,class_weight="balanced")#required for covering whole dataset
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        print("--confusion matrix--")
        conf_matrix=confusion_matrix(y_test,y_pred)
        print(conf_matrix)
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)
        print("ACCURACY : ",accuracy)
        print("PRECISION : ",precision)
        print("RECALL : ",recall)
        print("F1_SCORE: ",f1)
        fig,ax=plt.subplots(1,2,figsize=(14,10))
        sns.heatmap(conf_matrix,annot=True,cmap="Blues",ax=ax[0])
        ax[0].set_xlabel("predicted")
        ax[0].set_ylabel("actual")
        ax[0].set_title("Confusion_Matrix")
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        ax[1].bar(metrics.keys(),metrics.values())
        ax[1].set_xlabel("parameters")
        ax[1].set_ylabel("actual_values")
        ax[1].set_title("Comparision")
        plt.savefig("graph.png",dpi=200,bbox_inches="tight")
        plt.show()
        print("---Graph Saved Sucessfully---")
    elif n==5:
        break
    else:
        print("enter valid choice")
    








    

    
            