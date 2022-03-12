# -- import packages
# for data manipulation
import pandas as pd
import numpy as np

# for SARAH-1
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(random_state=0, n_estimators=10000)
from sklearn.feature_selection import SelectFromModel

# for SARAH-2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
randomforest2 = RandomForestClassifier(random_state=0, n_estimators=10000)

# -- useful functions
# weighted average fuctions
def wavg(val_col_name, wt_col_name):
    def inner(group):
        return (group[val_col_name] * group[wt_col_name]).sum() / group[wt_col_name].sum()
    inner.__name__ = 'wtd_avg'    
    return inner

def wavg_columns(dataframec,list1):
    cols = []
    for i in range(len(list1)):
        pivot = dataframec.groupby("code").apply(wavg(list1[i],"shift_length"))
        print((i+1)/len(list1))
        cols.append(pivot)
    pivot = pd.DataFrame(cols).transpose()
    pivot.columns = list1
    return pivot

# percentile rank function
def percent_rank(arr, score, sig_digits=8):
    arr = np.asarray(arr)
    arr = np.round(arr, sig_digits)
    score = np.round(score, sig_digits)
    if score in arr:
        small = (arr < score).sum()
        return small / (len(arr) - 1)
    else:
        if score < arr.min():
            return 0
        elif score > arr.max():
            return 1
        else:
            arr = np.sort(arr)
            position = np.searchsorted(arr, score)
            small = arr[position - 1]
            large = arr[position]
            small_rank = ((arr < score).sum() - 1) / (len(arr) - 1)
            large_rank = ((arr < large).sum()) / (len(arr) - 1)
            step = (score - small) / (large - small)
            rank = small_rank + step * (large_rank - small_rank)
            return rank

# -- import habits
habit = pd.read_csv("C:/Users/Admin/Desktop/Dev Project/Habits/Habits Data2.csv")
#habit = habit.drop(['Edgework Inside','Casting to Receive Puck','Follow through'], 1)

# -- import events
event_ww = pd.read_csv("C:/Users/Admin/Desktop/Dev Project/Habits/Events WW Data.csv")
event_og = pd.read_csv("C:/Users/Admin/Desktop/Dev Project/Habits/Events OG Data.csv")
event = pd.concat([event_ww,event_og])

# -- data manipulation
# modify event columns, drop irrelevant columns
event = event.drop(['SA','CF%',"FO%","Faceoffs lost","Faceoffs won"], 1)
event = event.replace("Sigrist Shannon Justin","Sigrist Shannon")

# merge all shots
event["Shots"] = event["Goals"] + event["Shots on goal"] + event["Missed shots"] + event["Blocked shots"]
event = event.drop(["Goals","Shots on goal","Missed shots","Blocked shots"], 1)

event = event[["Tournament"] + [c for c in event if c not in ["Tournament"]]]

# create list of habits and list of events
habit_list = list(habit.columns)[7:] # y variable SARAH-1
event_list = list(event.columns)[5:] # X variables SARAH-1

# adjust events per 60
event60 = event.copy()
for i in range(len(event)):
    print(i/len(event))
    event60.iloc[i,5:] = (event.iloc[i,5:] * 3600)/event60['shift_length'].iloc[i]

# merge habits and events on player name
df = habit.merge(event,left_on="code",right_on="code",how="left")
df60 = habit.merge(event60,left_on="code",right_on="code",how="left")

# -- SARAH-1 - Model 1 - Scouting Model - Predicting Events based on Habits
# 17 sub-models, 1 representing each event category

# habit dictionary to store results
habit_dico = {}
events_pred = []
for i in range(len(habit_list)):
    habit_dico[str(habit_list[i])] = []

variables = [] # empty list to store habits of interest
counter = [] # empty list to count the number of habits for each sub-model
for i in range(len(event_list)):
    print(event_list[i])
    
    # set X, y variables
    y = df60[event_list[i]] # X = events
    X = df60[habit_list] # y = habits
    model = randomforest.fit(X, y) # fit model

    sfm = SelectFromModel(model, threshold=0.0325) # threshold

    sfm.fit(X, y) # fit select from model
    variables_inside = [] # empty list to store habits of interest for each sub-model
    
    # feature selection for SARAH-1
    for feature_list_index in sfm.get_support(indices=True):
        variables_inside.append(X.columns[feature_list_index]) # habits that relate to events (SARAH-1 purposes)
        habit_dico[X.columns[feature_list_index]].append(event_list[i]) # events that relate to habits (SARAH-2 purposes)
    variables.append(variables_inside) # append variables from each sub-model
    counter.append(len(variables_inside)) # append number of habits for each sub-model
    

# dataframe that has all the event habit relationships
df_variables = pd.DataFrame([event_list,counter,variables]).transpose()

# -- SARAH-2 - Model 2 - Player Dev Model - Predicting Habits based on Events
# 30 sub-models, 1 representing each habit category
# success probability calculation

habits_pred2 = []
accuracys = []
for i in range(len(habit_list)):
    print(habit_list[i])
    
    y = df[habit_list[i]] # y = habit
    X = df60[habit_dico[habit_list[i]]] # X = events of interest
    X["Group_A"] = df["Group_A"] # dummy variable to differentiate group A and B 
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X)) # standardize X variables (events of interest)
    
    # test_train_split for accuracy calculation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model2 = randomforest2.fit(X_train,y_train)
    ypred_2 = model2.predict(X_test)
    acc = accuracy_score(y_test, ypred_2)
    accuracys.append(acc) 

    model2_2 = randomforest2.fit(X,y) # fit rf model
    y_pred2_2 = model2_2.predict_proba(X)[:,1] # make success probability prediction
    habits_pred2.append(y_pred2_2) # make rf predictions

# wavg (w on ice-time per period) prediction of habits on a player basis habits 
habits_pred2 = pd.DataFrame(habits_pred2).transpose()
habits_pred2.columns = habit_list
habits_pred2["code"] = df60["code"]
habits_pred2["shift_length"] = df60["shift_length"]

#habits_pivot2 = pd.pivot_table(habits_pred2,values=habit_list,index=['code'],aggfunc=np.mean)
habits_pivot2 = wavg_columns(habits_pred2,habit_list) # success probability model

# getting data ready for viz format
habits_results = pd.DataFrame(habits_pivot2.stack()).reset_index() # stack data in Tableau-friendly format
habits_results.columns = ["code","habit","success rate"] # change column names

# -- frequency model
# initial unweighted average attempt
pivot_frequency = pd.pivot_table(event,values=event_list,index=['code'],aggfunc=np.sum)
players = list(pivot_frequency.index)
pivot_toi = pd.pivot_table(event,values=['shift_length'],index=['code'],aggfunc=np.sum)

# use percentile, not min/max
#pivot_frequency = pd.DataFrame(scaler2.fit_transform(pivot_frequency),columns=event_list)

# get pivot event data per 60
for i in range(len(pivot_frequency)): # get pivot frequency of events adjusted per 60 minutes
    print(i/len(pivot_frequency))
    pivot_frequency.iloc[i,:] = (pivot_frequency.iloc[i,:] * 3600)/pivot_toi['shift_length'].iloc[i]

# reset index and select only players included in data set (without goalies)
pivot_frequency = pivot_frequency.reset_index()
pivot_frequency = pivot_frequency[pivot_frequency["code"].isin(habit["code"])]
pivot_frequency = pivot_frequency.reset_index(drop=True)

# calculate percent_rank of each event for players
count = 0
for i in range(len(pivot_frequency)): # i = rows
    for y in range(1,len(pivot_frequency.columns)): # y = columns
        print(count/int(len(pivot_frequency)*(len(pivot_frequency.columns)-1)))
        pivot_frequency.iloc[i,y] = percent_rank(pivot_frequency.iloc[:,y],pivot_frequency.iloc[i,y])
        count += 1

# get habit/event relationships mapped as 0s and 1s
list2 = []
for i in range(len(event_list)):
    list3 = []
    for y in range(len(habits_pivot2.columns)):
        if event_list[i] in habit_dico[list(habits_pivot2.columns)[y]]:
            list3.append(1)
        else:
            list3.append(0)
    list2.append(list3)
list2 = pd.DataFrame(list2,columns=habits_pivot2.columns)            
list2["event"] = event_list
list2 = list2.set_index("event")

# calculate frequency for habits      
freq_calc_l = []
count = 0
for i in range(len(habits_results)):
    row_code = pivot_frequency[pivot_frequency["code"]==habits_results["code"].iloc[i]].iloc[:,1:]
    count+=1
    for y in range(len(list2.columns)):
        if i%len(habit_list) == y%len(habit_list): # match remainder of euclidian division by number of habits to match i and y to correct player and event
            freq_calc = (row_code*list2.iloc[:,y]).values.sum()/sum(list2.iloc[:,y])
    print(i/len(habits_results))
    freq_calc_l.append(freq_calc) # append to frequency calculation list to include in df

habits_results["frequency"] = freq_calc_l 

# -- start merging results with player info
# correct name of players (format: Firstname Lastname)
names = [habits_results["code"].iloc[i].split(' ') for i in range(len(habits_results))]
names2 = [names[i][1]+" "+names[i][0] if len(names[i])==2 else names[i][2]+" "+names[i][1]+" "+names[i][0] for i in range(len(names))]
habits_results["name"] = names2

# import info data an manipulate height and weight to include measure symbol
info = pd.read_csv("C:/Users/Admin/Desktop/Dev Project/Habits/Player Info.csv")
info["Height"] = [info["Height"].iloc[i].split('/')[0]+" m"+" / "+info["Height"].iloc[i].split('/')[1] for i in range(len(info))]
info["Weight"] = [info["Weight"].iloc[i].split('/')[0]+" kg"+" / "+info["Weight"].iloc[i].split('/')[1] + " lbs" for i in range(len(info))]

# -- skill sets
# dico containing all skill sets (keys are skill sets and values are lists with all habits in given skill set)
dico_skillsets = {"Skating":[["Edgework Outside","Backwards Skating","Stride Recovery","Skating Mechanics","Crossovers","Shouldering Speed","Feet in motion"]],
                  "Puck Reception":[["Catching puck in Hip Pocket","Dynamic Catch","Getting off the boards"]],
                  "Stickhandling":[["Loading Puck to Hip Pocket","Underhandling of Puck","Handedness Versatility","Deception w/ puck"]],
                  "Physical":[["Initiating Contact","Puck Protection with Body","Fitness Level"]],
                  "Play Away from the Puck":[["Shoulder Checks","NZ Angling","Unassisted Stops","Jumping in Shot Lanes","Awareness without puck","Net Front Presence"]],
                  "Passing":[["Slip Passes","Leveraging & creating seams","Pass Placement","Vision"]],
                  "Shooting":[["Coordination","Weightransfer","Tip"]]}   

# convert dico to df and change column names              
skillsets = pd.DataFrame(dico_skillsets).transpose().reset_index()
skillsets.columns = ["skills","habit"]

# vlookup like formula where skill sets added to stacked df on the basis on matching habits to the correct skill set
# for Tableau-friendly format
skillsets_list = []
for i in range(len(habits_results)):
    for y in range(len(skillsets)):
        if habits_results["habit"].iloc[i] in skillsets["habit"].iloc[y]: # if habit is in skill set df, then assign that skill set to empty list
            skillsets_list.append(str(skillsets["skills"].iloc[y]))
            #print(habits_results["habit"].iloc[i])

habits_results["skillset"] = skillsets_list

# -- merge information with habits
# habits_results is final df to send to tableau for viz
habits_results = habits_results.merge(info,on="code")

# -- weighted average skill sets
w_freq = [] # empty list to store weighted frequency
for i in range(len(habit)):
    for y in range(len(skillsets)):
        print(i)
        data1 = habits_results[(habits_results["code"]==habit["code"].iloc[i])&(habits_results["skillset"]==skillsets["skills"].iloc[y])]
        for z in range(len(data1)):
            w_freq.append(data1["frequency"].iloc[z]/sum(data1["frequency"])) # weight calculation, the total is calculated in Tableau for viz purposes
            
habits_results["w_freq"] = w_freq
     
#habits_results.to_excel("C:/Users/Admin/Desktop/Dev Project/Habits/Tableau_Data2.xlsx")
