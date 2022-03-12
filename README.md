# WHKY Player Development Project
## Scouting Automated Ratings Analyzing Habits (SARAH)
### Co-authored by Noah Chaikof, Katia Clément-Heydra, Carleen Markey, Mikael Nahabedian, Adam Pilotte & Mairead Shaw 

### Summary
The project serves a two-fold purpose: to reduce the time that scouts and coaches spend trying to identify what players have foundational on-ice habits, and to streamline the process of evaluating the developmental progress of a players' habits. Essentially what we did was first look at the various national women's hockey teams and identify the set of "habits" a player regularly executes (i.e., edgework, catching the puck in the hip pocket, pass placement, etc). 

Combining the dataset of players' habits with a set of players' microstats (entries via pass/stickhandling, exits via stickhandling/pass, accurate/inaccurate passes, etc.), we developed a random forest classification model to accurately predict if a player possesses a certain habit based on their set of microstats. We also used random forest regression on our data to see how habits impacted each specific microstat. Combining this with an estimate of how frequently players used each habit, we created a Belfry performance matrix for a player's habits based entirely on their microstats. To help coaches, scouts, and anyone else access & use these tools, we've also created an interactive visualization for these models using our training dataset of national women's hockey teams in the last Worlds and Olympics.

### SARAH_paper.pdf
Contains a pdf with the paper outlining the process to build the SARAH models.

### player_dev_code.py
Contains the python code to manipulate the data, train the models and obtain the resulting habits for the player development matrix.

### SARAH Applications
Contains the Tableau dashboard with the player development matrix for the 262 skaters in the data set and 2 case studies outlining development plans for Ashton Bell and Abby Roque.
