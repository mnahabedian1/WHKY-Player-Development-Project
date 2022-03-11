# WHKY Player Development Project
Scouting Automated Ratings Analyzing Habits (SARAH)

Summary
The project serves a two-fold purpose: to reduce the time that scouts and coaches spend trying to identify what players have foundational on-ice habits, and to streamline the process of evaluating the developmental progress of a players' habits. Essentially what we did was first look at the various national women's hockey teams and identify the set of "habits" a player regularly executes (ie edgework, catching the puck in the hip pocket, accurate pass placement, etc). 

Combining the dataset of players' habits with a set of players' microstats (takeaways, accurate passes, etc.), we developed a random forest classification model to accurately predict if a player possesses a certain habit based on their set of microstats. We also used random forest regression on our data to see how habits impacted each specific microstat. Combining this with an estimate of how frequently players used each habit, we created a Belfry performance matrix for a player's habits based entirely on their microstats. To help coaches, scouts, and anyone else access & use these tools, we've also created an interactive visualization for these models using our training dataset of national women's hockey teams in the last Worlds and Olympics.
