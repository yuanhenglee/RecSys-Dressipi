# This is the dataset for the Dressipi RecSys Challenge 2022.

## train_purchases.csv
* columns: session_id, item_id

* The purchase that happened at the end of the session. One purchase per session.

## train_sessions.csv
* columns: session_id, item_id, date

* The items that were viewed in a session. The "date" column is a timestamp to miliseconds. A session is equal to a day, so a session is one user's activity on one day. The session goes up to and not including the first time the user viewed the item that they bought in the end. The last item in the session will be the last item viewed before viewing the item that they bought. To find they item they bought link to train_purchases.csv on session_id.

## item_features.csv
* columns: item_id, feature_category_id, feature_value_id

* The label data of items. A feature_category_id represents an aspect of the item such as "colour", the feature_value_id is the value for that aspect, e.g. "blue". Some items may not share many feature_cateogry_ids if they different types of items, for example trousers will share almost nothing with shirts. Even things like colour will not be shared, the colour aspect for trousers and shirts are two different feature_category_ids.

## candidate_items.csv
* columns: item_id

* The candidate items to recommend from. This is one list for both the validation and test set.

## test_leaderboard_sessions.csv
* columns: session_id, item_id, date

* The input sessions for prediction for the leaderboard.

## test_final_sessions.csv
* columns: session_id, item_id, date

* The input sessions for prediction for determining the final winners.

## clustering_allltems_file.csv
* columns: categoryTotalCount,category,categoryMemberItem

* All categories with its items.
* clustering by categoryID

## clustering_candidate_file.csv
* columns: categoryTotalCount,category,categoryMemberItem

* All categories with its candidate items.
* clustering by categoryID

## sessionClustering.csv
* columns: size, sessionID, itemID

* clustering by sessionID
***
## splitTrainSessions.py
It can split train_sessions.csv according to your specific time, and generate the data you want.

<font color=#FF0000>Attention: The set of month = {01, 02, 03, ..., 09, 10, 11, 12}</font>

* Example 1:
```py
python3 splitTrainSessions.py -y 2021 -d 03
```
Data will be stored in dataset/2021-03.csv.

* Example 2:
```py
python3 splitTrainSessions.py -y 2020
```
Data will be stored in dataset/2020.csv.

## clustering.cpp
