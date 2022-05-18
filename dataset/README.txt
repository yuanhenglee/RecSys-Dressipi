This is the dataset for the Dressipi RecSys Challenge 2022.

train_purchases.csv
columns: session_id, item_id
The purchase that happened at the end of the session. One purchase per session.

train_sessions.csv
columns: session_id, item_id, date
The items that were viewed in a session. The "date" column is a timestamp to miliseconds. A session is equal to a day, so a session is one user's activity on one day. The session goes up to and not including the first time the user viewed the item that they bought in the end. The last item in the session will be the last item viewed before viewing the item that they bought. To find they item they bought link to train_purchases.csv on session_id.

item_features.csv
columns: item_id, feature_category_id, feature_value_id
The label data of items. A feature_category_id represents an aspect of the item such as "colour", the feature_value_id is the value for that aspect, e.g. "blue". Some items may not share many feature_cateogry_ids if they different types of items, for example trousers will share almost nothing with shirts. Even things like colour will not be shared, the colour aspect for trousers and shirts are two different feature_category_ids.

candidate_items.csv
columns: item_id
The candidate items to recommend from. This is one list for both the validation and test set.

test_leaderboard_sessions.csv
columns: session_id, item_id, date
The input sessions for prediction for the leaderboard.

test_final_sessions.csv
columns: session_id, item_id, date
The input sessions for prediction for determining the final winners.

