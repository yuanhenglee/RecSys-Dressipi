# make sure dataset/train_features and dataset/test_features exist
mkdir -p dataset/train_features
mkdir -p dataset/test_features

python3 ./src/VSM/dump_purchase.py
python3 ./src/VSM/dump_session.py
python3 src/VSM/dump_session.py --session_path dataset/test_leaderboard_sessions.csv --output_path dataset/test_sessions.pickle
python3 ./src/VSM/dump_embedding.py
python3 src/VSM/feature_generate.py --with_purchase
python3 src/VSM/feature_generate.py --session_path dataset/test_sessions.pickle --output_path dataset/test_features/session.csv