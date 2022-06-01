python3 ./src/VSM/dump_purchase.py
python3 ./src/VSM/dump_session.py
python3 ./src/VSM/dump_embedding.py
python3 src/VSM/feature_generate.py --with_purchase
python3 src/VSM/feature_generate.py --session_path dataset/test_sessions.pickle --output_path dataset/test_features/session.csv