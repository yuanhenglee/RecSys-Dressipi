with open( '../../../dressipi_recsys2022/item_features.csv' ) as f:
    for line in f.readlines()[1:]:
        line_list = line.split(',')
        with open( '../../dataset/feature_value_sep/' + line_list[0] + '.txt', 'a') as doc:
            doc.write(' f'+line_list[1]+' v'+line_list[2].strip())
