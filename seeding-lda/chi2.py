import pandas as pd

from modules.embeddings import feature_select_chi2

# aspect_list = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']
aspect_list = ['giá', 'dịch_vụ', 'ship', 'hiệu_năng', 'chính_hãng', 'cấu_hình', 'phụ_kiện', 'mẫu_mã']
TRAIN_PATH = 'data/input/SP/tech_shopee.csv'
domain = 'tech_shopee'
if __name__ == '__main__':
    for aspect in aspect_list:
        TASK_NAME = 'SP_' + domain + '_' + aspect
        COL_LABEL_NAME = aspect
        train_df = pd.read_csv(TRAIN_PATH)
        print('============= Start Chi-Square Test for CSI =============================')
        feature_select_chi2(train_df['cmt'], train_df[COL_LABEL_NAME], task_name=TASK_NAME)
        print("Chi-Square Test for {} DONE!".format(TASK_NAME))
