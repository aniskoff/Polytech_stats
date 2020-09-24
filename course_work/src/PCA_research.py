import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

PATH_TO_DATA_FILE = '../data/data.xlsx'
PLOTS_DIR = 'plots'


def get_samples_dfs_from_file(filename=PATH_TO_DATA_FILE):
    xl_file = pd.ExcelFile(filename)

    samples_dfs = {sheet_name: xl_file.parse(sheet_name)
                   for sheet_name in xl_file.sheet_names}
    return samples_dfs['1'], samples_dfs['2']


def drop_extra_columns(df):
    return df.drop(columns=['№ п.п', 'проба'])


class DataPreprocesser:

    @staticmethod
    def fill_missing_values(df, fill_strategy='median'):
        tmp_df = df.apply(pd.to_numeric, errors='coerce')
        return tmp_df.fillna(getattr(tmp_df, fill_strategy)())

    @staticmethod
    def scale_data(df, Scaler=StandardScaler):
        return pd.DataFrame(Scaler().fit_transform(df))

    @staticmethod
    def drop_outliers_with_IQR(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        df_filtered = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_filtered


class Plotter:

    DEFAULT_IMAGE_EXT = 'png'

    @staticmethod
    def draw_event_plot(PCs_df, sample_1_size, description):
        plt.eventplot(PCs_df.iloc[:sample_1_size, 0], colors='r')
        plt.eventplot(PCs_df.iloc[sample_1_size:, 0], colors='b')
        plt.title(f'PC1 event plot of two given samples \n ({description})')
        fname = '.'.join(['event ' + description, Plotter.DEFAULT_IMAGE_EXT])
        plt.savefig(os.path.join(PLOTS_DIR, fname))
        plt.show()

    @staticmethod
    def draw_samples_on_scatter_plot(PCs_df, sample_1_size, description):
        plt.scatter(x=PCs_df.iloc[:sample_1_size, 0],
                    y=PCs_df.iloc[:sample_1_size, 1],
                    c='b',
                    label='sample 1',
                    alpha=0.4)
        plt.scatter(x=PCs_df.iloc[sample_1_size:, 0],
                    y=PCs_df.iloc[sample_1_size:, 1],
                    c='r',
                    label='sample 2',
                    alpha=0.4)
        plt.legend(loc='upper left')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Scatter plot of two given samples using PC1 and PC2 \n ({description})')
        fname = '.'.join(['scatter ' + description, Plotter.DEFAULT_IMAGE_EXT])
        plt.savefig(os.path.join(PLOTS_DIR, fname))
        plt.show()

    @staticmethod
    def draw_PCA_scree_plot(pca, description):
        PC_num = range(pca.n_components_)
        plt.bar(PC_num, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCs')
        plt.ylabel('explained variance ratio')
        plt.xticks(PC_num)
        plt.title(f'PCs explained variance distribution \n ({description})')
        fname = '.'.join(['scree ' + description, Plotter.DEFAULT_IMAGE_EXT])
        plt.savefig(os.path.join(PLOTS_DIR, fname))
        plt.show()


def analyze_component(pca, component_idx, features):
    PC = np.array(pca.components_[component_idx])
    top_features_idxs = np.argsort(np.abs(PC))[::-1]
    top_features = np.array(features)[top_features_idxs]
    return top_features, PC[top_features_idxs]


def pretty_print_feature_importance(top_features_pc, ranking):
    print('Название признака: {:<10} Вклад в PC1 '.format(''))
    for (idx, feature) in enumerate(top_features_pc):
        print('{}. {}{:<10}{:.5f}'.format(idx + 1, feature, '', ranking[feature]))


def main():
    sample_1_df, sample_2_df = get_samples_dfs_from_file()
    sample_1_df, sample_2_df = drop_extra_columns(sample_1_df), drop_extra_columns(sample_2_df)
    features = sample_1_df.columns.values

    preprocessor = DataPreprocesser()
    sample_1_df, sample_2_df = preprocessor.fill_missing_values(sample_1_df), \
                               preprocessor.fill_missing_values(sample_2_df)
    sample_1_df, sample_2_df = preprocessor.drop_outliers_with_IQR(sample_1_df), \
                               preprocessor.drop_outliers_with_IQR(sample_2_df)

    all_samples_df = sample_1_df.append(sample_2_df)

    all_samples_df = preprocessor.scale_data(all_samples_df)
    
    n_features = len(all_samples_df.columns)
    sample_1_size = len(sample_1_df)

    pca = PCA(n_components=n_features)
    pcs_df = pd.DataFrame(pca.fit_transform(all_samples_df))

    # description = 'Data is scaled AND outliers are dropped'
    # plotter = Plotter()
    # plotter.draw_PCA_scree_plot(pca, description=description)
    # plotter.draw_samples_on_scatter_plot(pcs_df, sample_1_size, description=description)
    # plotter.draw_event_plot(pcs_df, sample_1_size, description=description)
    pca_1 = PCA(n_components=11)
    pca_1.fit(sample_1_df)

    pca_2 = PCA(n_components=11)
    pca_2.fit(sample_2_df)

    print("Sample 1")
    top_features_pc1, top_coeffs_pc1 = analyze_component(pca_1, 0, features)
    print()

    d1 = pd.DataFrame.from_dict({'Feature name': top_features_pc1,
                                'Contribution to PC1': top_coeffs_pc1})
    print(d1)

    print("Sample 2")
    top_features_pc1, top_coeffs_pc1 = analyze_component(pca_2, 0, features)
    print()

    d2 = pd.DataFrame.from_dict({'Feature name': top_features_pc1,
                                'Contribution to PC1': top_coeffs_pc1})
    print(d2)

    print("Sample 1 + 2")
    top_features_pc1, top_coeffs_pc1 = analyze_component(pca, 0, features)
    print()

    d3 = pd.DataFrame.from_dict({'Feature name': top_features_pc1,
                                'Contribution to PC1': top_coeffs_pc1})
    print(d3)

    # print("Sample 2")
    # top_features_pc1, ranking_1 = analyze_component(pca_2, 0, features)
    # pretty_print_feature_importance(top_features_pc1, ranking_1)
    # print()
    #
    # print("Sample 1 + 2")
    # top_features_pc1, ranking_1 = analyze_component(pca, 0, features)
    # pretty_print_feature_importance(top_features_pc1, ranking_1)
    #
    #
    # # top_features_pc2, ranking_2 = analyze_component(pca, 1, features)
    # # print(top_features_pc2)
    # # print(ranking_2)
    # # print(sum(ranking_2.values()))
    #
    # pass

if __name__ == '__main__':
    main()
