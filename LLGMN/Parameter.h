#pragma once
#ifndef __INC_PARAMETER_H
#define __INC_PARAMETER_H

#include <vector>
#include <string>

//-----------------------------変数宣言-----------------------------------------------------------
//ネットワークパラメータ
const int    class_num = 4;			//クラス数
const int    component_num = 2;			//コンポーネント数
const int    data_size = 4;			//入力次元数
const double eta_w = 1e-2;		//重みwの学習定数
const int    max_times = 2000;		//学習回数の上限
const double correction = 1e-3;		//分散sigmaの行列式が計算できないとき用の補正項c

//
//データ
std::vector<std::vector<double>> training_data;			//教師データ
std::vector<std::vector<double>>	 training_label;		//教師データのラベル
std::vector<std::vector<double>> test_data;			//テストデータ
std::vector<std::vector<double>>    test_label;			//テストデータのラベル

//データパス
const std::string key_train = "";			//学習用データの番号など
const std::string key_test = "";			//テストデータの番号など

const std::string data_name = "train" + key_train + "";		//結果保存先ディレクトリの名前の末尾に追加される
const std::string data_folder = "D:\\nac_data\\";					//データ読み込みフォルダのパス
//const std::string result_folder = "C:\\Users\\watanabe\\Documents\\nacgmn_result\\0721\\";		//結果保存フォルダのパス
//const std::string data_name = "first_data";		//結果保存先ディレクトリの名前の末尾に追加される
//const std::string data_folder = "C:\\Users\\watanabe\\OneDrive - 横浜国立大学\\shimaLaboratory\\data\\created_data\\first_data\\";
const std::string result_folder = "C:\\Users\\watanabe\\Documents\\llgmn_result\\1015_1\\";

std::vector<std::string> get_training_data_paths() {
	std::vector<std::string> training_data_path_list;
	for (int i = 0; i < 7; ++i) {
		training_data_path_list.push_back(data_folder + "EMG1_train_data" + std::to_string(i) + ".csv");
	}
	return training_data_path_list;
}
std::vector<std::string> get_training_label_paths() {
	std::vector<std::string> training_label_path_list;
	for (int i = 0; i < 7; ++i) {
		training_label_path_list.push_back(data_folder + "EMG1_train_cls" + std::to_string(i) + ".csv");
	}
	return training_label_path_list;
}
std::vector<std::string> get_test_data_paths() {
	std::vector<std::string> test_data_path_list;
	for (int i = 0; i < 7; ++i) {
		test_data_path_list.push_back(data_folder + "EMG1_test_data" + std::to_string(i) + ".csv");
	}
	return test_data_path_list;
}
std::vector<std::string> get_test_label_paths() {
	std::vector<std::string> test_label_path_list;
	for (int i = 0; i < 7; ++i) {
		test_label_path_list.push_back(data_folder + "EMG1_test_cls" + std::to_string(i) + ".csv");
	}
	return test_label_path_list;
}

#endif

