// sum_dj_dwが小さくなりすぎるために結果的にgammaがすごく大きくなっておかしなことになってる．
// 例のlogの中が0になっている．
// どうにかする
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include"LLGMN.h"
#include"Parameter.h"
#include"utils.h"
using namespace std;
vector<vector<double>> get_vector_from_file(const string filename);
int main() {
	cout << "コンポーネント数を入力してね:";
	int component_size;
	//cin >> component_size;
	component_size = 2;
	cout << "学習率を入力してね:";
	double learning_rate;
	//cin >> learning_rate;
	learning_rate = 0.05;// ここ0.1にしたときerrorが-nan(ind)とかになるときがある．原因はわからん
	// データ読み込み
	vector<string> training_data_paths = get_training_data_paths();
	vector<string> training_label_paths = get_training_label_paths();
	vector<string> test_data_paths = get_test_data_paths();
	vector<string> test_label_paths = get_test_label_paths();
	try {// tryとcatchは特に関係がないので気にしないでください．
		// データ読み込み
		for (int j = 0; j < 7; ++j) {
			training_data = get_vector_from_file<double>(training_data_paths[j]);
			training_label = get_vector_from_file<double>(training_label_paths[j]);
			test_data1 = get_vector_from_file<double>(test_data_paths[j]);
			test_label1 = get_vector_from_file<double>(test_label_paths[j]);
			for (int i = 0; i < 20; ++i) {
				// データ処理部
				llgmn LL(data_size, class_num, component_size,
					learning_rate,result_folder,data_name);
				for (int i = 0; i < 1000; ++i) {
					LL.forward(training_data, training_label);
					cout << "loop:" << i << " error:" << LL.get_error() << endl;
					// スライド用データ取得
					vector<double> tmp_output = LL.get_output();
					//decltype(tmp_output)::iterator max_itr = max_element(tmp_output.begin(), tmp_output.end());

					if (i == 0) {
						LL.set_initial_J();// ターミナルアトラクタ用
					}
					LL.backward();
				}
				LL.forward(test_data1, test_label1);
				cout << "test error:" << LL.get_error() << endl;
				LL.evaluate(test_data1, test_label1, true, key_test1);
			}
		}
	}
	catch (const string estr) {
		cout << estr << endl;
		return -1;
	}
	return 0;
}
vector<vector<double>> get_vector_from_file(const string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		throw "Can't open "+ filename;
	}
	string str, str1;
	vector<vector<double>> data;
	while (getline(ifs, str)) {
		stringstream ss{ str };
		vector<double> tmp;
		while (getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}