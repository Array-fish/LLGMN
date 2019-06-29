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
using namespace std;
vector<vector<double>> get_vector_from_file(const string filename);
int main() {
	cout << "コンポーネント数を入力してね:";
	int component_size;
	//cin >> component_size;
	component_size = 2;
	cout << "学習率を入力してね:";
	double inpsilon;
	//cin >> inpsilon;
	inpsilon = 0.05;// ここ0.1にしたときerrorが-nan(ind)とかになるときがある．原因はわからん
	// データ読み込み
	vector<vector<double>> input_data;
	vector<vector<double>> output_data;
	vector<vector<double>> test_input_data;
	vector<vector<double>> test_output_data;
	try {// tryとcatchは特に関係がないので気にしないでください．
		// データ読み込み
		input_data = get_vector_from_file("lea_sig.csv");
		output_data = get_vector_from_file("lea_T_sig.csv");
		test_input_data = get_vector_from_file("dis_sig.csv");
		test_output_data = get_vector_from_file("dis_T_sig.csv");
		// スライド作るようデータ取得
		ofstream ofs("error_data005.csv");
		ofstream ofs1("output_data000.csv");
		// データ処理部
		llgmn LL(input_data[0].size(), output_data[0].size(), component_size, inpsilon);
		for (int i = 0; i < 1000; ++i) {
			LL.forward(input_data, output_data);
			cout << "loop:" << i << " error:" << LL.get_error() << endl;
			// スライド用データ取得
			ofs << i << "," << LL.get_error() << endl;
			vector<double> tmp_output = LL.get_output();
			for (auto out : tmp_output) {
				ofs1 << out << ",";
			}
			decltype(tmp_output)::iterator max_itr = max_element(tmp_output.begin(),tmp_output.end());
			for (decltype(tmp_output)::iterator itr = tmp_output.begin(); itr != tmp_output.end(); itr++) {
				if (itr == max_itr) {
					ofs1 << "1,";
				}
				else {
					ofs1 << "0,";
				}
			}
			ofs1 << endl;
			// slide 用終わり
			if (i == 0) {
				LL.set_initial_J();// ターミナルアトラクタ用
			}
			LL.backward();
		}
		LL.forward(test_input_data, test_output_data);
		cout << "test error:" << LL.get_error() << endl;
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