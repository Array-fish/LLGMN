#pragma once
#ifndef LLGMN_H
#define LLGMN_H
#include<vector>
using namespace std;
class llgmn {
private:
	vector<double> modified_input;
	const int modified_input_size;
	const int input_size;
	const int class_num;
	double initial_J;// ターミナルアトラクタ用
	// 学習率
	const double inpsilon;
	// クラスのコンポーネントの数　コンポーネントの数はクラスごとに変わるのか？
	// 今回は入力が面倒なので各クラスでコンポーネント数は同じとして実装する．
	//vector<int> component_num_for_class;
	const int component_size;
	// weight[入力][クラス][コンポーネント]
	vector<vector<vector<double>>> weight;
	// second_output[クラス][コンポーネント]
	vector<vector<double>> second_output;
	// クラス分の大きさがある
	vector<double> output;
	// 教師データ クラス数と同じ大きさ
	vector<double> teacher_data;
	// データ数 クラスの中で持つ必要があるか？
	int data_num;
	// 誤差算出用　蓄積する誤差 どっかで初期化しなきゃいけなかった希ガス　batchの中の全てのデータの更新量の和
	double sum_error;
	// 更新量の蓄積 重み更新用 batchの中の全てのデータの更新量の和
	vector<vector<vector<double>>> sum_update_val;
	// function
	void init_weight();
	void input_transformation(const vector<double>& input);
	vector<vector<double>> calc_second_layer_input();
	void calc_second_layer_output(const vector<vector<double>>& second_input);
	void approximate_second_layer_output(const vector<vector<double>>& second_input);
	void calc_third_layer_input();
	void pool_energy_func();
	void pool_update_val();
	void update_weight_patch();
	void init_sum_val();
	void set_teacher_data(const vector<double>& teacher_data);
	double terminal_attractor(double beta, const vector<vector<vector<double>>>& dj_dw, double learning_time);
public:
	llgmn(const int input_size, const int class_num, const int component_size, const double inpsilon);
	virtual ~llgmn() {};
	// 前方向に計算をしてエネルギー関数を出す．
	void forward(const vector<vector<double>>& input_data, const vector<vector<double>>& teacher_data);
	// 誤差伝搬をする
	void backward();
	// エネルギー関数の値をとる
	double get_error() const;
	vector<double> get_output() const;
	void set_initial_J();// ターミナルアトラクタ用の最初のエネルギー関数

};
inline double llgmn::get_error() const {
	return sum_error;
}
inline void llgmn::set_teacher_data(const vector<double>& teacher_data) {
	this->teacher_data = teacher_data;
}
inline void llgmn::set_initial_J() {
	this->initial_J = this->sum_error;
}
inline vector<double> llgmn::get_output() const {
	return output;
}
#endif LLGMN_H