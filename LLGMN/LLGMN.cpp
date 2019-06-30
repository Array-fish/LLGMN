#include"LLGMN.h"
#include<random>
#include<cmath>
#include<numeric>
#include<algorithm>
#include<string>
using namespace std;
llgmn::llgmn(const int input_size,const int class_num,const int component_size, const double inpsilon):
	input_size(input_size),
	class_num(class_num),
	component_size(component_size),
	inpsilon(inpsilon),
	modified_input_size(1 + input_size * (input_size + 3) / 2),
	sum_error(0)
{
	// vectorの要素数とかをコンストラクタ内でこうやって初期化するべきなのかわからない．
	weight = vector<vector<vector<double>>>(modified_input_size,vector<vector<double>>(class_num,vector<double>(component_size,0)));
	init_weight();
	output = vector<double>(class_num);
	teacher_data = vector<double>(class_num);
	modified_input.reserve(modified_input_size);
	second_output = vector<vector<double>>(class_num, vector<double>(component_size));
	sum_update_val = vector<vector<vector<double>>>(modified_input_size, vector<vector<double>>(class_num, vector<double>(component_size, 0)));
}
void llgmn::forward(const vector<vector<double>> &input_data, const vector<vector<double>>& teacher_data) {
	for (int n = 0; n < input_data.size(); ++n) {
		set_teacher_data(teacher_data[n]);
		input_loglinearization(input_data[n]);
		calc_second_layer_output(calc_second_layer_input());
		calc_third_layer_input();
		pool_update_val();
		pool_energy_func();
	}
}
void llgmn::backward() {
	update_weight_patch();
	init_sum_val();
}
void llgmn::init_weight() {
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0]
	for (int input = 0; input < modified_input_size; ++input) {
		for (int classi = 0; classi < class_num; ++classi) {
			for (int com = 0; com < component_size; ++com) {
				weight[input][classi][com] = rand01(mt);
			}
		}
		// なんか書いてあった，正し第2層の説明で書いてあったのでここで重みを0にするのが正しいかどうかはわからない
		// うまくいかなったら第2層の入力を計算するところで0として扱うように変更する．
		// 結局この0って何なんだろうね．
		weight[input][class_num - 1][component_size - 1] = 0;
	}
}

void llgmn::input_loglinearization(const vector<double>& input) {
	modified_input.clear();
	modified_input.push_back(1);
	for (int i = 0; i < input_size; ++i) {
		modified_input.push_back(input[i]);
	}
	for (int i = 0; i < input_size; ++i) {
		for (int n = i; n < input_size; ++n) {
			modified_input.push_back(input[i] * input[n]);
		}
	}
}
vector<vector<double>> llgmn::calc_second_layer_input() {
	vector <vector<double>> second_input(class_num,vector<double>(component_size,0));
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_size; ++com) {
			for (int mi = 0; mi < modified_input_size; ++mi) {
				second_input[cls][com] += modified_input[mi] * weight[mi][cls][com];
			}
		}
	}
	return second_input;
}
void llgmn::calc_second_layer_output(const vector<vector<double>>& second_input) {
	double sum_exp = 0;
	for (int cls = 0; cls < class_num; ++cls) {
		sum_exp += accumulate(second_input[cls].begin(), second_input[cls].end(), 0.0, [](double pool, double i) {return pool + exp(i); });
	}
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_size; ++com) {
			second_output[cls][com] = exp(second_input[cls][com]) / sum_exp;
		}
	}
}
void llgmn::calc_third_layer_input() {
	for (int cls = 0; cls < class_num; ++cls) {
		output[cls] = accumulate(second_output[cls].begin(), second_output[cls].end(), 0.0);
	}
}
void llgmn::pool_energy_func() {
	for (int cls = 0; cls < class_num; ++cls) {
		if (output[cls] <= 0) {
			if (output[cls] == 0) {
				throw "class " + to_string(cls) + " output[cls] is 0.";
			}
			throw "class " + to_string(cls) + " output[cls] is 0 or negative value.";
		}
		sum_error += -teacher_data[cls] * log(output[cls]);
	}
}
void llgmn::pool_update_val() {
	for (int mi = 0; mi < modified_input_size; ++mi) {
		for (int cls = 0; cls < class_num; ++cls) {
			double pre_calc = (output[cls] - teacher_data[cls]) * modified_input[mi] / output[cls];
			for (int com = 0; com < component_size; ++com) {
				sum_update_val[mi][cls][com] += pre_calc * second_output[cls][com];
			}
		}
	}
}
void llgmn::update_weight_patch() {
	for (int mi = 0; mi < modified_input_size; ++mi) {
		for (int cls = 0; cls < class_num; ++cls) {
			for (int com = 0; com < component_size; ++com) {
				// ターミナルラーニングではinpsilonのところをTAに変える．
				//weight[mi][cls][com] += -inpsilon*sum_update_val[mi][cls][com];
				weight[mi][cls][com] += -terminal_attractor(0.8,sum_update_val,1000)* sum_update_val[mi][cls][com];
			}
		}
	}
}
void llgmn::init_sum_val() {
	sum_error = 0;
	sum_update_val = vector<vector<vector<double>>>(modified_input_size, vector<vector<double>>(class_num, vector<double>(component_size, 0)));
}
double llgmn::terminal_attractor(double beta, const vector<vector<vector<double>>> &dj_dw,double learning_time) {
	double eta_t = pow(initial_J,1-beta)/learning_time/(1-beta);
	double sum_dj_dw = 0;
	for (int mi = 0; mi < modified_input_size; ++mi) {
		for (int cls = 0; cls < class_num; ++cls) {
			sum_dj_dw += accumulate(dj_dw[mi][cls].begin(), dj_dw[mi][cls].end(), 0.0, [](double acc, double i) {return acc + pow(i, 2); });
		}
	}
	//double unuse = sum_dj_dw;
	double gamma = pow(sum_error, beta) / sum_dj_dw;
	if (eta_t * gamma > 1000) {
		int nn=0;
	}
	return eta_t * gamma;
}