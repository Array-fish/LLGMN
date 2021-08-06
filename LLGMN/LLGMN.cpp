#include"LLGMN.h"
#include"utils.h"
#include<random>
#include<cmath>
#include<numeric>
#include<algorithm>
#include<string>
#include<fstream>
#include<iomanip>
#include<iostream>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;
llgmn::llgmn(const int input_dim,const int class_num,const int component_size, const double learning_rate, const string output_dir,const string data_name):
	input_dim(input_dim),
	class_num(class_num),
	component_size(component_size),
	learning_rate(learning_rate),
	modified_input_dim(1 + input_dim * (input_dim + 3) / 2),
	sum_error(0),
    time_prefix(output_dir+get_date_sec()+"_"+data_name)
{
    //実行時の日時でフォルダを作成（パラメータなど結果のファイルを格納）
    if (!fs::create_directories(time_prefix)) {
        cerr << time_prefix << " cannot be created." << endl;
    }
	// vectorの要素数とかをコンストラクタ内でこうやって初期化するべきなのかわからない．
	weight = vector<vector<vector<double>>>(modified_input_dim,vector<vector<double>>(class_num,vector<double>(component_size,0)));
	init_weight();
	output = vector<double>(class_num);
	teacher_data = vector<double>(class_num);
	modified_input.reserve(modified_input_dim);
	second_output = vector<vector<double>>(class_num, vector<double>(component_size));
	sum_update_val = vector<vector<vector<double>>>(modified_input_dim, vector<vector<double>>(class_num, vector<double>(component_size, 0)));
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

void llgmn::one_forward(vector<double>& data, vector<double>& posterior_probability) {
    input_loglinearization(data);
    calc_second_layer_output(calc_second_layer_input());
    calc_third_layer_input();
    posterior_probability = get_output();
}
void llgmn::backward() {
	update_weight_patch();
	init_sum_val();
}
void llgmn::init_weight() {
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0]
	for (int input = 0; input < modified_input_dim; ++input) {
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
	for (int i = 0; i < input_dim; ++i) {
		modified_input.push_back(input[i]);
	}
	for (int i = 0; i < input_dim; ++i) {
		for (int n = i; n < input_dim; ++n) {
			modified_input.push_back(input[i] * input[n]);
		}
	}
}
vector<vector<double>> llgmn::calc_second_layer_input() {
	vector <vector<double>> second_input(class_num,vector<double>(component_size,0));
	for (int cls = 0; cls < class_num; ++cls) {
		for (int com = 0; com < component_size; ++com) {
			for (int mi = 0; mi < modified_input_dim; ++mi) {
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
	for (int mi = 0; mi < modified_input_dim; ++mi) {
		for (int cls = 0; cls < class_num; ++cls) {
			double pre_calc = (output[cls] - teacher_data[cls]) * modified_input[mi] / output[cls];
			for (int com = 0; com < component_size; ++com) {
				sum_update_val[mi][cls][com] += pre_calc * second_output[cls][com];
			}
		}
	}
}
void llgmn::update_weight_patch() {
	for (int mi = 0; mi < modified_input_dim; ++mi) {
		for (int cls = 0; cls < class_num; ++cls) {
			for (int com = 0; com < component_size; ++com) {
				if (cls == class_num - 1 && com == component_size - 1)
					continue; // Final weight is always 0;
				// ターミナルラーニングではlearning_rateのところをTAに変える．
				//weight[mi][cls][com] += -learning_rate*sum_update_val[mi][cls][com];
				weight[mi][cls][com] += -terminal_attractor(0.8,sum_update_val,1000)* sum_update_val[mi][cls][com];
			}
		}
	}
}
void llgmn::init_sum_val() {
	sum_error = 0;
	sum_update_val = vector<vector<vector<double>>>(modified_input_dim, vector<vector<double>>(class_num, vector<double>(component_size, 0)));
}
double llgmn::terminal_attractor(double beta, const vector<vector<vector<double>>> &dj_dw,double learning_time) {
	double eta_t = pow(initial_J,1-beta)/learning_time/(1-beta);
	double sum_dj_dw = 0;
	for (int mi = 0; mi < modified_input_dim; ++mi) {
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
//テスト
vector<vector<double>> llgmn::evaluate(vector<vector<double>>& test_data, const vector<vector<double>>& class_label, const bool output2csv, string filenum)
{
    //変数宣言
    vector<int> class_true_positive(class_num + 1, 0);      //分類したクラスと正解のクラスが同じで,分類があっていた場合（陽性と予測し，実際に陽性）
    vector<int> class_true_negative(class_num + 1, 0);      //分類したクラスが正解のクラスではなく，分類があっていた場合（陰性と予測し，実際に陰性）
    vector<int> class_false_positive(class_num + 1, 0);     //分類したクラスが正解のクラスだったが,分類があっていなかった場合（陽性と予測し，実際は陰性）
    vector<int> class_false_negative(class_num + 1, 0);     //分類したクラスが正解のクラスではなく，分類があっていなかった場合（陰性と予測し，実際は陽性）
    vector<vector<int>> confusion_matrix(class_num + 1, vector<int>(class_num + 1, 0));    // confusion_matrix[true_class][predicted_class]
    int log_size = 1 + input_dim * (input_dim + 3) / 2;     //非線形変換後のデータサイズ
    vector<vector<double>> transform_data(test_data.size(), vector<double>(log_size));     //非線形変換後のデータ
    //posterior_probability.resize(test_data.size(), vector<double>(class_num + 1, 0));      //事後確率の配列を初期化（データサイズをテストデータ用に変更）
    vector<double> posterior_probability;
    int max_class_index;    //正解クラス
    int max_prob_index;     //予測クラス
    vector<vector<double>> result(class_num + 1, vector<double>(4, 0.0));    // 各種評価指標を格納 result[class][indeces] インデックスの順序は [accuracy, precision, recall, F-measure]
    vector<double> macro_average(4, 0.0);       //全クラスの評価指標の平均（単純平均のためマクロ平均[各セットで評価値を計算してから，それらを平均する]）
    double micro_average;                       //マイクロ平均[n回のテストを合計してから，評価値を計算する]

    //ラベルを出力
    string filename = time_prefix + "\\output_label" + filenum + ".csv";
    ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }


    for (int d = 0; d < test_data.size(); d++) {
        //ネットワークで事後確率を算出
        one_forward(test_data[d], posterior_probability);
        // データが未学習クラスを含んでいるけど、LLでは未学習クラスを想定していないので最初に0を追加する
        posterior_probability.push_back(posterior_probability.back());
        for (int i = class_num-1; i > 0; --i) {
            posterior_probability[i] = posterior_probability[i - 1];
        }
        posterior_probability[0] = 0;
        //事後確率からラベルを出力
        for (int c = 0; c <= class_num; c++) {
            if (_isnan(posterior_probability[c])) {  //nanだったら未学習クラスに分類
                if (c == 0) {
                    ofs << 1 << ",";
                }
                else {
                    ofs << 0 << ",";
                }
            }
            else {
                ofs << posterior_probability[c] << ",";
            }
        }
        ofs << endl;

        //正解のクラスと予測したクラスを格納
        max_class_index = distance(class_label[d].begin(), max_element(class_label[d].begin(), class_label[d].end()));
        max_prob_index = distance(posterior_probability.begin(), max_element(posterior_probability.begin(), posterior_probability.end()));
        //混同行列に追加
        confusion_matrix[max_class_index][max_prob_index]++;

        //正解クラスと予測クラスを比較し，混同行列に分類
        if (max_class_index == max_prob_index) {
            class_true_positive[max_class_index]++;
            for (int c = 0; c <= class_num; c++) {
                if (c != max_class_index)
                    class_true_negative[c]++;
            }
        }
        else {
            class_false_positive[max_prob_index]++;
            class_false_negative[max_class_index]++;
            for (int c = 0; c < class_num + 1; c++) {
                if (c != max_class_index && c != max_prob_index)
                    class_true_negative[c]++;
            }
        }
    }

    ofs.close();    //ラベルを出力したファイルを閉じる

    //混同行列の出力
    std::cout << "confusion_matrix" << endl;
    std::cout << "         class_0";
    for (int c = 1; c <= class_num; c++) {
        std::cout << "  class_" << c;
    }
    std::cout << endl;

    for (int true_cls = 0; true_cls <= class_num; true_cls++) {
        std::cout << "class_" << to_string(true_cls);
        for (int pred_cls = 0; pred_cls <= class_num; pred_cls++) {
            std::cout << "     " << setw(4) << confusion_matrix[true_cls][pred_cls];
        }
        std::cout << std::endl;
    }

    //各クラスごとに評価指標を出力 [accuracy, precision, recall, F-measure]
    for (int c = 0; c < class_num + 1; c++) {
        //accuracy
        result[c][0] = static_cast<double>(class_true_positive[c] + class_true_negative[c])
            / (class_true_positive[c] + class_false_positive[c] + class_false_negative[c] + class_true_negative[c]);
        //precision
        result[c][1] = static_cast<double>(class_true_positive[c]) / (class_true_positive[c] + class_false_positive[c]);
        //recall
        result[c][2] = static_cast<double>(class_true_positive[c]) / (class_true_positive[c] + class_false_negative[c]);
        //F-measure
        result[c][3] = 2 * result[c][1] * result[c][2] / (result[c][1] + result[c][2]);

        std::cout << "class: " << setw(3) << c
            << ", accuracy: " << fixed << setprecision(8) << result[c][0]
            << ", precision: " << fixed << setprecision(8) << result[c][1]
            << ", recall: " << fixed << setprecision(8) << result[c][2]
            << ", F-measure:" << fixed << setprecision(8) << result[c][3] << endl;
    }

    //マクロ平均を計算
    for (int i = 0; i < 4; ++i) {
        for (int c = 0; c < class_num + 1; c++) {
            macro_average[i] += result[c][i];
        }
        macro_average[i] /= (class_num + 1);
    }

    //全体の平均評価指標を出力（単純平均のためマクロ平均）
    std::cout << "macro_average" << endl
        << "accuracy: " << fixed << setprecision(8) << macro_average[0] << endl
        << "precision: " << fixed << setprecision(8) << macro_average[1] << endl
        << "recall: " << fixed << setprecision(8) << macro_average[2] << endl
        << "F-measure(ave): " << fixed << setprecision(8) << macro_average[3] << endl
        << "F-measure(calc): " << fixed << setprecision(8)
        << 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;

    //マイクロ平均を出力
    micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();     //真陽性の総和を取り，テストデータ総数で割る
    std::cout << "micro_average: " << fixed << setprecision(8) << micro_average << endl;

    //評価結果をcsvに出力
    if (output2csv) {
        ofstream ofs(time_prefix + "\\evaluate" + filenum + ".csv");

        ofs << "confusion matrix" << endl;
        for (int true_cls = 0; true_cls <= class_num; true_cls++) {
            for (int pred_cls = 0; pred_cls <= class_num; pred_cls++) {
                ofs << confusion_matrix[true_cls][pred_cls] << ",";
            }
            ofs << std::endl;
        }

        ofs << "class,accuracy,precision,recall,F-measure" << endl;
        for (int c = 0; c < class_num + 1; c++) {
            ofs << c << "," << result[c][0] << "," << result[c][1] << "," << result[c][2] << "," << result[c][3] << endl;
        }
        ofs << "macro_average" << endl
            << "accuracy," << macro_average[0] << endl
            << "precision," << macro_average[1] << endl
            << "recall," << macro_average[2] << endl
            << "F-measure(ave)," << macro_average[3] << endl
            << "F-measure(calc),"
            << 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;

        //micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();
        ofs << "micro_average," << micro_average << endl;
    }

    return result;
}