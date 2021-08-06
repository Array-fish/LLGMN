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
    //���s���̓����Ńt�H���_���쐬�i�p�����[�^�Ȃǌ��ʂ̃t�@�C�����i�[�j
    if (!fs::create_directories(time_prefix)) {
        cerr << time_prefix << " cannot be created." << endl;
    }
	// vector�̗v�f���Ƃ����R���X�g���N�^���ł�������ď���������ׂ��Ȃ̂��킩��Ȃ��D
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
		// �Ȃ񂩏����Ă������C������2�w�̐����ŏ����Ă������̂ł����ŏd�݂�0�ɂ���̂����������ǂ����͂킩��Ȃ�
		// ���܂������Ȃ������2�w�̓��͂��v�Z����Ƃ����0�Ƃ��Ĉ����悤�ɕύX����D
		// ���ǂ���0���ĉ��Ȃ񂾂낤�ˁD
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
				// �^�[�~�i�����[�j���O�ł�learning_rate�̂Ƃ����TA�ɕς���D
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
//�e�X�g
vector<vector<double>> llgmn::evaluate(vector<vector<double>>& test_data, const vector<vector<double>>& class_label, const bool output2csv, string filenum)
{
    //�ϐ��錾
    vector<int> class_true_positive(class_num + 1, 0);      //���ނ����N���X�Ɛ����̃N���X��������,���ނ������Ă����ꍇ�i�z���Ɨ\�����C���ۂɗz���j
    vector<int> class_true_negative(class_num + 1, 0);      //���ނ����N���X�������̃N���X�ł͂Ȃ��C���ނ������Ă����ꍇ�i�A���Ɨ\�����C���ۂɉA���j
    vector<int> class_false_positive(class_num + 1, 0);     //���ނ����N���X�������̃N���X��������,���ނ������Ă��Ȃ������ꍇ�i�z���Ɨ\�����C���ۂ͉A���j
    vector<int> class_false_negative(class_num + 1, 0);     //���ނ����N���X�������̃N���X�ł͂Ȃ��C���ނ������Ă��Ȃ������ꍇ�i�A���Ɨ\�����C���ۂ͗z���j
    vector<vector<int>> confusion_matrix(class_num + 1, vector<int>(class_num + 1, 0));    // confusion_matrix[true_class][predicted_class]
    int log_size = 1 + input_dim * (input_dim + 3) / 2;     //����`�ϊ���̃f�[�^�T�C�Y
    vector<vector<double>> transform_data(test_data.size(), vector<double>(log_size));     //����`�ϊ���̃f�[�^
    //posterior_probability.resize(test_data.size(), vector<double>(class_num + 1, 0));      //����m���̔z����������i�f�[�^�T�C�Y���e�X�g�f�[�^�p�ɕύX�j
    vector<double> posterior_probability;
    int max_class_index;    //�����N���X
    int max_prob_index;     //�\���N���X
    vector<vector<double>> result(class_num + 1, vector<double>(4, 0.0));    // �e��]���w�W���i�[ result[class][indeces] �C���f�b�N�X�̏����� [accuracy, precision, recall, F-measure]
    vector<double> macro_average(4, 0.0);       //�S�N���X�̕]���w�W�̕��ρi�P�����ς̂��߃}�N������[�e�Z�b�g�ŕ]���l���v�Z���Ă���C�����𕽋ς���]�j
    double micro_average;                       //�}�C�N������[n��̃e�X�g�����v���Ă���C�]���l���v�Z����]

    //���x�����o��
    string filename = time_prefix + "\\output_label" + filenum + ".csv";
    ofstream ofs(filename); if (ofs.fail()) { throw "Can't open " + filename; }


    for (int d = 0; d < test_data.size(); d++) {
        //�l�b�g���[�N�Ŏ���m�����Z�o
        one_forward(test_data[d], posterior_probability);
        // �f�[�^�����w�K�N���X���܂�ł��邯�ǁALL�ł͖��w�K�N���X��z�肵�Ă��Ȃ��̂ōŏ���0��ǉ�����
        posterior_probability.push_back(posterior_probability.back());
        for (int i = class_num-1; i > 0; --i) {
            posterior_probability[i] = posterior_probability[i - 1];
        }
        posterior_probability[0] = 0;
        //����m�����烉�x�����o��
        for (int c = 0; c <= class_num; c++) {
            if (_isnan(posterior_probability[c])) {  //nan�������疢�w�K�N���X�ɕ���
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

        //�����̃N���X�Ɨ\�������N���X���i�[
        max_class_index = distance(class_label[d].begin(), max_element(class_label[d].begin(), class_label[d].end()));
        max_prob_index = distance(posterior_probability.begin(), max_element(posterior_probability.begin(), posterior_probability.end()));
        //�����s��ɒǉ�
        confusion_matrix[max_class_index][max_prob_index]++;

        //�����N���X�Ɨ\���N���X���r���C�����s��ɕ���
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

    ofs.close();    //���x�����o�͂����t�@�C�������

    //�����s��̏o��
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

    //�e�N���X���Ƃɕ]���w�W���o�� [accuracy, precision, recall, F-measure]
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

    //�}�N�����ς��v�Z
    for (int i = 0; i < 4; ++i) {
        for (int c = 0; c < class_num + 1; c++) {
            macro_average[i] += result[c][i];
        }
        macro_average[i] /= (class_num + 1);
    }

    //�S�̂̕��ϕ]���w�W���o�́i�P�����ς̂��߃}�N�����ρj
    std::cout << "macro_average" << endl
        << "accuracy: " << fixed << setprecision(8) << macro_average[0] << endl
        << "precision: " << fixed << setprecision(8) << macro_average[1] << endl
        << "recall: " << fixed << setprecision(8) << macro_average[2] << endl
        << "F-measure(ave): " << fixed << setprecision(8) << macro_average[3] << endl
        << "F-measure(calc): " << fixed << setprecision(8)
        << 2 * macro_average[1] * macro_average[2] / (macro_average[1] + macro_average[2]) << endl;

    //�}�C�N�����ς��o��
    micro_average = accumulate(class_true_positive.begin(), class_true_positive.end(), 0.0) / test_data.size();     //�^�z���̑��a�����C�e�X�g�f�[�^�����Ŋ���
    std::cout << "micro_average: " << fixed << setprecision(8) << micro_average << endl;

    //�]�����ʂ�csv�ɏo��
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