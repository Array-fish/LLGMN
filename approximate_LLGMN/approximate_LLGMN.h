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
	double initial_J;// �^�[�~�i���A�g���N�^�p
	// �w�K��
	const double inpsilon;
	// �N���X�̃R���|�[�l���g�̐��@�R���|�[�l���g�̐��̓N���X���Ƃɕς��̂��H
	// ����͓��͂��ʓ|�Ȃ̂Ŋe�N���X�ŃR���|�[�l���g���͓����Ƃ��Ď�������D
	//vector<int> component_num_for_class;
	const int component_size;
	// weight[����][�N���X][�R���|�[�l���g]
	vector<vector<vector<double>>> weight;
	// second_output[�N���X][�R���|�[�l���g]
	vector<vector<double>> second_output;
	// �N���X���̑傫��������
	vector<double> output;
	// ���t�f�[�^ �N���X���Ɠ����傫��
	vector<double> teacher_data;
	// �f�[�^�� �N���X�̒��Ŏ��K�v�����邩�H
	int data_num;
	// �덷�Z�o�p�@�~�ς���덷 �ǂ����ŏ��������Ȃ��Ⴂ���Ȃ�������K�X�@batch�̒��̑S�Ẵf�[�^�̍X�V�ʂ̘a
	double sum_error;
	// �X�V�ʂ̒~�� �d�ݍX�V�p batch�̒��̑S�Ẵf�[�^�̍X�V�ʂ̘a
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
	// �O�����Ɍv�Z�����ăG�l���M�[�֐����o���D
	void forward(const vector<vector<double>>& input_data, const vector<vector<double>>& teacher_data);
	// �덷�`��������
	void backward();
	// �G�l���M�[�֐��̒l���Ƃ�
	double get_error() const;
	vector<double> get_output() const;
	void set_initial_J();// �^�[�~�i���A�g���N�^�p�̍ŏ��̃G�l���M�[�֐�

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