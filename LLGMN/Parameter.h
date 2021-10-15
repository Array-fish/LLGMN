#pragma once
#ifndef __INC_PARAMETER_H
#define __INC_PARAMETER_H

#include <vector>
#include <string>

//-----------------------------�ϐ��錾-----------------------------------------------------------
//�l�b�g���[�N�p�����[�^
const int    class_num = 4;			//�N���X��
const int    component_num = 2;			//�R���|�[�l���g��
const int    data_size = 4;			//���͎�����
const double eta_w = 1e-2;		//�d��w�̊w�K�萔
const int    max_times = 2000;		//�w�K�񐔂̏��
const double correction = 1e-3;		//���Usigma�̍s�񎮂��v�Z�ł��Ȃ��Ƃ��p�̕␳��c

//
//�f�[�^
std::vector<std::vector<double>> training_data;			//���t�f�[�^
std::vector<std::vector<double>>	 training_label;		//���t�f�[�^�̃��x��
std::vector<std::vector<double>> test_data;			//�e�X�g�f�[�^
std::vector<std::vector<double>>    test_label;			//�e�X�g�f�[�^�̃��x��

//�f�[�^�p�X
const std::string key_train = "";			//�w�K�p�f�[�^�̔ԍ��Ȃ�
const std::string key_test = "";			//�e�X�g�f�[�^�̔ԍ��Ȃ�

const std::string data_name = "train" + key_train + "";		//���ʕۑ���f�B���N�g���̖��O�̖����ɒǉ������
const std::string data_folder = "D:\\nac_data\\";					//�f�[�^�ǂݍ��݃t�H���_�̃p�X
//const std::string result_folder = "C:\\Users\\watanabe\\Documents\\nacgmn_result\\0721\\";		//���ʕۑ��t�H���_�̃p�X
//const std::string data_name = "first_data";		//���ʕۑ���f�B���N�g���̖��O�̖����ɒǉ������
//const std::string data_folder = "C:\\Users\\watanabe\\OneDrive - ���l������w\\shimaLaboratory\\data\\created_data\\first_data\\";
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

