#include"utils.h"
//#include"include/erslib.h"
#include<iostream>
#include<iomanip>
#include<sstream>
#include<string>
#include<filesystem>
#include<random>
#include<algorithm>
#include<vector>
#include<fstream>
#include<windows.h>
#include<stdlib.h>
#include<map>
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max) {
	if (rand_min > rand_max) std::swap(rand_min, rand_max);
	const size_t max_min_diff = static_cast<size_t>(rand_max - rand_min + 1);
	if (max_min_diff < size) throw std::invalid_argument("引数が異常です");

	std::vector<int> tmp;
	std::random_device rnd;													// 乱数発生のための初期シード
	std::mt19937 engine(rnd());												//  乱数（メルセンヌツイスターの32ビット版、引数は初期シード）
	std::uniform_int_distribution<int> distribution(rand_min, rand_max);	// [0, ?] 範囲の一様乱数

	const size_t make_size = static_cast<size_t>(size * 1.2);

	while (tmp.size() < size) {
		while (tmp.size() < make_size) tmp.push_back(distribution(engine));
		std::sort(tmp.begin(), tmp.end());
		auto unique_end = std::unique(tmp.begin(), tmp.end());

		if (size < std::distance(tmp.begin(), unique_end)) {
			unique_end = std::next(tmp.begin(), size);
		}
		tmp.erase(unique_end, tmp.end());
	}

	return std::move(tmp);
}

std::double_t make_rand(double rand_min, double rand_max) {
	std::random_device rnd;													// 乱数発生のための初期シード
	std::mt19937 engine(rnd());												//  乱数（メルセンヌツイスターの32ビット版、引数は初期シード）
	std::uniform_real_distribution<double> distribution(rand_min, rand_max);	// [0, ?] 範囲の一様乱数

	return distribution(engine);
}

std::string get_date_sec() {
	time_t t = time(nullptr);
	struct tm lt;
	const errno_t error = localtime_s(&lt, &t);

	// put to stringstream
	std::stringstream ss;
	ss << lt.tm_year + 1900;
	ss << "_";
	ss << lt.tm_mon + 1;
	ss << "_";
	ss << lt.tm_mday;
	ss << "_";
	ss << lt.tm_hour;
	ss << "_";
	ss << lt.tm_min;
	ss << "_";
	ss << lt.tm_sec;

	return ss.str();
}

//void serial_communication(const int portnum, const std::vector<std::vector<double>>& label) {
//	int tmp;		//関数の戻り値（成功の判定用）
//
//	//ポート開放
//	tmp = ERS_Open(portnum);
//
//	if (tmp != 0) {		//ハンドル取得に失敗したとき
//		printf("PORT COULD NOT OPEN\n");
//		exit(0);
//	}
//
//	//送信
//	//送信データ生成
//	int max_index;		//ラベルが最大のクラス番号
//
//	for (int i = 0; i < label.size(); i++) {
//		max_index = (int)std::distance(label[i].begin(), std::max_element(label[i].begin(), label[i].end()));
//		
//		switch (max_index)
//		{
//		case 0:
//			tmp = ERS_WPutc(portnum, '0');
//			break;
//		case 1:
//			tmp = ERS_WPutc(portnum, '1');
//			break;
//		case 2:
//			tmp = ERS_WPutc(portnum, '2');
//			break;
//		case 3:
//			tmp = ERS_WPutc(portnum, '3');
//			break;
//		default:
//			break;
//		}
//		std::cout << "class: " << max_index << std::endl;
//
//		if (tmp == 0) {		//失敗したとき
//			printf("WRITEFILE FAILED\n");
//			ERS_Close(7);
//			exit(0);
//		}
//
//		Sleep(10);
//	}
//
//	//終了処理
//	printf("FINISH\n");
//	ERS_Close(7);
//
//}

double sigmoid(double x){
	return 1.0 / (1 + exp(-x));
}
