#include "linemodLevelup.h"
#include <memory>
#include <iostream>
#include "linemod_icp.h"
#include <assert.h>
#include <chrono>  // for high_resolution_clock
#include <opencv2/rgbd.hpp>
#include "nms.h" 

using namespace std;
using namespace cv;

/**********************

Batch train

***********************/
void train()
{
	/***********params **********************/
	int numberOfData = 864;
	//std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";
	//std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;

	std::string depthBasePath = "F:\\ronaldwork\\testData\\temp\\set3\\depth\\";
	std::string rgbBasePath = "F:\\ronaldwork\\testData\\temp\\set3\\rgb\\";
	
	std::string className = "01_template";
	std::string classPath = "F:\\ronaldwork\\testData\\temp\\set3\\writeClasses\\";
	/***********main **********************/
	//cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINE();
	cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINEMOD();

	for (int i = 0; i < numberOfData; ++i)
	{
		char buffer[1024];
		snprintf(buffer, 1024, "%04d.png", i);

		Mat rgb = cv::imread(rgbBasePath + buffer);
		Mat depth = cv::imread(depthBasePath + buffer, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

		vector<Mat> sources;
		sources.push_back(rgb);
		sources.push_back(depth);

		
		detector->addTemplate(sources, className, cv::Mat());
		
		cout << "template done " <<i << "\n";
	}
	detector->writeClasses(classPath + "%s.yaml");
	cout << "template finish " << "\n";
}

/**********************

only train a datum

***********************/
void mini_train()
{
	/***********params **********************/
	int targetIdx = 2;
	//std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";
	//std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;

	std::string depthBasePath = "F:\\ronaldwork\\testData\\temp\\set2\\depth\\";
	std::string rgbBasePath = "F:\\ronaldwork\\testData\\temp\\rgb\\";

	std::string className = "02_template";
	std::string classPath = "F:\\ronaldwork\\testData\\temp\writeClasses\\";
	/***********main **********************/
	//cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINE();
	cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINEMOD();
	{
		char buffer[1024];
		snprintf(buffer, 1024, "%04d.png", targetIdx);

		Mat rgb = cv::imread(rgbBasePath + buffer);
		Mat depth = cv::imread(depthBasePath + buffer, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

		vector<Mat> sources;
		sources.push_back(rgb);
		sources.push_back(depth);


		detector->addTemplate(sources, className, cv::Mat());

		cout << "template done " << targetIdx << "\n";
	}
	detector->writeClasses(classPath + "%s.yaml");
}


/**********************

key image to a dark background
The new image size will be greater
than the old one in the form of square

***********************/
void keyTobackground(const cv::Mat &in,cv::Mat &out)
{
	int sizefactor = 16 * 5;
	int inputDimension = (in.size().width > in.size().height) ? in.size().width : in.size().height;

	int remainder = inputDimension % sizefactor;

	int Tdimension = 0;
	if (remainder == 0)
	{
		out = in.clone();
		return;
	}
	else
	{
		int temp = (float)inputDimension / (float)sizefactor;
		Tdimension = (temp + 1) * sizefactor;

		out = cv::Mat::zeros(cv::Size(Tdimension, Tdimension), in.type());

		cv::Rect roi = 
			cv::Rect(
			out.size().width / 2 - in.size().width / 2,
			out.size().height / 2 - in.size().height / 2,
			in.size().width,
			in.size().height);

		Mat out_roi = out(roi);
		in.copyTo(out_roi);
		
		return;
	}
}

/**********************

detect a datum

***********************/
void detect()
{
	/***********params **********************/
	int targetIdx = 31;
	std::string rgbBasePath = "F:\\ronaldwork\\testData\\temp\\set3\\rgb\\";
	std::string depthBasePath = "F:\\ronaldwork\\testData\\temp\\set3\\depth\\";;
	float score = 60;

	//cv::Rect roi = cv::Rect(0, 0, 640, 480);
	cv::Rect roi = cv::Rect(449, 249, 57, 71);
	std::string className = "01_template";
	std::string classPath = "F:\\ronaldwork\\testData\\temp\\set3\\writeClasses\\";

	/***********main **********************/
	char buffer[1024];
	snprintf(buffer, 1024, "%04d.png", targetIdx);

	Mat rgbOriginal = cv::imread(rgbBasePath + buffer);
	Mat depthOriginal = cv::imread(depthBasePath + buffer, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	Mat rgbCrop = rgbOriginal(roi);
	Mat depthCrop = depthOriginal(roi);

	Mat rgb;
	Mat depth;
	keyTobackground(rgbCrop, rgb);
	keyTobackground(depthCrop, depth);

	cv::imwrite("log\\inrgb.png", rgb);
	cv::imwrite("log\\indepth.png", depth);
	vector<Mat> sources;
	sources.push_back(rgb);
	sources.push_back(depth);

	vector<string> classes;
	classes.push_back(className);

	//cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINE();
	cv::Ptr<linemodLevelup::Detector> detector = linemodLevelup::getDefaultLINEMOD();
	detector->readClasses(classes, classPath + "%s.yaml");

	auto start_time = std::chrono::high_resolution_clock::now();
	vector<linemodLevelup::Match> matches = detector->match(sources, score, classes);
	auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
	cout << "match time: " << elapsed_time.count() / 1000000000.0 << "s" << endl;
	cout << "matches: " << matches.size() << endl;

	vector<Rect> boxes;
	vector<float> scores;
	vector<int> idxs;
	float r = 30;
	for (auto match : matches)
	{
		Rect box;
		box.x = match.x;
		box.y = match.y;
		box.width = r;
		box.height = r;
		boxes.push_back(box);
		scores.push_back(match.similarity);
	}
	NMSBoxes(boxes, scores, 0, 0.4, idxs);
	cout << "nms matches: " << idxs.size() << endl;

	Mat draw = rgb;
	for (int idx = 0; idx < idxs.size() && idx < 3; ++idx)
	{
		
		auto match = matches[idx];
		cout << "x: " << match.x << "\ty: " << match.y
			<< "\ttemplateID: " << match.template_id << "\n"
			<< "\tsimilarity: " << match.similarity << "\n";
		cv::rectangle(
			draw, 
			cv::Point(match.x, match.y), 
			cv::Point(match.x + r, match.y + r),
			cv::Scalar(255,0,255));
		
		cv::putText(draw, to_string(int(round(match.similarity))),
			Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 1.4, Scalar(0, 255, 255));

	}
	imshow("rgb", draw);
	waitKey(0);
}



/**********************

scale template and 
make data base
***********************/
void imagesTemplateGeneration()
{
	/***********params **********************/
	int numberOfData = 1313;
	std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;
	std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";

	std::string rgbBasePathOut = "F:\\ronaldwork\\temp\\rgb\\";
	std::string depthBasePathOut = "F:\\ronaldwork\\temp\\depth\\";

	/***********main **********************/
	std::vector<cv::Mat> rgbSrcs;
	std::vector<cv::Mat> depthSrcs;
	for (int i = 0; i < numberOfData; ++i)
	{
		char buffer[1024];
		snprintf(buffer, 1024, "%04d.png", i);

		Mat rgb = cv::imread(rgbBasePath + buffer);
		Mat depth = cv::imread(depthBasePath + buffer, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

		rgbSrcs.push_back(rgb);
		depthSrcs.push_back(depth);

		cout << "Load src  " << i << "\n";
	}

	int imageIdx = 0;
	std::vector<double> scales = { 0.4, 0.5, 0.6 };
	for (int i = 0; i < scales.size(); ++i)
	{
		for (int j = 0; j < rgbSrcs.size(); ++j)
		{
			char buffer[1024];
			snprintf(buffer, 1024, "%04d.png", imageIdx);

			cv::Mat rgb;
			cv::Mat depth;
			cv::resize(rgbSrcs.at(j), rgb, cv::Size(), scales.at(i), scales.at(i), CV_INTER_CUBIC);
			cv::resize(depthSrcs.at(j), depth, cv::Size(), scales.at(i), scales.at(i),  CV_INTER_CUBIC);

			cv::imwrite(rgbBasePathOut + buffer, rgb);
			cv::imwrite(depthBasePathOut + buffer, depth);
			imageIdx++;
			cout << "write template " << imageIdx<< "\n";
		}

	}
	cout << "template finish " << "\n";
}

int main()
{

	//imagesTemplateGeneration();
	//mini_train();
	//train();
	detect();
	std::cout << "press enter to continue...\n";
	getchar();
    return 0;
}
