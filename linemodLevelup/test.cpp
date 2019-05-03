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



// for test
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void train()
{
	/***********params **********************/
	int numberOfData = 3938;
	//std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";
	//std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;
	//std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";
	//std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;

	std::string depthBasePath = "F:\\ronaldwork\\temp\\depth\\";
	std::string rgbBasePath = "F:\\ronaldwork\\temp\\rgb\\";
	
	std::string className = "01_template";
	std::string classPath = "F:\\ronaldwork\\temp\\writeClasses\\";
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

void mini_train()
{
	/***********params **********************/
	int targetIdx = 156;
	std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";
	std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";;
	std::string className = "01_template";
	std::string classPath = "F:\\ronaldwork\\temp\\writeClasses\\";
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

void detect()
{
	/***********params **********************/
	int targetIdx = 156;
	std::string rgbBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\rgb\\";
	std::string depthBasePath = "F:\\ronaldwork\\3rd_source\\6DPose\\public\\datasets\\hinterstoisser\\train\\01\\depth\\";;
	float score = 60;

	//cv::Rect roi = cv::Rect(0, 0, 640, 480);
	cv::Rect roi = cv::Rect(169, 127, 240, 240);
	std::string className = "01_template";
	std::string classPath = "F:\\ronaldwork\\temp\\writeClasses\\";

	/***********main **********************/
	char buffer[1024];
	snprintf(buffer, 1024, "%04d.png", targetIdx);

	Mat rgbOriginal = cv::imread(rgbBasePath + buffer);
	Mat depthOriginal = cv::imread(depthBasePath + buffer, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	Mat rgb = rgbOriginal(roi);
	Mat depth = depthOriginal(roi);

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
	for (auto match : matches)
	{
		Rect box;
		box.x = match.x;
		box.y = match.y;
		box.width = 40;
		box.height = 40;
		boxes.push_back(box);
		scores.push_back(match.similarity);
	}
	NMSBoxes(boxes, scores, 0, 0.4, idxs);
	cout << "nms matches: " << idxs.size() << endl;

	Mat draw = rgb;
	for (auto idx : idxs)
	{
		auto match = matches[idx];
		int r = 40;
		cout << "x: " << match.x << "\ty: " << match.y
			<< "\ttemplateID: " << match.template_id << "\n"
			<< "\tsimilarity: " << match.similarity << "\n";
		cv::circle(draw, cv::Point(match.x + r, match.y + r), r, cv::Scalar(255, 0, 255), 2);
		cv::putText(draw, to_string(int(round(match.similarity))),
			Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 1.4, Scalar(0, 255, 255));

	}
	imshow("rgb", draw);
	waitKey(0);
}


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
	train();
	//mini_train();
	//detect();
	std::cout << "press enter to continue...\n";
	getchar();
    return 0;
}
