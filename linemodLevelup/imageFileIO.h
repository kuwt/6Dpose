#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class imageFileIO
{
public:
	static const int OK = 0;
	static const int ERR = -1;
	/*****************************
	* Load Image
	****************************/
	
	/***Load a single image as CV_LOAD_IMAGE_GRAYSCALE ******/
	static cv::Mat loadImage(const std::string& name);

	/** Load a single image, user specify the mode, e.g. CV_LOAD_IMAGE_GRAYSCALE ********/
	static cv::Mat loadImage(const std::string& name, int mode);

	/***Load a single image, User specify the mode, e.g. CV_LOAD_IMAGE_GRAYSCALE, return status ********/
	static int loadImage(const std::string& name, int mode, cv::Mat &image);

	static bool IsGrayScaleImage(const cv::Mat &image);

	/*****************************
	*	Load Tiff
	****************************/
	/****  Load float 32bit real Image  ***/
	static int FILE_LoadImageTiffR(cv::Mat &matImage, std::string strFilename);
	
	/*****************************
	*	Save Tiff
	****************************/
	/****  Save float 32bit real Image  ***/
	static int FILE_SaveImageTiffR(cv::Mat matImage, std::string strFilename);

	/****  Save int 32bit Image  ***/
	static int FILE_SaveImageTiffint(cv::Mat matImage, std::string strFilename);
	
	
	/*** Convert positive real image to 8 bit(0-255) range and save to strFilename *****/
	static int FileSaveImage_pverealto8bit(cv::Mat matImage, std::string strFilename);

	/*** Convert real image to 8 bit(0-255) range and save to strFilename *****/
	static int FileSaveImage_realto8bit(cv::Mat matImage, std::string strFilename);
};


