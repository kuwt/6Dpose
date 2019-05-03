
#include "imageFileIO.h"

#include <tiff.h>
#include <tiffio.h>
#include <iostream>
#include <fstream>

#include <math.h>
#include <string>
#include <experimental/filesystem>
#define	__MAX_PATH	256

std::string GetDirectoryFromPath(const std::string& str)
{
	size_t found;
	found = str.find_last_of("/\\");
	if (std::string::npos == found) //not found
	{
		return ".";
	}
	else //found
	{
		return str.substr(0, found);
	}
}


int imageFileIO::FILE_SaveImageTiffR(cv::Mat matImage,
	std::string strFilename)
{
	int	intStatus = 0;
	
	std::string directory = GetDirectoryFromPath(strFilename);
	if (!std::experimental::filesystem::exists(directory) || !std::experimental::filesystem::is_directory(directory))
	{
		std::cout << "directory = " << directory << " does not exist!\n";
		intStatus = -1;
		return intStatus;
	}

	
	CV_Assert(matImage.depth() == CV_32F);

	char* acBuffer;
	acBuffer = (char*)malloc(strFilename.size() + 1);
	memcpy(acBuffer, strFilename.c_str(), strFilename.size() + 1);

	unsigned int line_size = matImage.cols * 4;
	tdata_t row_buf = _TIFFmalloc(line_size);

	// call libtiff
	TIFF* tif = TIFFOpen(acBuffer, "w");
	if (tif == NULL)
	{
		intStatus = -1;
		goto Exit;
	}

	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, matImage.cols);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, matImage.rows);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * 4);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, 3);
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
	TIFFSetField(tif, TIFFTAG_COMPRESSION, 1);
	TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, 2);

	float* src = matImage.ptr<float>(0);

	for (int row = 0; row < matImage.rows; ++row)
	{
		memcpy(row_buf, src, line_size);
		TIFFWriteScanline(tif, row_buf, row);
		src += matImage.cols;
	}

Exit:
	_TIFFfree(row_buf);
	TIFFClose(tif);
	free(acBuffer);
	return intStatus;
}


int imageFileIO::FILE_LoadImageTiffR(cv::Mat &matImage, std::string strFilename)
{
	int	intStatus = 0;

	char* acBuffer;
	acBuffer = (char*)malloc(strFilename.size() + 1);
	memcpy(acBuffer, strFilename.c_str(), strFilename.size() + 1);

	// call libtiff
	TIFF* tif = TIFFOpen(acBuffer, "r");
	if (tif == NULL)
	{
		intStatus = -1;
		goto Exit;
	}

	uint32 imagelength;
	tsize_t scanline;
	tdata_t buf;
	uint32 row;
	uint32 col;

	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
	scanline = TIFFScanlineSize(tif);
	buf = _TIFFmalloc(scanline);

	int width = scanline / 4;
	int height = imagelength;
	matImage = cv::Mat(cv::Size(width, height), CV_32F);
	float* src = matImage.ptr<float>(0);
	for (row = 0; row < imagelength; row++)
	{
		TIFFReadScanline(tif, buf, row);
		memcpy(src, buf, scanline);
		src += matImage.cols;
		/*
		for (col = 0; col < scanline; col++)
		{
		printf("%f ", ((float*)buf)[col]);
		}
		*/
		//printf("\n");
	}
	_TIFFfree(buf);
	TIFFClose(tif);

Exit:
	free(acBuffer);
	return intStatus;
}

int imageFileIO::FILE_SaveImageTiffint(cv::Mat matImage, std::string strFilename)
{
	int	intStatus = 0;
	CV_Assert(matImage.depth() == CV_32S);

	char* acBuffer;
	acBuffer = (char*)malloc(strFilename.size() + 1);
	memcpy(acBuffer, strFilename.c_str(), strFilename.size() + 1);

	unsigned int line_size = matImage.cols * 4;
	tdata_t row_buf = _TIFFmalloc(line_size);

	// call libtiff
	TIFF* tif = TIFFOpen(acBuffer, "w");
	if (tif == NULL)
	{
		intStatus = -1;
		goto Exit;
	}

	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, matImage.cols);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, matImage.rows);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * 4);
	TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, 2);
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
	TIFFSetField(tif, TIFFTAG_COMPRESSION, 1);
	TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, 2);

	int* src = matImage.ptr<int>(0);



	for (int row = 0; row < matImage.rows; ++row)
	{
		memcpy(row_buf, src, line_size);
		TIFFWriteScanline(tif, row_buf, row);
		src += matImage.cols;
	}

Exit:
	_TIFFfree(row_buf);
	TIFFClose(tif);
	free(acBuffer);
	return intStatus;
}

int imageFileIO::FileSaveImage_pverealto8bit(cv::Mat matImage, std::string strFilename)
{
	int intStatus = 0;
	double dMin;
	double dMax;
	cv::minMaxIdx(matImage, &dMin, &dMax);
	cv::Mat matAdjustImage;
	cv::convertScaleAbs(matImage, matAdjustImage, 255 / dMax);
	intStatus = cv::imwrite(strFilename, matAdjustImage);
	return intStatus;
}

int imageFileIO::FileSaveImage_realto8bit(cv::Mat matImage, std::string strFilename)
{
	int intStatus = 0;
	double dMin;
	double dMax;
	cv::minMaxIdx(matImage, &dMin, &dMax);

	double dScalingFactor = abs(dMin) > abs(dMax) ? abs(dMin) : abs(dMax);

	cv::Mat matAdjustImage;
	matAdjustImage = matImage * (128.0 / dScalingFactor);
	//cv::minMaxIdx(adjimage, &min, &max);
	matAdjustImage = matAdjustImage + 127;
	//cv::minMaxIdx(adjimage, &min, &max);
	cv::Mat matAdjustImage2;
	matAdjustImage.convertTo(matAdjustImage2, CV_8U, 1.0);
	intStatus = imwrite(strFilename, matAdjustImage2);
	return intStatus;
}


cv::Mat imageFileIO::loadImage(const std::string& name)
{
	cv::Mat image = loadImage(name, CV_LOAD_IMAGE_GRAYSCALE);
	return image;
}

cv::Mat imageFileIO::loadImage(const std::string& name, int mode)
{
	cv::Mat image = cv::imread(name, mode);
	;
	if (image.empty())
	{
		std::cerr << "Can't load image - " << name << std::endl;
		exit(-1);
	}
#if DISPLAY_ALG_PROGRESS
	cv::imshow(name, image);
	cv::waitKey(1);
#endif
	return image;
}

int imageFileIO::loadImage(const std::string& name, int mode, cv::Mat &image)
{
	image = cv::imread(name, mode);

	if (image.empty())
	{
		std::cerr << "Can't load image - " << name << std::endl;
		return ERR;
	}
#if DISPLAY_ALG_PROGRESS
	cv::imshow(name, image);
	cv::waitKey(1);
#endif
	return OK;
}

bool imageFileIO::IsGrayScaleImage(const cv::Mat &image)
{
	bool is_colored = false;
	if (image.channels() == 3) {
		is_colored = true;
	}
	return (!is_colored);
}