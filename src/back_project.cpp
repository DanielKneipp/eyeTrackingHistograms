#include "back_project.h"

/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

/// Using 180 bins for hue
int h_bins = 16;
/// Using 256 bins for saturation
int s_bins = 256;
/// Using 256 bins for value
int v_bins = 256;
int histSize[] = { h_bins, s_bins, v_bins};
/// hue varies from 0 to 180, saturation and value varies from 0 to 255
float h_ranges[] = { 0.0, 180.0 };
float s_ranges[] = { 0.0, 256.0 };
float v_ranges[] = { 0.0, 256.0 };
const float* ranges[] = { h_ranges, s_ranges, v_ranges };
/// Use the 0,1,2 channels
int channels[] = { 0, 1, 2 };

/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

cv::Mat mask_backProject(
	const cv::Mat image, const cv::Mat imageROI,
	const int* fator_ajuste)
{
	cv::Mat imageROI_HSV;
	cv::Mat imageROI_H;
	cv::Mat imageROI_S;
	cv::Mat imageROI_V;

	cv::Mat image_HSV;

	cv::Mat maskImage;

	double minVal_H = 0, maxVal_H = 0;
	double minVal_S = 0, maxVal_S = 0;
	double minVal_V = 0, maxVal_V = 0;

	/// Valores de ajuste de limite superior e inferior para o inRange
	double ajuste_H = 0.0;
	double ajuste_S = 0.0;
	double ajuste_V = 0.0;

	/// Convertendo a imagem em BGR para HSV
	cvtColor(imageROI, imageROI_HSV, CV_BGR2HSV);
	cvtColor(image, image_HSV, CV_BGR2HSV);

	imageROI_H.create( imageROI_HSV.size(), imageROI_HSV.depth());
	imageROI_S.create( imageROI_HSV.size(), imageROI_HSV.depth());
	imageROI_V.create( imageROI_HSV.size(), imageROI_HSV.depth());

	/// Separando os canais da imagem em HSV
	int ch[] = { 0, 0 };
	cv::mixChannels(
		&imageROI_HSV,	// Imagem de origem
		1,				// Número de imagens de origem
		&imageROI_H,	// Imagem de destino
		1,				// Número de imagens de destino
		ch,				// Canal x da imagem de origem para _
						// canal y da imagem de destino
		1				// Número de pares do parâmetro acima.
	);
	ch[0] = 1;
	cv::mixChannels(&imageROI_HSV,1,&imageROI_S,1,ch,1);
	ch[0] = 2;
	cv::mixChannels(&imageROI_HSV,1,&imageROI_V,1,ch,1);

	cv::minMaxLoc(imageROI_H,&minVal_H,&maxVal_H);
	cv::minMaxLoc(imageROI_S,&minVal_S,&maxVal_S);
	cv::minMaxLoc(imageROI_V,&minVal_V,&maxVal_V);

	if ((*fator_ajuste) != 0) /// Valor 0 anula o ajuste
	{
		ajuste_H = (maxVal_H - minVal_H)/(*fator_ajuste); 
		ajuste_S = (maxVal_S - minVal_S)/(*fator_ajuste); 
		ajuste_V = (maxVal_V - minVal_V)/(*fator_ajuste); 
	}
	
	/*
		When using the hue component of a color, it is always important to take its saturation into
		account (which is the second entry of the vector). Indeed, when the saturation of a color is
		low, the hue information becomes unstable and unreliable. This is due to the fact that for
		low-saturated color, the B, G, and R components are almost equal. This makes it difficult
		to determine the exact color represented. In consequence, we decided to ignore the hue
		component of colors with low saturation.
		OpenCV_2_Cookbook -> pg. 112
	*/
	/// Usado para criar uma máscara para melhorar o resultado
	/// do backProject. A máscara é gerada com base nas partes da 
	/// imagem original que estão entre os valores da imagem de interesse.
	cv::inRange(
		image_HSV, 
		cv::Scalar(minVal_H + ajuste_H, MAX(23, minVal_S + ajuste_S), minVal_V + ajuste_V),
		cv::Scalar(maxVal_H - ajuste_H, maxVal_S - ajuste_S, maxVal_V - ajuste_V),
		maskImage);

	return maskImage;
}


void hist_backProject (
	const cv::Mat& imageROI, cv::MatND* hist_imageROI_HSV,
	const cv::Mat& maskROI)
{
	cv::Mat imageROI_HSV;

	cvtColor(imageROI, imageROI_HSV, CV_BGR2HSV);

	/// Calculando os histogramas e normalizado-os para valores entre 0 e 255  
	/// para serem usados no BackProject
	cv::calcHist( &imageROI_HSV, 1, &channels[0], maskROI, hist_imageROI_HSV[0],
		1, &histSize[0], &ranges[0]);
	cv::normalize(hist_imageROI_HSV[0], hist_imageROI_HSV[0], 0, 255, cv::NORM_MINMAX);
	cv::calcHist( &imageROI_HSV, 1, &channels[1], maskROI, hist_imageROI_HSV[1],
		1, &histSize[1], &ranges[1]);
	cv::normalize(hist_imageROI_HSV[1], hist_imageROI_HSV[1], 0, 255, cv::NORM_MINMAX);
	cv::calcHist( &imageROI_HSV, 1, &channels[2], maskROI, hist_imageROI_HSV[2],
		1, &histSize[2], &ranges[2]);
	cv::normalize(hist_imageROI_HSV[2], hist_imageROI_HSV[2], 0, 255, cv::NORM_MINMAX);
}	


void backProject(
	 const cv::Mat& image, const cv::MatND* hist_imageROI_HSV, 
	 cv::MatND* back_projections_HSV, const cv::Mat& maskImage)
 {
	cv::Mat image_HSV;

	/// Convertendo a imagem em BGR para HSV
	cvtColor(image, image_HSV, CV_BGR2HSV);

	cv::calcBackProject(
		&image_HSV,					// image source
		1,							// one image
		&channels[0],				// the channels used
		hist_imageROI_HSV[0],		// the histogram we are backprojecting
		back_projections_HSV[0],	// the resulting back projection image
		&ranges[0],					// the range of values, for each dimension
		1,							// a scaling factor
		true);

	cv::calcBackProject(
		&image_HSV,	1, &channels[1], hist_imageROI_HSV[1],		
		back_projections_HSV[1], &ranges[1], 1,	true);

	cv::calcBackProject(
		&image_HSV,	1, &channels[2], hist_imageROI_HSV[2],	
		back_projections_HSV[2], &ranges[2], 1, true);

	if (!maskImage.empty())
	{
		// cv::bitwise_and(b_p_hsv[*], maskImage, b_p_hsv[*]);
		back_projections_HSV[0] &= maskImage;
		back_projections_HSV[1] &= maskImage;
		back_projections_HSV[2] &= maskImage;
	}
 }