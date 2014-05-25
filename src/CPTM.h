#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define DIR_HAAR_FACE "haar/haarcascade_frontalface_alt.xml"
#define DIR_HAAR_EYE "haar/haarcascade_mcs_eyepair_big.xml"


/**
 * Function to detect human face and the eyes from an image (Haar Cascades).
 *
 * @param  im    The source image
 * @param  tpl   Will be filled with the eye template, if detection success.
 * @param  rect  Will be filled with the bounding box of the eye
 *
 * @return zero=failed, nonzero=success
 */
int detectEye(
	cv::Mat& im, 
	cv::Mat& tpl, 
	cv::Rect& rect); // Function to detect user's face and eye

/**
 * Perform template matching to search the user's eye in the given image (Template Matching).
 *
 * @param   im_src		The source image
 * @param   tpl_src		The eye template
 * @param   rect		The eye bounding box, will be updated with _
 *						the new location of the eye
 */
void trackEye(
	cv::Mat& im_src,
	cv::Mat& tpl_src, 
	cv::Rect& rect); // Function to track user's eye given its template

/**
 * Função para gerar uma imagem a partir de um histograma
 * de 1 canal
 *
 * @param	hist		O histograma
 *
 * @return	img_hist	A imagem gerada a aprtir do histograma
 */
cv::Mat imgHist(cv::MatND &hist); // Gera uma imagem de um histograma

/**
 * Função que que acha o novo tamanho da imagem mantendo sua proporção,
 * recebendo o novo tamanho desejado para o maior eixo da imagem.
 *
 * @param	frame			A imagem que se deseja determinar o novo tamanho
 * @param	new_frame_size	Estrutura do tipo Size que guradará o novo tamanho
 * @param	max_axis_size	Novo tamanho do maior eixo da imagem
 */
void find_new_size(
	cv::Mat& frame,
	cv::Size& new_frame_size,
	int max_axis_size);
