
#include <iostream>
#include <stdio.h>
#include <string>

#include "CPTM.h"
#include "back_project.h"
#include "opencv2/highgui/highgui.hpp"

/*
 * Problema em usar equalização de histogramas, aparentemente
 * diminui concideravelmente a diferença de dois histogramas
 */

void print () 
{
	for (int i = 0; i < 50; i++)
		std::cout << std::endl;
	
	std::cout << "\n\n******************************\n"
				 "----- Contador de Piscada ----\n"
				 "---- Baseado em comparacao ---\n"
				 "------- de histogramas -------\n"
				 "\n\n******************************\n";

	std::cout << "\nInstrucoes:"
				 "\nPrimeiramente, depois de ter detectado a\n"
				 "posicao correta dos olhos, e necessario saber como\n"
				 "sao os olhos fechados e abertos, para poder diferencialos,\n"
				 "entao tire uma foto dos olhos abertos e outra dos mesmos fechados.\n"
				 "E preciso que todas as fotas sejam tiradas nas mesmas posicoes\n"
				 "e condicoes de luminosidade (boas condicoes)\n";


	std::cout << "\nq - Sair \n"
				 "a - captura frame de olho aberto \n"
				 "f - capturar frame de olho fechado \n"
				 "b - habilitar Back Projection\n"
				 "m - ativar mascara (BP)\n"
				 "r - Redetectar os olhos (caso tenha travado em outro lugar)";
}

int main(int argn, char **argc)
{
    cv::VideoCapture cap;
	char key = '1';
	int contAltOlho = 0;
	int contaPiscada = 0;
	bool olhoAberto = false;
	bool olhoFechado = false;
	bool useMask = false;

	/// Using 256 bins for hue
	int h_bins = 256;
	int histSize[] = { h_bins };
	// hue varies from 0 to 255
	float h_ranges[] = { 0.0, 256.0 };
	const float* ranges[] = { h_ranges };
	// Use the o-th channel
	int channels[] = { 0 };

	bool enable_backProjection = false;
	cv::MatND back_projections_HSV [3];
	cv::MatND hist_imageROI_HSV [3];
	cv::Mat mask_BP;
	int fator_ajuste = 40;

    cv::Mat frame; 
    cv::Size size_frame;
    cv::Mat gray;
    cv::Mat eye_tpl;  // The eye template
	cv::Mat eye_tpl_bgr;
	cv::MatND histEye_bb;
	cv::MatND histEye_bb_old;
	cv::Mat imgHistEye_bb;
	double lim_comp_hist = 0.0;
	int lim_comp_hist_int = 194;
    cv::Rect eye_bb;  // The eye bounding box
	cv::Mat img_eye_bb;
	double fps;

	cv::Mat olhosAbertos;
	cv::MatND histOlhosAbertos;
	cv::Mat imgHistOlhosAbertos;

	cv::Mat olhosFechados;
	cv::MatND histOlhosFechados;
	cv::Mat imgHistOlhosFechados;

	cv::Scalar color(0,255,0);//verde

    if (argn > 1)
		cap.open(argc[1]); // Ler um arquivo de vídeo
	else
		cap.open(0); // 

	if (!cap.isOpened())
    {
        std::cout << "Falha ao abrir arquivo ou ativar a webcam" << std::endl;
		return EXIT_FAILURE;
    }

	/// Trabalha com no máximo 33 frames por segundo
	fps = ((cap.get(CV_CAP_PROP_FPS) < 33.0) ? cap.get(CV_CAP_PROP_FPS) : 33.0);

	print();

	/// Inicialização da câmera
	/// Depois de 1 segundo  de capturas de tela (ou 30 frames), a detecão começa
	for (int i = 0; i < fps; i++)
		cap >> frame;

	find_new_size(frame, size_frame, 800);

	cv::namedWindow( "video", 0 );
	cv::createTrackbar( "Fator do ajuste (BP)", "video", &fator_ajuste, 500, 0 );
	cv::createTrackbar( "Limiar para comparação de histogramas", "video", &lim_comp_hist_int, 1000, 0 );

    while(key != 'q') 
    {
        //cap >> frame;
		if (!cap.read(frame))
		{
			if (argn > 1)
			{
				cap.set(CV_CAP_PROP_POS_FRAMES, 0); /// Começar o vídeo denovo
                cap >> frame;
			}
			else
				break;
		}
		cv::resize(frame, frame, size_frame);

		gray.release();
        cv::cvtColor(frame, gray, CV_BGR2GRAY);

		lim_comp_hist = ((double)lim_comp_hist_int)/1000;

		/// Caso ainda não tenha sido detectado os olhos
        if (eye_bb.width == 0 || eye_bb.height == 0) 
		{			
            detectEye(gray, eye_tpl, eye_bb);	

			if (!eye_tpl.empty())
			{
				cv::imshow ("ROI modelo", eye_tpl);
				eye_tpl_bgr = frame(eye_bb);
				cv::Mat gray_eye_bb (gray(eye_bb)); // gcc não aceita &gray(eye_bb)
				calcHist( &gray_eye_bb, 1, channels, cv::Mat(), histEye_bb, 1, histSize, ranges);
				normalize( histEye_bb, histEye_bb, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
			}
		}
		/// Caso os olhos já tenham sido detectados
        else 
        {
			/// Procura eye_tpl dentro de gray (matching)
            trackEye(gray, eye_tpl, eye_bb);
			img_eye_bb = gray(eye_bb);

			/// Guarda o histograma da imagem anterior
			histEye_bb_old = histEye_bb.clone();
			calcHist( &img_eye_bb, 1, channels, cv::Mat(), histEye_bb, 1, histSize, ranges);

			/// Gera uma imagem do histograma dos olhos de frame atual pra visualizaçãoo
			imgHistEye_bb = imgHist(histEye_bb);
			cv::imshow("Histograma dos olhos agora", imgHistEye_bb);

			/// Normaliza o histograma do ROI do frame atual para comparação
			normalize( histEye_bb, histEye_bb, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

			/// Compara o histograma do ROI do frame anterior com o próximo
			if (cv::compareHist( histEye_bb, histEye_bb_old, CV_COMP_BHATTACHARYYA) > lim_comp_hist)
			{
				eye_bb.width = 0; eye_bb.height = 0;
				cv::destroyWindow("ROI modelo");
				eye_tpl.release();
				continue;
			}
			
			if (key == 'a') 
			{
				/// Grava uma imagem dos olhos abertos
				olhosAbertos = gray(eye_bb);

				calcHist( &olhosAbertos, 1, channels, cv::Mat(), histOlhosAbertos, 1, histSize, ranges);
				imgHistOlhosAbertos = imgHist(histOlhosAbertos);
				cv::normalize( histOlhosAbertos, histOlhosAbertos, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

				cv::imshow("Olhos Abertos", olhosAbertos);
				cv::imshow("Histograma dos olhos abertos",imgHistOlhosAbertos);
				key = '1';
			}
			else if (key == 'f')
			{
				/// grava uma imagem dos olhos fechados
				olhosFechados = gray(eye_bb);

				cv::calcHist( &olhosFechados, 1, channels, cv::Mat(), histOlhosFechados, 1, histSize, ranges);
				imgHistOlhosFechados = imgHist(histOlhosFechados);
				cv::normalize( histOlhosFechados, histOlhosFechados, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );				

				cv::imshow("Olhos Fechados", olhosFechados);
				cv::imshow("Histograma dos olhos fechados",imgHistOlhosFechados);
				key = '1';
			}
			else if (key == 'b')
			{
				if (enable_backProjection)
				{
					enable_backProjection = false;
					cv::destroyWindow("Back Projection | Hue");
					cv::destroyWindow("Back Projection | Saturation");
					cv::destroyWindow("Back Projection | Value");
				}
				else
				{
					hist_backProject (eye_tpl_bgr, hist_imageROI_HSV, (!mask_BP.empty() ? mask_BP(eye_bb) : cv::Mat()) );
					enable_backProjection = true;
				}
			}

			else if (key == 'm')
			{
				if (useMask)
				{
					mask_BP.release();
					hist_backProject (eye_tpl_bgr, hist_imageROI_HSV, cv::Mat());
					std::cout << "mascara desativada\n";
				}
				else
				{
					mask_BP = mask_backProject (frame, eye_tpl_bgr, &fator_ajuste);
					hist_backProject (eye_tpl_bgr, hist_imageROI_HSV, mask_BP(eye_bb));
					std::cout << "mascara ativada\n";
				}

				useMask = !useMask;
			}

			if (!olhosAbertos.empty() && !olhosFechados.empty())
			{
				//double base_base = compareHist(histEye_bb, histEye_bb, CV_COMP_BHATTACHARYYA); // sempre = 0
				double base_olhosAbertos = compareHist(histEye_bb, histOlhosAbertos, CV_COMP_BHATTACHARYYA);
				double base_olhosFechados = compareHist(histEye_bb, histOlhosFechados, CV_COMP_BHATTACHARYYA);

				if ((base_olhosAbertos/* - base_base*/) < (base_olhosFechados /*- base_base*/))
				{
					if (olhoFechado)
						contAltOlho++;

					color = cv::Scalar(255,0,0); // olhos abertos
					olhoAberto = true;
					olhoFechado = false;
				}
				else 
				{
					if (olhoAberto)
						contAltOlho++;

					color = cv::Scalar(0,0,255); // olhos fechados
					olhoAberto = false;
					olhoFechado = true;
				}
				if (contAltOlho == 2) // fechar e abrir os olhos
				{
					contAltOlho = 0;
					contaPiscada++;
					std::cout << contaPiscada << " piscadas " << std::endl;
				}
			}
			if (enable_backProjection)
			{
				if(useMask)
					mask_BP = mask_backProject (frame, eye_tpl_bgr, &fator_ajuste);
				
				backProject(frame, hist_imageROI_HSV, back_projections_HSV, mask_BP);
				cv::imshow("Back Projection | Hue", back_projections_HSV[0]);
				cv::imshow("Back Projection | Saturation", back_projections_HSV[1]);
				cv::imshow("Back Projection | Value", back_projections_HSV[2]);
			}

            cv::rectangle(frame, eye_bb, color);
        }

        cv::putText(frame,std::to_string(contaPiscada) + std::string(" piscadas"),
            cv::Point(30,30), CV_FONT_NORMAL, 1, cv::Scalar(255,255,255));
        cv::imshow("video", frame);

		key = cv::waitKey(1000/fps);
		if (key == 'r') 
		{
			contAltOlho = 0;
			contaPiscada = 0;
			olhoAberto = false;
			olhoFechado = false;

			eye_bb.width = 0;
			eye_bb.height = 0;
			eye_tpl.release();
			histEye_bb.release();
			imgHistEye_bb.release();
			cv::destroyWindow("Histograma dos olhos agora");

			if (enable_backProjection)
			{
				enable_backProjection = false;
				cv::destroyWindow("Back Projection | Hue");
				cv::destroyWindow("Back Projection | Saturation");
				cv::destroyWindow("Back Projection | Value");
			}
			if(!olhosAbertos.empty())
			{
				olhosAbertos.release();
				histOlhosAbertos.release();
				imgHistOlhosAbertos.release();
				cv::destroyWindow("Olhos Abertos");
				cv::destroyWindow("Histograma dos olhos abertos");
			}
			if(!olhosFechados.empty())
			{
				olhosFechados.release();
				histOlhosFechados.release();
				imgHistOlhosFechados.release();
				cv::destroyWindow("Olhos Fechados");
				cv::destroyWindow("Histograma dos olhos fechados");
			}
			
			color = cv::Scalar(0,255,0);
			key = '1';

			print();
		}
    }

	/// Finalizando
	cv::destroyAllWindows();
	cap.release();
	frame.release();
	eye_tpl.release();
	if(!olhosAbertos.empty())
	{
		olhosAbertos.release();
		histOlhosAbertos.release();
		imgHistOlhosAbertos.release();
	}
	if(!olhosFechados.empty())
	{
		olhosFechados.release();
		histOlhosFechados.release();
		imgHistOlhosFechados.release();
	}
	if(!histEye_bb.empty())
	{
		histEye_bb.release();
		imgHistEye_bb.release();
	}

	return EXIT_SUCCESS;
}
