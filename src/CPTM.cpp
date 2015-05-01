#include "CPTM.h"


/*
 * Referências:
 * https://opencv-code.com/tutorials/eye-detection-and-tracking/
 * http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
 * http://docs.opencv.org/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
 */


cv::Mat imgHist(cv::MatND& hist)
{
    cv::Mat img_hist (100, hist.rows, CV_8U, cv::Scalar(0));
    float bin_val = 0.0;
    double minVal, maxVal;

    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

    for (int i = 0; i < hist.rows; i++)
    {
        bin_val = hist.at<float>(i);

        cv::line (
            img_hist,
            cv::Point(i,img_hist.rows),
            cv::Point(i, img_hist.rows - static_cast<int>( bin_val * (img_hist.rows / maxVal) )),
            cv::Scalar(255,255,255));
    }
    return img_hist;
}


int detectEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
{
    cv::CascadeClassifier face_cascade (DIR_HAAR_FACE);
    cv::CascadeClassifier eyes_cascade (DIR_HAAR_EYE);

    cv::Mat im_copy = im.clone();
    std::vector<cv::Rect> faces, eyes;
    cv::equalizeHist( im_copy, im_copy );
    face_cascade.detectMultiScale(im_copy, faces, 1.1, 2, 
                                  0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));

    for (unsigned int i = 0; i < faces.size(); i++)
    {
        cv::Mat face = im_copy(faces[i]);
        cv::equalizeHist(face, face);
        eyes_cascade.detectMultiScale(face, eyes, 1.1, 2, 
                                  0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
        if (eyes.size()) 
        {
            /// Somente os olhos da primeira face
            rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
            tpl  = im(rect);
        }
    }

    return eyes.size();
}


void trackEye(cv::Mat& im_src, cv::Mat& tpl_src, cv::Rect& rect)
{
    cv::Mat im = im_src.clone();
    cv::Mat tpl = tpl_src.clone();

    cv::Size size(rect.width, rect.height);
    cv::Point point(size.width/2, size.height/2);
    cv::Rect window(rect + size - point);

    /// Manter window dentro dos limites de im
    window &= cv::Rect(0, 0, im.cols, im.rows);

    cv::equalizeHist(im, im);
    cv::equalizeHist(tpl, tpl);

    cv::Mat imRoi = im(window);
    cv::equalizeHist( imRoi, imRoi );

    cv::Mat dst(window.width -  tpl.cols + 1, window.height - tpl.rows + 1, CV_32FC1);
    cv::matchTemplate(imRoi, tpl, dst, CV_TM_CCOEFF);
    cv::normalize (dst, dst, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc, cv::Mat());

    /* 
       Condição não funciona, maxval > 0.95 sempre.
       Ensinar ao programa como saber quando o olho foi perdido.
       Detectar variação do histograma histEye_bb pode funcionkar
    */
    //if (maxval > 0.95)
    //{
        rect.x = window.x + maxloc.x;
        rect.y = window.y + maxloc.y;
    //}
    //else
    //    rect.x = rect.y = rect.width = rect.height = 0;
}


void find_new_size(cv::Mat& frame, cv::Size& new_frame_size, int max_axis_size)
{
    if (frame.cols >= frame.rows)
    {
        if (frame.cols > max_axis_size)
        {
            float perda_percent = ((frame.cols - max_axis_size) * 100) / frame.cols;
            new_frame_size.height = int (frame.rows - (frame.rows * (perda_percent / 100)));
            new_frame_size.width = max_axis_size;
        }
        else
        {
            new_frame_size.height = frame.rows;
            new_frame_size.width = frame.cols;
        }
    }
    else
    {
        if (frame.rows > max_axis_size)
        {
            float perda_percent = ((frame.rows - max_axis_size) * 100) / frame.rows;
            new_frame_size.width = int (frame.cols - (frame.cols * (perda_percent / 100)));
            new_frame_size.height = max_axis_size;
        }
        else
        {
            new_frame_size.height = frame.rows;
            new_frame_size.width = frame.cols;
        }
    }

}
