
/*Underwater Object Detection involving color correction algorithm
Author : Sagnik Basu

The MIT License
Copyright (c) 2015 Avi Sagnik Basu
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/








#include<opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include<fstream>
#include<string>
using namespace cv;
using namespace std;

double mul = 1;
int iter = 9;  //iterations required for color correction
Point center;
Point critp;
Point lp;
float s1 = 0; float s2 = 0;
int angle;
int scale = 1;
Mat rot_mat(2, 3, CV_32FC1);
VideoCapture vid("/home/shaggy/line_auv/GOPR0033.MP4");
ofstream myfile;
int k=0; //image counter
int lowcount = 0;
int critpoint = 0;
int gl, rl, bl, bh=255, rh=255, gh=255;
Mat img_bi, poly,dest;                      // stores the binary image

int img_x=100,img_y=100,img_width=1000,img_height=1000;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
vector<vector<Point> > polygons;
//Mat objects=Mat::zeros(1440,1920,CV_8UC1);
double aHigh=0;
int col_num=48,row_num=64;
/*----------------functions used--------------------*/

Mat correctGamma(Mat& , double);
Mat clahe_conversion(Mat);
void colorReduce(cv::Mat&, int );



/*--------------------------------------------------*/

Mat correctGamma(Mat& img, double gamma) {
    double inverse_gamma = gamma;

    Mat lut_matrix(1, 256, CV_8UC1);
    uchar * ptr = lut_matrix.ptr();
    for (int i = 0; i < 256; i++)
        ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

    Mat result;
    LUT(img, lut_matrix, result);

    return result;

}
//moments()
//VideoCapture front(0);

int main()
{
            int p;
            namedWindow("correction", CV_WINDOW_NORMAL);
            namedWindow("filter_image", CV_WINDOW_NORMAL);
            namedWindow("input", CV_WINDOW_NORMAL);
            namedWindow("Contours", CV_WINDOW_NORMAL);
            namedWindow("CLAHE", CV_WINDOW_NORMAL);
            // namedWindow("Color Reduce Image", CV_WINDOW_NORMAL);
            // cout<<"Now entering the loop";
            //namedWindow("", CV_WINDOW_NORMAL);
           /* createTrackbar("blue low", "Track", &bl, 255, NULL);
            createTrackbar("blue high", "Track", &bh, 255, NULL);
            createTrackbar("green low ", "Track", &gl, 255, NULL);
            createTrackbar("green high", "Track", &gh, 255, NULL);
            createTrackbar("red low", "Track", &rl, 255, NULL);
            createTrackbar("red high", "Track", &rh, 255, NULL);*/
           // myfile.open ("/home/shaggy/line_auv/example.txt");
           // Mat img = imread("/home/shaggy/line_auv/path.jpg");
           // dest=img.clone();
            Mat img,clahe_hsv;
            if(!vid.isOpened())
           {
                   cout<<"Check Video"<<"\n";
                   return 0;
               }

            //imshow("input",img);
            int i=12;
           // bilateralFilter(img,dest,i, i*2,i/2);


            while(1)
            {

                if(!vid.read(img))
                      {
                          continue;
                      }
                imshow("input",img);
                 Mat objects=Mat::zeros(img.size(),CV_8UC3);
		  Rect roi(img_x,img_y,img_width,img_height);
		  img=img(roi);

                    Mat clahe_conv=clahe_conversion(img);
                    imshow("CLAHE",clahe_conv);
                    //clahe_hsv=cvtColor(clahe_conv,clahe_hsv,CV_RGB2HSV);
                    imwrite("/home/shaggy/line_auv/CLAHE.jpg",clahe_conv);
                    imwrite("/home/shaggy/line_auv/input.jpg",img);
                    //imwrite("/home/shaggy/line_auv/clahe_hsv.jpg",clahe_hsv);
                    //cvtColor(img,img,CV_BGR2HSV);
                   // bilateralFilter(clahe_conv,dest,i, i*2,i/2);
                     //imshow("filter_image",dest);
                  //  Mat g_corr=correctGamma(img,1);
             //       imshow("CLAHE_HSV",clahe_hsv);
                    Mat imgc = img.clone();
                    Mat out(img.rows,img.cols,CV_8UC3);
                   // Mat objects(img.rows,img.cols,CV_8UC1);
                    //colorReduce(img,64);                                                          //reduce color in image
                    //imwrite("/home/shaggy/line_auv/color_reduce.jpg",img);
                    //imshow("Color Reduce Image",img);
                    // cout<<"image is k"<<k<<endl;
                     //++k;
                    cout<<"Now entering the loop";
                    for (int i = 0; i < iter; i++)
                    {
                       // cout<<"selfie"<<endl;
                       //  myfile << "image frame starts..........................................\n";
                        for ( p = 0; p < img.rows - img.rows / (row_num * mul); p += img.rows / (64 * mul))
                        {
                            for (int q = 0; q < img.cols - img.cols / (col_num * mul); q += img.cols / (48 * mul))
                            {
                                int bavg = 0;
                                int gavg = 0;
                                int ravg = 0;
                                for (int i = p; i < p + img.rows / (row_num* mul); i++)
                                {
                                    for (int j = q; j < q + img.cols / (col_num * mul); j++)
                                    {
                                       //Vec3b color = g_corr.at<Vec3b>(i, j);
                                        Vec3b color = clahe_conv.at<Vec3b>(i, j);
                                        int b = color[0];
                                        int g = color[1];
                                        int r = color[2];
                                        bavg += b;
                                        gavg += g;
                                        ravg += r;
                                    }
                                }
                                bavg = bavg * row_num * col_num* mul*mul / (img.rows*img.cols);
                                gavg = gavg * row_num * col_num * mul*mul / (img.rows*img.cols);
                                ravg = ravg * row_num * col_num * mul*mul / (img.rows*img.cols);
                               // int gg = 2 * gavg - (ravg + bavg);
                               // int rr = 2 * ravg - (gavg + bavg);
                                //cout<<"bavg="<<bavg<<"gavg="<<gavg<<"rvg="<<ravg<<endl;

                                // myfile << "Writing this to a file.\n";
                               //  myfile.close();
                                //myfile<<"bavg="<<bavg<<"\n"<<"rvg="<<ravg<<"\n"<<"gavg="<<gavg<<"\n";
                               // putText(img,to_string(bavg),Point(0,0));
                                //cout<<"iteration is..."<<p<<"and bavg="<<bavg<<"and red avg="<<ravg<<endl;
                                for (int i = p; i < p + img.rows / (row_num * mul); i++)
                                {
                                    for (int j = q; j < q + img.cols / (col_num * mul); j++)
                                    {
                                        Vec3b color =imgc.at<Vec3b>(i, j);
                                        /*if (2 * color[1] >= color[2] + color[0] + iter&&gavg >= 45)//&&2*color[1]>color[2]+color[0])
                                        {
                                            color[1] = 165;
                                            color[0] = 0;
                                            color[2] = 255;
                                        }*/
                                        if (/*(color[0]=bavg) &&( */(color[1]<=(gavg+20)))//)//&&2*color[2]>color[1]+color[0]) // for detecting balls
                                        {
                                            color[1] =0;
                                            color[0] =0;
                                            color[2] = 0;
                                        }
                                        else //if(color[2]>=ravg)
                                        {
                                            color[1] =255;
                                        color[0] =255;
                                        color[2] =255;
                                        }


                                out.at<Vec3b>(i, j) = color;
                            }
                        }

                    }

                }
                       // cout<<p<<endl;

                        // myfile << "Writing this to a file ends.....................\n";
            }

                //imshow("correction",out);
                cvtColor(out,out,CV_RGB2GRAY);
                erode(out,out,Mat(),Point(-1,-1),2,1,1);
                dilate(out,out,Mat(),Point(-1,-1),2,1,1);
                imshow("correction",out);
                findContours(out,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
                       //findContours(img_bi.clone(), contours,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
                 polygons.resize(contours.size());

                 cout<<"no. of contours"<<contours.size()<<endl;

                                 /*  for (int i = 0; i < contours.size(); i++)
                                   {
                                       approxPolyDP(Mat(contours[i]), polygons[i], arcLength(Mat(contours[i]), true)*0.019, true);
                                   }*/
                                   /*for (int i = 0; i < polygons.size(); i++)
                                   {



                                  }*/
                    int aHighIndex=0;
                    aHigh=0;
                 for( int i = 0; i< contours.size(); i++ )
                            {
                                if(contourArea(contours[i]) > aHigh)
                                {
                                   // aHighIndex = i;
                                    aHigh = contourArea(contours[i]);
                                    aHighIndex=i;
                                }
                            }

                                cout<<"largestarea"<<aHigh<<endl;
                                cout<<"index"<<aHighIndex<<endl;

                                   Scalar color = Scalar(0, 255, 255);
                                   drawContours(objects, contours, aHighIndex, color, 3, 8, hierarchy, 0, Point());

                //imwrite("/home/shaggy/line_auv/contours.jpg",objects);
                imshow("Contours",objects);


                if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
                       {
                                cout << "esc key is pressed by user" << endl;
                                break;
                       }

}


//    myfile.close();
    return 0;
}


Mat clahe_conversion(Mat input)
{
    vector<Mat> RGB; // Use the STL’s vector structure to store multiple Mat objects
    split(input, RGB); // split the image into separate color planes (R G B)
    ////    Enhance Local Contrast (CLAHE)
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    Mat RGB_eq;


    ////    Equalizes the histogram of a one channel image  (8UC1) using Contrast Limited Adaptive Histogram Equalization.
    clahe->apply(RGB[0],RGB[0]);
    clahe->apply(RGB[1],RGB[1]);
    clahe->apply(RGB[2],RGB[2]);

    merge(RGB,RGB_eq);              // now merge the results back
    return RGB_eq;
}
void colorReduce(Mat& image, int div)
{
    int nl = image.rows;                    // number of lines
    int nc = image.cols * image.channels(); // number of elements per line

    for (int j = 0; j < nl; j++)
    {
        // get the address of row j
        uchar* data = image.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            // process each pixel
            data[i] = data[i] / div * div + div / 2;
        }
    }
}


