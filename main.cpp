/* 
 * File:   main.cpp
 * Author: swl
 *
 * Created on November 29, 2015, 11:52 AM
 */

#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include "SketchRenderer.h"

using namespace std;

int main(int argc, char** argv) 
{
    SketchRenderer::init();
    //SketchRenderer::load("/home/swl/Documents/Data/psb/db/0/m32/m32.off");
    SketchRenderer::load("/home/swl/Documents/Data/psb/db/0/m76/m76.off");
    cv::Mat1f sketch = SketchRenderer::genSketch(glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
    
    cv::namedWindow("Sketch");
    cv::Mat1b disp = sketch*255.0;
    cv::imshow("Sketch", disp);
    cv::waitKey(0);
    //glutMainLoop();
    return 0;
}

