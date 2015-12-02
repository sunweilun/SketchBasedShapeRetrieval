/* 
 * File:   hardCodeFunctions.h
 * Author: swl
 *
 * Created on December 1, 2015, 5:15 PM
 */

#ifndef HARDCODEFUNCTIONS_H
#define	HARDCODEFUNCTIONS_H

#include <sys/time.h>
#include <sys/stat.h>

#include "Retriever.h"

std::string libPath = "/home/swl/Documents/Data/psb/";

void train()
{
    Retriever retriever;
    retriever.init();
    retriever.setLibPath(libPath);
    retriever.train(libPath+"cluster_pa=1e-1.bin", libPath+"dict_pa=1e-1.bin");   
    retriever.saveTrainingData(libPath+"sbsr_pa=1e-1.bin");
}

inline double diff(const struct timeval& ts, const struct timeval& te) 
{
    return (te.tv_sec - ts.tv_sec)*1e6+te.tv_usec-ts.tv_usec;
}

void show()
{
    Retriever retriever;
    retriever.init();
    retriever.loadTrainingData(libPath+"sbsr.bin");
    retriever.buildTopKDTree();
    
    struct timeval ts, te;
    
    char line[1024];
    while(1)
    {
        printf("Input File Path:");
        gets(line);
        
        std::string folder_name = line;
        
        if(strcmp(line, "q") == 0)
            break;
        
        mkdir(line, ACCESSPERMS);
        
        std::string input_path = "/home/swl/Documents/Data/test_sketches/";
        input_path = input_path + line + ".png";
        
        cv::Mat sketch = cv::imread(input_path, CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat1f input = sketch / 255.0;
        
        gettimeofday(&ts, NULL);

        std::vector<Retriever::RetrievalInfo> retrievalList = retriever.retrieveAll(input);

        gettimeofday(&te, NULL);

        FILE* file = fopen((folder_name+"/info.txt").c_str(), "w");

        fprintf(file, "time = %0.3lf ms\n", diff(ts, te)*1e-3);

        for(unsigned i=0; i<10; i++)
        {
            //printf("%s -- %f\n", retrievalList[i].path.c_str(), retrievalList[i].score);
            SketchRenderer::load(retrievalList[i].path.c_str());

            cv::imwrite(folder_name+"/input.png", input*255.0);

            cv::Mat1f sketch = SketchRenderer::genSketch(
                    retrievalList[i].front, 
                    retrievalList[i].up
                    );

            char fn[1024];
            sprintf(fn, "%s/sketch%u.png", folder_name.c_str(), i+1);
            cv::imwrite(fn, sketch*255.0);
            // dbg(sketch);

            cv::Mat1f image = SketchRenderer::genShading(
                    retrievalList[i].front, 
                    retrievalList[i].up
                    );

            sprintf(fn, "%s/model%u.png", folder_name.c_str(), i+1);
            cv::imwrite(fn, image*255.0);
            // dbg(image);
        }

        gettimeofday(&ts, NULL);

        retrievalList = retriever.retrieveTop(input);

        gettimeofday(&te, NULL);

        fprintf(file, "knn_time = %0.3lf ms\n", diff(ts, te)*1e-3);

        for(unsigned i=0; i<10; i++)
        {
            //printf("%s -- %f\n", retrievalList[i].path.c_str(), retrievalList[i].score);
            SketchRenderer::load(retrievalList[i].path.c_str());

            cv::Mat1f sketch = SketchRenderer::genSketch(
                    retrievalList[i].front, 
                    retrievalList[i].up
                    );

            
            
            char fn[1024];
            sprintf(fn, "%s/knn_sketch%u.png", folder_name.c_str(), i+1);
            cv::imwrite(fn, sketch*255.0);
            // dbg(sketch);

            cv::Mat1f image = SketchRenderer::genShading(
                    retrievalList[i].front, 
                    retrievalList[i].up
                    );

            sprintf(fn, "%s/knn_model%u.png", folder_name.c_str(), i+1);
            cv::imwrite(fn, image*255.0);
            // dbg(image);
        }
        fclose(file);
    }
}

#endif	/* HARDCODEFUNCTIONS_H */

