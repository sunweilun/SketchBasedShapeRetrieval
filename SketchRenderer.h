/* 
 * File:   SketchRenderer.h
 * Author: swl
 *
 * Created on November 29, 2015, 12:02 PM
 */

#ifndef SKETCHRENDERER_H
#define	SKETCHRENDERER_H

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <vector>

class SketchRenderer {
public:
    static void init();
    static void load(const char* path);
    static cv::Mat1f genShading(const glm::vec3& front, const glm::vec3& up);
    static cv::Mat1f genSketch(const glm::vec3& front, const glm::vec3& up);
protected:
    static void renderView(const glm::vec3& front, const glm::vec3& up);
    static void render();
    static std::vector<glm::vec3> vertices;
    static GLuint listID;
    static GLuint colorTexID;
    static GLuint depthBufferID;
    static GLuint frameBufferID;
    static GLuint renderSize;
    static float shrinkage;
    static float depthThres;
    static float normalAngleThres; // in degrees
    static void loadOFF(const char* path);
};

#endif	/* SKETCHRENDERER_H */

