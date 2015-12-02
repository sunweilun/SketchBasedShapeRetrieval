/* 
 * File:   SketchRenderer.cpp
 * Author: swl
 * 
 * Created on November 29, 2015, 12:02 PM
 */


#include <GL/glew.h>

#include "SketchRenderer.h"

GLuint SketchRenderer::colorTexID = 0;
GLuint SketchRenderer::depthBufferID = 0;
GLuint SketchRenderer::frameBufferID = 0;
GLuint SketchRenderer::listID = 0;
GLuint SketchRenderer::renderSize = 256;
float SketchRenderer::shrinkage = 0.9;
float SketchRenderer::depthThres = 0.01;
float SketchRenderer::normalAngleThres = 60;
std::vector<glm::vec3> SketchRenderer::vertices;

void SketchRenderer::init()
{
    int ac = 0;
    char** av;
    glutInit(&ac, av);
    glutInitWindowSize(renderSize, renderSize);
    glutCreateWindow("SketchRenderer");
    glewInit();
    listID = glGenLists(1);
    glutDisplayFunc(render);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGB);
    glClearColor(0, 0, 0, 0);
    
    glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
    glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
    glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
    
    glGenTextures(1, &colorTexID);
    glBindTexture(GL_TEXTURE_2D, colorTexID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //NULL means reserve texture memory, but texels are undefined
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, renderSize, renderSize, 
            0, GL_RGB, GL_FLOAT, NULL);
    //-------------------------
    glGenFramebuffers(1, &frameBufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    //Attach 2D texture to this FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexID, 0);
    //-------------------------
    glGenRenderbuffers(1, &depthBufferID);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, 
            renderSize, renderSize);
    //-------------------------
    //Attach depth buffer to FBO
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBufferID);
    glutHideWindow();
}

void SketchRenderer::load(const char* path)
{
    loadOFF(path);
}

void SketchRenderer::renderView(const glm::vec3& front, const glm::vec3& up)
{
    glm::vec3 right = glm::normalize(glm::cross(front, up));
    glm::vec3 rectUp = glm::cross(right, front); 
    glm::vec2 xb, yb;
    for (unsigned i=0; i<vertices.size(); i++)
    {
        float x = glm::dot(vertices[i], right);
        float y = glm::dot(vertices[i], rectUp);
        if (i == 0)
        {
            xb = glm::vec2(x, x);
            yb = glm::vec2(y, y);
        }
        xb[0] = std::min(x, xb[0]);
        xb[1] = std::max(x, xb[1]);
        yb[0] = std::min(y, yb[0]);
        yb[1] = std::max(y, yb[1]);
    }
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    float dx = xb[1] - xb[0];
    float dy = yb[1] - yb[0];
    float len = std::max(dx, dy) / shrinkage;
    glOrtho(-len*0.5, len*0.5, -len*0.5, len*0.5, -5, 5);
    
    float xc = (xb[0] + xb[1])*0.5;
    float yc = (yb[0] + yb[1])*0.5;
    glm::vec3 eye = xc*right+yc*rectUp;
    glm::vec3 center = eye+front;
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye.x, eye.y, eye.z, 
            center.x, center.y, center.z,
            rectUp.x, rectUp.y, rectUp.z);
    
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glCallList(listID);
    glDisable(GL_DEPTH_TEST);
    glFlush();
}

cv::Mat1f SketchRenderer::genShading(const glm::vec3& front, const glm::vec3& up)
{
    cv::Mat1f image(renderSize, renderSize);
    image.setTo(0.0);
    renderView(front, up);
    std::vector<glm::vec3> normalMap(renderSize*renderSize);
    glReadPixels(0, 0, renderSize, renderSize, GL_RGB, GL_FLOAT, normalMap.data());
    
    for(unsigned y=0; y<renderSize; y++)
    {
        for(unsigned x=0; x<renderSize; x++)
        {
            glm::vec3 n = normalMap[(renderSize-1-y)*renderSize+x];
            if(glm::length(n) < 0 || std::isnan(n.x))
                continue;
            n = glm::normalize(n);
            image.at<float>(y, x) = fabs(glm::dot(n, front));
        }
    }
    return image;
}

cv::Mat1f SketchRenderer::genSketch(const glm::vec3& front, const glm::vec3& up)
{
    cv::Mat1f sketch(renderSize, renderSize);
    sketch.setTo(0.0);
    
    renderView(front, up);
    std::vector<glm::vec3> normalMap(renderSize*renderSize);
    glReadPixels(0, 0, renderSize, renderSize, GL_RGB, GL_FLOAT, normalMap.data());
    std::vector<float> depthMap(renderSize*renderSize);
    glReadPixels(0, 0, renderSize, renderSize, GL_DEPTH_COMPONENT, GL_FLOAT, depthMap.data());
    
    std::vector<glm::ivec2> offList;
    offList.push_back(glm::ivec2(-1, -1));
    offList.push_back(glm::ivec2(-1, 0));
    offList.push_back(glm::ivec2(-1, 1));
    offList.push_back(glm::ivec2(0, -1));
    offList.push_back(glm::ivec2(0, 1));
    offList.push_back(glm::ivec2(1, -1));
    offList.push_back(glm::ivec2(1, 0));
    offList.push_back(glm::ivec2(1, 1));
    
    for(unsigned y=1; y+1<renderSize; y++)
    {
        for(unsigned x=1; x+1<renderSize; x++)
        {
            const glm::vec3 &n = normalMap[(renderSize-1-y)*renderSize+x];
            const float &d = depthMap[(renderSize-1-y)*renderSize+x];
            for(unsigned k=0; k<offList.size(); k++)
            {
                unsigned nx = x+offList[k].x;
                unsigned ny = y+offList[k].y;
                const glm::vec3 &nn = normalMap[(renderSize-1-ny)*renderSize+nx];
                const float &nd = depthMap[(renderSize-1-ny)*renderSize+nx];
                
                if(fabs(nd-d) > depthThres)
                {
                    sketch.at<float>(y, x) = 1.0f;
                }
                else if(glm::length(n) > 0.5)
                {
                    if(glm::length(nn) < 0.5 || 
                            acosf(glm::dot(n, nn))*180*M_1_PI > normalAngleThres)
                        sketch.at<float>(y, x) = 1.0f;
                }
            }
        }
    }
    return sketch;
}

void SketchRenderer::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, colorTexID);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glFlush();
}

void SketchRenderer::loadOFF(const char* path)
{
    FILE* file = fopen(path, "r");
    char line[1024];
    fgets(line, 1024, file);
    fgets(line, 1024, file);
    unsigned nv, nf;
    sscanf(line, "%u %u", &nv, &nf);
    vertices.resize(nv);
    glm::vec3 ub, lb;
    for(unsigned i=0; i<nv; i++)
    {
        glm::vec3& v = vertices[i];
        fgets(line, 1024, file);
        sscanf(line, "%f %f %f", &v.x, &v.y, &v.z);
        if(i == 0)
        {
            ub = lb = v;
        }
        else
        {
            ub = glm::max(ub, v);
            lb = glm::min(lb, v);
        }
    }
    glm::vec3 db = ub-lb;
    glm::vec3 cb = (ub+lb)*0.5f;
    float d = std::max(db.x, std::max(db.y, db.z));
    for(unsigned i=0; i<nv; i++)
    {
        vertices[i] = (vertices[i] - cb) / d;
    }
    glNewList(listID, GL_COMPILE);
    
    
    for(unsigned i=0; i<nf; i++)
    {
        glBegin(GL_TRIANGLES);
        glm::uvec3 f;
        unsigned num = 0;
        fscanf(file, "%u", &num);
        fscanf(file, "%u %u %u", &f.x, &f.y, &f.z);
        
        for(unsigned j=0; j+3<=num; j++)
        {
            glm::vec3 normal = glm::cross(vertices[f[1]] - vertices[f[0]],
                vertices[f[2]] - vertices[f[1]]);
            normal = glm::normalize(normal);
            if(std::isnan(normal.x))
                normal = glm::vec3(0, 0, 0);
            for(unsigned k=0; k<3; k++)
            {
                glColor3f(normal.x, normal.y, normal.z);
                glVertex3fv((float*)&vertices[f[k]]);
            }
            f.y = f.z;
            if(j+3 != num)
                fscanf(file, "%u", &f.z);
        }
        //fgets(line, 1024, file);
        glEnd();
    }
    
    
    glEndList();
    fclose(file);
}