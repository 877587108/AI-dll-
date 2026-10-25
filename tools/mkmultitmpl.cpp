// 从 OK 样本生成多模板库并写入模型目录
#include <opencv2/core.hpp>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif
#include "chardet.hpp"
#include "common/console_utf8.hpp"
namespace fs=std::filesystem;
namespace{
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p,cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);std::streamoff n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read(reinterpret_cast<char*>(b.data()),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3){std::printf("usage: mkmultitmpl <modelDir> <ok image...>\n");return 2;}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm)){std::printf("model fail\n");return 1;}
    cm.templateSize=64;
    std::vector<chardet::CharUnit> base=chardet::toAllUnits(cm);
    const int nChar=(int)base.size();
    cm.charTemplates.assign(nChar,{});
    int nT=0;
    for(size_t i=2;i<a.size();++i){
        cv::Mat g; if(!rd(a[i],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        if(!cm.ref.empty()){
            cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
            cv::Rect ia; for(const auto& u:base) ia=ia.area()>0?(ia|u.bbox):u.bbox;
            ia.x=std::max(0,ia.x-20);ia.y=std::max(0,ia.y-20);
            ia.width=std::min(sig.cols-ia.x,ia.width+40);ia.height=std::min(sig.rows-ia.y,ia.height+40);
            cv::Mat rC,sC; r32(ia).copyTo(rC); s32(ia).copyTo(sC);
            auto sh=cv::phaseCorrelate(sC,rC,cv::noArray());
            if(std::abs(sh.x)<60&&std::abs(sh.y)<60){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
                cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}
        }
        for(int c=0;c<nChar;++c){
            cv::Rect box=base[c].bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(box.width<=4||box.height<=4)continue;
            cv::Mat patch; sig(box).convertTo(patch,CV_32F);
            cv::threshold(patch,patch,0.25,0.0,cv::THRESH_TOZERO);
            cv::Mat r; cv::resize(patch,r,cv::Size(cm.templateSize,cm.templateSize),0,0,cv::INTER_LINEAR);
            double mx=0; cv::minMaxLoc(r,nullptr,&mx);
            if(mx>1e-6) r*=(1.0/mx);
            cm.charTemplates[c].push_back(r);
        }
        ++nT;
    }
    std::printf("从 %d 张样本生成模板库：%d 字符 x %d 张\n",nT,nChar,nT);
    if(!chardet::saveModel(a[1],cm)){std::printf("save fail\n");return 1;}
    std::printf("已写入 %s\n",a[1].c_str());
    return 0;
}