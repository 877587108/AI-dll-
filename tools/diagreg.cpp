// 只用字符区域做相位相关配准，看能否得到 +8 的平移
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
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
bool rd(const std::string& p,cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}

    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);

        // 全图相位相关
        cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
        auto shFull=cv::phaseCorrelate(s32,r32,cv::noArray());

        // 只用字符区域（编号行）做配准
        cv::Rect cbox=all[0].bbox;
        for(auto&u:all){ cv::Rect b=u.bbox; if(b.y>800){ cbox|=b; } }   // 下方两行
        cbox.x=std::max(0,cbox.x-20); cbox.y=std::max(0,cbox.y-20);
        cbox.width=std::min(sig.cols-cbox.x,cbox.width+40); cbox.height=std::min(sig.rows-cbox.y,cbox.height+40);

        cv::Mat rC,sC; r32(cbox).copyTo(rC); s32(cbox).copyTo(sC);
        auto shChar=cv::phaseCorrelate(sC,rC,cv::noArray());

        fs::path fp=u8p(a[fi]);
        std::printf("%-42s 全图(%+.1f,%+.1f)  仅字符区(%+.1f,%+.1f)\n",
            std::string((const char*)fp.filename().u8string().c_str()).substr(0,42).c_str(),
            -shFull.x,-shFull.y,-shChar.x,-shChar.y);
    }
    return 0;
}