// 两种配准的置信度对比（相位相关的峰值响应无法直接取，改用配准后模板匹配峰值）
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
        cv::Mat sig0=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig0.convertTo(s32,CV_32F);

        // 字符区
        cv::Rect ia;
        for(auto&u:all){ cv::Rect b=u.bbox; if(b.y>800){ ia|=b; } }
        ia.x=std::max(0,ia.x-20); ia.y=std::max(0,ia.y-20);
        ia.width=std::min(sig0.cols-ia.x,ia.width+40); ia.height=std::min(sig0.rows-ia.y,ia.height+40);

        auto shFull=cv::phaseCorrelate(s32,r32,cv::noArray());
        cv::Mat rC,sC; r32(ia).copyTo(rC); s32(ia).copyTo(sC);
        auto shChar=cv::phaseCorrelate(sC,rC,cv::noArray());

        // 用编号行第一个字符的模板匹配峰值来评价配准质量
        auto evalAfter=[&](double sx,double sy)->double{
            cv::Mat s2;
            cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sx,0,1,-sy);
            cv::warpAffine(sig0,s2,M,sig0.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
            double sumP=0; int cnt=0;
            for(size_t i=0;i<all.size()&&i<8;++i){
                cv::Rect box=all[i].bbox&cv::Rect(0,0,s2.cols,s2.rows);
                if(box.width<=4||box.height<=4)continue;
                cv::Mat mf,tf; cm.freq(box).convertTo(mf,CV_32F,1.0/255.0); s2(box).convertTo(tf,CV_32F);
                cv::Mat res; cv::matchTemplate(tf,mf,res,cv::TM_CCOEFF_NORMED);
                double mx=0; cv::minMaxLoc(res,nullptr,&mx); sumP+=mx; ++cnt;
            }
            return cnt>0?sumP/cnt:0;
        };
        double pFull=evalAfter(shFull.x,shFull.y);
        double pChar=evalAfter(shChar.x,shChar.y);

        fs::path fp=u8p(a[fi]);
        std::printf("%-38s 全图(%+.1f,%+.1f)peak%.3f  字符区(%+.1f,%+.1f)peak%.3f  %s\n",
            std::string((const char*)fp.filename().u8string().c_str()).substr(0,38).c_str(),
            -shFull.x,-shFull.y,pFull,-shChar.x,-shChar.y,pChar,
            pChar>pFull?"→ 用字符区":"→ 用全图");
    }
    return 0;
}