// 逐字符输出：找出改笔画样本中差异最大的字符
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
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    std::vector<chardet::CharUnit> base;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)base.push_back(x);}

    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        std::vector<chardet::CharUnit> units=base;
        // 配准
        if(!cm.ref.empty()){
            cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
            cv::Rect ia; for(const auto& u:units) ia=ia.area()>0?(ia|u.bbox):u.bbox;
            ia.x=std::max(0,ia.x-20);ia.y=std::max(0,ia.y-20);
            ia.width=std::min(sig.cols-ia.x,ia.width+40);ia.height=std::min(sig.rows-ia.y,ia.height+40);
            cv::Mat rC,sC; r32(ia).copyTo(rC); s32(ia).copyTo(sC);
            auto sh=cv::phaseCorrelate(sC,rC,cv::noArray());
            if(std::abs(sh.x)<60&&std::abs(sh.y)<60){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
                cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}
        }
        for(auto& u:units){
            cv::Rect box=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(box.width<=4||box.height<=4)continue;
            cv::Rect sr(box.x-12,box.y-12,box.width+24,box.height+24);
            sr&=cv::Rect(0,0,sig.cols,sig.rows);
            if(sr.width<box.width||sr.height<box.height)continue;
            cv::Mat mS,tS; cm.freq(sr).convertTo(mS,CV_32F,1.0/255.0); sig(sr).convertTo(tS,CV_32F);
            cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-sr.x,box.y-sr.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
            cv::Point mL; double mv=0; cv::minMaxLoc(res,nullptr,&mv,nullptr,&mL);
            if(mv<0.30)continue;
            cv::Rect nb(box.x+mL.x-12,box.y+mL.y-12,box.width,box.height);
            if((nb&cv::Rect(0,0,sig.cols,sig.rows))!=nb)continue;
            u.bbox=nb;
        }
        // 输出每个字符
        fs::path fp=u8p(a[fi]);
        std::string nm((const char*)fp.filename().u8string().c_str());
        std::printf("%s\n", nm.substr(0,34).c_str());
        std::vector<std::pair<double,int>> v;
        for(size_t i=0;i<units.size();++i){
            auto nc=chardet::diffNcc(sig,cm.freq,units[i],cm.config);
            if(!nc.valid)continue;
            v.push_back({std::max(nc.missRatio,nc.extraRatio),(int)i});
        }
        std::sort(v.begin(),v.end(),[](auto&x,auto&y){return x.first>y.first;});
        for(size_t i=0;i<v.size()&&i<4;++i){
            auto nc=chardet::diffNcc(sig,cm.freq,units[v[i].second],cm.config);
            std::printf("  #%-2d miss=%.3f extra=%.3f local=%.3f score=%.2f\n", v[i].second, nc.missRatio, nc.extraRatio, nc.localExtra, nc.score);

        }
    }
    return 0;
}