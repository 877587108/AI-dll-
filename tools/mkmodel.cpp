// 把 GUI 的 model.png + regions.conf 转成 chardet 格式，验证格式可用
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
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
namespace fs = std::filesystem;
namespace {
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p, cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3){std::printf("用法: markloc_mkmodel <模型图png> <regions.conf> <输出目录>\n");return 2;}
    cv::Mat freq; if(!rd(a[1],freq)){std::printf("模型图读取失败\n");return 1;}

    chardet::Model m;
    m.view=0; m.imageSize=freq.size(); m.config=chardet::Config();
    m.freq=freq.clone();

    // 解析 regions.conf
    std::ifstream f(a[2]);
    if(!f){std::printf("regions.conf 打不开\n");return 1;}
    std::string line; int nRegion=0;
    while(std::getline(f,line)){ if(!line.empty()&&line[0]!='#'){ std::istringstream iss(line); if(iss>>nRegion) break; } }
    for(int i=0;i<nRegion;i++){
        std::getline(f,line); std::istringstream iss(line);
        chardet::RegionInfo ri; int nChar=0;
        if(!(iss>>ri.rect.x>>ri.rect.y>>ri.rect.width>>ri.rect.height>>nChar)) break;
        ri.index=i;
        for(int k=0;k<nChar;k++){
            std::getline(f,line); std::istringstream cs(line);
            chardet::CharBox cb;
            if(!(cs>>cb.bbox.x>>cb.bbox.y>>cb.bbox.width>>cb.bbox.height)) break;
            cb.extend=cv::Rect(cb.bbox.x-4,cb.bbox.y-4,cb.bbox.width+8,cb.bbox.height+8);
            cb.index=k;
            ri.chars.push_back(cb);
        }
        m.regions.push_back(std::move(ri));
    }
    std::printf("解析: %d 区域, %d 字符\n",(int)m.regions.size(),m.totalChars());
    if(!chardet::saveModel(a[3],m)){std::printf("保存失败\n");return 1;}
    std::printf("已写入 chardet 格式: %s\n",a[3].c_str());

    // 读回验证
    chardet::Model back;
    if(chardet::loadModel(a[3],back)){
        std::printf("读回验证: %d 区域, %d 字符\n",(int)back.regions.size(),back.totalChars());
    } else { std::printf("读回失败\n"); return 1; }
    return 0;
}